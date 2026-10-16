#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Verification ladder for a deploy bundle.

L1  conversion : framework (torch / onnxruntime) vs ncnn fp32 on identical input tensors (cosine)
L2  quantization: ncnn int8 vs ncnn fp32 (cosine per output)
L3  end-to-end : ncnn + paired post-processing vs the framework's own predict() on real images
                 (detections matched by class + IoU; classify top-1 agreement)
L3c C++ runtime: same images through the C++ demo, compared with the Python pipeline (see cpp_check)
Host latency is reported for orientation only; board latency must be measured on the board.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..utils import log, write_json
from .meta import ModelMeta
from .postprocess import Detection
from .quant import mean_similarity, output_similarity
from .runner import NcnnPipeline, read_image

CONFIDENT = 0.5   # classify: reference top-1 probability above which top-1 must match exactly


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    if a.shape != b.shape:
        return -1.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    return float(a @ b / (na * nb))


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class MatchStats:
    matched: int
    n_a: int
    n_b: int
    mean_iou: float
    max_score_diff: float

    @property
    def f1(self) -> float:
        """2*matched / (n_a + n_b); 1.0 when both are empty."""
        tot = self.n_a + self.n_b
        return 1.0 if tot == 0 else 2.0 * self.matched / tot


def match_detections(a: Sequence[Detection], b: Sequence[Detection], iou_thr: float = 0.5) -> MatchStats:
    """Greedy one-to-one matching (same class, highest IoU first)."""
    pairs = []
    for i, da in enumerate(a):
        for j, db in enumerate(b):
            if da.cls != db.cls:
                continue
            iou = box_iou((da.x1, da.y1, da.x2, da.y2), (db.x1, db.y1, db.x2, db.y2))
            if iou >= iou_thr:
                pairs.append((iou, i, j))
    pairs.sort(key=lambda t: -t[0])
    used_a, used_b, ious, sdiff = set(), set(), [], []
    for iou, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        ious.append(iou)
        sdiff.append(abs(a[i].score - b[j].score))
    return MatchStats(len(ious), len(a), len(b), float(np.mean(ious)) if ious else 1.0,
                      float(max(sdiff)) if sdiff else 0.0)


def _agg(stats: list[MatchStats]) -> dict:
    if not stats:
        return {}
    return {"images": len(stats), "f1_mean": round(float(np.mean([s.f1 for s in stats])), 4),
            "f1_min": round(float(min(s.f1 for s in stats)), 4),
            "matched": int(sum(s.matched for s in stats)), "dets_a": int(sum(s.n_a for s in stats)),
            "dets_b": int(sum(s.n_b for s in stats)),
            "mean_iou": round(float(np.mean([s.mean_iou for s in stats])), 4),
            "max_score_diff": round(float(max(s.max_score_diff for s in stats)), 4)}


# ------------------------------------------------------------------ references
class UltralyticsReference:
    """Framework ground truth: the original .pt run through Ultralytics predict with matching settings."""

    def __init__(self, weights: Path, meta: ModelMeta):
        from ultralytics import YOLO

        self.yolo = YOLO(str(weights))
        self.meta = meta
        self.e2e = meta.post.method == "e2e_topk"

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        p = self.meta.post
        r = self.yolo.predict(img_bgr, imgsz=[self.meta.input.h, self.meta.input.w], conf=p.conf, iou=p.iou,
                              max_det=p.max_det, agnostic_nms=p.agnostic, nms=False if self.e2e else None,
                              rect=False, device="cpu", verbose=False)[0]
        b = r.boxes
        xyxy, conf, cls = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()
        return [Detection(*map(float, xyxy[i]), float(conf[i]), int(cls[i])) for i in range(len(conf))]

    def classify(self, img_bgr: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        r = self.yolo.predict(img_bgr, imgsz=self.meta.input.h, device="cpu", verbose=False)[0]
        probs = r.probs.data.cpu().numpy()
        order = np.argsort(-probs, kind="stable")[:k]
        return [(int(i), float(probs[i])) for i in order]


def torch_forward_fn(module) -> Callable[[np.ndarray], list[np.ndarray]]:
    from .export import torch_reference

    return lambda chw: torch_reference(module, chw)


def onnx_forward_fn(onnx_path: Path) -> Callable[[np.ndarray], list[np.ndarray]]:
    """L1 reference for generic ONNX: the prepared (post-rules) ONNX on the same CHW tensor ncnn gets."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    def fn(chw: np.ndarray) -> list[np.ndarray]:
        outs = sess.run(None, {name: chw[None].astype(np.float32)})
        return [np.squeeze(o, 0) if o.ndim >= 2 and o.shape[0] == 1 else o for o in outs]

    return fn


# ------------------------------------------------------------------ main
def verify_bundle(bundle_dir: Path, images: Sequence[Path], reference_fn: Callable | None = None,
                  ul_weights: Path | None = None, min_fp32_cos: float = 0.999, min_int8_cos: float = 0.93,
                  min_e2e_match: float = 0.9, int8_iou: float = 0.5, fp32_iou: float = 0.8) -> dict[str, Any]:
    """Run L1-L3 on `images`. Returns a report dict with `passed` and per-check details."""
    from .preprocess import preprocess_ncnn

    bundle_dir = Path(bundle_dir)
    fp32 = NcnnPipeline.from_bundle(bundle_dir, "fp32")
    meta = fp32.meta
    int8 = NcnnPipeline.from_bundle(bundle_dir, "int8") if "int8" in meta.files else None
    ref = UltralyticsReference(ul_weights, meta) if ul_weights and meta.family in ("yolo", "classify") else None
    out_names = [o.name for o in meta.outputs]
    report: dict[str, Any] = {"bundle": str(bundle_dir), "model": meta.name, "family": meta.family,
                              "images": len(images), "checks": {}, "notes": []}
    l1, l2 = [[] for _ in out_names], [[] for _ in out_names]
    l2_task: list[float] = []
    e2e_fp32, e2e_int8, e2e_int8_vs_ref, cls_agree, cls_int8 = [], [], [], [], []
    cls_in_top5, cls_confident = [], []
    lat = {"fp32": [], "int8": []}
    used = 0
    for p in images:
        try:
            img = read_image(p)
        except Exception as exc:  # noqa: BLE001
            report["notes"].append(f"skip {p}: {exc}")
            continue
        used += 1
        mat, g = preprocess_ncnn(img, meta.input)
        chw = np.array(mat, dtype=np.float32)
        t = time.perf_counter()
        o32 = fp32.net.forward(chw, meta.input.blob, out_names)
        lat["fp32"].append(time.perf_counter() - t)
        if reference_fn is not None:
            for k, (a, b) in enumerate(zip(reference_fn(chw), o32)):
                l1[k].append(cosine(a, b))
        if int8 is not None:
            t = time.perf_counter()
            o8 = int8.net.forward(chw, meta.input.blob, out_names)
            lat["int8"].append(time.perf_counter() - t)
            for k, (a, b) in enumerate(zip(o32, o8)):
                l2[k].append(cosine(a, b))
            l2_task.append(output_similarity(o8, o32, meta.box_channels if meta.task == "detect" else None))
        if meta.task == "detect":
            d32 = fp32.detect(img)
            d8 = int8.detect(img) if int8 else None
            if ref is not None:
                dref = ref.detect(img)
                e2e_fp32.append(match_detections(dref, d32, fp32_iou))
                if d8 is not None:
                    e2e_int8_vs_ref.append(match_detections(dref, d8, int8_iou))
            if d8 is not None:
                e2e_int8.append(match_detections(d32, d8, int8_iou))
        elif meta.family == "classify":
            t32 = fp32.classify(img)
            if ref is not None:
                r5 = ref.classify(img, k=5)
                cls_agree.append(int(r5[0][0] == t32[0][0]))
                cls_in_top5.append(int(t32[0][0] in [c for c, _ in r5]))
                if r5[0][1] >= CONFIDENT:
                    cls_confident.append(int(r5[0][0] == t32[0][0]))
            if int8 is not None:
                cls_int8.append(int(int8.classify(img)[0][0] == t32[0][0]))
    report["images_used"] = used
    checks = report["checks"]

    def add(name: str, value: float | None, thr: float, detail: Any = None, group: str = "fp32", gate: bool = True,
            measured: bool = True):
        # measured=False: the check ran but had nothing to compare (e.g. no objects on either side);
        # recorded with passed=None so it can neither pass nor fail a group on its own
        if not measured:
            checks[name] = {"value": None, "threshold": thr, "passed": None, "group": group, "gate": gate,
                            "detail": detail}
            return
        if value is None:
            return
        checks[name] = {"value": round(float(value), 6), "threshold": thr, "passed": bool(value >= thr),
                        "group": group, "gate": gate, "detail": detail}

    def add_match(name: str, stats: list[MatchStats], thr: float, group: str = "fp32", gate: bool = True):
        a = _agg(stats)
        empty = a["dets_a"] + a["dets_b"] == 0
        add(name, a["f1_mean"], thr, a, group=group, gate=gate, measured=not empty)
        if empty:
            report["notes"].append(f"{name}: neither side detected anything at conf={meta.post.conf} on the "
                                   "verification images, so detection agreement is not measurable")

    if any(l1):
        per = [float(np.min(v)) for v in l1]
        add("L1_fp32_vs_framework_cos_min", min(per), min_fp32_cos, {"per_output_min": per})
    if l2_task:
        per = [float(np.mean(v)) for v in l2]
        metric = "detection-aware (relevant class scores + object boxes)" if meta.task == "detect" else "cosine"
        informative = sum(x is not None for x in l2_task)
        add("L2_int8_vs_fp32_similarity", mean_similarity(l2_task) if informative else None, min_int8_cos,
            {"metric": metric, "raw_cos_per_output_mean": per, "images_with_signal": informative}, group="int8",
            measured=informative > 0)
        if informative == 0:
            report["notes"].append("no verification image contains objects the model responds to: "
                                   "L2 int8 similarity is not measurable")
    if e2e_fp32:
        add_match("L3_fp32_vs_ultralytics_f1", e2e_fp32, min_e2e_match)
    if e2e_int8:
        add_match("L3_int8_vs_fp32_f1", e2e_int8, min_e2e_match - 0.1, group="int8", gate=False)
    if e2e_int8_vs_ref:
        add_match("L3_int8_vs_ultralytics_f1", e2e_int8_vs_ref, min_e2e_match - 0.1, group="int8", gate=False)
    if cls_agree:
        # Ultralytics resizes with antialias, ncnn bilinear does not: low-confidence images may swap top-1/top-2,
        # so the gate is "our top-1 within the reference top-5" + "exact top-1 where the reference is confident"
        add("L3_fp32_top1_in_ultralytics_top5", float(np.mean(cls_in_top5)), min_e2e_match, None)
        if cls_confident:
            add("L3_fp32_vs_ultralytics_top1_agree_confident", float(np.mean(cls_confident)), min_e2e_match,
                {"images": len(cls_confident), "reference_min_prob": CONFIDENT})
        add("L3_fp32_vs_ultralytics_top1_agree", float(np.mean(cls_agree)), min_e2e_match, None, gate=False)
        report["notes"].append("classify: Ultralytics resizes with antialias (PIL); ncnn bilinear does not, "
                               "so top-1 may swap with top-2 on low-confidence images (strict top-1 is informational)")
    if cls_int8:
        add("L3_int8_vs_fp32_top1_agree", float(np.mean(cls_int8)), min_e2e_match - 0.1, None, group="int8",
            gate=False)
    report["host_latency_ms"] = {k: round(1000 * float(np.median(v)), 2) for k, v in lat.items() if v}
    if used == 0:
        report["notes"].append("no readable verification images")
    finalize_verdict(report, has_int8=int8 is not None)
    if int8 is not None and report["passed_int8"] is False and meta.post.method == "e2e_topk":
        report["notes"].append("YOLO26 end-to-end (one-to-one) heads are quantization-sensitive: ship fp32, or "
                               "redeploy with deploy.e2e: false (NMS head) which usually quantizes better")
    for name, c in checks.items():
        log.info("%-40s %s (thr %.3f) %s%s", name, "   n/a" if c["value"] is None else f"{c['value']:.4f}",
                 c["threshold"], _result(c["passed"]), "" if c["gate"] else " [info]")
    return report


def _result(passed: bool | None) -> str:
    return "UNMEASURED" if passed is None else ("PASS" if passed else "FAIL")


def finalize_verdict(report: dict, has_int8: bool) -> dict:
    """passed_fp32 / passed_int8 from the gated checks of each group; `passed` = fp32 deliverable is valid;
    `recommended_precision` = int8 if it passed, else fp32.

    A group passes only if every *measured* gated check passes and at least one gated check was measured.
    Unmeasured checks (passed=None) never count as a pass: int8 without any measured gate is listed in
    report["unverified"] and reported as not passed, so a model that detects nothing cannot ship as int8."""
    checks = report.get("checks", {})
    used = report.get("images_used", 0) > 0
    notes = report.setdefault("notes", [])

    def note(text: str) -> None:
        if text not in notes:
            notes.append(text)

    def gated(g: str) -> list[dict]:
        return [c for c in checks.values() if c.get("group") == g and c.get("gate", True)]

    def group_ok(g: str) -> bool:
        return all(c["passed"] for c in gated(g) if c["passed"] is not None)

    def measured(g: str) -> bool:
        return any(c["passed"] is not None for c in gated(g))

    unverified = []
    has_fp32_gate = measured("fp32")
    if used and not has_fp32_gate:
        note("no gated fp32 check was measured (no framework reference, or nothing to compare): "
             "fp32 conversion is unverified")
    report["passed_fp32"] = used and has_fp32_gate and group_ok("fp32") and group_ok("cpp_fp32")
    if has_int8:
        int8_ok = used and report["passed_fp32"] and group_ok("int8") and group_ok("cpp_int8")
        if int8_ok and not measured("int8"):
            unverified.append("int8")
            note("int8 could not be verified (no gated int8 check was measurable): recommending fp32; rerun "
                 "`evp verify --bundle <bundle> --images <dir>` with representative images containing objects")
        report["passed_int8"] = int8_ok and measured("int8")
    else:
        report["passed_int8"] = None
    report["unverified"] = unverified
    report["passed"] = report["passed_fp32"]
    report["recommended_precision"] = "int8" if report["passed_int8"] else "fp32"
    return report


def write_report(report: dict, out_dir: Path, extra: dict | None = None) -> tuple[Path, Path]:
    data = dict(report)
    if extra:
        data.update(extra)
    jp = write_json(Path(out_dir) / "report.json", data)
    mp = Path(out_dir) / "report.md"
    mp.write_text(render_markdown(data), encoding="utf-8", newline="\n")
    return jp, mp


def render_markdown(r: dict) -> str:
    def verdict(v, group: str) -> str:
        if group in (r.get("unverified") or []):
            return "UNVERIFIED"
        return "not built" if v is None else ("PASS" if v else "FAIL")

    fp32 = verdict(r.get("passed_fp32", r.get("passed")), "fp32")
    lines = [f"# Deploy report: {r.get('model', '?')}", "",
             f"- family: `{r.get('family')}`", f"- verification images: {r.get('images_used', 0)}",
             f"- fp32: **{fp32}**, int8: **{verdict(r.get('passed_int8'), 'int8')}**",
             f"- recommended precision: **{r.get('recommended_precision', 'fp32')}**", ""]
    if r.get("checks"):
        lines += ["| check | value | threshold | result |", "|---|---|---|---|"]
        for name, c in r["checks"].items():
            res = _result(c["passed"]) + ("" if c.get("gate", True) else " (info)")
            value = "n/a" if c["value"] is None else f"{c['value']:.4f}"
            lines.append(f"| {name} | {value} | {c['threshold']} | {res} |")
        lines.append("")
    if r.get("host_latency_ms"):
        lines += ["Host latency (x86, 1 thread, orientation only): "
                  + ", ".join(f"{k} {v} ms" for k, v in r["host_latency_ms"].items()), ""]
    q = r.get("quant")
    if q:
        lines += ["## INT8", "", f"- method: {q.get('method')}, calibration images: {q.get('calib_count')}",
                  f"- cosine all-int8: {q.get('cos_full_int8')}, final: {q.get('cos_final')} "
                  f"(target {q.get('target_cos')}, met: {q.get('target_met')})",
                  f"- fp32 layers ({len(q.get('fp32_layers', []))}): {', '.join(q.get('fp32_layers', [])) or '-'}", ""]
    c = r.get("convert")
    if c:
        lines += ["## Conversion", "", f"- ncnn layers: {c.get('layer_count')}",
                  "- layer types: " + ", ".join(f"{k}x{v}" for k, v in (c.get("layer_types") or {}).items()),
                  "- ONNX rules applied: " + (", ".join(c.get("rules_applied") or []) or "none"), ""]
    cpp = r.get("cpp")
    if cpp:
        lines += ["## C++ runtime", "", f"- result: **{'PASS' if cpp.get('passed') else 'FAIL'}**",
                  f"- {cpp.get('summary', '')}", ""]
    if r.get("notes"):
        lines += ["## Notes", ""] + [f"- {n}" for n in r["notes"]] + [""]
    return "\n".join(lines)
