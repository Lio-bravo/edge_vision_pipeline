#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Accuracy against ground truth: mAP of the framework model and of the ncnn bundle (fp32 / int8) on a labeled
YOLO split, all through ONE metric implementation so the numbers are comparable (verify.py compares models with
each other; this module compares them with the labels).

Metric: COCO-style AP (101-point interpolated precision envelope), greedy per-class matching by score, IoU
thresholds 0.50:0.95. mAP is computed from low-confidence predictions (`map_conf`); precision / recall / F1 are
reported at the deployment confidence (`conf`), which is what the board will see.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import DataError
from ..utils import list_images, log

IOU_THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)


@dataclass(frozen=True)
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes, shapes (N,4) x (M,4) -> (N,M)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), np.float64)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def _match(preds: list[Box], gts: list[Box], thr: float) -> np.ndarray:
    """Greedy matching of score-sorted preds to unmatched GT of one image and class; returns TP flags."""
    tp = np.zeros(len(preds), bool)
    if not preds or not gts:
        return tp
    ious = iou_matrix(np.array([[p.x1, p.y1, p.x2, p.y2] for p in preds]),
                      np.array([[g.x1, g.y1, g.x2, g.y2] for g in gts]))
    used = np.zeros(len(gts), bool)
    for i in range(len(preds)):
        cand = np.where(~used, ious[i], -1.0)
        j = int(np.argmax(cand))
        if cand[j] >= thr:
            used[j] = True
            tp[i] = True
    return tp


def average_precision(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    """COCO 101-point interpolated AP from TP flags and scores of all predictions of one class."""
    if n_gt == 0:
        return float("nan")
    if tp.size == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tpc = np.cumsum(tp[order])
    fpc = np.cumsum(~tp[order])
    recall = tpc / n_gt
    precision = tpc / np.maximum(tpc + fpc, 1e-12)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, rs, side="left")
    return float(np.mean([envelope[i] if i < len(envelope) else 0.0 for i in idx]))


def evaluate(preds: dict[str, list[Box]], gts: dict[str, list[Box]], names: Sequence[str],
             conf: float = 0.25) -> dict:
    """mAP50, mAP50-95, per-class AP50 and P/R/F1 (IoU 0.5, score >= conf). Images missing from `preds` count
    as images with no predictions. Classes without GT are excluded from the means (AP is undefined)."""
    nc = len(names)
    if nc == 0:
        raise DataError("evaluate: class names are empty")
    for k, boxes in list(preds.items()) + list(gts.items()):
        for b in boxes:
            if not 0 <= b.cls < nc:
                raise DataError(f"evaluate: class id {b.cls} out of range [0,{nc}) in {k}")
    n_gt = np.zeros(nc, int)
    for boxes in gts.values():
        for b in boxes:
            n_gt[b.cls] += 1
    ap = np.full((nc, len(IOU_THRS)), np.nan)
    tp_at_conf = fp_at_conf = 0
    for c in range(nc):
        scores, flags = [], [[] for _ in IOU_THRS]
        for img in gts.keys() | preds.keys():
            p = sorted((b for b in preds.get(img, []) if b.cls == c), key=lambda b: -b.score)
            g = [b for b in gts.get(img, []) if b.cls == c]
            scores += [b.score for b in p]
            for t, thr in enumerate(IOU_THRS):
                m = _match(p, g, float(thr))
                flags[t].append(m)
                if t == 0:
                    keep = np.array([b.score >= conf for b in p], bool)
                    tp_at_conf += int((m & keep).sum())
                    fp_at_conf += int((~m & keep).sum())
        s = np.array(scores, np.float64)
        for t in range(len(IOU_THRS)):
            f = np.concatenate(flags[t]) if flags[t] else np.zeros(0, bool)
            ap[c, t] = average_precision(f, s, int(n_gt[c]))
    valid = n_gt > 0
    total_gt = int(n_gt.sum())
    precision = tp_at_conf / (tp_at_conf + fp_at_conf) if tp_at_conf + fp_at_conf else 0.0
    recall = tp_at_conf / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "images": len(gts),
        "instances": total_gt,
        "mAP50": float(np.nanmean(ap[valid, 0])) if valid.any() else None,
        "mAP50_95": float(np.nanmean(ap[valid])) if valid.any() else None,
        "precision": precision, "recall": recall, "f1": f1, "conf": conf,
        "per_class": {names[c]: {"instances": int(n_gt[c]),
                                 "AP50": None if np.isnan(ap[c, 0]) else float(ap[c, 0]),
                                 "AP50_95": None if np.isnan(ap[c]).all() else float(np.nanmean(ap[c]))}
                      for c in range(nc)},
    }


def load_yolo_split(root: Path, split: str = "val") -> tuple[list[Path], dict[str, list[Box]]]:
    """images/<split> + labels/<split> (normalized cx cy w h) -> image list and pixel GT keyed by image stem.
    An image without a label file has no objects."""
    from ..data.labelme import image_size

    img_dir, lbl_dir = Path(root) / "images" / split, Path(root) / "labels" / split
    images = list_images(img_dir) if img_dir.is_dir() else []
    if not images:
        raise DataError(f"no images in {img_dir}",
                        "point --data at a YOLO dataset root (images/<split>, labels/<split>)")
    gts: dict[str, list[Box]] = {}
    for p in images:
        w, h = image_size(p)
        boxes = []
        lf = lbl_dir / f"{p.stem}.txt"
        if lf.is_file():
            for ln, line in enumerate(lf.read_text(encoding="utf-8").splitlines(), 1):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 5:
                    raise DataError(f"{lf}:{ln}: expected 5 fields (detection labels), got {len(parts)}")
                c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                boxes.append(Box(c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        gts[p.stem] = boxes
    return images, gts


def class_map(model_names: Sequence[str], names: Sequence[str]) -> dict[int, int]:
    """Model class id -> dataset class id by name; error if no overlap (the same rule as prelabel)."""
    pos = {n: i for i, n in enumerate(names)}
    m = {i: pos[n] for i, n in enumerate(model_names) if n in pos}
    if not m:
        raise DataError(f"model classes {list(model_names)[:10]} share no name with dataset classes {list(names)}")
    return m


def predict_bundle(bundle: Path, precision: str, threads: int = 1,
                   map_conf: float = 0.001) -> tuple[Callable, list[str]]:
    from .runner import NcnnPipeline, read_image

    pipe = NcnnPipeline.from_bundle(Path(bundle), precision, num_threads=threads)
    if pipe.meta.task != "detect":
        raise DataError(f"evaluate needs a detection bundle, got family={pipe.meta.family}")
    pipe.post.conf = map_conf
    pipe.post.max_det = max(pipe.post.max_det, 300)

    def fn(p: Path) -> list[Box]:
        return [Box(d.cls, d.x1, d.y1, d.x2, d.y2, d.score) for d in pipe.detect(read_image(p))]

    return fn, list(pipe.meta.labels)


def predict_ultralytics(weights: Path, imgsz: int, map_conf: float = 0.001, device: str = "cpu",
                        iou: float = 0.45) -> tuple[Callable, list[str]]:
    from ultralytics import YOLO

    model = YOLO(str(weights))

    def fn(p: Path) -> list[Box]:
        r = model.predict(source=str(p), conf=map_conf, iou=iou, imgsz=imgsz, device=device, max_det=300,
                          verbose=False)[0]
        if r.boxes is None or not len(r.boxes):
            return []
        return [Box(int(c), *xyxy, float(s)) for xyxy, c, s in
                zip(r.boxes.xyxy.cpu().tolist(), r.boxes.cls.cpu().tolist(), r.boxes.conf.cpu().tolist())]

    names = model.names
    return fn, [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def run_eval(predict: Callable[[Path], list[Box]], model_names: Sequence[str], images: Sequence[Path],
             gts: dict[str, list[Box]], names: Sequence[str], conf: float) -> dict:
    """Predict every image, remap classes by name, evaluate; adds mean latency per image (ms, this host)."""
    cmap = class_map(model_names, names)
    preds: dict[str, list[Box]] = {}
    t = []
    for p in images:
        t0 = time.perf_counter()
        boxes = predict(p)
        t.append((time.perf_counter() - t0) * 1000)
        preds[p.stem] = [Box(cmap[b.cls], b.x1, b.y1, b.x2, b.y2, b.score) for b in boxes if b.cls in cmap]
    res = evaluate(preds, gts, names, conf)
    res["latency_ms"] = float(np.mean(t)) if t else None
    return res


def render_eval_markdown(results: dict[str, dict], names: Sequence[str]) -> str:
    def f(v):
        return "n/a" if v is None else f"{v:.3f}"

    lines = ["| model | mAP50 | mAP50-95 | P | R | F1 | ms/img (host) |", "|---|---|---|---|---|---|---|"]
    for k, r in results.items():
        lines.append(f"| {k} | {f(r['mAP50'])} | {f(r['mAP50_95'])} | {f(r['precision'])} | {f(r['recall'])} | "
                     f"{f(r['f1'])} | {r['latency_ms']:.1f} |")
    lines += ["", "| class | instances | " + " | ".join(f"AP50 {k}" for k in results) + " |",
              "|---|---|" + "---|" * len(results)]
    first = next(iter(results.values()))
    for n in names:
        lines.append(f"| {n} | {first['per_class'][n]['instances']} | "
                     + " | ".join(f(r["per_class"][n]["AP50"]) for r in results.values()) + " |")
    conf = first["conf"]
    lines += ["", f"mAP from predictions with conf >= 0.001; P/R/F1 at conf >= {conf} (IoU 0.5). "
              "Latency is the Python reference runtime on this host, not the board."]
    return "\n".join(lines) + "\n"


def log_eval(name: str, r: dict) -> None:
    log.info("eval %-8s mAP50=%s mAP50-95=%s P=%.3f R=%.3f (%d images, %d instances)", name,
             "n/a" if r["mAP50"] is None else f"{r['mAP50']:.3f}",
             "n/a" if r["mAP50_95"] is None else f"{r['mAP50_95']:.3f}", r["precision"], r["recall"], r["images"],
             r["instances"])
