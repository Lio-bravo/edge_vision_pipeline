#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Workbench back end: everything the web UI does, without importing gradio (unit-testable).

The UI never keeps project data in memory: every action reloads the project YAML, so edits made on disk or
by the CLI are always picked up, and the CLI and the workbench can be used on the same project.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_config
from ..data.labelme import LABELME_VERSION, find_image_for_json, image_size
from ..data.merge import list_split_images
from ..errors import ConfigError, DataError, EvpError
from ..utils import IMAGE_EXTS, ensure_dir, list_images, log, read_json, write_json

TEMPLATE = Path(__file__).resolve().parents[2] / "configs" / "project.example.yaml"
REVIEW_FILE = ".evp_review"  # must not end in .json: ingest / LabelMe treat every *.json as an annotation
PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
           "#469990", "#dcbeff", "#9a6324", "#800000", "#aaffc3", "#808000", "#000075"]


# ------------------------------------------------------------------ project
def load_project(config_path: str | Path, overrides: dict | None = None) -> dict:
    """load_config with a friendlier error for an empty path; classes must be non-empty for the workbench."""
    if not str(config_path or "").strip():
        raise ConfigError("no project loaded", "enter the path of a project.yaml (or create a new project)")
    cfg = load_config(Path(str(config_path).strip().strip('"')), overrides)
    if not cfg.get("classes"):
        raise ConfigError("config.classes is empty", "list your class names under `classes:` in the project YAML")
    return cfg


def parse_classes(text: str) -> list[str]:
    """'scratch, dent\\nstain' -> ['scratch', 'dent', 'stain']; rejects empty and duplicate names."""
    names = [n.strip() for n in re.split(r"[,\n，;；]+", text or "") if n.strip()]
    if not names:
        raise ConfigError("no class names given", "enter class names separated by commas, e.g. scratch, dent")
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        raise ConfigError(f"duplicate class names: {dups}")
    bad = [n for n in names if any(c in n for c in "[]{}:#'\"")]
    if bad:
        raise ConfigError(f"class names may not contain []{{}}:#'\" : {bad}")
    return names


def create_project(project_dir: str | Path, classes: Sequence[str], force: bool = False) -> Path:
    """New project folder from the template with `classes` filled in; creates the workspace folders."""
    root = ensure_dir(Path(str(project_dir).strip().strip('"')).resolve())
    dst = root / "project.yaml"
    if dst.exists() and not force:
        raise ConfigError(f"{dst} already exists", "load it instead, or pick another folder")
    if not TEMPLATE.is_file():
        raise ConfigError(f"template missing: {TEMPLATE}")
    text = TEMPLATE.read_text(encoding="utf-8")
    flow = json.dumps(list(classes), ensure_ascii=False)
    text, n = re.subn(r"^classes:.*$", f"classes: {flow}", text, count=1, flags=re.M)
    if n != 1:
        raise ConfigError("template has no single-line `classes:` entry")
    dst.write_text(text, encoding="utf-8")
    cfg = load_project(dst)
    for k in ("raw_images", "labelme_dir", "batch_dataset", "corpus_root", "runs", "deploy_out"):
        ensure_dir(Path(cfg["paths"][k]))
    ensure_dir(root / "models")
    return dst


def list_bundles(deploy_out: str | Path) -> list[Path]:
    root = Path(deploy_out)
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir() and (p / "meta.json").is_file()),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def list_weights(cfg: dict) -> list[Path]:
    """best.pt of every train run (newest first) plus .pt / .onnx files under <project>/models."""
    out = sorted(Path(cfg["paths"]["runs"]).glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime,
                 reverse=True) if Path(cfg["paths"]["runs"]).is_dir() else []
    models = Path(cfg["_config_dir"]) / "models"
    if models.is_dir():
        out += sorted(p for p in models.rglob("*") if p.suffix.lower() in (".pt", ".onnx"))
    return out


def project_status(cfg: dict) -> dict[str, Any]:
    p = cfg["paths"]

    def count_images(d: str | Path) -> int:
        d = Path(d)
        return len(list_images(d)) if d.is_dir() else 0

    lm = Path(p["labelme_dir"])
    jsons = sorted(lm.glob("*.json")) if lm.is_dir() else []
    reviewed = load_reviewed(lm)
    corpus = Path(p["corpus_root"]) / "images"
    return {
        "classes": list(cfg["classes"]),
        "raw_images": count_images(p["raw_images"]),
        "to_review": len(jsons),
        "reviewed": sum(1 for j in jsons if j.stem in reviewed),
        "corpus_train": len(list_split_images(corpus / "train")),
        "corpus_val": len(list_split_images(corpus / "val")),
        "weights": [str(w) for w in list_weights(cfg)][:5],
        "bundles": [b.name for b in list_bundles(p["deploy_out"])],
    }


def status_markdown(cfg: dict) -> str:
    s = project_status(cfg)
    return (f"**类别** {', '.join(s['classes'])}　|　**待标注图片** {s['raw_images']}　|　"
            f"**审核进度** {s['reviewed']}/{s['to_review']}　|　**训练集** {s['corpus_train']} / "
            f"**验证集** {s['corpus_val']}　|　**部署包** {len(s['bundles'])}")


# ------------------------------------------------------------------ review / annotation
def load_reviewed(labelme_dir: Path) -> set[str]:
    f = Path(labelme_dir) / REVIEW_FILE
    if not f.is_file():
        return set()
    try:
        data = read_json(f)
    except EvpError:
        log.warning("ignoring broken %s", f)
        return set()
    return {str(s) for s in data.get("reviewed", [])} if isinstance(data, dict) else set()


def _set_reviewed(labelme_dir: Path, stem: str, value: bool = True) -> None:
    done = load_reviewed(labelme_dir)
    (done.add if value else done.discard)(stem)
    write_json(Path(labelme_dir) / REVIEW_FILE, {"reviewed": sorted(done)})


def review_items(labelme_dir: str | Path) -> list[str]:
    """Stems of the LabelMe json in the review folder, sorted."""
    d = Path(labelme_dir)
    return sorted(j.stem for j in d.glob("*.json")) if d.is_dir() else []


def stage_for_manual_labeling(raw_images: str | Path, labelme_dir: str | Path) -> int:
    """Cold start without a prelabel model: copy raw images into the review folder with EMPTY LabelMe json.
    Existing json (prelabeled or reviewed) are never replaced. Returns the number of json created."""
    images = list_images(Path(raw_images))
    if not images:
        raise DataError(f"no images under {raw_images}", "collect images first")
    out = ensure_dir(Path(labelme_dir))
    n = 0
    for img in images:
        jp = out / f"{img.stem}.json"
        if jp.exists():
            continue
        dst = out / img.name
        if img.resolve() != dst.resolve():
            shutil.copy2(img, dst)
        w, h = image_size(dst)
        write_json(jp, {"version": LABELME_VERSION, "flags": {}, "shapes": [], "imagePath": dst.name,
                        "imageData": None, "imageHeight": h, "imageWidth": w})
        n += 1
    return n


def class_colors(names: Sequence[str]) -> list[str]:
    return [PALETTE[i % len(PALETTE)] for i in range(len(names))]


def _hex_rgb(h: str) -> tuple[int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def load_annotation(labelme_dir: str | Path, stem: str, names: Sequence[str]) -> dict:
    """LabelMe json -> image_annotator value {"image": path, "boxes": [...]}. Polygons are shown as their
    bounding box (detection training uses the box anyway); other shape types are not shown but are kept
    on save."""
    jp = Path(labelme_dir) / f"{stem}.json"
    if not jp.is_file():
        raise DataError(f"annotation not found: {jp}")
    try:
        data = json.loads(jp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DataError(f"broken LabelMe json: {jp}: {exc}", "fix or delete it") from exc
    img = find_image_for_json(jp, data.get("imagePath"))
    if img is None:
        raise DataError(f"image for {jp.name} not found", "the image must sit next to its json")
    colors = dict(zip(names, class_colors(names)))
    boxes = []
    for s in data.get("shapes") or []:
        if (s.get("shape_type") or "rectangle").lower() not in ("rectangle", "polygon"):
            continue
        pts = s.get("points") or []
        if len(pts) < 2:
            continue
        xs, ys = [float(p[0]) for p in pts], [float(p[1]) for p in pts]
        label = str(s.get("label", ""))
        boxes.append({"xmin": round(min(xs)), "ymin": round(min(ys)), "xmax": round(max(xs)), "ymax": round(max(ys)),
                      "label": label, "color": _hex_rgb(colors.get(label, "#ff0000"))})
    return {"image": str(img), "boxes": boxes}


def save_annotation(labelme_dir: str | Path, stem: str, value: dict | None, names: Sequence[str]) -> dict:
    """image_annotator value -> LabelMe json (atomic write) and mark the image as reviewed.
    Boxes are clipped to the image; boxes under 1 px and unknown labels are errors (nothing is written), so a
    typo never silently becomes a missing label. Returns {"boxes": n, "path": json}."""
    jp = Path(labelme_dir) / f"{stem}.json"
    if not jp.is_file():
        raise DataError(f"annotation not found: {jp}")
    if not value:
        raise DataError("nothing to save (no image loaded)")
    if value.get("orientation"):
        raise DataError("the image is rotated in the editor", "rotate it back to 0 before saving")
    data = json.loads(jp.read_text(encoding="utf-8"))
    img = find_image_for_json(jp, data.get("imagePath"))
    if img is None:
        raise DataError(f"image for {jp.name} not found")
    w, h = image_size(img)
    known = set(names)
    shapes, errors = [], []
    for i, b in enumerate(value.get("boxes") or [], 1):
        label = str(b.get("label") or "").strip()
        if label not in known:
            errors.append(f"box {i}: label {label!r} not in classes {list(names)}")
            continue
        x1, x2 = sorted((min(max(float(b["xmin"]), 0.0), w), min(max(float(b["xmax"]), 0.0), w)))
        y1, y2 = sorted((min(max(float(b["ymin"]), 0.0), h), min(max(float(b["ymax"]), 0.0), h)))
        if x2 - x1 < 1 or y2 - y1 < 1:
            errors.append(f"box {i} ({label}) is smaller than 1 px after clipping to the image")
            continue
        shapes.append({"label": label, "points": [[x1, y1], [x2, y2]], "group_id": None, "description": "",
                       "shape_type": "rectangle", "flags": {}})
    if errors:
        raise DataError("; ".join(errors), "fix the boxes (pick a label from the list), then save again")
    keep = [s for s in data.get("shapes") or [] if (s.get("shape_type") or "rectangle").lower()
            not in ("rectangle", "polygon")]
    data.update({"shapes": shapes + keep, "imageWidth": w, "imageHeight": h, "imagePath": img.name,
                 "imageData": None, "version": data.get("version") or LABELME_VERSION})
    write_json(jp, data)
    _set_reviewed(Path(labelme_dir), stem, True)
    return {"boxes": len(shapes), "path": str(jp)}


def delete_review_item(labelme_dir: str | Path, stem: str) -> None:
    """Drop one image (json + image) from the review batch, e.g. an unusable frame."""
    d = Path(labelme_dir)
    jp = d / f"{stem}.json"
    if not jp.is_file():
        raise DataError(f"annotation not found: {jp}")
    try:
        img = find_image_for_json(jp, json.loads(jp.read_text(encoding="utf-8")).get("imagePath"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        img = None
    jp.unlink()
    if img is not None and img.parent.resolve() == d.resolve():
        img.unlink()
    _set_reviewed(d, stem, False)


# ------------------------------------------------------------------ background jobs
class _QueueHandler(logging.Handler):
    def __init__(self, sink: deque):
        super().__init__(logging.INFO)
        self.sink = sink
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.sink.append(self.format(record))
        except Exception:  # noqa: BLE001
            self.handleError(record)


@dataclass
class Job:
    name: str
    started: float = field(default_factory=time.time)
    finished: float | None = None
    result: Any = None
    error: str | None = None
    lines: deque = field(default_factory=lambda: deque(maxlen=2000))

    @property
    def running(self) -> bool:
        return self.finished is None

    def log_text(self) -> str:
        return "\n".join(self.lines)


class JobRunner:
    """One long job (train / deploy / prelabel) at a time, in a thread, with the evp and ultralytics log
    records captured for the UI. Starting a second job while one runs is an error, not a queue."""

    LOGGERS = ("evp", "ultralytics")

    def __init__(self):
        self._lock = threading.Lock()
        self.job: Job | None = None

    def start(self, name: str, fn: Callable[[], Any]) -> Job:
        with self._lock:
            if self.job is not None and self.job.running:
                raise EvpError(f"job '{self.job.name}' is still running", "wait for it to finish")
            job = Job(name)
            self.job = job
        handler = _QueueHandler(job.lines)

        def run():
            loggers = [logging.getLogger(n) for n in self.LOGGERS]
            for lg in loggers:
                lg.addHandler(handler)
            try:
                job.result = fn()
            except EvpError as exc:
                job.error = str(exc)
                job.lines.append(f"[ERROR] {exc}")
            except Exception as exc:  # noqa: BLE001
                job.error = f"{type(exc).__name__}: {exc}"
                job.lines.append(f"[ERROR] {job.error}")
                log.exception("job %s failed", name)
            finally:
                for lg in loggers:
                    lg.removeHandler(handler)
                job.finished = time.time()

        threading.Thread(target=run, name=f"evp-job-{name}", daemon=True).start()
        return job

    def wait(self, timeout: float | None = None) -> Job | None:
        t0 = time.time()
        while self.job is not None and self.job.running:
            if timeout is not None and time.time() - t0 > timeout:
                break
            time.sleep(0.05)
        return self.job


# ------------------------------------------------------------------ inference / benchmark
def draw_detections(img_bgr: np.ndarray, dets, labels: Sequence[str]) -> np.ndarray:
    import cv2

    vis = img_bgr.copy()
    colors = [_hex_rgb(c) for c in class_colors(labels or ["x"])]
    th = max(1, round(max(vis.shape[:2]) / 400))
    for d in dets:
        r, g, b = colors[d.cls % len(colors)]
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(vis, p1, p2, (b, g, r), th)
        name = labels[d.cls] if d.cls < len(labels) else str(d.cls)
        cv2.putText(vis, f"{name} {d.score:.2f}", (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 * th + 0.1, (b, g, r), th)
    return vis


_PIPE_CACHE: dict[tuple, Any] = {}


def get_pipeline(bundle: str | Path, precision: str, threads: int):
    """NcnnPipeline cached by (bundle, precision, threads, meta mtime) so a re-deploy is picked up."""
    from ..deploy.runner import NcnnPipeline

    bundle = Path(bundle)
    if not (bundle / "meta.json").is_file():
        raise DataError(f"not a deploy bundle (no meta.json): {bundle}")
    key = (str(bundle.resolve()), precision, int(threads), (bundle / "meta.json").stat().st_mtime_ns)
    if key not in _PIPE_CACHE:
        if len(_PIPE_CACHE) > 8:
            _PIPE_CACHE.clear()
        _PIPE_CACHE[key] = NcnnPipeline.from_bundle(bundle, precision, num_threads=int(threads))
    return _PIPE_CACHE[key]


def infer_image(bundle: str | Path, img_bgr: np.ndarray, precision: str = "fp32", threads: int = 1,
                conf: float | None = None) -> tuple[np.ndarray, list[dict], float]:
    """One image through the bundle (Python ncnn reference runtime). Returns (vis BGR, rows, latency ms)."""
    if img_bgr is None or img_bgr.size == 0:
        raise DataError("no image given")
    pipe = get_pipeline(bundle, precision, threads)
    old = pipe.post.conf
    if conf is not None:
        pipe.post.conf = float(conf)
    try:
        t0 = time.perf_counter()
        res = pipe.run(img_bgr)
        ms = (time.perf_counter() - t0) * 1000
    finally:
        pipe.post.conf = old
    labels = pipe.meta.labels
    if res.detections is not None:
        rows = [{"label": labels[d.cls] if d.cls < len(labels) else str(d.cls), "score": round(d.score, 3),
                 "x1": round(d.x1, 1), "y1": round(d.y1, 1), "x2": round(d.x2, 1), "y2": round(d.y2, 1)}
                for d in res.detections]
        return draw_detections(img_bgr, res.detections, labels), rows, ms
    if res.topk is not None:
        rows = [{"label": labels[c] if c < len(labels) else str(c), "score": round(s, 4)} for c, s in res.topk]
        return img_bgr, rows, ms
    return img_bgr, [{"output": str(list(o.shape))} for o in res.outputs], ms


def benchmark(bundle: str | Path, images: Sequence[np.ndarray], precisions: Sequence[str] = ("fp32", "int8"),
              threads: Sequence[int] = (1,), runs: int = 20, warmup: int = 3) -> list[dict]:
    """End-to-end latency (preprocess + forward + postprocess) of the Python runtime on THIS machine.
    Precisions missing from the bundle are skipped. These are host numbers, not board numbers."""
    from ..deploy.meta import load_meta

    if not images:
        raise DataError("benchmark needs at least one image")
    if runs < 1 or warmup < 0:
        raise DataError(f"runs must be >= 1 and warmup >= 0, got runs={runs} warmup={warmup}")
    bundle = Path(bundle)
    meta = load_meta(bundle)
    rows = []
    for prec in precisions:
        if prec not in meta.files:
            continue
        size_mb = (bundle / meta.files[prec].bin).stat().st_size / 1e6
        for t in threads:
            pipe = get_pipeline(bundle, prec, int(t))
            for i in range(warmup):
                pipe.run(images[i % len(images)])
            times = []
            for i in range(runs):
                t0 = time.perf_counter()
                pipe.run(images[i % len(images)])
                times.append((time.perf_counter() - t0) * 1000)
            times.sort()
            mean = statistics.fmean(times)
            rows.append({"precision": prec, "threads": int(t), "input": f"{meta.input.w}x{meta.input.h}",
                         "weights_MB": round(size_mb, 2), "mean_ms": round(mean, 2),
                         "p50_ms": round(times[len(times) // 2], 2),
                         "p90_ms": round(times[min(len(times) - 1, int(len(times) * 0.9))], 2),
                         "fps": round(1000.0 / mean, 1) if mean > 0 else None})
    if not rows:
        raise DataError(f"bundle has none of the precisions {list(precisions)} (has {sorted(meta.files)})")
    return rows


def read_report(bundle: str | Path) -> str:
    f = Path(bundle) / "report.md"
    return f.read_text(encoding="utf-8") if f.is_file() else "_该部署包没有 report.md（部署时跳过了验证？）_"


def bundle_files(bundle: str | Path) -> list[dict]:
    b = Path(bundle)
    return [{"file": p.name, "KB": round(p.stat().st_size / 1024, 1)} for p in sorted(b.iterdir()) if p.is_file()]


def sample_images(cfg: dict, limit: int = 8) -> list[Path]:
    """A few project images for benchmarking: corpus val, then corpus train, then the review folder."""
    p = cfg["paths"]
    for d in (Path(p["corpus_root"]) / "images" / "val", Path(p["corpus_root"]) / "images" / "train",
              Path(p["labelme_dir"]), Path(p["raw_images"])):
        if d.is_dir():
            imgs = [x for x in list_images(d) if x.suffix.lower() in IMAGE_EXTS]
            if imgs:
                return imgs[:limit]
    return []
