#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Prelabel raw images into LabelMe JSON for human review.

Backends:
  - ultralytics: a .pt model (default, same as the old SOP)
  - ncnn: a deploy bundle produced by `evp deploy` (also exercises the board pipeline on real data)
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from ..errors import DataError
from ..utils import ensure_dir, list_images, log, resolve_device
from .labelme import build_labelme_payload, image_size


def _predict_ultralytics(model_path: Path, conf: float, iou: float, imgsz: int, device: str):
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if getattr(model, "task", "detect") != "detect":
        raise DataError(f"prelabel needs a detect model, got task={model.task}: {model_path}")

    def predict(img_path: Path):
        r = model.predict(source=str(img_path), conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)[0]
        out = []
        if r.boxes is not None and len(r.boxes):
            for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.cls.cpu().tolist(),
                                              r.boxes.conf.cpu().tolist()):
                out.append((x1, y1, x2, y2, int(c), float(s)))
        return out

    names = model.names
    model_names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    return predict, model_names


def _predict_ncnn(bundle_dir: Path, conf: float, iou: float, precision: str):
    from ..deploy.runner import NcnnPipeline

    pipe = NcnnPipeline.from_bundle(bundle_dir, precision=precision)
    pipe.post.conf = conf
    pipe.post.iou = iou

    if pipe.meta.task != "detect":
        raise DataError(f"prelabel needs a detect bundle, got family={pipe.meta.family}: {bundle_dir}")

    def predict(img_path: Path):
        from ..deploy.runner import read_image

        dets = pipe.detect(read_image(img_path))
        return [(d.x1, d.y1, d.x2, d.y2, d.cls, d.score) for d in dets]

    return predict, list(pipe.meta.labels)


def map_class_ids(model_names: list[str], names: list[str]) -> dict[int, int]:
    """Model class id -> config class id, matched by NAME (ids differ when class orders differ).
    Model classes missing from the config are dropped; no overlap at all is an error."""
    if list(model_names) == list(names):
        return {i: i for i in range(len(names))}
    pos = {n: i for i, n in enumerate(names)}
    id_map = {mid: pos[n] for mid, n in enumerate(model_names) if n in pos}
    if not id_map:
        raise DataError(f"prelabel model classes {list(model_names)[:10]} share no name with config classes {names}",
                        "use a model trained on these classes, or make config.classes use the model's class names")
    dropped = [n for n in model_names if n not in pos]
    log.warning("model classes differ from config classes; mapped by name %s%s",
                {model_names[m]: names[c] for m, c in id_map.items()},
                f", dropping model-only classes {dropped[:10]}" if dropped else "")
    return id_map


def prelabel(source: Path, out_dir: Path, names: list[str], model: Path | None = None,
             bundle: Path | None = None, conf: float = 0.25, iou: float = 0.45, imgsz: int = 640,
             device: str = "cpu", embed_image_data: bool = False, precision: str = "fp32",
             overwrite: bool = False) -> int:
    """Predict every image in `source` and write <out_dir>/<image> + <stem>.json for LabelMe review.

    Uses the Ultralytics `model` (.pt) or an ncnn deploy `bundle`. Existing json are kept unless
    `overwrite` (they may already be human-reviewed). Returns the number of json written."""
    images = list_images(Path(source))
    if not images:
        raise DataError(f"no images under {source}", "put new images into paths.raw_images")
    stems: dict[str, list[str]] = {}
    for p in images:
        stems.setdefault(p.stem.lower(), []).append(p.name)
    dups = {k: v for k, v in stems.items() if len(v) > 1}
    if dups:
        raise DataError(f"images share a file name stem (they would share one LabelMe json): {dups}",
                        "rename them so every image has a unique name (case-insensitive)")
    ensure_dir(out_dir)
    if bundle is not None:
        predict, model_names = _predict_ncnn(Path(bundle), conf, iou, precision)
    else:
        if model is None or not Path(model).is_file():
            raise DataError(f"prelabel model not found: {model}", "set paths.prelabel_model")
        predict, model_names = _predict_ultralytics(Path(model), conf, iou, imgsz,
                                                    resolve_device(device, "prelabel.device"))
    id_map = map_class_ids(model_names, list(names))

    t0 = time.time()
    written = skipped = 0
    for img in images:
        dst_img = Path(out_dir) / img.name
        json_path = dst_img.with_suffix(".json")
        if json_path.exists() and not overwrite:
            skipped += 1  # never clobber a human-reviewed json
            continue
        if img.resolve() != dst_img.resolve():
            shutil.copy2(img, dst_img)
        w, h = image_size(dst_img)
        boxes = [(x1, y1, x2, y2, id_map[c], s) for x1, y1, x2, y2, c, s in predict(dst_img) if c in id_map]
        payload = build_labelme_payload(dst_img, w, h, boxes, names, embed_image_data)
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        written += 1
    log.info("prelabel: wrote %d json, kept %d existing (%.1fs) -> %s", written, skipped, time.time() - t0, out_dir)
    log.info("next: review with `labelme %s`, then run `evp ingest`", out_dir)
    return written
