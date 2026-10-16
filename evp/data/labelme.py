#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""LabelMe JSON <-> YOLO detect txt conversion."""

from __future__ import annotations

import json
import shutil
from base64 import b64encode
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import DataError
from ..utils import IMAGE_EXTS, ensure_dir, log

LABELME_VERSION = "5.1.1"


@dataclass
class ConvertStats:
    files: int = 0
    objects: int = 0
    empty: int = 0
    missing_image: int = 0
    skipped_shapes: int = 0
    unknown_labels: dict[str, int] = field(default_factory=dict)


EXIF_ORIENTATION = 0x0112


def image_size(path: Path) -> tuple[int, int]:
    """Display (w, h) from the image header: EXIF orientations 5-8 swap w/h, as LabelMe, cv2.imread and
    Ultralytics all apply the orientation before annotating / training. Does not decode pixels."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            w, h = int(im.width), int(im.height)
            try:
                orient = int(im.getexif().get(EXIF_ORIENTATION, 1))
            except Exception:  # noqa: BLE001
                orient = 1
    except Exception as exc:  # noqa: BLE001
        raise DataError(f"cannot read image size: {path}: {exc}") from exc
    return (h, w) if orient in (5, 6, 7, 8) else (w, h)


def shape_to_yolo(points: Sequence[Sequence[float]], img_w: int,
                  img_h: int) -> tuple[float, float, float, float] | None:
    """Axis-aligned bbox of points, clipped to the image, normalized (cx, cy, w, h). None if degenerate."""
    if img_w <= 0 or img_h <= 0:
        raise DataError(f"invalid image size {img_w}x{img_h}")
    if len(points) < 2:
        return None
    xs = [min(max(float(p[0]), 0.0), float(img_w)) for p in points]
    ys = [min(max(float(p[1]), 0.0), float(img_h)) for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bw, bh = x2 - x1, y2 - y1
    if bw < 1.0 or bh < 1.0:
        return None
    return ((x1 + bw / 2.0) / img_w, (y1 + bh / 2.0) / img_h, bw / img_w, bh / img_h)


def find_image_for_json(json_path: Path, image_path_field: str | None) -> Path | None:
    """imagePath relative to the json, then its basename next to the json, then <json stem>.<image ext>."""
    candidates: list[Path] = []
    if image_path_field:
        rel = Path(image_path_field.replace("\\", "/"))
        if not rel.is_absolute():
            candidates.append(json_path.parent / rel)
        candidates.append(json_path.parent / rel.name)
    candidates += [json_path.with_suffix(ext) for ext in IMAGE_EXTS]
    candidates += [json_path.with_suffix(ext.upper()) for ext in IMAGE_EXTS]
    for c in candidates:
        if c.is_file():
            return c
    return None


def labelme_to_yolo_lines(data: dict, name_to_id: dict[str, int], img_w: int, img_h: int,
                          stats: ConvertStats, src: str = "") -> list[str]:
    """Rectangle/polygon shapes -> YOLO lines. Other shape types, unknown labels and degenerate boxes are
    skipped and counted in `stats` (callers decide whether unknown labels are fatal)."""
    lines: list[str] = []
    for shape in data.get("shapes") or []:
        stype = (shape.get("shape_type") or "rectangle").lower()
        if stype not in ("rectangle", "polygon"):
            stats.skipped_shapes += 1
            log.warning("skip shape_type=%s in %s", stype, src)
            continue
        label = str(shape.get("label", "")).strip()
        if label not in name_to_id:
            stats.unknown_labels[label] = stats.unknown_labels.get(label, 0) + 1
            continue
        box = shape_to_yolo(shape.get("points") or [], img_w, img_h)
        if box is None:
            stats.skipped_shapes += 1
            log.warning("skip degenerate box (label=%s) in %s", label, src)
            continue
        cx, cy, w, h = box
        lines.append(f"{name_to_id[label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def convert_dir(labelme_dir: Path, out_root: Path, names: list[str], split: str = "train",
                copy_images: bool = True) -> ConvertStats:
    """Convert every LabelMe json in `labelme_dir` into out_root/{images,labels}/<split>."""
    labelme_dir = Path(labelme_dir)
    if not labelme_dir.is_dir():
        raise DataError(f"LabelMe dir not found: {labelme_dir}")
    if not names:
        raise DataError("class names are empty", "set `classes:` in the project config")
    json_files = sorted(labelme_dir.glob("*.json"))
    if not json_files:
        raise DataError(f"no json under {labelme_dir}", "run prelabel first, then review in LabelMe")

    name_to_id = {n: i for i, n in enumerate(names)}
    img_dir = ensure_dir(Path(out_root) / "images" / split)
    lbl_dir = ensure_dir(Path(out_root) / "labels" / split)
    stats = ConvertStats()
    items: list[tuple[Path, dict, Path]] = []
    by_stem: dict[str, list[str]] = {}
    for jp in json_files:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DataError(f"broken LabelMe json: {jp}: {exc}", "fix or delete it, then rerun") from exc
        if not isinstance(data, dict):
            raise DataError(f"not a LabelMe json object: {jp}", "fix or delete it, then rerun")
        img = find_image_for_json(jp, data.get("imagePath"))
        if img is None:
            stats.missing_image += 1
            log.warning("json without image, skipped: %s", jp.name)
            continue
        items.append((jp, data, img))
        by_stem.setdefault(img.stem.lower(), []).append(f"{jp.name} -> {img.name}")
    dups = {k: v for k, v in by_stem.items() if len(v) > 1}
    if dups:
        raise DataError(f"several json files map to the same image name (labels would overwrite each other): {dups}",
                        "keep one json per image and give every image a unique file name (case-insensitive)")
    for jp, data, img in items:
        w, h = image_size(img)
        jw, jh = int(data.get("imageWidth") or 0), int(data.get("imageHeight") or 0)
        if (jw, jh) != (w, h) and jw and jh:
            log.warning("%s: json size %dx%d != image %dx%d, using image size", jp.name, jw, jh, w, h)
        lines = labelme_to_yolo_lines(data, name_to_id, w, h, stats, jp.name)
        (lbl_dir / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if copy_images:
            dst = img_dir / img.name
            if img.resolve() != dst.resolve():
                shutil.copy2(img, dst)
        stats.files += 1
        stats.objects += len(lines)
        stats.empty += 0 if lines else 1
    if stats.unknown_labels:
        log.warning("labels not in config.classes (ignored): %s", stats.unknown_labels)
    return stats


def build_labelme_payload(image_path: Path, img_w: int, img_h: int,
                          boxes: Sequence[tuple[float, float, float, float, int, float]],
                          names: list[str], embed_image_data: bool = False) -> dict:
    """boxes: (x1, y1, x2, y2, class_id, score) in pixels."""
    shapes = []
    for x1, y1, x2, y2, cid, score in boxes:
        cid = int(cid)
        if not 0 <= cid < len(names):
            log.warning("skip unknown class id %d for %s", cid, image_path.name)
            continue
        shapes.append({
            "label": names[cid],
            "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
            "group_id": None,
            "description": f"score={score:.3f}",
            "shape_type": "rectangle",
            "flags": {},
        })
    return {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": b64encode(image_path.read_bytes()).decode("ascii") if embed_image_data else None,
        "imageHeight": int(img_h),
        "imageWidth": int(img_w),
    }
