#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Append a batch into the long-term corpus (never re-split existing val)."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import yaml

from ..errors import DataError
from ..utils import IMAGE_EXTS, clear_dir_contents, ensure_dir, log

SPLITS = ("train", "val")


def list_split_images(split_img_dir: Path) -> list[Path]:
    if not split_img_dir.is_dir():
        return []
    return [p for p in sorted(split_img_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def unique_pair_dest(img_dst_dir: Path, lbl_dst_dir: Path, img_name: str) -> tuple[Path, Path]:
    """foo.jpg -> foo__b1.jpg, foo__b2.jpg ... until neither image nor label (any ext) collides."""
    stem, suffix = Path(img_name).stem, Path(img_name).suffix
    n = 0
    while True:
        s = stem if n == 0 else f"{stem}__b{n}"
        img_dst = img_dst_dir / f"{s}{suffix}"
        lbl_dst = lbl_dst_dir / f"{s}.txt"
        same_stem_images = [img_dst_dir / f"{s}{e}" for e in IMAGE_EXTS]
        if not lbl_dst.exists() and not any(p.exists() for p in same_stem_images):
            return img_dst, lbl_dst
        n += 1


def write_data_yaml(dataset_root: Path, names: list[str]) -> Path:
    content = {
        "path": Path(dataset_root).resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": list(names),
    }
    out = Path(dataset_root) / "data.yaml"
    out.write_text(yaml.safe_dump(content, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out


def check_label_ids(label_file: Path, nc: int) -> None:
    """Every line must be `cls cx cy w h` with 0 <= cls < nc and normalized coords in [0, 1]."""
    for ln, line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise DataError(f"{label_file}:{ln}: expected 5 fields, got {len(parts)}")
        cid = int(parts[0])
        if not 0 <= cid < nc:
            raise DataError(f"{label_file}:{ln}: class id {cid} out of range [0,{nc})",
                            "classes in config changed? fix labels or config.classes")
        vals = [float(v) for v in parts[1:]]
        if any(v < 0 or v > 1 for v in vals):
            raise DataError(f"{label_file}:{ln}: normalized coords out of [0,1]: {vals}")


def merge_split(batch_root: Path, corpus_root: Path, split: str, nc: int | None = None) -> tuple[int, int]:
    """Move image+label pairs of one split into the corpus (renaming on collision). Returns (moved, skipped)."""
    src_img = Path(batch_root) / "images" / split
    src_lbl = Path(batch_root) / "labels" / split
    dst_img = ensure_dir(Path(corpus_root) / "images" / split)
    dst_lbl = ensure_dir(Path(corpus_root) / "labels" / split)
    ok = skipped = 0
    for img in list_split_images(src_img):
        lbl = src_lbl / f"{img.stem}.txt"
        if not lbl.is_file():
            log.warning("skip image without label: %s", img.name)
            skipped += 1
            continue
        if nc is not None:
            check_label_ids(lbl, nc)
        img_dst, lbl_dst = unique_pair_dest(dst_img, dst_lbl, img.name)
        if img_dst.name != img.name:
            log.info("rename to avoid collision: %s -> %s", img.name, img_dst.name)
        shutil.move(str(img), str(img_dst))
        shutil.move(str(lbl), str(lbl_dst))
        ok += 1
    return ok, skipped


def merge_batch(batch_root: Path, corpus_root: Path, names: list[str]) -> dict:
    """Validate all labels of the batch, then move train/val into the corpus, rewrite data.yaml and write a
    merge log (corpus/merge_logs/). Returns before/merged/after counts."""
    batch_root, corpus_root = Path(batch_root), Path(corpus_root)
    if not batch_root.is_dir():
        raise DataError(f"batch dataset not found: {batch_root}")
    if not any(list_split_images(batch_root / "images" / s) for s in SPLITS):
        raise DataError(f"batch dataset is empty: {batch_root}", "run ingest after LabelMe review")
    # validate everything before moving anything, so a bad label never half-merges a batch
    for s in SPLITS:
        for img in list_split_images(batch_root / "images" / s):
            lbl = batch_root / "labels" / s / f"{img.stem}.txt"
            if lbl.is_file():
                check_label_ids(lbl, len(names))
    for s in SPLITS:
        ensure_dir(corpus_root / "images" / s)
        ensure_dir(corpus_root / "labels" / s)
    before = {s: len(list_split_images(corpus_root / "images" / s)) for s in SPLITS}
    result = {"before": before, "merged": {}, "skipped": 0}
    for s in SPLITS:
        ok, sk = merge_split(batch_root, corpus_root, s)
        result["merged"][s] = ok
        result["skipped"] += sk
    result["after"] = {s: len(list_split_images(corpus_root / "images" / s)) for s in SPLITS}
    result["data_yaml"] = str(write_data_yaml(corpus_root, names))
    log_dir = ensure_dir(corpus_root / "merge_logs")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"merge_{stamp}.yaml"
    log_path.write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    result["log"] = str(log_path)
    log.info("merged train=%d val=%d skipped=%d; corpus now train=%d val=%d", result["merged"]["train"],
             result["merged"]["val"], result["skipped"], result["after"]["train"], result["after"]["val"])
    return result


def clear_staging(raw_images: Path, labelme_dir: Path, batch_dataset: Path) -> None:
    """Empty the per-batch folders after a successful merge (the folders themselves are kept)."""
    for p in (raw_images, labelme_dir, batch_dataset):
        n = clear_dir_contents(Path(p))
        ensure_dir(Path(p))
        log.info("cleared %d entries under %s", n, p)
    for s in SPLITS:
        ensure_dir(Path(batch_dataset) / "images" / s)
        ensure_dir(Path(batch_dataset) / "labels" / s)
