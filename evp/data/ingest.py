#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Ingest: reviewed LabelMe json -> YOLO batch -> split -> merge into the corpus -> clear staging."""

from __future__ import annotations

from pathlib import Path

from ..errors import ConfigError, DataError
from ..utils import log
from .labelme import convert_dir
from .merge import clear_staging, list_split_images, merge_batch
from .split import split_batch


def ingest(cfg: dict, keep_staging: bool = False) -> dict:
    """Shared by `evp ingest` and the workbench. Any failure (broken json, unknown label, bad id) stops before
    merge/clear, so reviewed work is kept. Returns the merge result (before/merged/after, val_moved)."""
    p, ds = cfg["paths"], cfg["dataset"]
    names = [str(c) for c in cfg.get("classes") or []]
    if not names:
        raise ConfigError("config.classes is empty", "list your class names under `classes:`")
    batch = Path(p["batch_dataset"])
    stats = convert_dir(Path(p["labelme_dir"]), batch, names, split="train")
    log.info("json2yolo: %s", stats)
    if stats.files == 0:
        raise DataError("no reviewed images converted", "check LabelMe json files and their image paths")
    if stats.unknown_labels:
        raise DataError(f"labels not in config.classes: {stats.unknown_labels}",
                        f"fix the labels in LabelMe or add them to `classes:` ({names}), then rerun ingest; "
                        f"staging folders were not touched")
    n_val = split_batch(batch, float(ds["val_ratio"]), int(ds["seed"]))
    res = merge_batch(batch, Path(p["corpus_root"]), names)
    res["val_moved"] = n_val
    if not list_split_images(Path(p["corpus_root"]) / "images" / "val"):
        log.warning("corpus has no val images yet (this batch had < 2 labeled images); "
                    "`evp train` needs at least one, so ingest more reviewed images first")
    if ds.get("clear_after_merge", True) and not keep_staging:
        clear_staging(Path(p["raw_images"]), Path(p["labelme_dir"]), batch)
    return res
