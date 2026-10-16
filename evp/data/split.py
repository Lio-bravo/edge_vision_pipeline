#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Split THIS batch only (train -> val). Corpus val is never re-split."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from ..errors import DataError
from ..utils import IMAGE_EXTS, ensure_dir, log


def split_batch(batch_root: Path, val_ratio: float, seed: int = 42, move: bool = True) -> int:
    """Move (or copy, move=False) round(n * val_ratio) labeled pairs (at least 1, never all) from batch
    train to batch val. Deterministic for a given seed and file set.
    Returns the number moved (0 if < 2 labeled images)."""
    if not 0 < val_ratio < 1:
        raise DataError(f"val_ratio must be in (0,1), got {val_ratio}")
    batch_root = Path(batch_root)
    train_img = batch_root / "images" / "train"
    train_lbl = batch_root / "labels" / "train"
    val_img = ensure_dir(batch_root / "images" / "val")
    val_lbl = ensure_dir(batch_root / "labels" / "val")
    if not train_img.is_dir():
        raise DataError(f"missing {train_img}", "run ingest/json2yolo first")
    pairs = [p for p in sorted(train_img.iterdir())
             if p.suffix.lower() in IMAGE_EXTS and (train_lbl / f"{p.stem}.txt").is_file()]
    if len(pairs) < 2:
        log.warning("batch has %d labeled image(s); nothing moved to val", len(pairs))
        return 0
    n_val = max(1, int(round(len(pairs) * val_ratio)))
    n_val = min(n_val, len(pairs) - 1)  # always keep at least one train sample
    chosen = random.Random(seed).sample(pairs, n_val)
    op = shutil.move if move else shutil.copy2
    for img in chosen:
        op(str(img), str(val_img / img.name))
        op(str(train_lbl / f"{img.stem}.txt"), str(val_lbl / f"{img.stem}.txt"))
    log.info("split: %d -> val, %d stay train (ratio=%.2f seed=%d)", n_val, len(pairs) - n_val, val_ratio, seed)
    return n_val
