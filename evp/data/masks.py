#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Segmentation masks -> detection boxes (for anomaly / defect datasets that ship pixel masks, e.g. VisA, MVTec).

Defect masks are fragmented (a scratch breaks into several blobs, a melt spot comes with 1-3 px specks), so per
mask value: dilate by `merge_px` to join fragments that belong together, take connected components of the dilated
map, box the ORIGINAL pixels of each component, then drop boxes whose longer side is under `min_side`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import DataError


@dataclass(frozen=True)
class MaskBox:
    value: int      # mask value (defect type id in the dataset's own numbering)
    x1: int
    y1: int
    x2: int         # exclusive
    y2: int
    pixels: int     # defect pixels inside the box

    @property
    def max_side(self) -> int:
        return max(self.x2 - self.x1, self.y2 - self.y1)


def mask_to_boxes(mask: np.ndarray, merge_px: int = 15, min_side: int = 8,
                  values: dict[int, object] | None = None) -> list[MaskBox]:
    """Boxes for every non-zero value of a 2-D integer mask (see module doc). `values` restricts to those keys.
    A mask with no defect pixels gives []. Raises DataError for non-2-D input or negative parameters."""
    import cv2

    if mask is None or mask.ndim != 2:
        raise DataError(f"mask must be a 2-D array, got shape {None if mask is None else mask.shape}")
    if merge_px < 0 or min_side < 0:
        raise DataError(f"merge_px and min_side must be >= 0, got {merge_px}, {min_side}")
    out: list[MaskBox] = []
    present = [int(v) for v in np.unique(mask) if v != 0]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * merge_px + 1,) * 2) if merge_px else None
    for v in present:
        if values is not None and v not in values:
            continue
        binary = (mask == v).astype(np.uint8)
        grouped = cv2.dilate(binary, kernel) if kernel is not None else binary
        n, labels = cv2.connectedComponents(grouped, connectivity=8)
        for k in range(1, n):
            ys, xs = np.nonzero((labels == k) & (binary > 0))
            if xs.size == 0:
                continue
            b = MaskBox(v, int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1, int(xs.size))
            if b.max_side >= min_side:
                out.append(b)
    return sorted(out, key=lambda b: (b.value, b.y1, b.x1))
