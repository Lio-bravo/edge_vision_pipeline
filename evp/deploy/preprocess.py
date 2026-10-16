#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Reference preprocessing. Mirrors runtime/cpp/src/preprocess.cpp line by line.

Geometry uses Python `round` (half-to-even) on doubles, matching Ultralytics LetterBox; C++ uses
`std::nearbyint` (FE_TONEAREST == half-to-even) on doubles for the same results.
Pixel ops use ncnn's own resize/border/normalize so Python and C++ produce identical tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .meta import InputSpec


@dataclass(frozen=True)
class Geometry:
    """Maps a source image (src_w x src_h) into the network input (dst_w x dst_h)."""
    src_w: int
    src_h: int
    dst_w: int
    dst_h: int
    resize_w: int        # size the source is resized to before pad / crop
    resize_h: int
    scale_x: float       # resized = src * scale
    scale_y: float
    pad_left: int = 0    # letterbox border
    pad_top: int = 0
    pad_right: int = 0
    pad_bottom: int = 0
    crop_x: int = 0      # center_crop offset inside the resized image
    crop_y: int = 0

    def to_source(self, x: float, y: float) -> tuple[float, float]:
        """Network-input pixel coords -> source image pixel coords."""
        return ((x - self.pad_left + self.crop_x) / self.scale_x,
                (y - self.pad_top + self.crop_y) / self.scale_y)


def letterbox_geometry(src_w: int, src_h: int, dst_w: int, dst_h: int, scaleup: bool = True) -> Geometry:
    """Ultralytics LetterBox(auto=False, center=True): uniform scale, then pad; the +-0.1 before rounding
    puts the odd padding pixel at the bottom/right exactly like Ultralytics."""
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"invalid source size {src_w}x{src_h}")
    r = min(dst_h / src_h, dst_w / src_w)
    if not scaleup:
        r = min(r, 1.0)
    # >= 1: extreme aspect ratios would otherwise round to an empty image (Ultralytics crashes there)
    new_w, new_h = max(1, int(round(src_w * r))), max(1, int(round(src_h * r)))
    dw, dh = (dst_w - new_w) / 2.0, (dst_h - new_h) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # guard: rounding can never make the canvas differ from dst
    if new_w + left + right != dst_w:
        right = dst_w - new_w - left
    if new_h + top + bottom != dst_h:
        bottom = dst_h - new_h - top
    return Geometry(src_w, src_h, dst_w, dst_h, new_w, new_h, r, r, left, top, right, bottom)


def stretch_geometry(src_w: int, src_h: int, dst_w: int, dst_h: int) -> Geometry:
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"invalid source size {src_w}x{src_h}")
    return Geometry(src_w, src_h, dst_w, dst_h, dst_w, dst_h, dst_w / src_w, dst_h / src_h)


def center_crop_geometry(src_w: int, src_h: int, dst_w: int, dst_h: int) -> Geometry:
    """torchvision Resize(short side) + CenterCrop, as used by Ultralytics classify_transforms."""
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"invalid source size {src_w}x{src_h}")
    size = min(dst_w, dst_h) if dst_w != dst_h else dst_w
    short, long_ = (src_w, src_h) if src_w <= src_h else (src_h, src_w)
    new_short, new_long = size, int(size * long_ / short)
    new_w, new_h = (new_short, new_long) if src_w <= src_h else (new_long, new_short)
    if dst_w != dst_h:
        # non-square crop: scale so both sides cover the crop (aspect preserved)
        r = max(dst_w / src_w, dst_h / src_h)
        new_w, new_h = max(dst_w, int(round(src_w * r))), max(dst_h, int(round(src_h * r)))
    crop_x = int(round((new_w - dst_w) / 2.0))
    crop_y = int(round((new_h - dst_h) / 2.0))
    return Geometry(src_w, src_h, dst_w, dst_h, new_w, new_h, new_w / src_w, new_h / src_h,
                    crop_x=crop_x, crop_y=crop_y)


def make_geometry(spec: InputSpec, src_w: int, src_h: int) -> Geometry:
    """Geometry for the meta's resize mode (letterbox | stretch | center_crop)."""
    if spec.resize == "letterbox":
        return letterbox_geometry(src_w, src_h, spec.w, spec.h)
    if spec.resize == "stretch":
        return stretch_geometry(src_w, src_h, spec.w, spec.h)
    if spec.resize == "center_crop":
        return center_crop_geometry(src_w, src_h, spec.w, spec.h)
    raise ValueError(f"unknown resize mode {spec.resize}")


def preprocess_ncnn(img_bgr: np.ndarray, spec: InputSpec):
    """BGR uint8 HxWx3 -> (ncnn.Mat CHW float, Geometry). Same calls as the C++ runtime."""
    import ncnn

    if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3 or img_bgr.dtype != np.uint8:
        raise ValueError("expected BGR uint8 image of shape HxWx3")
    img_bgr = np.ascontiguousarray(img_bgr)
    h, w = img_bgr.shape[:2]
    g = make_geometry(spec, w, h)
    pix = ncnn.Mat.PixelType.PIXEL_BGR2RGB if spec.color == "RGB" else ncnn.Mat.PixelType.PIXEL_BGR
    m = ncnn.Mat.from_pixels_resize(img_bgr, pix, w, h, g.resize_w, g.resize_h)
    if spec.resize == "letterbox" and (g.pad_left or g.pad_top or g.pad_right or g.pad_bottom):
        out = ncnn.Mat()
        ncnn.copy_make_border(m, out, g.pad_top, g.pad_bottom, g.pad_left, g.pad_right,
                              ncnn.BorderType.BORDER_CONSTANT, float(spec.pad_value))
        m = out
    elif spec.resize == "center_crop" and (g.resize_w != spec.w or g.resize_h != spec.h):
        out = ncnn.Mat()
        ncnn.copy_cut_border(m, out, g.crop_y, g.resize_h - spec.h - g.crop_y,
                             g.crop_x, g.resize_w - spec.w - g.crop_x)
        m = out
    if m.w != spec.w or m.h != spec.h:
        raise RuntimeError(f"preprocess produced {m.w}x{m.h}, expected {spec.w}x{spec.h}")
    m.substract_mean_normalize(list(map(float, spec.mean)), list(map(float, spec.norm)))
    return m, g


def preprocess_numpy(img_bgr: np.ndarray, spec: InputSpec) -> tuple[np.ndarray, Geometry]:
    """ncnn-free reference using cv2 (used for calibration .npy and as a cross-check)."""
    import cv2

    h, w = img_bgr.shape[:2]
    g = make_geometry(spec, w, h)
    img = img_bgr[:, :, ::-1] if spec.color == "RGB" else img_bgr
    if (g.resize_w, g.resize_h) != (w, h):
        img = cv2.resize(np.ascontiguousarray(img), (g.resize_w, g.resize_h), interpolation=cv2.INTER_LINEAR)
    if spec.resize == "letterbox":
        img = cv2.copyMakeBorder(np.ascontiguousarray(img), g.pad_top, g.pad_bottom, g.pad_left, g.pad_right,
                                 cv2.BORDER_CONSTANT, value=(spec.pad_value,) * 3)
    elif spec.resize == "center_crop":
        img = img[g.crop_y:g.crop_y + spec.h, g.crop_x:g.crop_x + spec.w]
    x = img.astype(np.float32).transpose(2, 0, 1)
    mean = np.asarray(spec.mean, np.float32)[:, None, None]
    norm = np.asarray(spec.norm, np.float32)[:, None, None]
    return np.ascontiguousarray((x - mean) * norm), g
