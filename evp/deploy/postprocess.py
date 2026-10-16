#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Reference post-processing (numpy). Mirrors runtime/cpp/src/postprocess.cpp.

Inputs are raw head tensors in ncnn layout (C, H, W), one per stride, C = 4*reg_max + nc:
  reg_max > 1 : DFL bins per side (YOLOv5u/v8/v11/v12), decoded by softmax expectation
  reg_max == 1: ltrb distances in grid units (YOLO26)
then class logits. Post methods:
  nms      : Ultralytics non_max_suppression (class-aware unless agnostic, IoU > thr suppresses)
  e2e_topk : Ultralytics Detect.postprocess for one-to-one heads (two-stage top-k, no NMS)
Classify output is (nc,) logits or probabilities.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .meta import ModelMeta, PostSpec
from .preprocess import Geometry

MAX_WH = 7680.0   # Ultralytics class offset for class-aware NMS
MAX_NMS = 30000   # Ultralytics caps NMS input to the top-30000 boxes
E2E_MIN_K = 300   # Ultralytics predictor runs the one-to-one top-k with k = max(max_det, 300)


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2, self.score, float(self.cls)]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def inv_sigmoid(p: float) -> float:
    p = min(max(p, 1e-7), 1 - 1e-7)
    return math.log(p / (1 - p))


def dfl_decode(bins: np.ndarray, reg_max: int) -> np.ndarray:
    """bins [N, 4*reg_max] -> distances [N, 4] (softmax expectation per side)."""
    b = bins.reshape(-1, 4, reg_max)
    b = b - b.max(-1, keepdims=True)
    e = np.exp(b)
    p = e / e.sum(-1, keepdims=True)
    return (p * np.arange(reg_max, dtype=np.float64)).sum(-1)


@dataclass
class Candidates:
    """Cells whose best class score > conf, across all strides."""
    boxes: np.ndarray    # [N, 4] xyxy in network-input pixels
    scores: np.ndarray   # [N, nc] sigmoid class scores (full vector, needed by e2e multi-class)
    best: np.ndarray     # [N] best class id
    best_score: np.ndarray  # [N]


def gather_candidates(outputs: Sequence[np.ndarray], strides: Sequence[int], nc: int, reg_max: int,
                      conf: float) -> Candidates:
    """Decode every cell whose best class score > conf into an xyxy box (network-input pixels).

    Anchors are cell centers (x + 0.5, y + 0.5) * stride, iterated stride by stride in row-major order;
    the C++ runtime uses the same order so ties resolve identically. Logits are pre-filtered against
    inv_sigmoid(conf) only as a speed-up; the final test is the strict `score > conf` on sigmoid values."""
    if len(outputs) != len(strides):
        raise ValueError(f"{len(outputs)} outputs but {len(strides)} strides")
    box_ch = 4 * reg_max
    thr = inv_sigmoid(conf) if conf > 0 else -np.inf
    boxes, scores = [], []
    for out, s in zip(outputs, strides):
        out = np.asarray(out, np.float32)
        if out.ndim != 3 or out.shape[0] != box_ch + nc:
            raise ValueError(f"head output shape {out.shape} != ({box_ch}+{nc}, H, W)")
        logits = out[box_ch:]
        ys, xs = np.nonzero(logits.max(0) > thr)
        if len(ys) == 0:
            continue
        raw = out[:box_ch, ys, xs].T.astype(np.float64)
        dist = dfl_decode(raw, reg_max) if reg_max > 1 else raw
        ax, ay = xs + 0.5, ys + 0.5
        boxes.append(np.stack([(ax - dist[:, 0]) * s, (ay - dist[:, 1]) * s,
                               (ax + dist[:, 2]) * s, (ay + dist[:, 3]) * s], 1))
        scores.append(sigmoid(logits[:, ys, xs].T.astype(np.float64)))
    if not boxes:
        z = np.zeros((0,))
        return Candidates(np.zeros((0, 4)), np.zeros((0, nc)), z.astype(np.int64), z)
    b = np.concatenate(boxes, 0)
    sc = np.concatenate(scores, 0)
    best = sc.argmax(1)
    best_score = sc[np.arange(len(sc)), best]
    keep = best_score > conf   # strict, like Ultralytics (logit pre-filter is only a speed-up)
    return Candidates(b[keep], sc[keep], best[keep], best_score[keep])


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = a + b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def nms(dets: np.ndarray, iou_thr: float, agnostic: bool, max_det: int) -> np.ndarray:
    """dets [N, 6] -> kept rows, greedy on score order; suppress when IoU > iou_thr."""
    if len(dets) == 0:
        return dets
    order = np.argsort(-dets[:, 4], kind="stable")[:MAX_NMS]
    dets = dets[order]
    boxes = dets[:, :4].copy()
    if not agnostic:
        boxes += dets[:, 5:6] * MAX_WH
    keep: list[int] = []
    suppressed = np.zeros(len(dets), bool)
    for i in range(len(dets)):
        if suppressed[i]:
            continue
        keep.append(i)
        if len(keep) >= max_det:
            break
        rest = np.nonzero(~suppressed[i + 1:])[0] + i + 1
        if len(rest):
            suppressed[rest[iou_one_to_many(boxes[i], boxes[rest]) > iou_thr]] = True
    return dets[keep]


def e2e_topk(c: Candidates, conf: float, max_det: int, agnostic: bool) -> np.ndarray:
    """Ultralytics Detect.get_topk_index: top-k anchors by best score, then top-k (anchor, class) pairs."""
    if len(c.boxes) == 0:
        return np.zeros((0, 6))
    k = max(max_det, E2E_MIN_K)
    a_order = np.argsort(-c.best_score, kind="stable")[:k]
    if agnostic:
        rows = [(*c.boxes[a], c.best_score[a], c.best[a]) for a in a_order[:max_det]]
        return np.asarray(rows, np.float64).reshape(-1, 6)
    sub = c.scores[a_order]                       # [k, nc]
    flat = sub.reshape(-1)
    p_order = np.argsort(-flat, kind="stable")[:k]
    rows = []
    for p in p_order:
        s = flat[p]
        if s <= conf or len(rows) >= max_det:
            break
        a = a_order[p // sub.shape[1]]
        rows.append((*c.boxes[a], s, p % sub.shape[1]))
    return np.asarray(rows, np.float64).reshape(-1, 6)


def to_source(dets: np.ndarray, g: Geometry) -> list[Detection]:
    """Network-space [N, 6] rows -> Detection in source-image pixels, clipped to the image (Ultralytics
    scale_boxes behaviour)."""
    out = []
    for x1, y1, x2, y2, s, cl in dets:
        sx1, sy1 = g.to_source(x1, y1)
        sx2, sy2 = g.to_source(x2, y2)
        sx1, sx2 = min(max(sx1, 0.0), g.src_w), min(max(sx2, 0.0), g.src_w)
        sy1, sy2 = min(max(sy1, 0.0), g.src_h), min(max(sy2, 0.0), g.src_h)
        out.append(Detection(float(sx1), float(sy1), float(sx2), float(sy2), float(s), int(cl)))
    return out


def detect_network_space(outputs: Sequence[np.ndarray], meta: ModelMeta, post: PostSpec | None = None) -> np.ndarray:
    """[N, 6] (x1, y1, x2, y2, score, cls) in network-input pixels, before un-letterboxing."""
    post = post or meta.post
    c = gather_candidates(outputs, [o.stride for o in meta.outputs], meta.num_classes, meta.reg_max, post.conf)
    if post.method == "e2e_topk":
        return e2e_topk(c, post.conf, post.max_det, post.agnostic)
    dets = np.concatenate([c.boxes, c.best_score[:, None], c.best[:, None].astype(np.float64)], 1) \
        if len(c.boxes) else np.zeros((0, 6))
    return nms(dets, post.iou, post.agnostic, post.max_det)


def postprocess_detect(outputs: Sequence[np.ndarray], meta: ModelMeta, g: Geometry,
                       post: PostSpec | None = None) -> list[Detection]:
    return to_source(detect_network_space(outputs, meta, post), g)


def postprocess_classify(output: np.ndarray, post: PostSpec) -> list[tuple[int, float]]:
    """Top-k (class, score), optional softmax first; stable sort so equal scores keep class order."""
    v = np.asarray(output, np.float64).reshape(-1)
    if post.softmax:
        v = np.exp(v - v.max())
        v = v / v.sum()
    k = min(post.topk, len(v))
    order = np.argsort(-v, kind="stable")[:k]
    return [(int(i), float(v[i])) for i in order]
