#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Python ncnn runtime: the reference implementation of the deployed bundle (meta + param/bin)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..errors import EvpError
from .meta import ModelMeta, PostSpec, load_meta
from .postprocess import Detection, postprocess_classify, postprocess_detect
from .preprocess import Geometry, preprocess_ncnn


class NcnnNet:
    """Thin wrapper over ncnn.Net. fp16 storage/arithmetic are off by default so x86 results match
    Cortex-A7 (no fp16 arithmetic) and the fp32 reference."""

    def __init__(self, param: Path, bin_: Path, num_threads: int = 1, use_fp16: bool = False,
                 use_int8: bool = True):
        import ncnn

        param, bin_ = Path(param), Path(bin_)
        for f in (param, bin_):
            if not f.is_file():
                raise EvpError(f"ncnn model file not found: {f}")
        self.net = ncnn.Net()
        o = self.net.opt
        o.use_vulkan_compute = False
        o.num_threads = int(num_threads)
        o.use_fp16_packed = use_fp16
        o.use_fp16_storage = use_fp16
        o.use_fp16_arithmetic = use_fp16
        o.use_bf16_storage = False
        o.use_int8_inference = use_int8
        if self.net.load_param(str(param)) != 0:
            raise EvpError(f"ncnn failed to load param {param}", "is it a valid ncnn .param (magic 7767517)?")
        if self.net.load_model(str(bin_)) != 0:
            raise EvpError(f"ncnn failed to load weights {bin_}", "param/bin mismatch or truncated .bin")
        self.param_path, self.bin_path = param, bin_

    def forward(self, x, input_name: str, output_names: Sequence[str]) -> list[np.ndarray]:
        """x: ncnn.Mat or float32 CHW / 1-D numpy array. Returns numpy copies (C,H,W) / (W,)."""
        import ncnn

        mat = x if isinstance(x, ncnn.Mat) else ncnn.Mat(np.ascontiguousarray(x, dtype=np.float32))
        ex = self.net.create_extractor()
        if ex.input(input_name, mat) != 0:
            raise EvpError(f"ncnn input blob {input_name!r} not found")
        outs = []
        for name in output_names:
            ret, m = ex.extract(name)
            if ret != 0:
                raise EvpError(f"ncnn extract({name!r}) failed with {ret}")
            outs.append(np.array(m, dtype=np.float32, copy=True))
        return outs


@dataclass
class PipelineResult:
    geometry: Geometry
    outputs: list[np.ndarray]
    detections: list[Detection] | None = None
    topk: list[tuple[int, float]] | None = None


class NcnnPipeline:
    """meta-driven preprocess -> ncnn forward -> post-processing; the Python twin of the C++ evp::Model.
    `post` is a private copy of meta.post, so callers may tweak conf/iou without touching the meta."""

    def __init__(self, meta: ModelMeta, net: NcnnNet, precision: str):
        self.meta = meta
        self.net = net
        self.precision = precision
        self.post = replace(meta.post)

    @classmethod
    def from_bundle(cls, bundle_dir: Path, precision: str = "fp32", num_threads: int = 1) -> NcnnPipeline:
        bundle_dir = Path(bundle_dir)
        meta = load_meta(bundle_dir)
        if precision not in meta.files:
            raise EvpError(f"bundle {bundle_dir} has no {precision!r} model (available: {sorted(meta.files)})")
        f = meta.files[precision]
        return cls(meta, NcnnNet(bundle_dir / f.param, bundle_dir / f.bin, num_threads=num_threads), precision)

    def forward_image(self, img_bgr: np.ndarray) -> tuple[list[np.ndarray], Geometry]:
        mat, g = preprocess_ncnn(img_bgr, self.meta.input)
        return self.net.forward(mat, self.meta.input.blob, [o.name for o in self.meta.outputs]), g

    def run(self, img_bgr: np.ndarray, post: PostSpec | None = None) -> PipelineResult:
        outs, g = self.forward_image(img_bgr)
        post = post or self.post
        res = PipelineResult(g, outs)
        if self.meta.task == "detect":
            res.detections = postprocess_detect(outs, self.meta, g, post)
        elif self.meta.family == "classify":
            res.topk = postprocess_classify(outs[0], post)
        return res

    def detect(self, img_bgr: np.ndarray, post: PostSpec | None = None) -> list[Detection]:
        if self.meta.task != "detect":
            raise EvpError(f"model family {self.meta.family!r} is not a detector")
        return self.run(img_bgr, post).detections or []

    def classify(self, img_bgr: np.ndarray, post: PostSpec | None = None) -> list[tuple[int, float]]:
        if self.meta.family != "classify":
            raise EvpError(f"model family {self.meta.family!r} is not a classifier")
        return self.run(img_bgr, post).topk or []


def read_image(path: Path) -> np.ndarray:
    """cv2.imread that works with non-ASCII Windows paths."""
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise EvpError(f"cannot read image {path}")
    return img
