#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Write a tiny generic ONNX used by the operator-rewrite example.

Graph: uint8 NHWC input -> Cast(float) -> Transpose(NHWC to NCHW) -> ReduceL2.
See docs/DEPLOY_TOOLCHAIN.md, section 「配套」.

    python examples/onnx_rewrite/build_demo_onnx.py
    evp inspect-onnx examples/onnx_rewrite/cast_transpose_l2.onnx --out examples/onnx_rewrite/prepared.onnx
"""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper

OUT = Path(__file__).resolve().parent / "cast_transpose_l2.onnx"


def build() -> onnx.ModelProto:
    """uint8 NHWC, then Cast, NHWC->NCHW Transpose, and ReduceL2 on the channel axis."""
    inp = helper.make_tensor_value_info("images", TensorProto.UINT8, [1, 8, 8, 3])
    out = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 8, 8])
    nodes = [
        helper.make_node("Cast", ["images"], ["x"], to=TensorProto.FLOAT),
        helper.make_node("Transpose", ["x"], ["y"], perm=[0, 3, 1, 2]),
        helper.make_node("ReduceL2", ["y"], ["z"], axes=[1], keepdims=0),
    ]
    graph = helper.make_graph(nodes, "demo", [inp], [out])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])


def main() -> None:
    model = build()
    onnx.save(model, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
