#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Every rule must keep the graph numerically equivalent (onnxruntime before vs after), with any work moved
out of the graph reflected in meta updates."""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from evp.deploy.onnx_rules import (  # noqa: E402
    apply_rules,
    cut_outputs,
    einsum_to_matmul,
    extract_input_normalization,
    merge_meta_updates,
    op_inventory,
    reduce_l2_decompose,
    risky_ops,
    strip_input_cast,
    strip_input_transpose,
    suggest_cut,
)
from evp.errors import ConvertError, UnsupportedOpError  # noqa: E402

RNG = np.random.default_rng(0)


def run(model, feeds: dict) -> list[np.ndarray]:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


def make_model(nodes, inputs, outputs, inits=(), opset=17):
    """Outputs may be declared without a shape; it is filled from shape inference so the checker accepts them."""
    g = helper.make_graph(nodes, "g", inputs, outputs, initializer=list(inits))
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
    m.ir_version = 8
    inferred = onnx.shape_inference.infer_shapes(m)
    known = {v.name: v for v in list(inferred.graph.value_info) + list(inferred.graph.output)}
    for o in m.graph.output:
        if not o.type.tensor_type.HasField("shape") and o.name in known:
            o.type.tensor_type.shape.CopyFrom(known[o.name].type.tensor_type.shape)
    onnx.checker.check_model(m)
    return m


def run_named(model, feeds: dict, name: str) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([name], feeds)[0]


def conv_model(prefix_nodes, prefix_inits, x_shape=(1, 3, 8, 8), x_type=TensorProto.FLOAT, last="xn"):
    w = numpy_helper.from_array(RNG.normal(0, 1, (4, 3, 3, 3)).astype(np.float32), "W")
    nodes = list(prefix_nodes) + [helper.make_node("Conv", [last, "W"], ["y"], pads=[1, 1, 1, 1])]
    return make_model(nodes, [helper.make_tensor_value_info("x", x_type, list(x_shape))],
                      [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)], [w, *prefix_inits])


def const(name, arr):
    return numpy_helper.from_array(np.asarray(arr, np.float32), name)


# ---------------------------------------------------------------- normalization extraction
def apply_norm(x_chw_pixels, mean, norm):
    return ((x_chw_pixels - np.asarray(mean, np.float32)[:, None, None]) * np.asarray(norm, np.float32)[:, None, None])


def test_extract_normalization_div_sub_div():
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    m = conv_model([helper.make_node("Div", ["x", "s255"], ["a"]), helper.make_node("Sub", ["a", "mean"], ["b"]),
                    helper.make_node("Div", ["b", "std"], ["xn"])],
                   [const("s255", 255.0), const("mean", mean), const("std", std)])
    x = RNG.uniform(0, 255, (1, 3, 8, 8)).astype(np.float32)
    ref = run(m, {"x": x})[0]
    r = extract_input_normalization(m)
    assert r.changed and op_inventory(m) == {"Conv": 1}
    xin = apply_norm(x[0], r.meta_updates["input.mean"], r.meta_updates["input.norm"])[None]
    np.testing.assert_allclose(run(m, {"x": xin.astype(np.float32)})[0], ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(r.meta_updates["input.mean"], (mean * 255).reshape(-1), rtol=1e-6)


def test_extract_normalization_mul_add():
    m = conv_model([helper.make_node("Mul", ["x", "a"], ["t"]), helper.make_node("Add", ["t", "b"], ["xn"])],
                   [const("a", [[[2.0]], [[3.0]], [[4.0]]]), const("b", [[[1.0]], [[-1.0]], [[0.5]]])])
    x = RNG.uniform(0, 1, (1, 3, 8, 8)).astype(np.float32)
    ref = run(m, {"x": x})[0]
    r = extract_input_normalization(m)
    assert r.changed
    xin = apply_norm(x[0], r.meta_updates["input.mean"], r.meta_updates["input.norm"])[None].astype(np.float32)
    np.testing.assert_allclose(run(m, {"x": xin})[0], ref, rtol=1e-4, atol=1e-4)


def test_extract_normalization_rejects_1d_constant_on_nchw():
    """A [C] constant on an NCHW tensor broadcasts along W; folding it as per-channel would be wrong."""
    m = conv_model([helper.make_node("Sub", ["x", "m"], ["xn"])], [const("m", [1.0, 2.0, 3.0])], x_shape=(1, 3, 8, 3))
    assert not extract_input_normalization(m).changed


def test_extract_normalization_nhwc_then_transpose():
    """Normalization in NHWC before the layout Transpose: fold, then strip the transpose (default rule order)."""
    m = conv_model([helper.make_node("Sub", ["x", "m"], ["a"]), helper.make_node("Mul", ["a", "s"], ["b"]),
                    helper.make_node("Transpose", ["b"], ["xn"], perm=[0, 3, 1, 2])],
                   [const("m", [10.0, 20.0, 30.0]), const("s", [0.1, 0.2, 0.3])], x_shape=(1, 8, 8, 3))
    x = RNG.uniform(0, 255, (1, 8, 8, 3)).astype(np.float32)
    ref = run(m, {"x": x})[0]
    m, results = apply_rules(m)
    upd = merge_meta_updates(results)
    assert op_inventory(m) == {"Conv": 1}
    assert [i.dim_value for i in m.graph.input[0].type.tensor_type.shape.dim] == [1, 3, 8, 8]
    xin = apply_norm(x[0].transpose(2, 0, 1), upd["input.mean"], upd["input.norm"])[None].astype(np.float32)
    np.testing.assert_allclose(run(m, {"x": xin})[0], ref, rtol=1e-4, atol=1e-3)


def test_extract_normalization_skips_shared_input_and_output_feed():
    # input consumed twice -> not a pure preprocessing chain
    w = numpy_helper.from_array(RNG.normal(0, 1, (4, 3, 1, 1)).astype(np.float32), "W")
    m = make_model([helper.make_node("Sub", ["x", "m"], ["a"]), helper.make_node("Conv", ["a", "W"], ["y"]),
                    helper.make_node("Conv", ["x", "W"], ["z"])],
                   [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])],
                   [helper.make_tensor_value_info("y", TensorProto.FLOAT, None),
                    helper.make_tensor_value_info("z", TensorProto.FLOAT, None)], [w, const("m", 1.0)])
    assert not extract_input_normalization(m).changed
    m2 = make_model([helper.make_node("Sub", ["x", "m"], ["y"])],
                    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])],
                    [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)], [const("m", 1.0)])
    assert not extract_input_normalization(m2).changed


def test_extract_normalization_div_by_zero_not_folded():
    m = conv_model([helper.make_node("Div", ["x", "z"], ["xn"])], [const("z", [[[1.0]], [[0.0]], [[1.0]]])])
    assert not extract_input_normalization(m).changed


def test_merge_meta_updates_composes():
    from evp.deploy.onnx_rules import RuleResult

    r1 = RuleResult("a", True, meta_updates={"input.mean": [10.0] * 3, "input.norm": [0.5] * 3})
    r2 = RuleResult("b", True, meta_updates={"input.mean": [2.0] * 3, "input.norm": [0.1] * 3, "input.layout": "NCHW"})
    upd = merge_meta_updates([r1, r2])
    p = 100.0
    assert ((p - upd["input.mean"][0]) * upd["input.norm"][0]) == pytest.approx(((p - 10) * 0.5 - 2) * 0.1)
    assert upd["input.layout"] == "NCHW"


# ---------------------------------------------------------------- input transpose / cast
def test_strip_input_transpose():
    m = conv_model([helper.make_node("Transpose", ["x"], ["xn"], perm=[0, 3, 1, 2])], [], x_shape=(1, 8, 8, 3))
    x = RNG.uniform(0, 1, (1, 8, 8, 3)).astype(np.float32)
    ref = run(m, {"x": x})[0]
    r = strip_input_transpose(m)
    assert r.changed
    np.testing.assert_allclose(run(m, {"x": x.transpose(0, 3, 1, 2).copy()})[0], ref, rtol=1e-5, atol=1e-5)
    assert not strip_input_transpose(m).changed


def test_strip_input_transpose_other_perm_untouched():
    m = conv_model([helper.make_node("Transpose", ["x"], ["xn"], perm=[0, 1, 3, 2])], [], x_shape=(1, 3, 8, 8))
    assert not strip_input_transpose(m).changed


def test_strip_input_cast():
    m = conv_model([helper.make_node("Cast", ["x"], ["xn"], to=TensorProto.FLOAT)], [], x_type=TensorProto.UINT8)
    x = RNG.integers(0, 255, (1, 3, 8, 8), dtype=np.uint8)
    ref = run(m, {"x": x})[0]
    assert strip_input_cast(m).changed
    np.testing.assert_allclose(run(m, {"x": x.astype(np.float32)})[0], ref, rtol=1e-5)


# ---------------------------------------------------------------- op replacement
@pytest.mark.parametrize("eq, a_shape, b_shape", [
    ("ij,jk->ik", (4, 5), (5, 6)),
    ("bij,bjk->bik", (2, 4, 5), (2, 5, 3)),
    ("bhij,bhjd->bhid", (1, 2, 4, 5), (1, 2, 5, 3)),
    ("bhid,bhjd->bhij", (1, 2, 4, 8), (1, 2, 6, 8)),
    ("bid,bjd->bij", (2, 4, 8), (2, 6, 8)),
    ("ik,jk->ij", (4, 8), (6, 8)),
])
def test_einsum_to_matmul(eq, a_shape, b_shape):
    m = make_model([helper.make_node("Einsum", ["a", "b"], ["y"], equation=eq)],
                   [helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape)),
                    helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b_shape))],
                   [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)])
    feeds = {"a": RNG.normal(0, 1, a_shape).astype(np.float32), "b": RNG.normal(0, 1, b_shape).astype(np.float32)}
    ref = run(m, feeds)[0]
    r = einsum_to_matmul(m)
    assert r.changed and "Einsum" not in op_inventory(m)
    np.testing.assert_allclose(run(m, feeds)[0], ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("eq, want", [
    ("ij,jk->ik", False), ("nqd,nkd->nqk", True), ("bhqk,bhkd->bhqd", False), ("ik,jk->ij", True),
    ("ij,ij->i", None), ("ij,jk->ki", None), ("bij,jk->bik", None), ("...ij,...jk->...ik", None),
    ("iij,jk->ik", None), ("ij,jk", None), ("i,i->", None), ("bij,bjk,bkl->bil", None), ("bij,cjk->bik", None),
])
def test_einsum_matmul_plan(eq, want):
    from evp.deploy.onnx_rules import einsum_matmul_plan

    assert einsum_matmul_plan(eq) is want


def test_einsum_unknown_pattern_reported():
    m = make_model([helper.make_node("Einsum", ["a", "b"], ["y"], equation="ij,ij->i")],
                   [helper.make_tensor_value_info("a", TensorProto.FLOAT, [3, 4]),
                    helper.make_tensor_value_info("b", TensorProto.FLOAT, [3, 4])],
                   [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)])
    r = einsum_to_matmul(m)
    assert not r.changed and "not handled" in r.notes[0]


@pytest.mark.parametrize("opset, keepdims", [(13, 1), (13, 0), (18, 1)])
def test_reduce_l2_decompose(opset, keepdims):
    if opset >= 18:
        node = helper.make_node("ReduceL2", ["x", "axes"], ["y"], keepdims=keepdims)
        inits = [numpy_helper.from_array(np.array([1], np.int64), "axes")]
    else:
        node = helper.make_node("ReduceL2", ["x"], ["y"], axes=[1], keepdims=keepdims)
        inits = []
    m = make_model([node], [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 5, 3])],
                   [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)], inits, opset=opset)
    x = RNG.normal(0, 1, (2, 5, 3)).astype(np.float32)
    ref = run(m, {"x": x})[0]
    assert reduce_l2_decompose(m).changed
    assert op_inventory(m) == {"ReduceSumSquare": 1, "Sqrt": 1}
    np.testing.assert_allclose(run(m, {"x": x})[0], ref, rtol=1e-5)


# ---------------------------------------------------------------- cut / risky / apply_rules
def postproc_model():
    """conv -> head (out 'feat') -> Sigmoid -> TopK -> outputs (post-processing that ncnn can't run)."""
    w = numpy_helper.from_array(RNG.normal(0, 1, (6, 3, 1, 1)).astype(np.float32), "W")
    k = numpy_helper.from_array(np.array([5], np.int64), "K")
    nodes = [helper.make_node("Conv", ["x", "W"], ["feat"]),
             helper.make_node("Reshape", ["feat", "shape"], ["flat"]),
             helper.make_node("Sigmoid", ["flat"], ["sig"]),
             helper.make_node("TopK", ["sig", "K"], ["vals", "idx"], axis=-1)]
    shape = numpy_helper.from_array(np.array([1, -1], np.int64), "shape")
    return make_model(nodes, [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])],
                      [helper.make_tensor_value_info("vals", TensorProto.FLOAT, None),
                       helper.make_tensor_value_info("idx", TensorProto.INT64, None)], [w, k, shape])


def test_risky_and_suggest_cut():
    m = postproc_model()
    assert "TopK" in risky_ops(m)
    assert suggest_cut(m) == ["sig"]


def test_apply_rules_raises_with_suggested_cut():
    with pytest.raises(UnsupportedOpError) as ei:
        apply_rules(postproc_model())
    assert ei.value.ops == ["TopK"] and "sig" in str(ei.value)


def test_apply_rules_auto_cut():
    src = postproc_model()
    ref = onnx.ModelProto()
    ref.CopyFrom(src)
    ref.graph.output.append(helper.make_tensor_value_info("sig", TensorProto.FLOAT, [1, 96]))
    m, results = apply_rules(src, cut="auto")
    assert [o.name for o in m.graph.output] == ["sig"]
    assert "TopK" not in op_inventory(m)
    x = RNG.uniform(0, 1, (1, 3, 4, 4)).astype(np.float32)
    np.testing.assert_allclose(run(m, {"x": x})[0], run_named(ref, {"x": x}, "sig"), rtol=1e-6)


def test_cut_outputs_errors_and_noop():
    m = postproc_model()
    assert not cut_outputs(m, []).changed
    with pytest.raises(ConvertError, match="not in graph"):
        cut_outputs(m, ["nope"])


def test_cut_prunes_unused_initializers():
    m = postproc_model()
    cut_outputs(m, ["feat"])
    assert {t.name for t in m.graph.initializer} == {"W"}
    assert op_inventory(m) == {"Conv": 1}


def test_apply_rules_unknown_rule():
    with pytest.raises(ConvertError, match="unknown ONNX rule"):
        apply_rules(postproc_model(), rules=["bogus"], cut=["sig"])


def test_apply_rules_none_keeps_graph():
    m = conv_model([helper.make_node("Sub", ["x", "m"], ["xn"])], [const("m", 1.0)])
    m, results = apply_rules(m, rules="none")
    assert op_inventory(m) == {"Conv": 1, "Sub": 1} and results == []


def test_apply_rules_not_failing_when_disabled():
    m, _ = apply_rules(postproc_model(), fail_on_risky=False)
    assert "TopK" in op_inventory(m)
