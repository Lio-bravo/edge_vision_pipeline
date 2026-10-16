#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""ONNX graph rules: inventory, risky-op detection, cut points and operator replacement.

Every rule is numerically verified in tests/test_onnx_rules.py with onnxruntime (before vs after).
Rules that move work out of the graph return `meta_updates` so pre/post-processing stays paired:
  - extract_input_normalization -> meta.input.mean / meta.input.norm
  - strip_input_transpose       -> network takes CHW (ncnn native)
  - strip_input_cast            -> network takes float (ncnn native)
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..errors import ConvertError, UnsupportedOpError

# ops that ncnn cannot run (or pnnx cannot lower) -> why + what to do
RISKY_OPS: dict[str, str] = {
    "TopK": "post-processing; cut before it and do top-k in C++",
    "NonMaxSuppression": "post-processing; cut before it and use the C++ NMS",
    "NonZero": "data-dependent shape; cut before it",
    "ScatterND": "no ncnn equivalent; rewrite in PyTorch (index_put) or cut",
    "ScatterElements": "no ncnn equivalent; rewrite or cut",
    "GatherND": "no ncnn equivalent; rewrite with Slice/Reshape or cut",
    "GatherElements": "no ncnn equivalent; rewrite or cut",
    "Loop": "control flow; unroll in PyTorch before export",
    "If": "control flow; export each branch as its own model",
    "Unique": "data-dependent shape; cut",
    "Compress": "data-dependent shape; cut",
    "RoiAlign": "use ncnn ROIAlign via a custom post stage",
    "Range": "dynamic shape; export with fixed input size and constant folding",
    "ConstantOfShape": "dynamic shape; fix input size and fold constants",
    "Shape": "dynamic shape; fix input size and fold constants",
}
POSTPROCESS_OPS = {"TopK", "NonMaxSuppression", "NonZero", "Unique", "Compress"}


@dataclass
class RuleResult:
    name: str
    changed: bool
    notes: list[str] = field(default_factory=list)
    meta_updates: dict = field(default_factory=dict)


def _onnx():
    import onnx

    return onnx


def op_inventory(model) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for n in model.graph.node:
        out[n.op_type] += 1
    return dict(sorted(out.items()))


def risky_ops(model) -> dict[str, str]:
    return {op: RISKY_OPS[op] for op in op_inventory(model) if op in RISKY_OPS}


def _consumers(graph) -> dict[str, list]:
    cons: dict[str, list] = defaultdict(list)
    for n in graph.node:
        for i in n.input:
            if i:
                cons[i].append(n)
    return cons


def _producer(graph) -> dict[str, object]:
    return {o: n for n in graph.node for o in n.output}


def _single_input(graph):
    """The only non-initializer graph input, or None (old IR versions list initializers as inputs)."""
    init_names = {t.name for t in graph.initializer}
    real = [i for i in graph.input if i.name not in init_names]
    return real[0] if len(real) == 1 else None


def _initializers(graph) -> dict[str, np.ndarray]:
    from onnx import numpy_helper

    return {t.name: numpy_helper.to_array(t) for t in graph.initializer}


def _const_value(graph, name: str, inits: dict[str, np.ndarray] | None = None) -> np.ndarray | None:
    from onnx import numpy_helper

    inits = inits if inits is not None else _initializers(graph)
    if name in inits:
        return inits[name]
    for n in graph.node:
        if n.op_type == "Constant" and n.output and n.output[0] == name:
            for a in n.attribute:
                if a.name == "value":
                    return numpy_helper.to_array(a.t)
    return None


def prune_dead(model) -> int:
    """Remove nodes/initializers not reachable from graph outputs. Returns removed node count."""
    g = model.graph
    prod = _producer(g)
    needed = {o.name for o in g.output}
    keep = set()
    stack = list(needed)
    while stack:
        t = stack.pop()
        n = prod.get(t)
        if n is None or id(n) in keep:
            continue
        keep.add(id(n))
        stack.extend(i for i in n.input if i)
    old = list(g.node)
    new = [n for n in old if id(n) in keep]
    removed = len(old) - len(new)
    if removed:
        del g.node[:]
        g.node.extend(new)
    used = {i for n in g.node for i in n.input} | {o.name for o in g.output}
    inits = [t for t in g.initializer if t.name in used]
    if len(inits) != len(g.initializer):
        del g.initializer[:]
        g.initializer.extend(inits)
    return removed


def cut_outputs(model, names: list[str]) -> RuleResult:
    """Make `names` the graph outputs and drop everything downstream."""
    onnx = _onnx()
    if not names:
        return RuleResult("cut_outputs", False)
    g = model.graph
    known = {o for n in g.node for o in n.output} | {i.name for i in g.input}
    missing = [n for n in names if n not in known]
    if missing:
        raise ConvertError(f"cut_outputs: tensors not in graph: {missing}",
                           "list tensor names with `evp inspect-onnx model.onnx`")
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
        vi = {v.name: v for v in list(inferred.graph.value_info) + list(inferred.graph.output)}
    except Exception:  # noqa: BLE001
        vi = {}
    del g.output[:]
    for n in names:
        if n in vi:
            g.output.append(vi[n])
        else:
            g.output.append(onnx.helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, None))
    removed = prune_dead(model)
    return RuleResult("cut_outputs", True, [f"outputs -> {names}; removed {removed} downstream nodes"])


def suggest_cut(model, ops: set[str] | None = None) -> list[str]:
    """Tensors at the boundary of the post-processing subgraph (taint = risky op + descendants)."""
    g = model.graph
    ops = ops or POSTPROCESS_OPS
    cons = _consumers(g)
    tainted: set[int] = set()
    q = deque(n for n in g.node if n.op_type in ops)
    while q:
        n = q.popleft()
        if id(n) in tainted:
            continue
        tainted.add(id(n))
        for o in n.output:
            q.extend(cons.get(o, []))
    if not tainted:
        return []
    prod = _producer(g)
    inits = {t.name for t in g.initializer}
    frontier: list[str] = []
    for n in g.node:
        if id(n) not in tainted:
            continue
        for i in n.input:
            p = prod.get(i)
            if not i or i in inits or (p is not None and p.op_type == "Constant"):
                continue
            if p is None:   # a graph input: nothing to cut before it
                continue
            if id(p) not in tainted:
                if i not in frontier:
                    frontier.append(i)
    for o in g.output:
        p = prod.get(o.name)
        if p is not None and id(p) not in tainted and o.name not in frontier:
            frontier.append(o.name)
    return frontier


def _broadcast_c(v: np.ndarray, c: int, layout: str = "NCHW") -> np.ndarray | None:
    """Scalar or per-channel constant -> float64 [C]; None if it is not per-channel for this layout.

    NCHW: [C,1,1] / [1,C,1,1]  (a 1-D [C] would broadcast along W, not channels)
    NHWC: [C] / [1,1,1,C] / [1,1,C]
    """
    v = np.asarray(v, np.float64)
    if v.size == 1:
        return np.full(c, float(v.reshape(-1)[0]))
    if v.size != c:
        return None
    if layout == "NCHW" and v.ndim >= 3 and v.shape[-1] == 1 and v.shape[-2] == 1 and v.shape[-3] == c:
        return v.reshape(-1)
    if layout == "NHWC" and v.shape[-1] == c:
        return v.reshape(-1)
    return None


def _input_layout(inp, channels: int) -> str | None:
    """NCHW / NHWC from a 4-D input shape; None if it cannot be told apart."""
    dims = inp.type.tensor_type.shape.dim
    if len(dims) != 4:
        return None
    shape = [d.dim_value if d.HasField("dim_value") else -1 for d in dims]
    if shape[1] == channels and shape[3] != channels:
        return "NCHW"
    if shape[3] == channels and shape[1] != channels:
        return "NHWC"
    return "NCHW" if shape[1] == channels else None


def extract_input_normalization(model, channels: int = 3) -> RuleResult:
    """Fold a leading chain of Sub/Add/Mul/Div-by-constant on the input into meta mean/norm.

    y = a*x + b per channel  ==  (x - mean) * norm  with norm = a, mean = -b / a.
    """
    g = model.graph
    init_names = {t.name for t in g.initializer}
    real_inputs = [i for i in g.input if i.name not in init_names]
    if len(real_inputs) != 1:
        return RuleResult("extract_input_normalization", False, ["skip: model has multiple inputs"])
    layout = _input_layout(real_inputs[0], channels)
    if layout is None:
        return RuleResult("extract_input_normalization", False, ["skip: cannot tell the channel axis of the input"])
    inp = real_inputs[0].name
    inits = _initializers(g)
    cons = _consumers(g)
    a = np.ones(channels)
    b = np.zeros(channels)
    chain = []
    cur = inp
    while True:
        users = cons.get(cur, [])
        if len(users) != 1:
            break
        n = users[0]
        if n.op_type not in ("Sub", "Add", "Mul", "Div") or len(n.input) != 2 or n.input[0] != cur:
            break
        c = _const_value(g, n.input[1], inits)
        if c is None:
            break
        cv = _broadcast_c(c, channels, layout)
        if cv is None or (n.op_type == "Div" and np.any(cv == 0)):
            break
        if n.op_type == "Sub":
            b = b - cv
        elif n.op_type == "Add":
            b = b + cv
        elif n.op_type == "Mul":
            a, b = a * cv, b * cv
        else:
            a, b = a / cv, b / cv
        chain.append(n)
        cur = n.output[0]
    if not chain:
        return RuleResult("extract_input_normalization", False)
    if np.any(a == 0):
        return RuleResult("extract_input_normalization", False, ["skip: zero scale"])
    last_out = chain[-1].output[0]
    if any(o.name == n.output[0] for o in g.output for n in chain):
        return RuleResult("extract_input_normalization", False, ["skip: normalization feeds output directly"])
    for n in g.node:
        for k, i in enumerate(n.input):
            if i == last_out:
                n.input[k] = inp
    for n in chain:
        g.node.remove(n)
    prune_dead(model)
    mean = (-b / a).tolist()
    norm = a.tolist()
    return RuleResult("extract_input_normalization", True,
                      [f"removed {len(chain)} input op(s) {[n.op_type for n in chain]}; mean={mean} norm={norm}"],
                      {"input.mean": mean, "input.norm": norm})


def strip_input_transpose(model) -> RuleResult:
    """NHWC input + Transpose(0,3,1,2) -> NCHW input (ncnn feeds CHW)."""
    onnx = _onnx()
    g = model.graph
    inp = _single_input(g)
    if inp is None:
        return RuleResult("strip_input_transpose", False)
    users = _consumers(g).get(inp.name, [])
    if len(users) != 1 or users[0].op_type != "Transpose":
        return RuleResult("strip_input_transpose", False)
    t = users[0]
    perm = next((list(a.ints) for a in t.attribute if a.name == "perm"), None)
    if perm != [0, 3, 1, 2]:
        return RuleResult("strip_input_transpose", False)
    dims = inp.type.tensor_type.shape.dim
    shape = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in dims]
    new_shape = [shape[0], shape[3], shape[1], shape[2]] if len(shape) == 4 else None
    out = t.output[0]
    for n in g.node:
        for k, i in enumerate(n.input):
            if i == out:
                n.input[k] = inp.name
    g.node.remove(t)
    if new_shape is not None:
        elem = inp.type.tensor_type.elem_type
        g.input.remove(inp)
        g.input.insert(0, onnx.helper.make_tensor_value_info(inp.name, elem, new_shape))
    return RuleResult("strip_input_transpose", True, ["input is now NCHW; removed leading Transpose"],
                      {"input.layout": "NCHW"})


def strip_input_cast(model) -> RuleResult:
    """uint8 input + Cast(to=float) -> float input (ncnn always feeds float)."""
    onnx = _onnx()
    g = model.graph
    inp = _single_input(g)
    if inp is None:
        return RuleResult("strip_input_cast", False)
    users = _consumers(g).get(inp.name, [])
    if len(users) != 1 or users[0].op_type != "Cast":
        return RuleResult("strip_input_cast", False)
    c = users[0]
    to = next((a.i for a in c.attribute if a.name == "to"), None)
    if to != onnx.TensorProto.FLOAT:
        return RuleResult("strip_input_cast", False)
    out = c.output[0]
    for n in g.node:
        for k, i in enumerate(n.input):
            if i == out:
                n.input[k] = inp.name
    g.node.remove(c)
    inp.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    return RuleResult("strip_input_cast", True, ["input is now float32; removed leading Cast"])


def einsum_matmul_plan(eq: str) -> bool | None:
    """For a 2-operand einsum equal to a batched MatMul, return whether B must be transposed
    (False: A[..., i, k] @ B[..., k, j]; True: A[..., i, k] @ B[..., j, k]^T). None if not a plain matmul."""
    eq = eq.replace(" ", "")
    if "->" not in eq or "." in eq:
        return None
    lhs, out = eq.split("->")
    ops = lhs.split(",")
    if len(ops) != 2:
        return None
    a, b = ops
    if any(len(set(s)) != len(s) for s in (a, b, out)) or len(a) < 2 or len(a) != len(b) or len(out) != len(a):
        return None
    batch = a[:-2]
    if b[:-2] != batch or out[:-2] != batch:
        return None
    i, k = a[-2:]
    if len({i, k} | set(batch)) != len(batch) + 2:
        return None
    if b[-2] == k and b[-1] not in a:
        j, trans = b[-1], False
    elif b[-1] == k and b[-2] not in a:
        j, trans = b[-2], True
    else:
        return None
    return trans if out[-2:] == i + j else None


def einsum_to_matmul(model) -> RuleResult:
    """Replace simple 2-operand Einsum patterns with MatMul (+Transpose)."""
    onnx = _onnx()
    g = model.graph
    notes = []
    changed = False
    for n in list(g.node):
        if n.op_type != "Einsum" or len(n.input) != 2:
            continue
        eq = next((a.s.decode() if isinstance(a.s, bytes) else a.s for a in n.attribute if a.name == "equation"), "")
        eq = eq.replace(" ", "")
        trans = einsum_matmul_plan(eq)
        if trans is None:
            notes.append(f"einsum '{eq}' not handled (node {n.name})")
            continue
        a_in, b_in = n.input
        idx = list(g.node).index(n)
        new_nodes = []
        if trans:
            rank = len(eq.split("->")[0].split(",")[1])
            perm = list(range(rank - 2)) + [rank - 1, rank - 2]
            bt = f"{n.output[0]}__bT"
            new_nodes.append(onnx.helper.make_node("Transpose", [b_in], [bt], perm=perm, name=f"{n.name}_bT"))
            b_in = bt
        new_nodes.append(onnx.helper.make_node("MatMul", [a_in, b_in], [n.output[0]], name=f"{n.name}_matmul"))
        g.node.remove(n)
        for k, nn in enumerate(new_nodes):
            g.node.insert(idx + k, nn)
        notes.append(f"einsum '{eq}' -> MatMul ({n.name})")
        changed = True
    return RuleResult("einsum_to_matmul", changed, notes)


def reduce_l2_decompose(model) -> RuleResult:
    """ReduceL2(x) -> Sqrt(ReduceSumSquare(x)) (keeps opset attributes/axes input)."""
    onnx = _onnx()
    g = model.graph
    changed = False
    for n in list(g.node):
        if n.op_type != "ReduceL2":
            continue
        idx = list(g.node).index(n)
        mid = f"{n.output[0]}__sumsq"
        rss = onnx.helper.make_node("ReduceSumSquare", list(n.input), [mid], name=f"{n.name}_sumsq")
        rss.attribute.extend(n.attribute)
        sq = onnx.helper.make_node("Sqrt", [mid], [n.output[0]], name=f"{n.name}_sqrt")
        g.node.remove(n)
        g.node.insert(idx, rss)
        g.node.insert(idx + 1, sq)
        changed = True
    return RuleResult("reduce_l2_decompose", changed, ["ReduceL2 -> Sqrt(ReduceSumSquare)"] if changed else [])


RULES: dict[str, Callable] = {
    "strip_input_cast": strip_input_cast,
    "strip_input_transpose": strip_input_transpose,
    "extract_input_normalization": extract_input_normalization,
    "einsum_to_matmul": einsum_to_matmul,
    "reduce_l2_decompose": reduce_l2_decompose,
}
# normalization may sit before (NHWC) or after (NCHW) an input Transpose, so it runs on both sides
DEFAULT_RULES = ["strip_input_cast", "extract_input_normalization", "strip_input_transpose",
                 "extract_input_normalization", "einsum_to_matmul", "reduce_l2_decompose"]


def merge_meta_updates(results: list[RuleResult]) -> dict:
    """Combine rule meta updates in application order. Successive normalizations compose:
    x1 = (p - m0) * n0 then x2 = (x1 - m1) * n1  ==  (p - (m0 + m1 / n0)) * (n0 * n1)."""
    out: dict = {}
    for r in results:
        upd = dict(r.meta_updates or {})
        if "input.mean" in upd and "input.mean" in out:
            m0, n0 = out["input.mean"], out["input.norm"]
            m1, n1 = upd.pop("input.mean"), upd.pop("input.norm")
            out["input.mean"] = [a + b / n for a, b, n in zip(m0, m1, n0)]
            out["input.norm"] = [a * b for a, b in zip(n0, n1)]
        out.update(upd)
    return out


def apply_rules(model, rules: list[str] | str = "default", cut: list[str] | str | None = None,
                fail_on_risky: bool = True) -> tuple[object, list[RuleResult]]:
    """Run cut + rules in a fixed order, then check remaining risky ops."""
    onnx = _onnx()
    results: list[RuleResult] = []
    if cut == "auto":
        cut = suggest_cut(model)
        if cut:
            results.append(RuleResult("suggest_cut", True, [f"auto cut at {cut}"]))
    if cut:
        results.append(cut_outputs(model, list(cut)))
    names = DEFAULT_RULES if rules in ("default", None) else ([] if rules == "none" else list(rules))
    for r in names:
        if r not in RULES:
            raise ConvertError(f"unknown ONNX rule {r!r}", f"available: {sorted(RULES)}")
        results.append(RULES[r](model))
    try:
        onnx.checker.check_model(model)
    except Exception as exc:  # noqa: BLE001
        raise ConvertError(f"ONNX invalid after rules: {exc}") from exc
    left = risky_ops(model)
    if left and fail_on_risky:
        cut_hint = suggest_cut(model)
        raise UnsupportedOpError(list(left), (
            "; ".join(f"{k}: {v}" for k, v in left.items())
            + (f". Suggested deploy.onnx.cut_outputs: {cut_hint}" if cut_hint else "")))
    return model, results
