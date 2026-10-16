#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""ONNX -> (slim + operator rules) -> pnnx -> ncnn fp32, with structural and numeric checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..errors import ConvertError, UnsupportedOpError
from ..utils import ensure_dir, log
from .ncnn_param import NcnnParam, load_param, unsupported_layers
from .onnx_rules import RuleResult, apply_rules, merge_meta_updates, op_inventory
from .tools import ToolRunner, default_runner


@dataclass
class ConvertResult:
    """Everything the deploy pipeline needs from a conversion; `meta_updates` carries work moved out of the
    graph by the rules (input.mean / input.norm / input.layout) and must be folded into the meta."""
    onnx: Path                      # processed ONNX fed to pnnx
    param: Path
    bin: Path
    ncnn: NcnnParam
    input_name: str
    output_names: list[str]         # ncnn blob names, same order as the ONNX outputs
    rules: list[RuleResult] = field(default_factory=list)
    meta_updates: dict = field(default_factory=dict)
    onnx_ops: dict = field(default_factory=dict)
    onnx_input_shape: list[int] = field(default_factory=list)


def slim(model):
    """Constant folding / shape inference so Shape/Range/ConstantOfShape chains disappear before rule checks."""
    try:
        import onnxslim
    except ImportError:
        log.warning("onnxslim not installed; skipping graph simplification")
        return model
    try:
        return onnxslim.slim(model)
    except Exception as exc:  # noqa: BLE001
        log.warning("onnxslim failed (%s); continuing with the unsimplified graph", exc)
        return model


def _graph_input(model):
    graph_inputs = [i for i in model.graph.input if i.name not in {t.name for t in model.graph.initializer}]
    if len(graph_inputs) != 1:
        raise ConvertError(f"model must have exactly 1 input, found {[i.name for i in graph_inputs]}",
                           "multi-input models are not supported by the deploy pipeline")
    return graph_inputs[0]


def _static_input_shape(model) -> list[int]:
    return [int(d.dim_value) if d.dim_value > 0 else -1 for d in _graph_input(model).type.tensor_type.shape.dim]


def _is_nhwc(shape: list[int]) -> bool:
    return len(shape) == 4 and shape[3] in (1, 3) and shape[1] not in (1, 3)


def match_input_shape(nchw: list[int], model_shape: list[int]) -> list[int]:
    """The pipeline describes the input as NCHW; map it onto an NHWC model input when the model is NHWC."""
    if len(nchw) == 4 and _is_nhwc(model_shape):
        n, c, h, w = nchw
        return [n, h, w, c]
    return list(nchw)


def prepare_onnx(src: Path, dst: Path, rules="default", cut=None, fail_on_risky: bool = True,
                 input_shape: list[int] | None = None):
    """Load, fix dynamic dims, slim, apply operator rules, slim again, save.
    Returns (dst, rule_results, merged_meta_updates, op_inventory, input_shape_after_rules)."""
    import onnx

    src = Path(src)
    if not src.is_file():
        raise ConvertError(f"ONNX file not found: {src}")
    try:
        model = onnx.load(str(src))
    except Exception as exc:  # noqa: BLE001
        raise ConvertError(f"cannot load ONNX {src}: {exc}") from exc
    shape = _static_input_shape(model)
    if input_shape:
        input_shape = match_input_shape(input_shape, shape)
        if len(input_shape) != len(shape):
            raise ConvertError(f"input_shape rank {len(input_shape)} != model input rank {len(shape)} ({shape})")
        for i, (want, have) in enumerate(zip(input_shape, shape)):
            if have > 0 and want != have:
                raise ConvertError(f"input dim {i}: model has fixed {have}, config asks {want} (model input {shape})",
                                   "set deploy.imgsz to the model's [h, w]")
        dim = _graph_input(model).type.tensor_type.shape.dim
        for d, v in zip(dim, input_shape):
            d.ClearField("dim_param")
            d.dim_value = int(v)
        shape = list(input_shape)
    if any(s <= 0 for s in shape):
        raise ConvertError(f"model input has dynamic dims {shape}", "set deploy.imgsz so the shape can be fixed")
    model = slim(model)
    model, results = apply_rules(model, rules=rules, cut=cut, fail_on_risky=fail_on_risky)
    if any(r.changed for r in results):
        model = slim(model)
    meta_updates = merge_meta_updates(results)
    # strip_input_transpose turns NHWC into NCHW; re-read the shape after rules
    shape = _static_input_shape(model)
    dst = Path(dst)
    ensure_dir(dst.parent)
    onnx.save(model, str(dst))
    return dst, results, meta_updates, op_inventory(model), shape


def run_pnnx(onnx_path: Path, work_dir: Path, input_shape: list[int], runner: ToolRunner | None = None,
             fp16_weights: bool = False, stem: str = "model") -> tuple[Path, Path]:
    """ONNX -> <stem>.ncnn.param/.bin in `work_dir` (fixed `input_shape`). Every pnnx output path is passed
    explicitly so all artifacts land in work_dir under predictable names (stale ones are deleted first)."""
    runner = runner or default_runner()
    tool = runner.get("pnnx")
    work_dir = ensure_dir(work_dir)
    out = {k: work_dir / f"{stem}.{ext}" for k, ext in (
        ("ncnnparam", "ncnn.param"), ("ncnnbin", "ncnn.bin"), ("ncnnpy", "ncnn.py"),
        ("pnnxparam", "pnnx.param"), ("pnnxbin", "pnnx.bin"), ("pnnxpy", "pnnx.py"), ("pnnxonnx", "pnnx.onnx"))}
    for p in out.values():
        if p.exists():
            p.unlink()
    shape_s = "[" + ",".join(str(int(s)) for s in input_shape) + "]"
    args = [tool.path_arg(onnx_path), f"inputshape={shape_s}", f"fp16={1 if fp16_weights else 0}", "optlevel=2"]
    args += [f"{k}={tool.path_arg(v)}" for k, v in out.items()]
    proc = tool.run(args, cwd=work_dir)
    param, binf = out["ncnnparam"], out["ncnnbin"]
    if not param.is_file() or not binf.is_file():
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-20:])
        raise ConvertError(f"pnnx produced no ncnn model for {onnx_path}\n{tail}")
    (work_dir / "pnnx.log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    return param, binf


def check_param(param: Path, expected_outputs: int) -> NcnnParam:
    """Structural check of the converted model: only registered ncnn layers, 1 input, all outputs kept."""
    p = load_param(param)
    bad = unsupported_layers(p)
    if bad:
        raise UnsupportedOpError([l.type for l in bad],
                                 "pnnx emitted layers the ncnn runtime does not implement; add an ONNX "
                                 "replacement rule or cut the graph before them (docs/DEPLOY_TOOLCHAIN.md)")
    if len(p.inputs) != 1:
        raise ConvertError(f"ncnn model has {len(p.inputs)} inputs {p.inputs}; expected 1")
    if len(p.outputs) != expected_outputs:
        raise ConvertError(f"ncnn model has outputs {p.outputs}, ONNX had {expected_outputs}",
                           "pnnx may have merged/dropped outputs; inspect work/model.pnnx.param")
    return p


def _ordered_outputs(p: NcnnParam, n: int) -> list[str]:
    """pnnx names graph outputs out0..out{n-1} in ONNX order; fall back to param order."""
    want = [f"out{i}" for i in range(n)]
    return want if set(want) <= set(p.outputs) else list(p.outputs)


def convert_onnx(src: Path, work_dir: Path, rules="default", cut=None, fail_on_risky: bool = True,
                 input_shape: list[int] | None = None, runner: ToolRunner | None = None,
                 fp16_weights: bool = False, check_numeric: bool = True, min_cos: float = 0.999) -> ConvertResult:
    """Full ONNX -> ncnn fp32 conversion: prepare_onnx (rules) -> pnnx -> check_param -> ORT vs ncnn cosine.

    rules / cut / fail_on_risky are passed to onnx_rules.apply_rules; input_shape is NCHW (mapped onto NHWC
    models automatically). Raises ConvertError / UnsupportedOpError with an actionable hint."""
    import onnx

    work_dir = ensure_dir(work_dir)
    prepared, results, meta_updates, ops, shape = prepare_onnx(
        src, work_dir / "model.prepared.onnx", rules, cut, fail_on_risky, input_shape)
    for r in results:
        if r.changed:
            log.info("rule %s: %s", r.name, "; ".join(r.notes))
    n_out = len(onnx.load(str(prepared), load_external_data=False).graph.output)
    param, binf = run_pnnx(prepared, work_dir, shape, runner, fp16_weights)
    p = check_param(param, n_out)
    res = ConvertResult(prepared, param, binf, p, p.inputs[0], _ordered_outputs(p, n_out), results,
                        meta_updates, ops, shape)
    if check_numeric:
        cos = compare_with_onnxruntime(res, min_cos=min_cos)
        log.info("ncnn fp32 vs onnxruntime cosine: %s", ", ".join(f"{c:.6f}" for c in cos))
    return res


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    if a.shape != b.shape:
        return -1.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def compare_with_onnxruntime(res: ConvertResult, seed: int = 0, min_cos: float = 0.999) -> list[float]:
    """Feed the same random input to ORT (prepared ONNX) and ncnn; every output must match by cosine.

    Also proves the out0..outN -> ONNX output mapping (a swapped mapping gives cos ~0 or shape mismatch)."""
    import onnxruntime as ort

    from .runner import NcnnNet

    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=res.onnx_input_shape).astype(np.float32)
    sess = ort.InferenceSession(str(res.onnx), providers=["CPUExecutionProvider"])
    ref = sess.run(None, {sess.get_inputs()[0].name: x})
    net = NcnnNet(res.param, res.bin)
    chw = x[0] if x.ndim == 4 else x
    outs = net.forward(chw, res.input_name, res.output_names)
    cos = []
    for name, r, o in zip(res.output_names, ref, outs):
        r = np.squeeze(r, 0) if r.ndim >= 2 and r.shape[0] == 1 else r
        if r.size != o.size:
            raise ConvertError(f"ncnn output {name} has {o.size} values, ONNX has {r.size} ({o.shape} vs {r.shape})")
        cos.append(_cos(r, o))
    bad = [(n, c) for n, c in zip(res.output_names, cos) if c < min_cos]
    if bad:
        raise ConvertError(f"ncnn fp32 output differs from onnxruntime: {bad} (min_cos {min_cos})",
                           "a pnnx pass mis-converted an op; try rules=none, a different cut, or report the op")
    return cos
