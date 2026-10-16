#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""INT8 post-training quantization with ncnn2table / ncnn2int8.

* Calibration tensors are produced by the deploy preprocessing itself (preprocess_ncnn -> .npy, type=1),
  so letterbox/normalisation during calibration is exactly what the board will feed.
* Mixed precision: a layer stays fp32 when both its activation entry (`name`) and weight entry
  (`name_param_0`) are removed from the table (ncnn2int8 skips layers without an activation scale).
* Head output convolutions (raw box/class logits) are kept fp32 by default.
* If int8-vs-fp32 cosine is below target, layers are moved to fp32 greedily by leave-one-out
  sensitivity until the target or `max_fp32_layers` is reached.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..errors import EvpError, QuantError
from ..utils import ensure_dir, log
from .meta import InputSpec
from .ncnn_param import QUANTIZABLE, NcnnParam, load_param
from .preprocess import preprocess_ncnn
from .tools import ToolRunner, default_runner

WEIGHT_SUFFIX = "_param_0"
METHODS = ("kl", "aciq", "eq")
AUTO_METHODS = ("kl", "aciq")   # `auto` calibrates with both and keeps the better full-int8 score


@dataclass
class QuantResult:
    """Outcome of `quantize`: final int8 model files, which layers stayed fp32, and the similarity trail
    (`history`: one entry per evaluated table; `sensitivity`: per-layer gain when kept fp32)."""
    param: Path
    bin: Path
    table: Path
    fp32_layers: list[str]
    int8_layers: list[str]
    cos_full_int8: float
    cos_final: float
    target_cos: float
    history: list[dict] = field(default_factory=list)
    sensitivity: dict[str, float] = field(default_factory=dict)
    calib_count: int = 0
    method: str = "kl"

    @property
    def target_met(self) -> bool:
        return self.cos_final >= self.target_cos

    def to_dict(self) -> dict:
        return {
            "method": self.method, "calib_count": self.calib_count, "target_cos": self.target_cos,
            "cos_full_int8": round(self.cos_full_int8, 6), "cos_final": round(self.cos_final, 6),
            "target_met": self.target_met, "fp32_layers": self.fp32_layers,
            "int8_layer_count": len(self.int8_layers), "history": self.history,
            "sensitivity_top": dict(sorted(self.sensitivity.items(), key=lambda kv: -kv[1])[:20]),
        }


# ---------------------------------------------------------------- calibration data
def sample_images(images: Sequence[Path], count: int, seed: int = 0) -> list[Path]:
    images = sorted({Path(p) for p in images})
    if not images:
        raise QuantError("no calibration images", "set deploy.quant.calib_images to a folder of representative images")
    if count <= 0 or count >= len(images):
        return images
    rng = random.Random(seed)
    return sorted(rng.sample(images, count))


def make_calib_npy(images: Sequence[Path], spec: InputSpec, out_dir: Path, warn_below: int = 16) -> list[Path]:
    """Preprocess each image exactly like the runtime and store CHW float32 .npy files (0000.npy, ...)."""
    from .runner import read_image

    out_dir = ensure_dir(out_dir)
    for old in out_dir.glob("*.npy"):
        old.unlink()
    out: list[Path] = []
    for p in images:
        try:
            img = read_image(p)
        except EvpError as exc:
            log.warning("calibration: skip %s (%s)", p, exc.message)
            continue
        mat, _ = preprocess_ncnn(img, spec)
        arr = np.ascontiguousarray(np.array(mat, dtype=np.float32))
        if arr.shape != (3, spec.h, spec.w):
            raise QuantError(f"calibration tensor shape {arr.shape} != (3, {spec.h}, {spec.w})")
        f = out_dir / f"{len(out):04d}.npy"
        np.save(f, arr)
        out.append(f)
    if not out:
        raise QuantError("no readable calibration images")
    if len(out) < warn_below:
        log.warning("only %d calibration images; 64-200 representative images are recommended", len(out))
    return out


# ---------------------------------------------------------------- table handling
@dataclass
class Table:
    """An ncnn2table calibration table. Layer `X` has a weight line `X_param_0 <scales...>` and an
    activation line `X <scale>`; writing it with X in `fp32` drops both, so ncnn2int8 keeps X in fp32."""
    lines: list[str]                       # original lines, order preserved
    weights: dict[str, int]                # layer -> line index
    acts: dict[str, int]

    def layers(self) -> list[str]:
        return [n for n in self.acts if n in self.weights]

    def write(self, path: Path, fp32: set[str] = frozenset()) -> Path:
        drop = {self.weights[n] for n in fp32 if n in self.weights} | {self.acts[n] for n in fp32 if n in self.acts}
        text = "\n".join(l for i, l in enumerate(self.lines) if i not in drop) + "\n"
        Path(path).write_text(text, encoding="utf-8", newline="\n")
        return Path(path)


def parse_table(text: str, layer_names: set[str] | None = None) -> Table:
    """`layer_names` (quantizable ncnn layers) disambiguates layers whose own name ends in `_param_0`."""
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    weights: dict[str, int] = {}
    acts: dict[str, int] = {}
    for i, l in enumerate(lines):
        tok = l.split()
        if len(tok) < 2:
            raise QuantError(f"malformed table line {i + 1}: {l!r}")
        name = tok[0]
        base = name[: -len(WEIGHT_SUFFIX)] if name.endswith(WEIGHT_SUFFIX) else None
        if base is not None and (layer_names is None or base in layer_names):
            weights[base] = i
        else:
            acts[name] = i
    return Table(lines, weights, acts)


def output_conv_layers(p: NcnnParam) -> list[str]:
    """Quantizable layers whose result reaches a graph output through non-quantizable layers only."""
    outs = set(p.outputs)
    producers = {t: l for l in p.layers for t in l.tops}
    found: list[str] = []
    seen: set[str] = set()
    stack = list(outs)
    while stack:
        blob = stack.pop()
        if blob in seen:
            continue
        seen.add(blob)
        l = producers.get(blob)
        if l is None or l.type == "Input":
            continue
        if l.type in QUANTIZABLE:
            if l.name not in found:
                found.append(l.name)
            continue
        stack.extend(l.bottoms)
    return found


# ---------------------------------------------------------------- tools
def run_ncnn2table(param: Path, bin_: Path, list_file: Path, table: Path, spec: InputSpec, method: str,
                   work_dir: Path, runner: ToolRunner, threads: int = 0) -> Path:
    """Calibrate from the .npy list (type=1: CHW float32, shape=[w,h,3]). List entries are relative to
    `work_dir` because ncnn2table reads them with %255s (no spaces, short paths)."""
    if method not in METHODS:
        raise QuantError(f"unknown calibration method {method!r}", f"one of {METHODS}")
    tool = runner.get("ncnn2table")
    args = [tool.path_arg(param), tool.path_arg(bin_), _rel(list_file, work_dir), _rel(table, work_dir),
            f"shape=[{spec.w},{spec.h},3]", f"method={method}", "type=1"]
    if threads > 0:
        args.append(f"thread={threads}")
    try:
        proc = tool.run(args, cwd=work_dir)
    except EvpError as exc:
        hint = ("ncnn2table method=eq crashes on some models (upstream bug); use method aciq, kl or auto"
                if method == "eq" else "check the calibration list and that the fp32 model loads")
        raise QuantError(f"ncnn2table failed (method={method}): {exc.message}", hint) from exc
    if not table.is_file() or table.stat().st_size == 0:
        raise QuantError(f"ncnn2table produced no table\n{proc.stderr[-2000:]}")
    return table


def run_ncnn2int8(param: Path, bin_: Path, table: Path, out_param: Path, out_bin: Path, work_dir: Path,
                  runner: ToolRunner) -> tuple[Path, Path]:
    tool = runner.get("ncnn2int8")
    for f in (out_param, out_bin):
        if f.exists():
            f.unlink()
    tool.run([tool.path_arg(param), tool.path_arg(bin_), tool.path_arg(out_param), tool.path_arg(out_bin),
              tool.path_arg(table)], cwd=work_dir)
    if not out_param.is_file() or not out_bin.is_file():
        raise QuantError("ncnn2int8 produced no model")
    return out_param, out_bin


def _rel(p: Path, base: Path) -> str:
    """Relative POSIX path (ncnn2table reads list entries with %255s: no spaces, short)."""
    rel = Path(p).resolve().relative_to(Path(base).resolve()).as_posix()
    if " " in rel or len(rel) > 250:
        raise QuantError(f"path unusable by ncnn2table: {rel!r}")
    return rel


# ---------------------------------------------------------------- evaluation
def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    return float(a @ b / (na * nb))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


SCORE_FLOOR = 0.02


def detection_similarity(outs: Sequence[np.ndarray], refs: Sequence[np.ndarray], box_channels: int,
                         floor: float = SCORE_FLOOR) -> float | None:
    """Detection-aware similarity of raw head outputs (all strides pooled):
      class part: cosine of sigmoid scores where either side scores > floor (background noise ignored)
      box part  : cosine of box channels at cells where fp32 sees an object (max score > floor)
    Raw-tensor cosine is dominated by the 4*reg_max box bins and hides broken class scores; plain sigmoid
    cosine is dominated by background noise on images without objects. Returns None if nothing is relevant
    (background-only image: no signal, must not count as a perfect score)."""
    cls_a, cls_b, box_a, box_b = [], [], [], []
    for o, r in zip(outs, refs):
        so, sr = _sigmoid(o[box_channels:]), _sigmoid(r[box_channels:])
        m = (so > floor) | (sr > floor)
        cls_a.append(so[m])
        cls_b.append(sr[m])
        cells = sr.max(0) > floor
        box_a.append(o[:box_channels, cells].reshape(-1))
        box_b.append(r[:box_channels, cells].reshape(-1))
    ca, cb = np.concatenate(cls_a), np.concatenate(cls_b)
    ba, bb = np.concatenate(box_a), np.concatenate(box_b)
    if not ca.size and not ba.size:
        return None
    c = _cos(ca, cb) if ca.size else 1.0
    b = _cos(ba, bb) if ba.size else 1.0
    return min(c, b)


def output_similarity(outs: Sequence[np.ndarray], refs: Sequence[np.ndarray],
                      box_channels: int | None = None) -> float | None:
    """Similarity of one image's outputs: detection-aware for detect heads (None = nothing relevant),
    else min per-output cosine."""
    if box_channels is not None and all(o.ndim == 3 and o.shape[0] > box_channels for o in outs):
        return detection_similarity(outs, refs, box_channels)
    return min(_cos(o, r) for o, r in zip(outs, refs))


def mean_similarity(vals: Sequence[float | None]) -> float:
    """Mean over images that carry signal; 1.0 (with a warning) if none does."""
    v = [x for x in vals if x is not None]
    if not v:
        log.warning("no evaluation image contains objects: int8 similarity cannot be measured (reported as 1.0); "
                    "use representative images with objects")
        return 1.0
    return float(np.mean(v))


class Evaluator:
    """Mean over images of the minimum per-output similarity(int8, fp32)."""

    def __init__(self, fp32_param: Path, fp32_bin: Path, inputs: list[np.ndarray], in_name: str,
                 out_names: list[str], box_channels: int | None = None):
        from .runner import NcnnNet

        if not inputs:
            raise QuantError("no evaluation inputs for quantization")
        self.inputs, self.in_name, self.out_names = inputs, in_name, out_names
        self.box_channels = box_channels
        ref = NcnnNet(fp32_param, fp32_bin)
        self.refs = [ref.forward(x, in_name, out_names) for x in inputs]

    def score(self, param: Path, bin_: Path) -> float:
        from .runner import NcnnNet

        net = NcnnNet(param, bin_)
        vals = []
        for x, ref in zip(self.inputs, self.refs):
            outs = net.forward(x, self.in_name, self.out_names)
            vals.append(output_similarity(outs, ref, self.box_channels))
        return mean_similarity(vals)


# ---------------------------------------------------------------- main entry
def quantize(fp32_param: Path, fp32_bin: Path, spec: InputSpec, out_names: list[str], calib_images: Sequence[Path],
             work_dir: Path, method: str = "auto", calib_count: int = 64, target_cos: float = 0.97,
             keep_output_layers_fp32: bool = True, auto_mixed_precision: bool = True, max_fp32_layers: int = 16,
             eval_images: Sequence[Path] | None = None, eval_count: int = 16, fp32_layers: Sequence[str] = (),
             runner: ToolRunner | None = None, seed: int = 0, box_channels: int | None = None) -> QuantResult:
    """box_channels: 4*reg_max for detect heads (enables the detection-aware metric), None otherwise."""
    runner = runner or default_runner()
    work_dir = ensure_dir(Path(work_dir).resolve())
    p = load_param(fp32_param)
    quant_names = {l.name for l in p.quantizable_layers()}
    if not quant_names:
        raise QuantError("model has no Convolution/InnerProduct layers to quantize")

    if method != "auto" and method not in METHODS:
        raise QuantError(f"unknown calibration method {method!r}", f"one of auto, {', '.join(METHODS)}")
    imgs = sample_images(calib_images, calib_count, seed)
    npys = make_calib_npy(imgs, spec, work_dir / "calib")
    list_file = work_dir / "calib_list.txt"
    list_file.write_text("\n".join(_rel(f, work_dir) for f in npys) + "\n", encoding="utf-8", newline="\n")

    if eval_images:
        ev_imgs = sample_images(eval_images, eval_count, seed + 1)
        ev_inputs = [np.load(f) for f in make_calib_npy(ev_imgs, spec, work_dir / "eval", warn_below=0)]
    else:
        ev_inputs = [np.load(f) for f in npys[:max(1, eval_count)]]
    ev = Evaluator(fp32_param, fp32_bin, ev_inputs, p.inputs[0], out_names, box_channels)

    table: Table | None = None

    def build(fp32: set[str], tag: str, tbl: Table | None = None) -> tuple[Path, Path, float]:
        t = (tbl or table).write(work_dir / f"{tag}.table", fp32)
        prm, bn = run_ncnn2int8(fp32_param, fp32_bin, t, work_dir / f"{tag}.param", work_dir / f"{tag}.bin",
                                work_dir, runner)
        return prm, bn, ev.score(prm, bn)

    history: list[dict] = []
    best: tuple[float, str, Table] | None = None
    for m in (AUTO_METHODS if method == "auto" else (method,)):
        tf = run_ncnn2table(fp32_param, fp32_bin, list_file, work_dir / f"full_{m}.table", spec, m, work_dir, runner)
        tbl = parse_table(tf.read_text(encoding="utf-8"), quant_names)
        if not tbl.layers():
            raise QuantError("calibration table has no layer entries", "is the calibration list empty or unreadable?")
        _, _, c = build(set(), f"int8_full_{m}", tbl)
        history.append({"step": f"full_int8[{m}]", "fp32": [], "cos": round(c, 6)})
        log.info("int8 (all layers, %s) similarity vs fp32: %.6f", m, c)
        if best is None or c > best[0]:
            best = (c, m, tbl)
    assert best is not None
    cos_full, method, table = best

    forced = set(fp32_layers) & quant_names
    unknown = set(fp32_layers) - quant_names
    if unknown:
        log.warning("deploy.quant fp32_layers not found in model (ignored): %s", sorted(unknown))
    if keep_output_layers_fp32:
        forced |= set(output_conv_layers(p)) & quant_names
    fp32 = set(forced)
    prm, bn, cos = build(fp32, "int8_base")
    history.append({"step": "base", "fp32": sorted(fp32), "cos": round(cos, 6)})
    log.info("int8 with %d fp32 head layers: similarity %.6f", len(fp32), cos)

    sens: dict[str, float] = {}
    if cos < target_cos and auto_mixed_precision and max_fp32_layers > len(fp32):
        candidates = [n for n in table.layers() if n not in fp32]
        log.info("similarity %.6f < target %.4f: leave-one-out sensitivity over %d layers",
                 cos, target_cos, len(candidates))
        for n in candidates:
            _, _, c = build(fp32 | {n}, "int8_probe")
            sens[n] = c - cos
        order = sorted(candidates, key=lambda n: -sens[n])
        for n in order:
            if cos >= target_cos or len(fp32) >= max_fp32_layers:
                break
            if sens[n] <= 0:
                break
            trial = fp32 | {n}
            tprm, tbn, tc = build(trial, "int8_trial")
            history.append({"step": f"+{n}", "fp32": sorted(trial), "cos": round(tc, 6)})
            if tc > cos:
                fp32, cos = trial, tc
                log.info("  keep %s fp32 -> similarity %.6f", n, cos)
        prm, bn, cos = build(fp32, "int8_final")
    else:
        prm, bn, cos = build(fp32, "int8_final")
    if cos < target_cos:
        log.warning("int8 similarity %.6f below target %.4f (fp32 layers: %d). Ship fp32 or add calibration data.",
                    cos, target_cos, len(fp32))
    final_table = table.write(work_dir / "final.table", fp32)
    int8_layers = load_param(prm).int8_layers()
    return QuantResult(prm, bn, final_table, sorted(fp32), int8_layers, cos_full, cos, target_cos, history, sens,
                       len(npys), method)
