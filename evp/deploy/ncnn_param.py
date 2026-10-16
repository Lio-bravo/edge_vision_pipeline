#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Parser / writer for ncnn .param text files and operator support checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..errors import ConvertError

MAGIC = "7767517"

# Layers that ncnn2int8 can quantize (weights + input activation).
QUANTIZABLE = ("Convolution", "ConvolutionDepthWise", "InnerProduct")

# Built-in ncnn layer types (src/layer/*.h, ncnn 2024+). Used only when the python binding cannot
# tell us; the authoritative check is `ncnn.layer_to_index`.
NCNN_LAYERS = frozenset("""
AbsVal ArgMax BatchNorm Bias BNLL Cast Celu Clip Concat Convolution Convolution1D Convolution3D
ConvolutionDepthWise ConvolutionDepthWise1D ConvolutionDepthWise3D CopyTo Crop CumulativeSum
Deconvolution Deconvolution1D Deconvolution3D DeconvolutionDepthWise DeconvolutionDepthWise1D
DeconvolutionDepthWise3D DeepCopy DeformableConv2D Dequantize Diag Dropout Einsum Eltwise ELU Embed
Erf Exp ExpandDims Flatten Fold GELU GLU Gemm GridSample GroupNorm GRU HardSigmoid HardSwish
InnerProduct Input InstanceNorm Interp LayerNorm Log LRN LSTM MatMul MemoryData Mish
MultiHeadAttention MVN Noop Normalize Packing Padding Permute PixelShuffle Pooling Pooling1D
Pooling3D Power PReLU PriorBox Quantize Reduction ReLU Reorg Requantize Reshape RMSNorm RNN ROIAlign
ROIPooling Scale SELU Shrink ShuffleChannel Sigmoid Slice Softmax Softplus Spectrogram Split SPP
Squeeze StatisticsPooling Swish TanH Threshold Tile UnaryOp Unfold YoloDetectionOutput Yolov3DetectionOutput
BinaryOp CopyTo InverseSpectrogram Rotary SDPA
""".split())


@dataclass
class Layer:
    """One .param line: `type name n_bottom n_top bottoms... tops... k=v...`."""
    type: str
    name: str
    bottoms: list[str]
    tops: list[str]
    params: list[str] = field(default_factory=list)   # raw "k=v" tokens, order preserved

    def param_dict(self) -> dict[str, str]:
        out = {}
        for tok in self.params:
            k, _, v = tok.partition("=")
            out[k] = v
        return out

    def to_line(self) -> str:
        parts = [f"{self.type:<24}", f"{self.name:<24}", str(len(self.bottoms)), str(len(self.tops))]
        parts += self.bottoms + self.tops + self.params
        return " ".join(parts)


@dataclass
class NcnnParam:
    """Parsed .param graph. `outputs` are blobs no layer consumes; `to_text` round-trips the file."""
    layers: list[Layer]

    @property
    def blob_count(self) -> int:
        blobs = set()
        for l in self.layers:
            blobs.update(l.bottoms)
            blobs.update(l.tops)
        return len(blobs)

    def layer_types(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for l in self.layers:
            out[l.type] = out.get(l.type, 0) + 1
        return dict(sorted(out.items()))

    @property
    def inputs(self) -> list[str]:
        return [t for l in self.layers if l.type == "Input" for t in l.tops]

    @property
    def outputs(self) -> list[str]:
        consumed = {b for l in self.layers for b in l.bottoms}
        return [t for l in self.layers for t in l.tops if t not in consumed]

    def producer(self, blob: str) -> Layer | None:
        for l in self.layers:
            if blob in l.tops:
                return l
        return None

    def quantizable_layers(self) -> list[Layer]:
        return [l for l in self.layers if l.type in QUANTIZABLE]

    def int8_layers(self) -> list[str]:
        """Layers carrying int8 weights after ncnn2int8 (param key 8 = int8_scale_term > 0)."""
        out = []
        for l in self.layers:
            if l.type in QUANTIZABLE:
                v = l.param_dict().get("8")
                if v is not None and v not in ("0", ""):
                    out.append(l.name)
        return out

    def to_text(self) -> str:
        lines = [MAGIC, f"{len(self.layers)} {self.blob_count}"] + [l.to_line() for l in self.layers]
        return "\n".join(lines) + "\n"

    def save(self, path: Path) -> None:
        Path(path).write_text(self.to_text(), encoding="utf-8", newline="\n")


def parse_param(text: str) -> NcnnParam:
    """Strict parse: bad magic, header, field counts or layer count raise ConvertError with the line number."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines or lines[0] != MAGIC:
        raise ConvertError("not an ncnn param file (bad magic)")
    if len(lines) < 2:
        raise ConvertError("truncated ncnn param file")
    try:
        n_layers, _n_blobs = (int(x) for x in lines[1].split()[:2])
    except ValueError as exc:
        raise ConvertError(f"bad ncnn param header: {lines[1]!r}") from exc
    layers = []
    for ln, line in enumerate(lines[2:], 3):
        tok = line.split()
        if len(tok) < 4:
            raise ConvertError(f"param line {ln}: too few fields")
        try:
            nb, nt = int(tok[2]), int(tok[3])
        except ValueError as exc:
            raise ConvertError(f"param line {ln}: bad bottom/top counts") from exc
        if len(tok) < 4 + nb + nt:
            raise ConvertError(f"param line {ln}: missing blob names")
        layers.append(Layer(tok[0], tok[1], tok[4:4 + nb], tok[4 + nb:4 + nb + nt], tok[4 + nb + nt:]))
    if len(layers) != n_layers:
        raise ConvertError(f"param declares {n_layers} layers but has {len(layers)}")
    return NcnnParam(layers)


def load_param(path: Path) -> NcnnParam:
    path = Path(path)
    if not path.is_file():
        raise ConvertError(f"param not found: {path}")
    return parse_param(path.read_text(encoding="utf-8"))


def is_registered(layer_type: str) -> bool:
    """True if ncnn knows this layer type (python binding first, static list fallback)."""
    if "." in layer_type or "::" in layer_type:
        return False  # pnnx leftovers such as torch.topk / Tensor.index / aten::xxx
    try:
        import ncnn

        fn = getattr(ncnn, "layer_to_index", None)
        if fn is not None:
            return int(fn(layer_type)) >= 0
    except Exception:  # noqa: BLE001
        pass
    return layer_type in NCNN_LAYERS


def unsupported_layers(param: NcnnParam) -> list[Layer]:
    return [l for l in param.layers if not is_registered(l.type)]
