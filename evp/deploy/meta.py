#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Model metadata: the single source of truth for pre/post-processing.

`meta.json` is consumed by the Python reference pipeline; `model.meta` (flat key=value) is generated
from the same object for the C++ runtime so the board never needs a JSON parser. Both are written by
`save_meta`; never edit one without regenerating the other.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..errors import EvpError
from ..utils import read_json, write_json

SCHEMA_VERSION = 1
# yolo: anchor-free Ultralytics detect heads (v5u/v8/v11/v12 reg_max=16 with DFL, YOLO26 reg_max=1)
FAMILIES = ("yolo", "classify", "raw")
RESIZE_MODES = ("letterbox", "stretch", "center_crop")
POST_METHODS = ("nms", "e2e_topk", "topk", "none")


class MetaError(EvpError):
    pass


@dataclass
class InputSpec:
    w: int
    h: int
    blob: str = "in0"
    color: str = "RGB"            # channel order fed to the network
    resize: str = "letterbox"     # letterbox | stretch | center_crop
    pad_value: int = 114          # letterbox border value (all channels)
    mean: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    norm: list[float] = field(default_factory=lambda: [1 / 255.0] * 3)  # x' = (x - mean) * norm


@dataclass
class OutputSpec:
    name: str
    shape: list[int]              # ncnn (c, h, w) or (w,) for 1-D
    stride: int = 0               # detect heads only


@dataclass
class PostSpec:
    method: str = "nms"           # nms (one-to-many heads) | e2e_topk (YOLO26 one-to-one) | topk (classify) | none
    conf: float = 0.25
    iou: float = 0.45
    max_det: int = 100
    agnostic: bool = False
    topk: int = 5                 # classify
    softmax: bool = False         # classify: apply softmax on CPU (False if graph already outputs probs)


@dataclass
class ModelFile:
    param: str
    bin: str
    sha256_bin: str = ""


@dataclass
class ModelMeta:
    """The pre/post-processing contract of one bundle (docs/META_SPEC.md). `validate` enforces the same
    invariants the C++ parser checks, so a meta that saves here also loads on the board."""
    name: str
    family: str
    input: InputSpec
    outputs: list[OutputSpec]
    labels: list[str]
    post: PostSpec = field(default_factory=PostSpec)
    reg_max: int = 16
    files: dict[str, ModelFile] = field(default_factory=dict)   # precision -> files
    quant: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    @property
    def task(self) -> str:
        return {"yolo": "detect", "classify": "classify"}.get(self.family, "raw")

    @property
    def box_channels(self) -> int:
        return 4 * self.reg_max

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise MetaError(f"unsupported meta schema_version {self.schema_version} (expected {SCHEMA_VERSION})")
        if self.family not in FAMILIES:
            raise MetaError(f"unknown family {self.family!r}", f"one of {FAMILIES}")
        i = self.input
        if i.w <= 0 or i.h <= 0:
            raise MetaError(f"invalid input size {i.w}x{i.h}")
        if i.color not in ("RGB", "BGR"):
            raise MetaError(f"input.color must be RGB/BGR, got {i.color!r}")
        if i.resize not in RESIZE_MODES:
            raise MetaError(f"input.resize must be one of {RESIZE_MODES}, got {i.resize!r}")
        if not 0 <= i.pad_value <= 255:
            raise MetaError(f"input.pad_value out of range: {i.pad_value}")
        if len(i.mean) != 3 or len(i.norm) != 3:
            raise MetaError("input.mean / input.norm must have 3 values")
        if not self.outputs:
            raise MetaError("meta has no outputs")
        names = [o.name for o in self.outputs]
        if len(set(names)) != len(names):
            raise MetaError(f"duplicate output names: {names}")
        for lab in self.labels:
            if not isinstance(lab, str) or not lab or "\n" in lab or "\r" in lab:
                raise MetaError(f"invalid label {lab!r} (must be non-empty single-line text)")
        p = self.post
        if p.method not in POST_METHODS:
            raise MetaError(f"post.method must be one of {POST_METHODS}, got {p.method!r}")
        if self.task == "detect":
            if not self.labels:
                raise MetaError("detect model needs labels")
            if self.reg_max < 1:
                raise MetaError(f"reg_max must be >= 1, got {self.reg_max}")
            if p.method not in ("nms", "e2e_topk"):
                raise MetaError(f"detect model needs post.method nms|e2e_topk, got {p.method!r}")
            want_c = self.box_channels + self.num_classes
            for o in self.outputs:
                if o.stride <= 0:
                    raise MetaError(f"detect output {o.name} needs stride > 0")
                if len(o.shape) != 3 or o.shape[0] != want_c:
                    raise MetaError(f"output {o.name} shape {o.shape}: expected channels {want_c} "
                                    f"(reg_max={self.reg_max}, nc={self.num_classes})")
                if o.shape[1] * o.stride != i.h or o.shape[2] * o.stride != i.w:
                    raise MetaError(f"output {o.name} grid {o.shape[1:]} x stride {o.stride} != input {i.h}x{i.w}")
        if self.family == "classify":
            if len(self.outputs) != 1:
                raise MetaError("classify model must have exactly 1 output")
            if self.labels and self.outputs[0].shape[-1] != self.num_classes:
                raise MetaError(f"classify output {self.outputs[0].shape} != num labels {self.num_classes}")
            if p.method != "topk":
                raise MetaError(f"classify model needs post.method=topk, got {p.method!r}")
        if not (0 <= p.conf <= 1 and 0 <= p.iou <= 1 and p.max_det >= 1 and p.topk >= 1):
            raise MetaError(f"invalid post params: {p}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["task"] = self.task
        d["num_classes"] = self.num_classes
        return d

    @staticmethod
    def from_dict(d: dict) -> ModelMeta:
        try:
            m = ModelMeta(
                name=d["name"],
                family=d["family"],
                input=InputSpec(**d["input"]),
                outputs=[OutputSpec(**o) for o in d["outputs"]],
                labels=list(d.get("labels", [])),
                post=PostSpec(**d.get("post", {})),
                reg_max=int(d.get("reg_max", 16)),
                files={k: ModelFile(**v) for k, v in (d.get("files") or {}).items()},
                quant=dict(d.get("quant") or {}),
                source=dict(d.get("source") or {}),
                schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            )
        except (KeyError, TypeError) as exc:
            raise MetaError(f"malformed meta: {exc}") from exc
        m.validate()
        return m


def _fmt_float(v: float) -> str:
    return repr(float(v))


def to_flat(meta: ModelMeta) -> str:
    """Serialize to the flat `key=value` format read by runtime/cpp/src/meta.cpp."""
    meta.validate()
    i, p = meta.input, meta.post
    lines = [
        "# generated by evp; do not edit, regenerate with `evp deploy`",
        f"schema_version={meta.schema_version}",
        f"name={meta.name}",
        f"family={meta.family}",
        f"input.w={i.w}",
        f"input.h={i.h}",
        f"input.blob={i.blob}",
        f"input.color={i.color}",
        f"input.resize={i.resize}",
        f"input.pad_value={i.pad_value}",
        "input.mean=" + ",".join(_fmt_float(v) for v in i.mean),
        "input.norm=" + ",".join(_fmt_float(v) for v in i.norm),
        f"reg_max={meta.reg_max}",
        f"num_classes={meta.num_classes}",
        f"output.count={len(meta.outputs)}",
    ]
    for k, o in enumerate(meta.outputs):
        lines += [f"output.{k}.name={o.name}", f"output.{k}.stride={o.stride}",
                  "output.{}.shape={}".format(k, ",".join(str(s) for s in o.shape))]
    lines += [
        f"post.method={p.method}",
        f"post.conf={_fmt_float(p.conf)}",
        f"post.iou={_fmt_float(p.iou)}",
        f"post.max_det={p.max_det}",
        f"post.agnostic={int(p.agnostic)}",
        f"post.topk={p.topk}",
        f"post.softmax={int(p.softmax)}",
        f"labels.count={len(meta.labels)}",
    ]
    lines += [f"label.{k}={lab}" for k, lab in enumerate(meta.labels)]
    for prec, f in sorted(meta.files.items()):
        lines += [f"file.{prec}.param={f.param}", f"file.{prec}.bin={f.bin}"]
    return "\n".join(lines) + "\n"


def parse_flat(text: str) -> dict[str, str]:
    """Inverse of the line format (used by tests to prove the C++ contract round-trips)."""
    out: dict[str, str] = {}
    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip("\r")
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise MetaError(f"model.meta line {ln}: missing '='")
        k, v = line.split("=", 1)
        if k in out:
            raise MetaError(f"model.meta line {ln}: duplicate key {k}")
        out[k] = v
    return out


def save_meta(meta: ModelMeta, out_dir: Path) -> tuple[Path, Path]:
    meta.validate()
    out_dir = Path(out_dir)
    jp = write_json(out_dir / "meta.json", meta.to_dict())
    fp = out_dir / "model.meta"
    fp.write_text(to_flat(meta), encoding="utf-8", newline="\n")
    (out_dir / "labels.txt").write_text("\n".join(meta.labels) + ("\n" if meta.labels else ""), encoding="utf-8",
                                        newline="\n")
    return jp, fp


def load_meta(path: Path) -> ModelMeta:
    path = Path(path)
    if path.is_dir():
        path = path / "meta.json"
    d = read_json(path)
    d.pop("task", None)
    d.pop("num_classes", None)
    return ModelMeta.from_dict(d)
