#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""PyTorch-level adapters: patch Ultralytics heads to emit raw tensors, then export to ONNX.

`imgsz` is (h, w) everywhere, like Ultralytics.

Cut point for detect heads (all anchor-free Ultralytics families):
    per stride i: concat(box_head[i](feat_i), cls_head[i](feat_i))  -> (1, 4*reg_max + nc, H_i, W_i)
DFL/decode/sigmoid/NMS/top-k are NOT in the graph; runtime post-processing does them (paired via meta).
"""

from __future__ import annotations

import copy
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import ExportError
from ..utils import ensure_dir, log, sha256_file
from .meta import InputSpec, ModelMeta, OutputSpec, PostSpec


@dataclass
class ExportResult:
    onnx: Path
    meta: ModelMeta
    module: Any = None                      # patched torch module (for reference inference)
    notes: list[str] = field(default_factory=list)


def _load_ultralytics(weights: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise ExportError("ultralytics is not installed", "pip install ultralytics") from exc
    if not Path(weights).is_file():
        raise ExportError(f"weights not found: {weights}")
    try:
        return YOLO(str(weights))
    except Exception as exc:  # noqa: BLE001
        raise ExportError(f"cannot load {weights} with ultralytics: {exc}") from exc


def detect_family(yolo) -> str:
    """Map the Ultralytics head type to a meta family. Only the exact Detect class counts as `yolo`:
    subclasses (Segment, Pose, OBB) add outputs the detect post-processing would silently ignore."""
    from ultralytics.nn.modules.head import Classify, Detect

    head = yolo.model.model[-1]
    if type(head) is Detect:
        return "yolo"
    if isinstance(head, Classify):
        return "classify"
    raise ExportError(
        f"unsupported head {type(head).__name__} (task={yolo.task})",
        "supported: Detect (v5u/v8/v11/v12/YOLO26) and Classify. Segment/Pose/OBB adapters are on the roadmap; "
        "use the ONNX path with deploy.family=raw meanwhile")


def _names_list(names) -> list[str]:
    return [str(names[i]) for i in sorted(names)] if isinstance(names, dict) else [str(n) for n in names]


def _patch_detect_head(model, e2e: bool) -> tuple[int, list[int], int]:
    """Make model(x) return a tuple of raw per-stride tensors. Returns (nc, strides, reg_max)."""
    head = model.model[-1]
    has_o2o = getattr(head, "one2one_cv2", None) is not None
    if e2e and not has_o2o:
        raise ExportError("deploy.e2e=true but this model has no one-to-one head",
                          "only YOLO26 (and v10-style) models are NMS-free; set deploy.e2e: false")
    box_heads = head.one2one_cv2 if e2e else head.cv2
    cls_heads = head.one2one_cv3 if e2e else head.cv3
    if box_heads is None or cls_heads is None:
        raise ExportError("selected detect branch was removed (fused checkpoint?)",
                          "flip deploy.e2e or re-export from the training checkpoint")
    nl = head.nl

    def forward(self, x):
        import torch

        return tuple(torch.cat((box_heads[i](x[i]), cls_heads[i](x[i])), 1) for i in range(nl))

    head.forward = types.MethodType(forward, head)
    strides = [int(s) for s in head.stride.tolist()]
    return int(head.nc), strides, int(head.reg_max)


def _patch_classify_head(model) -> int:
    head = model.model[-1]
    head.export = True   # Classify returns softmax probabilities in export mode
    return int(head.linear.out_features)


def export_ultralytics(weights: Path, imgsz: tuple[int, int], out_dir: Path, e2e: bool = False,
                       family: str = "auto", input_color: str = "RGB", post_cfg: dict | None = None,
                       name: str | None = None, opset: int = 17) -> ExportResult:
    """Export a head-patched Ultralytics model to ONNX (fixed shape, batch 1) and build its meta skeleton."""
    import torch

    h, w = int(imgsz[0]), int(imgsz[1])
    yolo = _load_ultralytics(Path(weights))
    fam = detect_family(yolo)
    if family not in ("auto", fam):
        raise ExportError(f"deploy.family={family} but weights look like {fam}")
    model = copy.deepcopy(yolo.model).float().eval().cpu()
    head = model.model[-1]
    notes: list[str] = []
    if fam == "yolo" and hasattr(head, "one2one_cv2"):
        head.end2end = bool(e2e)  # decides which branch fuse() keeps
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.fuse(verbose=False) if hasattr(model, "fuse") else model

    labels = _names_list(yolo.names)
    post_cfg = post_cfg or {}
    if fam == "yolo":
        nc, strides, reg_max = _patch_detect_head(model, e2e)
        if any(h % s or w % s for s in strides):
            raise ExportError(f"imgsz {h}x{w} not divisible by strides {strides}")
        if len(labels) != nc:
            raise ExportError(f"model has {nc} classes but names has {len(labels)}")
        outputs = [OutputSpec(f"out{i}", [4 * reg_max + nc, h // s, w // s], s) for i, s in enumerate(strides)]
        post = PostSpec(method="e2e_topk" if e2e else "nms", conf=float(post_cfg.get("conf", 0.25)),
                        iou=float(post_cfg.get("iou", 0.45)), max_det=int(post_cfg.get("max_det", 100)),
                        agnostic=bool(post_cfg.get("agnostic_nms", False)))
        inp = InputSpec(w=w, h=h, color=input_color, resize="letterbox", pad_value=114)
        branch = "one2one" if e2e else "one2many"
        notes.append(f"detect head: reg_max={reg_max} strides={strides} nc={nc} branch={branch}")
    else:
        nc = _patch_classify_head(model)
        reg_max = 0
        if len(labels) != nc:
            raise ExportError(f"classifier has {nc} outputs but names has {len(labels)}")
        outputs = [OutputSpec("out0", [nc], 0)]
        post = PostSpec(method="topk", topk=int(post_cfg.get("topk", 5)), softmax=False)
        inp = InputSpec(w=w, h=h, color=input_color, resize="center_crop", pad_value=0)
        notes.append(f"classify head: nc={nc}, softmax kept in graph")

    x = torch.zeros(1, 3, h, w)
    with torch.no_grad():
        y = model(x)
    ys = y if isinstance(y, (tuple, list)) else (y,)
    if len(ys) != len(outputs):
        raise ExportError(f"model produced {len(ys)} outputs, expected {len(outputs)}")
    for o, t in zip(outputs, ys):
        got = list(t.shape[1:]) if fam == "yolo" else [int(t.shape[-1])]
        if got != o.shape:
            raise ExportError(f"output {o.name} shape {got} != expected {o.shape}")

    out_dir = ensure_dir(out_dir)
    stem = name or Path(weights).stem
    onnx_path = out_dir / f"{stem}.onnx"
    export_onnx(model, x, onnx_path, ["in0"], [o.name for o in outputs], opset)
    meta = ModelMeta(name=stem, family=fam, input=inp, outputs=outputs, labels=labels, post=post,
                     reg_max=reg_max if fam == "yolo" else 16,
                     source={"weights": str(Path(weights).resolve()), "weights_sha256": sha256_file(Path(weights)),
                             "framework": "ultralytics", "ultralytics_version": _ul_version(),
                             "task": yolo.task, "e2e": bool(e2e)})
    if fam == "classify":
        meta.reg_max = 16  # unused for classify; keep schema default
    meta.validate()
    log.info("exported %s -> %s (%s)", weights, onnx_path, "; ".join(notes))
    return ExportResult(onnx_path, meta, model, notes)


def export_onnx(module, example, path: Path, input_names: list[str], output_names: list[str], opset: int = 17) -> Path:
    """TorchScript-based ONNX export (fixed shapes). The dynamo exporter is avoided on purpose:
    it emits opset-18+ functions that older converters mishandle."""
    import warnings

    import torch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.onnx.export(module, example, str(path), opset_version=opset, dynamo=False,
                              input_names=input_names, output_names=output_names, do_constant_folding=True)
        except TypeError:  # torch < 2.5 has no `dynamo` kwarg
            torch.onnx.export(module, example, str(path), opset_version=opset, input_names=input_names,
                              output_names=output_names, do_constant_folding=True)
        except Exception as exc:  # noqa: BLE001
            raise ExportError(f"ONNX export failed: {exc}",
                              "custom modules may need a PyTorch-level replacement; see docs/ADAPTERS.md") from exc
    if not Path(path).is_file():
        raise ExportError(f"ONNX export produced no file: {path}")
    return Path(path)


def _ul_version() -> str:
    try:
        import ultralytics

        return str(ultralytics.__version__)
    except Exception:  # noqa: BLE001
        return "unknown"


def torch_reference(module, chw: np.ndarray) -> list[np.ndarray]:
    """Run the patched module on one preprocessed CHW float32 tensor; returns outputs without batch dim."""
    import torch

    with torch.no_grad():
        y = module(torch.from_numpy(np.ascontiguousarray(chw, np.float32))[None])
    ys = y if isinstance(y, (tuple, list)) else (y,)
    return [t[0].cpu().numpy().astype(np.float32) for t in ys]
