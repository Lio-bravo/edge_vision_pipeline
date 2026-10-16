#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Project config: YAML -> validated dict with defaults; relative paths resolve against the config file."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

DEFAULTS: dict[str, Any] = {
    "project": {"name": "evp_project"},
    "classes": [],
    "paths": {
        "prelabel_model": "models/best.pt",
        "raw_images": "workspace/raw_images",
        "labelme_dir": "workspace/labelme",
        "batch_dataset": "workspace/batch_dataset",
        "corpus_root": "workspace/corpus",
        "runs": "workspace/runs",
        "deploy_out": "workspace/deploy",
    },
    "collect": {
        "dedup_hamming": -1,    # near-duplicate dHash distance; -1 = off (exact duplicates are always dropped)
        "min_sharpness": 0.0,   # variance-of-Laplacian gate; 0 = off
        "video_every_sec": 1.0,
        "max_frames": 0,        # 0 = no cap
        "camera": 0,            # camera index or stream URL
        "camera_count": 20,
        "camera_interval": 1.0,
        "jpeg_quality": 95,
        "dedup_against_corpus": True,
    },
    "prelabel": {"conf": 0.25, "iou": 0.45, "imgsz": 640, "device": "auto", "embed_image_data": False},
    "dataset": {"val_ratio": 0.1, "seed": 42, "clear_after_merge": True},
    "train": {
        "base_weights": "yolo11n.pt",
        "epochs": 100,
        "batch": 16,
        "imgsz": 640,
        "workers": 8,
        "device": "auto",      # auto = GPU 0 if CUDA is available, else cpu
        "amp": True,
        "close_mosaic": 10,
        "run_name": "train",
        "task": "detect",
        "extra": {},
    },
    "deploy": {
        "weights": "auto",     # auto = latest train run best.pt; or a .pt (Ultralytics) / .onnx (generic) path
        "name": "",            # bundle name; default = weights stem
        "family": "auto",
        "e2e": False,          # YOLO26: True = one-to-one NMS-free head, False = one-to-many + NMS
        "imgsz": [320, 320],   # [h, w]
        "input_color": "RGB",
        "weights_fp16": False,  # store ncnn weights as fp16 (half size, tiny accuracy cost)
        # generic ONNX (family raw) or manual overrides; null = keep adapter defaults
        "input": {"resize": None, "mean": None, "norm": None, "pad_value": None},
        "labels": [],          # generic ONNX only (Ultralytics weights carry their own names)
        "post": {"conf": 0.25, "iou": 0.45, "max_det": 100, "agnostic_nms": False, "topk": 5},
        "quant": {
            "enable": True,
            "calib_images": "auto",
            "calib_count": 64,
            "method": "auto",       # auto (best of kl/aciq) | kl | aciq | eq
            "target_cos": 0.97,     # detection-aware similarity int8 vs fp32 (see quant.detection_similarity)
            "keep_output_layers_fp32": True,
            "auto_mixed_precision": True,
            "max_fp32_layers": 16,
            "fp32_layers": [],   # ncnn layer names forced to stay fp32
            "eval_count": 16,
        },
        "verify": {
            "images": "auto",
            "count": 32,
            "min_fp32_cos": 0.999,
            "min_int8_cos": 0.93,
            "min_e2e_match": 0.9,
        },
        "onnx": {"cut_outputs": [], "rules": "default"},
        "tools_dir": "",
        "wsl_distro": "",
        "cpp_runtime": True,
    },
}

PATH_KEYS = ("prelabel_model", "raw_images", "labelme_dir", "batch_dataset", "corpus_root", "runs",
             "deploy_out")


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def _require(cond: bool, msg: str, hint: str | None = None) -> None:
    if not cond:
        raise ConfigError(msg, hint)


def validate(cfg: dict) -> None:
    """Fail fast with ConfigError on any invalid value (types, ranges, enums); normalizes int imgsz to [h, w]."""
    classes = cfg.get("classes")
    _require(isinstance(classes, list), "config.classes must be a list of names")
    _require(len(set(map(str, classes))) == len(classes), f"duplicate class names: {classes}")
    for c in classes:
        _require(isinstance(c, str) and c.strip() != "", f"invalid class name: {c!r}")

    col = cfg["collect"]
    _require(-1 <= _num(col.get("dedup_hamming"), "collect.dedup_hamming", int) <= 64,
             "collect.dedup_hamming must be in [-1, 64] (-1 = off)")
    _require(_num(col.get("min_sharpness"), "collect.min_sharpness") >= 0, "collect.min_sharpness must be >= 0")
    _require(_num(col.get("video_every_sec"), "collect.video_every_sec") > 0, "collect.video_every_sec must be > 0")
    _require(_num(col.get("max_frames"), "collect.max_frames", int) >= 0, "collect.max_frames must be >= 0")
    _require(_num(col.get("camera_count"), "collect.camera_count", int) >= 1, "collect.camera_count must be >= 1")
    _require(_num(col.get("camera_interval"), "collect.camera_interval") >= 0, "collect.camera_interval must be >= 0")
    _require(1 <= _num(col.get("jpeg_quality"), "collect.jpeg_quality", int) <= 100,
             "collect.jpeg_quality must be in [1, 100]")
    _require(isinstance(col.get("camera"), (int, str)) and not isinstance(col.get("camera"), bool),
             f"collect.camera must be an index or a stream URL, got {col.get('camera')!r}")

    ds = cfg["dataset"]
    vr = ds.get("val_ratio")
    _require(isinstance(vr, (int, float)) and 0 < float(vr) < 1,
             f"dataset.val_ratio must be in (0,1), got {vr!r}")

    dep = cfg["deploy"]
    imgsz = dep.get("imgsz")
    if isinstance(imgsz, int):
        dep["imgsz"] = [imgsz, imgsz]
        imgsz = dep["imgsz"]
    _require(isinstance(imgsz, list) and len(imgsz) == 2 and all(isinstance(x, int) for x in imgsz),
             f"deploy.imgsz must be int or [h, w], got {imgsz!r}")
    _require(all(x > 0 for x in imgsz), f"deploy.imgsz must be positive, got {imgsz}",
             "detectors need multiples of the max stride (32), e.g. 256, 320, 416, 640")
    inp = dep.get("input") or {}
    _require(inp.get("resize") in (None, "letterbox", "stretch", "center_crop"),
             f"deploy.input.resize invalid: {inp.get('resize')!r}", "letterbox | stretch | center_crop")
    for k in ("mean", "norm"):
        v = inp.get(k)
        _require(v is None or (isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v)),
                 f"deploy.input.{k} must be null or 3 numbers, got {v!r}")
    pv = inp.get("pad_value")
    _require(pv is None or (isinstance(pv, int) and 0 <= pv <= 255), f"deploy.input.pad_value invalid: {pv!r}")
    _require(isinstance(dep.get("labels"), list), "deploy.labels must be a list")
    _require(dep.get("input_color") in ("RGB", "BGR"), "deploy.input_color must be RGB or BGR")
    _require(dep.get("family") in ("auto", "yolo", "classify", "raw"),
             f"deploy.family invalid: {dep.get('family')!r}",
             "one of auto|yolo|classify|raw")
    _require(isinstance(dep.get("e2e"), bool), "deploy.e2e must be true/false")
    q = dep["quant"]
    _require(q.get("method") in ("auto", "kl", "aciq", "eq"), f"deploy.quant.method invalid: {q.get('method')!r}")
    _require(0 < _num(q.get("target_cos"), "deploy.quant.target_cos") <= 1, "deploy.quant.target_cos must be in (0,1]")
    for k in ("calib_count", "eval_count", "max_fp32_layers"):
        _require(_num(q.get(k), f"deploy.quant.{k}", int) >= (0 if k == "max_fp32_layers" else 1),
                 f"deploy.quant.{k} out of range: {q.get(k)!r}")
    _require(isinstance(q.get("fp32_layers"), list), "deploy.quant.fp32_layers must be a list of ncnn layer names")
    post = dep["post"]
    _require(0 <= _num(post["conf"], "deploy.post.conf") <= 1, "deploy.post.conf must be in [0,1]")
    _require(0 <= _num(post["iou"], "deploy.post.iou") <= 1, "deploy.post.iou must be in [0,1]")
    _require(_num(post["max_det"], "deploy.post.max_det", int) >= 1, "deploy.post.max_det must be >= 1")
    _require(_num(post["topk"], "deploy.post.topk", int) >= 1, "deploy.post.topk must be >= 1")
    v = dep["verify"]
    for k in ("min_fp32_cos", "min_int8_cos", "min_e2e_match"):
        _require(0 <= _num(v.get(k), f"deploy.verify.{k}") <= 1, f"deploy.verify.{k} must be in [0,1]")
    _require(_num(v.get("count"), "deploy.verify.count", int) >= 1, "deploy.verify.count must be >= 1")
    ox = dep["onnx"]
    cut = ox.get("cut_outputs")
    _require(cut in (None, "auto") or (isinstance(cut, list) and all(isinstance(c, str) and c for c in cut)),
             f"deploy.onnx.cut_outputs must be a list of tensor names or 'auto', got {cut!r}",
             "find names with `evp inspect-onnx model.onnx`")
    rules = ox.get("rules")
    from .deploy.onnx_rules import RULES

    _require(rules in (None, "default", "none") or (isinstance(rules, list) and all(r in RULES for r in rules)),
             f"deploy.onnx.rules must be default | none | a list of {sorted(RULES)}, got {rules!r}")


def _num(v, name: str, kind=float):
    """Numeric config value or ConfigError (bools are rejected; ints must be integral)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (kind is int and float(v) != int(v)):
        raise ConfigError(f"{name} must be {'an integer' if kind is int else 'a number'}, got {v!r}")
    return kind(v)


def load_config(path: str | Path | None, overrides: dict | None = None) -> dict:
    """Load YAML (or defaults if path is None). Adds `_config_dir` and resolves path keys."""
    if path is None:
        raw: dict = {}
        base_dir = Path.cwd()
    else:
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"config not found: {path}", "copy configs/project.example.yaml and edit it")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"config root must be a mapping: {path}")
        base_dir = path.resolve().parent

    cfg = _deep_merge(DEFAULTS, raw)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    validate(cfg)
    paths = cfg["paths"]
    for key in PATH_KEYS:
        if paths.get(key):
            paths[key] = str(resolve_path(paths[key], base_dir))
    dep = cfg["deploy"]
    if dep.get("weights") not in (None, "", "auto"):
        dep["weights"] = str(resolve_path(dep["weights"], base_dir))
    for sect, key in (("quant", "calib_images"), ("verify", "images")):
        v = dep[sect].get(key)
        if v not in (None, "", "auto"):
            dep[sect][key] = str(resolve_path(v, base_dir))
    if dep.get("tools_dir"):
        dep["tools_dir"] = str(resolve_path(dep["tools_dir"], base_dir))
    cfg["_config_dir"] = str(base_dir)
    return cfg
