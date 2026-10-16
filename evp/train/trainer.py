#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Train on the corpus with Ultralytics; always reads corpus/data.yaml, never SOP staging."""

from __future__ import annotations

from pathlib import Path

from ..data.merge import list_split_images
from ..errors import DataError
from ..utils import ensure_dir, log, resolve_device


def find_latest_best(runs_dir: Path, run_name: str | None = None) -> Path:
    """Newest (mtime) weights/best.pt under runs_dir, preferring runs whose name starts with run_name."""
    runs_dir = Path(runs_dir)
    pattern = f"{run_name}*/weights/best.pt" if run_name else "*/weights/best.pt"
    cands = sorted(runs_dir.glob(pattern))
    if not cands:
        cands = sorted(runs_dir.rglob("weights/best.pt"))
    if not cands:
        raise DataError(f"no best.pt under {runs_dir}", "train first, or set deploy.weights explicitly")
    return max(cands, key=lambda p: p.stat().st_mtime)


def check_corpus(corpus_root: Path) -> Path:
    """data.yaml exists and both splits have images; returns the data.yaml path."""
    corpus_root = Path(corpus_root)
    data_yaml = corpus_root / "data.yaml"
    if not data_yaml.is_file():
        raise DataError(f"data.yaml not found: {data_yaml}", "run `evp ingest` to merge a reviewed batch first")
    for split in ("train", "val"):
        if not list_split_images(corpus_root / "images" / split):
            hint = ("a batch needs >= 2 labeled images to give val one; ingest more reviewed images"
                    if split == "val" else "run `evp ingest` with reviewed images")
            raise DataError(f"corpus {split} images empty: {corpus_root / 'images' / split}", hint)
    return data_yaml


def train(corpus_root: Path, runs_dir: Path, train_cfg: dict) -> Path:
    """Ultralytics model.train on corpus/data.yaml with the `train:` config (`extra` is passed through).
    Returns the best.pt path."""
    data_yaml = check_corpus(corpus_root)
    device = resolve_device(train_cfg.get("device", "auto"), "train.device")
    from ultralytics import YOLO

    ensure_dir(runs_dir)
    name = str(train_cfg.get("run_name", "train"))
    args = dict(
        data=str(data_yaml),
        imgsz=int(train_cfg.get("imgsz", 640)),
        epochs=int(train_cfg.get("epochs", 100)),
        batch=train_cfg.get("batch", 16),
        workers=int(train_cfg.get("workers", 8)),
        device=device,
        amp=bool(train_cfg.get("amp", True)) and device not in ("cpu", "mps"),
        close_mosaic=int(train_cfg.get("close_mosaic", 10)),
        project=str(Path(runs_dir).resolve()),
        name=name,
        exist_ok=True,
    )
    args.update(train_cfg.get("extra") or {})
    log.info("train: weights=%s data=%s epochs=%s imgsz=%s device=%s", train_cfg.get("base_weights"), data_yaml,
             args["epochs"], args["imgsz"], device)
    model = YOLO(str(train_cfg.get("base_weights", "yolo11n.pt")))
    model.train(**args)
    best = Path(runs_dir) / name / "weights" / "best.pt"
    if not best.is_file():
        best = find_latest_best(runs_dir, name)
    log.info("train done: %s", best)
    return best
