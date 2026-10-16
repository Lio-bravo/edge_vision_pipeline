#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PRETRAINED = ROOT / "models" / "pretrained"


def _assets_dir() -> Path | None:
    try:
        import ultralytics

        d = Path(ultralytics.__file__).parent / "assets"
        return d if d.is_dir() else None
    except ImportError:
        return None


ASSETS = _assets_dir()


def write_image(path: Path, w: int, h: int, seed: int = 0) -> Path:
    import cv2

    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    assert ok
    buf.tofile(str(path))
    return path


def write_labelme(json_path: Path, image_name: str, w: int, h: int, shapes: list[dict]) -> Path:
    json_path.write_text(json.dumps({"version": "5.4.1", "flags": {}, "shapes": shapes, "imagePath": image_name,
                                     "imageData": None, "imageWidth": w, "imageHeight": h}), encoding="utf-8")
    return json_path


def rect(label: str, x1, y1, x2, y2) -> dict:
    return {"label": label, "points": [[x1, y1], [x2, y2]], "shape_type": "rectangle", "group_id": None,
            "flags": {}}


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """A project folder with a config using classes [a, b] and relative workspace paths."""
    cfg = {
        "classes": ["a", "b"],
        "paths": {"raw_images": "ws/raw", "labelme_dir": "ws/labelme", "batch_dataset": "ws/batch",
                  "corpus_root": "ws/corpus", "runs": "ws/runs", "deploy_out": "ws/deploy",
                  "prelabel_model": "models/p.pt"},
        "dataset": {"val_ratio": 0.25, "seed": 1, "clear_after_merge": True},
    }
    import yaml

    (tmp_path / "project.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return tmp_path


def needs_weights(name: str):
    return pytest.mark.skipif(not (PRETRAINED / name).is_file(), reason=f"missing {PRETRAINED / name}")


def toolchain_available() -> bool:
    try:
        from evp.deploy.tools import ToolRunner

        r = ToolRunner()
        r.get("pnnx")
        r.get("ncnn2table")
        r.get("ncnn2int8")
        return True
    except Exception:  # noqa: BLE001
        return False


_TOOLCHAIN: bool | None = None


@pytest.fixture(scope="session")
def toolchain() -> bool:
    global _TOOLCHAIN
    if _TOOLCHAIN is None:
        _TOOLCHAIN = toolchain_available()
    if not _TOOLCHAIN:
        pytest.skip("pnnx / ncnn tools not available (run scripts/setup_tools_wsl.ps1 or scripts/setup_tools_linux.sh)")
    return True
