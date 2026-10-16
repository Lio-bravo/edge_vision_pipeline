#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""End-to-end deploys with the real toolchain (pnnx + ncnn tools + C++ runtime build).

Run with:  pytest -m integration       (several minutes; needs models/pretrained/*.pt and the tools)
Images: $EVP_TEST_IMAGES, else the Ultralytics sample images shipped for tests."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from evp.cli import main
from evp.config import load_config
from evp.utils import list_images
from tests.conftest import ASSETS, PRETRAINED, needs_weights

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_CANDIDATES = [os.environ.get("EVP_TEST_IMAGES"), str(ASSETS) if ASSETS else None]


def _image_dir() -> Path | None:
    for c in _CANDIDATES:
        if c and Path(c).is_dir() and list_images(Path(c), recursive=True):
            return Path(c)
    return None


IMAGES = _image_dir()
needs_images = pytest.mark.skipif(IMAGES is None, reason="no test images")


def cfg_for(tmp: Path, weights: Path, **deploy) -> dict:
    over = {"paths": {"deploy_out": str(tmp / "deploy"), "runs": str(tmp / "runs"), "corpus_root": str(tmp / "c"),
                      "raw_images": str(tmp / "raw")},
            "deploy": {"weights": str(weights),
                       "quant": {"calib_images": str(IMAGES), "calib_count": 32, "eval_count": 8},
                       "verify": {"images": str(IMAGES), "count": 12}, **deploy}}
    return load_config(None, over)


def run_deploy(tmp: Path, weights: Path, **deploy) -> tuple[dict, dict, Path]:
    from evp.deploy.pipeline import deploy as do_deploy

    res = do_deploy(cfg_for(tmp, weights, **deploy))
    bundle = Path(res["bundle"])
    report = json.loads((bundle / "report.json").read_text(encoding="utf-8"))
    for f in ("meta.json", "model.meta", "report.md", "runtime/CMakeLists.txt", "runtime/LICENSE"):
        assert (bundle / f).is_file(), f
    return res, report, bundle


def assert_cpp_fp32(report: dict) -> None:
    c = report["checks"].get("L3c_cpp_vs_python_fp32")
    if c is None:
        pytest.skip("C++ runtime could not be built here")
    assert c["passed"], c["detail"]
    assert c["detail"]["input_max_abs_diff"] == 0.0


@needs_images
@needs_weights("yolo11n.pt")
def test_deploy_yolo11n(tmp_path, toolchain, capsys):
    res, report, bundle = run_deploy(tmp_path, PRETRAINED / "yolo11n.pt", imgsz=[320, 320])
    assert res["passed"] is True
    ch = report["checks"]
    assert ch["L1_fp32_vs_framework_cos_min"]["value"] >= 0.999
    assert ch["L3_fp32_vs_ultralytics_f1"]["value"] >= 0.95
    assert_cpp_fp32(report)
    assert (bundle / "yolo11n.int8.param").is_file() and report["quant"]["cos_final"] > 0.9
    # CLI infer on the bundle, both precisions
    img = list_images(IMAGES, recursive=True)[0]
    vis = tmp_path / "vis"
    for prec in ("fp32", "int8"):
        assert main(["infer", "--bundle", str(bundle), str(img), "--precision", prec, "--save-vis", str(vis)]) == 0
        row = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert "detections" in row
    assert any(vis.iterdir())


@needs_images
@needs_weights("yolo26n.pt")
@pytest.mark.parametrize("e2e", [True, False])
def test_deploy_yolo26n(tmp_path, toolchain, e2e):
    res, report, bundle = run_deploy(tmp_path, PRETRAINED / "yolo26n.pt", imgsz=[320, 320], e2e=e2e,
                                     name=f"y26_{int(e2e)}")
    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    assert meta["post"]["method"] == ("e2e_topk" if e2e else "nms")
    assert res["passed"] is True, report["checks"]
    assert report["checks"]["L3_fp32_vs_ultralytics_f1"]["value"] >= 0.95
    assert_cpp_fp32(report)
    if e2e and res["passed_int8"] is False:
        assert any("deploy.e2e: false" in n for n in report["notes"])


@needs_images
@needs_weights("yolo11n-cls.pt")
def test_deploy_classify(tmp_path, toolchain):
    res, report, bundle = run_deploy(tmp_path, PRETRAINED / "yolo11n-cls.pt", imgsz=[224, 224])
    assert res["passed"] is True, report["checks"]
    assert report["checks"]["L1_fp32_vs_framework_cos_min"]["value"] >= 0.999
    assert_cpp_fp32(report)


@needs_images
def test_deploy_generic_onnx_with_in_graph_preprocessing(tmp_path, toolchain, capsys):
    """NHWC float input -> Transpose -> (x/255 - mean)/std -> CNN -> softmax. The rules must strip the
    transpose and normalization, the meta must carry the composed mean/norm, and verification must pass."""
    torch = pytest.importorskip("torch")

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.f = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, 2, 1), torch.nn.ReLU(),
                                         torch.nn.Conv2d(8, 16, 3, 2, 1), torch.nn.ReLU(),
                                         torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(16, 5))

        def forward(self, x):
            x = x.permute(0, 3, 1, 2)
            x = (x / 255.0 - mean) / std
            return torch.softmax(self.f(x), 1)

    onnx_path = tmp_path / "tiny_cls.onnx"
    torch.onnx.export(Net().eval(), torch.zeros(1, 64, 64, 3), str(onnx_path), opset_version=17,
                      input_names=["image"], output_names=["prob"], dynamo=False)
    res, report, bundle = run_deploy(tmp_path, onnx_path, family="classify", imgsz=[64, 64],
                                     labels=["a", "b", "c", "d", "e"], input={"resize": "stretch"})
    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    inp = meta["input"]
    np.testing.assert_allclose(inp["mean"], (mean.view(-1) * 255).tolist(), rtol=1e-4)
    np.testing.assert_allclose(inp["norm"], (1 / (std.view(-1) * 255)).tolist(), rtol=1e-4)
    assert meta["post"]["softmax"] is False                       # graph already outputs probabilities
    assert "strip_input_transpose" in res["convert"]["rules_applied"]
    assert "extract_input_normalization" in res["convert"]["rules_applied"]
    assert res["passed"] is True, report["checks"]
    assert report["checks"]["L1_fp32_vs_framework_cos_min"]["value"] >= 0.999
    assert_cpp_fp32(report)
    img = list_images(IMAGES, recursive=True)[0]
    assert main(["infer", "--bundle", str(bundle), str(img)]) == 0
    row = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert len(row["topk"]) == 5 and row["topk"][0]["label"] in "abcde"
    assert sum(t["score"] for t in row["topk"]) == pytest.approx(1.0, abs=1e-3)


@needs_images
@needs_weights("yolo11n.pt")
def test_sop_end_to_end(tmp_path, toolchain, capsys):
    """The full SOP through the CLI: prelabel (.pt) -> review (unchanged) -> ingest -> train (1 CPU epoch)
    -> deploy -> prelabel the next batch with the bundle -> incremental ingest (old val is not re-split)."""
    import shutil

    import yaml

    imgs = list_images(IMAGES, recursive=True)
    if len(imgs) < 20:
        pytest.skip("need >= 20 test images")
    proj = tmp_path / "proj"
    raw = proj / "ws" / "raw"
    raw.mkdir(parents=True)
    cfg = {"classes": ["person", "car", "chair"],
           "paths": {"prelabel_model": str(PRETRAINED / "yolo11n.pt"), "raw_images": "ws/raw",
                     "labelme_dir": "ws/labelme", "batch_dataset": "ws/batch", "corpus_root": "ws/corpus",
                     "runs": "ws/runs", "deploy_out": "ws/deploy"},
           "prelabel": {"device": "cpu"},
           "dataset": {"val_ratio": 0.2, "seed": 0},
           "train": {"base_weights": str(PRETRAINED / "yolo11n.pt"), "epochs": 1, "batch": 8, "imgsz": 160,
                     "workers": 0, "device": "auto", "run_name": "sop", "extra": {"plots": False}},
           "deploy": {"imgsz": [160, 160], "cpp_runtime": False,
                      "quant": {"calib_count": 8, "eval_count": 4}, "verify": {"count": 4}}}
    pj = proj / "project.yaml"
    pj.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    c = ["-c", str(pj)]

    def last_json() -> dict:
        return json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    for p in imgs[:16]:
        shutil.copy2(p, raw / p.name)
    assert main(["prelabel", *c]) == 0
    assert last_json()["written"] == 16
    assert main(["ingest", *c]) == 0
    merged = last_json()
    assert merged["after"]["train"] + merged["after"]["val"] == 16 and merged["after"]["val"] >= 1
    assert not any(raw.iterdir())                                  # staging cleared after merge

    assert main(["train", *c]) == 0
    best = Path(last_json()["best"])
    assert best.is_file()

    assert main(["deploy", *c]) == 0
    res = last_json()
    assert res["passed"] is True                                   # fp32 is always measurable through L1
    report = json.loads(Path(res["report"]).read_text(encoding="utf-8"))
    meta = json.loads((Path(res["bundle"]) / "meta.json").read_text(encoding="utf-8"))
    assert Path(meta["source"]["weights"]) == best.resolve()        # deploy weights=auto picked the new run
    if "int8" in report.get("unverified", []):
        assert res["recommended_precision"] == "fp32"

    for p in imgs[16:20]:
        shutil.copy2(p, raw / p.name)
    assert main(["prelabel", *c, "--bundle", res["bundle"]]) == 0
    assert last_json()["written"] == 4
    assert main(["ingest", *c]) == 0
    again = last_json()
    assert again["before"] == merged["after"]
    assert again["after"]["val"] >= merged["after"]["val"]
    assert again["after"]["train"] + again["after"]["val"] == 20


def test_deploy_rejects_yolo_onnx(tmp_path):
    from evp.deploy.pipeline import deploy as do_deploy
    from evp.errors import ConfigError

    fake = tmp_path / "y.onnx"
    fake.write_bytes(b"")
    cfg = load_config(None, {"deploy": {"weights": str(fake), "family": "yolo"},
                             "paths": {"deploy_out": str(tmp_path / "d")}})
    with pytest.raises(ConfigError, match="Ultralytics .pt"):
        do_deploy(cfg)
