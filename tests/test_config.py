#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
from pathlib import Path

import pytest
import yaml

from evp.cli import parse_overrides
from evp.config import DEFAULTS, load_config
from evp.errors import ConfigError


def test_defaults_load_without_file():
    cfg = load_config(None)
    assert cfg["deploy"]["imgsz"] == [320, 320]
    assert cfg["deploy"]["quant"]["method"] == "auto"
    assert Path(cfg["paths"]["corpus_root"]).is_absolute()


def test_relative_paths_resolve_against_config_dir(tmp_project: Path):
    cfg = load_config(tmp_project / "project.yaml")
    assert Path(cfg["paths"]["raw_images"]) == (tmp_project / "ws" / "raw").resolve()
    assert cfg["classes"] == ["a", "b"]


def test_deploy_paths_resolve(tmp_project: Path):
    cfg = load_config(tmp_project / "project.yaml", {"deploy": {"weights": "m/x.pt",
                                                                "quant": {"calib_images": "calib"}}})
    assert Path(cfg["deploy"]["weights"]) == (tmp_project / "m" / "x.pt").resolve()
    assert Path(cfg["deploy"]["quant"]["calib_images"]) == (tmp_project / "calib").resolve()
    assert cfg["deploy"]["verify"]["images"] == "auto"


def test_int_imgsz_expands():
    assert load_config(None, {"deploy": {"imgsz": 256}})["deploy"]["imgsz"] == [256, 256]


@pytest.mark.parametrize("override, fragment", [
    ({"dataset": {"val_ratio": 0}}, "val_ratio"),
    ({"dataset": {"val_ratio": 1.5}}, "val_ratio"),
    ({"deploy": {"imgsz": [320]}}, "imgsz"),
    ({"deploy": {"imgsz": [0, 320]}}, "imgsz"),
    ({"deploy": {"input_color": "GRAY"}}, "input_color"),
    ({"deploy": {"family": "segment"}}, "family"),
    ({"deploy": {"e2e": "yes"}}, "e2e"),
    ({"deploy": {"quant": {"method": "minmax"}}}, "method"),
    ({"deploy": {"post": {"conf": 2}}}, "conf"),
    ({"deploy": {"input": {"mean": [1, 2]}}}, "mean"),
    ({"deploy": {"input": {"resize": "crop"}}}, "resize"),
    ({"deploy": {"input": {"pad_value": 300}}}, "pad_value"),
    ({"classes": ["a", "a"]}, "duplicate"),
    ({"classes": ["a", ""]}, "invalid class"),
])
def test_invalid_values_raise(override, fragment):
    with pytest.raises(ConfigError) as ei:
        load_config(None, override)
    assert fragment in str(ei.value)


def test_missing_config_file(tmp_path: Path):
    with pytest.raises(ConfigError, match="config not found"):
        load_config(tmp_path / "nope.yaml")


def test_non_mapping_yaml(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(p)


def test_invalid_yaml(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("a: [1, 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(p)


def test_defaults_not_mutated(tmp_project: Path):
    before = yaml.safe_dump(DEFAULTS)
    load_config(tmp_project / "project.yaml", {"deploy": {"post": {"conf": 0.5}}})
    assert yaml.safe_dump(DEFAULTS) == before


def test_parse_overrides_nested_and_typed():
    o = parse_overrides(["deploy.e2e=true", "deploy.imgsz=[256,320]", "classes=[x, y]", "deploy.name=abc"])
    assert o == {"deploy": {"e2e": True, "imgsz": [256, 320], "name": "abc"}, "classes": ["x", "y"]}


@pytest.mark.parametrize("bad", ["noequals", "=1", "a..=1"])
def test_parse_overrides_errors(bad):
    if bad == "a..=1":
        assert parse_overrides([bad]) == {"a": 1}
        return
    with pytest.raises(ConfigError):
        parse_overrides([bad])


def test_parse_overrides_conflict():
    with pytest.raises(ConfigError, match="conflicting"):
        parse_overrides(["a=1", "a.b=2"])


@pytest.mark.parametrize("override, msg", [
    ({"deploy": {"onnx": {"cut_outputs": "a,b"}}}, "cut_outputs"),
    ({"deploy": {"onnx": {"cut_outputs": [""]}}}, "cut_outputs"),
    ({"deploy": {"onnx": {"rules": ["nope"]}}}, "deploy.onnx.rules"),
    ({"deploy": {"onnx": {"rules": "all"}}}, "deploy.onnx.rules"),
    ({"deploy": {"post": {"conf": "abc"}}}, "deploy.post.conf must be a number"),
    ({"deploy": {"post": {"max_det": 1.5}}}, "must be an integer"),
    ({"deploy": {"post": {"topk": 0}}}, "topk"),
    ({"deploy": {"quant": {"calib_count": True}}}, "calib_count"),
    ({"deploy": {"quant": {"fp32_layers": "conv_1"}}}, "fp32_layers"),
    ({"deploy": {"verify": {"min_int8_cos": 2}}}, "min_int8_cos"),
    ({"deploy": {"verify": {"count": 0}}}, "verify.count"),
])
def test_deploy_validation(override, msg):
    with pytest.raises(ConfigError, match=msg):
        load_config(None, override)


def test_deploy_validation_accepts_valid_onnx_options():
    cfg = load_config(None, {"deploy": {"onnx": {"cut_outputs": "auto", "rules": ["einsum_to_matmul"]}}})
    assert cfg["deploy"]["onnx"]["cut_outputs"] == "auto"
    load_config(None, {"deploy": {"onnx": {"cut_outputs": ["/m/Concat_output_0"], "rules": "none"}}})
