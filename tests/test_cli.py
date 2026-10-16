#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
import json

import numpy as np
import pytest

from evp.cli import main, parse_overrides
from evp.config import load_config
from evp.errors import ConfigError


def test_parse_overrides():
    o = parse_overrides(["deploy.e2e=true", "classes=[x, y]", "deploy.quant.calib_count=8", "name=a=b"])
    assert o == {"deploy": {"e2e": True, "quant": {"calib_count": 8}}, "classes": ["x", "y"], "name": "a=b"}
    assert parse_overrides(None) == {}


@pytest.mark.parametrize("bad, msg", [("novalue", "key=value"), ("=1", "empty key"), ("a=[1,", "cannot parse"),
                                      ("a=1", None)])
def test_parse_overrides_errors(bad, msg):
    if msg is None:
        items = [bad, "a.b=2"]
        with pytest.raises(ConfigError, match="conflicting"):
            parse_overrides(items)
        return
    with pytest.raises(ConfigError, match=msg):
        parse_overrides([bad])


def test_init_creates_project(tmp_path, capsys):
    d = tmp_path / "proj"
    assert main(["init", str(d)]) == 0
    cfg = load_config(d / "project.yaml")
    for k in ("raw_images", "labelme_dir", "corpus_root", "deploy_out"):
        assert (d / "project.yaml").is_file() and __import__("pathlib").Path(cfg["paths"][k]).is_dir()
    assert main(["init", str(d)]) == 2               # exists, no --force
    assert main(["init", str(d), "--force"]) == 0


def test_bad_override_exit_code(tmp_project):
    assert main(["tools", "-c", str(tmp_project / "project.yaml"), "--set", "deploy.quant.method=bogus"]) == 2


def test_infer_missing_bundle(tmp_path):
    assert main(["infer", "--bundle", str(tmp_path / "nope"), str(tmp_path / "a.jpg")]) == 2


def test_inspect_onnx(tmp_path, capsys):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    w = numpy_helper.from_array(np.ones((2, 3, 1, 1), np.float32), "W")
    s = numpy_helper.from_array(np.array(255.0, np.float32), "s")
    k = numpy_helper.from_array(np.array([3], np.int64), "K")
    shape = numpy_helper.from_array(np.array([1, -1], np.int64), "shape")
    g = helper.make_graph(
        [helper.make_node("Transpose", ["x"], ["t"], perm=[0, 3, 1, 2]), helper.make_node("Div", ["t", "s"], ["n"]),
         helper.make_node("Conv", ["n", "W"], ["f"]), helper.make_node("Reshape", ["f", "shape"], ["r"]),
         helper.make_node("TopK", ["r", "K"], ["v", "i"], axis=-1)],
        "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 4, 3])],
        [helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 3]),
         helper.make_tensor_value_info("i", TensorProto.INT64, [1, 3])], [w, s, k, shape])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    src = tmp_path / "m.onnx"
    onnx.save(m, str(src))

    assert main(["inspect-onnx", str(src)]) == 0
    info = json.loads(capsys.readouterr().out)
    assert "TopK" in info["risky"] and info["suggested_cut"] == ["r"]

    out = tmp_path / "m.prepared.onnx"
    assert main(["inspect-onnx", str(src), "--out", str(out), "--cut", "auto"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert out.is_file()
    assert "TopK" not in info["ops_after"] and "Transpose" not in info["ops_after"]
    assert info["input_shape"] == [1, 3, 4, 4]
    assert info["meta_updates"]["input.layout"] == "NCHW"
    assert info["meta_updates"]["input.norm"] == pytest.approx([1 / 255] * 3)

    assert main(["inspect-onnx", str(tmp_path / "missing.onnx")]) == 2
