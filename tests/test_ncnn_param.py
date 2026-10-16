#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
import pytest

from evp.deploy.ncnn_param import is_registered, load_param, parse_param, unsupported_layers
from evp.errors import ConvertError

PARAM = """7767517
6 7
Input            in0      0 1 in0
Convolution      conv_0   1 1 in0 1 0=16 1=3 4=1 5=1 6=432
Split            splitncnn_0 1 2 1 2 3
Convolution      conv_1   1 1 2 out0 0=4 1=1 5=1 6=64 8=2
Sigmoid          sig_0    1 1 3 4
torch.topk       topk_0   1 2 4 out1 out2
"""


def test_parse_roundtrip(tmp_path):
    p = parse_param(PARAM)
    assert p.inputs == ["in0"]
    assert p.outputs == ["out0", "out1", "out2"]
    assert p.layer_types()["Convolution"] == 2
    assert p.blob_count == 8
    assert p.producer("out0").name == "conv_1"
    assert p.producer("nope") is None
    assert [l.name for l in p.quantizable_layers()] == ["conv_0", "conv_1"]
    assert p.int8_layers() == ["conv_1"]
    assert p.layers[1].param_dict() == {"0": "16", "1": "3", "4": "1", "5": "1", "6": "432"}
    f = tmp_path / "m.param"
    p.save(f)
    q = load_param(f)
    assert [(l.type, l.name, l.bottoms, l.tops, l.params) for l in q.layers] == \
           [(l.type, l.name, l.bottoms, l.tops, l.params) for l in p.layers]


def test_unsupported_layers():
    p = parse_param(PARAM)
    assert [l.type for l in unsupported_layers(p)] == ["torch.topk"]
    assert is_registered("Convolution") and not is_registered("aten::foo") and not is_registered("Tensor.index")


@pytest.mark.parametrize("text, msg", [
    ("", "bad magic"),
    ("123\n1 1\n", "bad magic"),
    ("7767517\n", "truncated"),
    ("7767517\nx y\n", "bad ncnn param header"),
    ("7767517\n1 1\nInput in0 0\n", "too few fields"),
    ("7767517\n1 1\nInput in0 a 1 in0\n", "bad bottom/top counts"),
    ("7767517\n1 1\nInput in0 0 2 in0\n", "missing blob names"),
    ("7767517\n2 1\nInput in0 0 1 in0\n", "declares 2 layers but has 1"),
])
def test_parse_errors(text, msg):
    with pytest.raises(ConvertError, match=msg):
        parse_param(text)


def test_load_missing(tmp_path):
    with pytest.raises(ConvertError, match="param not found"):
        load_param(tmp_path / "none.param")
