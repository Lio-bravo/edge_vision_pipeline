#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
import numpy as np
import pytest

from evp.deploy.ncnn_param import parse_param
from evp.deploy.quant import (
    _rel,
    detection_similarity,
    output_conv_layers,
    output_similarity,
    parse_table,
    sample_images,
)
from evp.errors import QuantError

TABLE = """conv_0_param_0 12.5 13.0
conv_1_param_0 7.0
conv_0 40.1
conv_1 22.2
"""


def test_parse_and_write_table(tmp_path):
    t = parse_table(TABLE, {"conv_0", "conv_1"})
    assert t.layers() == ["conv_0", "conv_1"]
    out = t.write(tmp_path / "t.table", {"conv_1"})
    assert out.read_text(encoding="utf-8") == "conv_0_param_0 12.5 13.0\nconv_0 40.1\n"
    assert t.write(tmp_path / "full.table").read_text(encoding="utf-8") == TABLE
    # unknown names in fp32 set are ignored
    assert "conv_0" in t.write(tmp_path / "x.table", {"nope"}).read_text(encoding="utf-8")


def test_parse_table_layer_filter_and_malformed():
    t = parse_table(TABLE, {"conv_0"})
    assert t.layers() == ["conv_0"]
    with pytest.raises(QuantError, match="malformed table line 2"):
        parse_table("a 1\nbroken\n")


def test_output_conv_layers_through_non_quantizable():
    p = parse_param("""7767517
6 6
Input        in0    0 1 in0
Convolution  c0     1 1 in0 a 0=8 1=3 6=216
Split        sp     1 2 a b c
Convolution  head_box 1 1 b d 0=4 1=1 6=32
Convolution  head_cls 1 1 c e 0=2 1=1 6=16
Concat       cat    2 1 d e out0
""")
    assert sorted(output_conv_layers(p)) == ["head_box", "head_cls"]


def test_sample_images(tmp_path):
    imgs = [tmp_path / f"{i}.jpg" for i in range(10)]
    assert sample_images(imgs, 0) == sorted(imgs)
    assert sample_images(imgs, 20) == sorted(imgs)
    a, b = sample_images(imgs, 4, seed=1), sample_images(imgs, 4, seed=1)
    assert a == b and len(a) == 4
    with pytest.raises(QuantError, match="no calibration images"):
        sample_images([], 4)


def test_rel_rejects_spaces(tmp_path):
    (tmp_path / "a b").mkdir()
    assert _rel(tmp_path / "x" / "y.npy", tmp_path) == "x/y.npy"
    with pytest.raises(QuantError, match="unusable"):
        _rel(tmp_path / "a b" / "c.npy", tmp_path)


# ---------------------------------------------------------------- similarity metric
def head(rng, box=64, nc=4, hw=(4, 4), obj_cells=((1, 1, 2),)):
    """One stride output [box+nc, h, w] with mostly background and a few confident (cell, class) entries."""
    o = np.full((box + nc, *hw), -8.0, np.float32)
    o[:box] = rng.normal(0, 1, (box, *hw))
    for y, x, c in obj_cells:
        o[box + c, y, x] = 4.0
    return o


def test_detection_similarity_identical_and_background():
    rng = np.random.default_rng(0)
    o = [head(rng), head(rng, hw=(2, 2), obj_cells=())]
    assert detection_similarity(o, o, 64) == pytest.approx(1.0)
    bg = [head(rng, obj_cells=())]
    assert detection_similarity(bg, bg, 64) is None   # nothing relevant: no signal


def test_mean_similarity_ignores_background_images():
    from evp.deploy.quant import mean_similarity

    assert mean_similarity([0.5, None, None, 0.7]) == pytest.approx(0.6)
    assert mean_similarity([None, None]) == 1.0
    assert mean_similarity([]) == 1.0


def test_detection_similarity_detects_broken_class_scores():
    rng = np.random.default_rng(1)
    ref = [head(rng)]
    bad = [ref[0].copy()]
    bad[0][64 + 2, 1, 1] = -8.0          # object lost
    bad[0][64 + 0, 3, 3] = 4.0           # phantom object
    assert detection_similarity(bad, ref, 64) < 0.2
    # raw cosine would hide it because the box bins dominate
    assert output_similarity(bad, ref, None) > 0.9


def test_detection_similarity_box_noise_only_matters_at_objects():
    rng = np.random.default_rng(2)
    ref = [head(rng)]
    noisy_bg = [ref[0].copy()]
    noisy_bg[0][:64, 0, 0] += 50.0       # background cell boxes don't matter
    assert detection_similarity(noisy_bg, ref, 64) == pytest.approx(1.0)
    noisy_obj = [ref[0].copy()]
    noisy_obj[0][:64, 1, 1] = -noisy_obj[0][:64, 1, 1]
    assert detection_similarity(noisy_obj, ref, 64) < 0.0


def test_output_similarity_falls_back_for_non_detect_shapes():
    a = [np.array([1.0, 2.0, 3.0])]
    assert output_similarity(a, a, 64) == pytest.approx(1.0)
    assert output_similarity([np.zeros(3)], [np.zeros(3)], None) == 1.0
    assert output_similarity([np.zeros(3)], [np.ones(3)], None) == 0.0
