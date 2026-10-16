#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
import numpy as np
import pytest

from evp.deploy.meta import InputSpec
from evp.deploy.preprocess import (
    center_crop_geometry,
    letterbox_geometry,
    make_geometry,
    preprocess_ncnn,
    preprocess_numpy,
    stretch_geometry,
)

SIZES = [(810, 1080), (640, 481), (640, 482), (1, 1), (4000, 10), (10, 4000), (333, 777), (1920, 1080), (320, 320),
         (321, 319), (100, 33)]


@pytest.mark.parametrize("sw, sh", SIZES)
@pytest.mark.parametrize("dw, dh", [(320, 320), (640, 384), (256, 320)])
def test_letterbox_geometry_invariants(sw, sh, dw, dh):
    g = letterbox_geometry(sw, sh, dw, dh)
    assert g.resize_w + g.pad_left + g.pad_right == dw
    assert g.resize_h + g.pad_top + g.pad_bottom == dh
    assert min(g.pad_left, g.pad_right, g.pad_top, g.pad_bottom) >= 0
    assert g.resize_w >= 1 and g.resize_h >= 1
    assert abs(g.pad_left - g.pad_right) <= 1 and abs(g.pad_top - g.pad_bottom) <= 1


@pytest.mark.parametrize("sw, sh", SIZES)
def test_to_source_inverts_letterbox(sw, sh):
    g = letterbox_geometry(sw, sh, 320, 320)
    x, y = g.to_source(g.pad_left + g.resize_w, g.pad_top + g.resize_h)
    assert x == pytest.approx(g.resize_w / g.scale_x) and y == pytest.approx(g.resize_h / g.scale_y)
    assert g.to_source(g.pad_left, g.pad_top) == pytest.approx((0.0, 0.0))


def test_invalid_source_sizes():
    for fn in (letterbox_geometry, stretch_geometry, center_crop_geometry):
        with pytest.raises(ValueError):
            fn(0, 10, 32, 32)
    with pytest.raises(ValueError):
        make_geometry(InputSpec(w=32, h=32, resize="bogus"), 10, 10)


@pytest.mark.parametrize("sw, sh", [(640, 480), (480, 640), (300, 301), (224, 224), (1000, 50)])
def test_center_crop_matches_torchvision(sw, sh):
    tv = pytest.importorskip("torchvision.transforms.functional")
    import torch

    g = center_crop_geometry(sw, sh, 224, 224)
    x = torch.arange(sw * sh, dtype=torch.float64).reshape(1, sh, sw)
    compute = getattr(tv, "_compute_resized_output_size", None)
    resized_size = list(compute((sh, sw), [224])) if compute is not None else None
    if resized_size is not None:
        assert (g.resize_h, g.resize_w) == tuple(resized_size)
    # crop offsets: torchvision center_crop uses int(round((H - h) / 2.0))
    assert g.crop_y == int(round((g.resize_h - 224) / 2.0)) and g.crop_x == int(round((g.resize_w - 224) / 2.0))
    del x


def _img(w, h, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("sw, sh", [(810, 1080), (640, 481), (100, 33), (320, 320), (1, 1)])
def test_letterbox_pixels_match_ultralytics(sw, sh):
    aug = pytest.importorskip("ultralytics.data.augment")
    img = _img(sw, sh)
    ref = aug.LetterBox(new_shape=(320, 320), auto=False, scaleup=True, center=True)(image=img)
    x, g = preprocess_numpy(img, InputSpec(w=320, h=320, color="BGR", norm=[1.0, 1.0, 1.0]))
    assert x.shape == (3, 320, 320)
    np.testing.assert_array_equal(x.transpose(1, 2, 0).astype(np.uint8), ref)


@pytest.mark.parametrize("resize", ["letterbox", "stretch", "center_crop"])
@pytest.mark.parametrize("color", ["RGB", "BGR"])
def test_ncnn_vs_numpy_preprocess(resize, color):
    spec = InputSpec(w=96, h=64, color=color, resize=resize, mean=[10.0, 20.0, 30.0], norm=[0.01, 0.02, 0.03])
    img = _img(150, 77, seed=3)
    m, g1 = preprocess_ncnn(img, spec)
    a = np.array(m)
    b, g2 = preprocess_numpy(img, spec)
    assert g1 == g2 and a.shape == b.shape == (3, 64, 96)
    # ncnn bilinear uses fixed-point weights; allow 1-2 grey levels after normalisation
    assert np.abs(a - b).max() <= 2.0 * max(spec.norm) + 1e-6


def test_ncnn_preprocess_padding_value_and_channel_order():
    img = np.zeros((10, 20, 3), np.uint8)
    img[..., 0], img[..., 2] = 50, 200   # B=50, R=200
    m, g = preprocess_ncnn(img, InputSpec(w=32, h=32, color="RGB", norm=[1.0, 1.0, 1.0]))
    a = np.array(m)
    assert g.pad_top == 8 and g.pad_bottom == 8
    assert a[0, 0, 0] == 114                       # border
    assert a[0, 16, 16] == 200 and a[2, 16, 16] == 50   # RGB order inside


def test_ncnn_preprocess_rejects_bad_input():
    spec = InputSpec(w=32, h=32)
    for bad in (None, np.zeros((10, 10), np.uint8), np.zeros((10, 10, 3), np.float32), np.zeros((10, 10, 4), np.uint8)):
        with pytest.raises(ValueError):
            preprocess_ncnn(bad, spec)
