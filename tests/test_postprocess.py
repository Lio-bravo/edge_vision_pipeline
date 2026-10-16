#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Post-processing vs Ultralytics / torchvision reference implementations on synthetic head outputs."""

import numpy as np
import pytest

from evp.deploy.meta import InputSpec, ModelMeta, OutputSpec, PostSpec
from evp.deploy.postprocess import (
    Candidates,
    dfl_decode,
    e2e_topk,
    gather_candidates,
    inv_sigmoid,
    nms,
    postprocess_classify,
    postprocess_detect,
    sigmoid,
    to_source,
)
from evp.deploy.preprocess import letterbox_geometry

torch = pytest.importorskip("torch")


def make_outputs(nc: int, reg_max: int, h: int = 64, w: int = 64, seed: int = 0, n_obj: int = 30):
    """Random raw head tensors with a few confident cells (so NMS / top-k have real work)."""
    rng = np.random.default_rng(seed)
    outs, strides = [], [8, 16, 32]
    for s in strides:
        gh, gw = h // s, w // s
        o = rng.normal(0, 1, (4 * reg_max + nc, gh, gw)).astype(np.float32)
        o[4 * reg_max:] -= 6.0
        if reg_max == 1:
            o[:4] = np.abs(o[:4]) * 2 + 0.5
        for _ in range(n_obj // len(strides)):
            y, x, c = rng.integers(gh), rng.integers(gw), rng.integers(nc)
            o[4 * reg_max + c, y, x] = rng.uniform(-1, 4)
        outs.append(o)
    return outs, strides


def meta_for(nc, reg_max, method="nms", conf=0.25, iou=0.45, max_det=100, agnostic=False, h=64, w=64):
    outs, strides = make_outputs(nc, reg_max, h, w)
    return ModelMeta(name="t", family="yolo", input=InputSpec(w=w, h=h),
                     outputs=[OutputSpec(f"out{i}", list(o.shape), s) for i, (o, s) in enumerate(zip(outs, strides))],
                     labels=[str(i) for i in range(nc)], reg_max=reg_max,
                     post=PostSpec(method=method, conf=conf, iou=iou, max_det=max_det, agnostic=agnostic))


def ultralytics_decode(outs, strides, nc, reg_max):
    """Ultralytics-style decode: DFL (softmax expectation) + dist2bbox(xyxy) * stride + sigmoid scores."""
    from ultralytics.utils.tal import dist2bbox, make_anchors

    feats = [torch.from_numpy(o[None].astype(np.float64)) for o in outs]
    anchors, stride_t = make_anchors(feats, torch.tensor(strides, dtype=torch.float64), 0.5)
    x = torch.cat([f.view(1, 4 * reg_max + nc, -1) for f in feats], 2)
    box, cls = x.split((4 * reg_max, nc), 1)
    if reg_max > 1:
        b, _, a = box.shape
        box = box.view(b, 4, reg_max, a).transpose(2, 1).softmax(1)
        box = (box * torch.arange(reg_max, dtype=torch.float64).view(1, reg_max, 1, 1)).sum(1)
    anchors, stride_t = anchors.transpose(0, 1), stride_t.transpose(0, 1)   # as in Detect._inference
    dbox = dist2bbox(box, anchors.unsqueeze(0), xywh=False, dim=1) * stride_t
    return torch.cat((dbox, cls.sigmoid()), 1)  # [1, 4+nc, A]


# ---------------------------------------------------------------- primitives
def test_sigmoid_inverse():
    for p in (0.01, 0.25, 0.5, 0.9, 0.999):
        assert sigmoid(np.array(inv_sigmoid(p))) == pytest.approx(p)
    assert np.isfinite(inv_sigmoid(0.0)) and np.isfinite(inv_sigmoid(1.0))


def test_dfl_decode_matches_manual_softmax():
    rng = np.random.default_rng(0)
    bins = rng.normal(0, 3, (5, 64))
    d = dfl_decode(bins, 16)
    for i in range(5):
        for side in range(4):
            b = bins[i, side * 16:(side + 1) * 16]
            p = np.exp(b - b.max()) / np.exp(b - b.max()).sum()
            assert d[i, side] == pytest.approx((p * np.arange(16)).sum())


def test_gather_shape_errors():
    with pytest.raises(ValueError):
        gather_candidates([np.zeros((10, 2, 2))], [8, 16], 2, 1, 0.25)
    with pytest.raises(ValueError):
        gather_candidates([np.zeros((7, 2, 2))], [8], 2, 1, 0.25)


@pytest.mark.parametrize("reg_max", [1, 16])
def test_gather_matches_ultralytics_decode(reg_max):
    pytest.importorskip("ultralytics")
    nc = 5
    outs, strides = make_outputs(nc, reg_max)
    ref = ultralytics_decode(outs, strides, nc, reg_max)[0].T.numpy()   # [A, 4+nc]
    keep = ref[:, 4:].max(1) > 0.25
    c = gather_candidates(outs, strides, nc, reg_max, 0.25)
    assert len(c.boxes) == keep.sum() > 0
    np.testing.assert_allclose(c.boxes, ref[keep, :4], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(c.scores, ref[keep, 4:], rtol=1e-6, atol=1e-9)


def test_gather_conf_zero_keeps_everything():
    outs, strides = make_outputs(3, 1)
    c = gather_candidates(outs, strides, 3, 1, 0.0)
    assert len(c.boxes) == sum(o.shape[1] * o.shape[2] for o in outs)


# ---------------------------------------------------------------- NMS
@pytest.mark.parametrize("agnostic", [False, True])
@pytest.mark.parametrize("seed", range(5))
def test_nms_matches_torchvision(agnostic, seed):
    tvops = pytest.importorskip("torchvision.ops")
    rng = np.random.default_rng(seed)
    n = 300
    xy = rng.uniform(0, 300, (n, 2))
    wh = rng.uniform(5, 80, (n, 2))
    dets = np.concatenate([xy, xy + wh, rng.uniform(0.3, 1, (n, 1)), rng.integers(0, 4, (n, 1))], 1)
    ours = nms(dets, 0.45, agnostic, 1000)
    b = torch.from_numpy(dets[:, :4])
    s = torch.from_numpy(dets[:, 4])
    keep = tvops.nms(b, s, 0.45) if agnostic else tvops.batched_nms(b, s, torch.from_numpy(dets[:, 5]).long(), 0.45)
    np.testing.assert_allclose(ours, dets[keep.numpy()])


def test_nms_max_det_and_empty():
    assert nms(np.zeros((0, 6)), 0.5, False, 10).shape == (0, 6)
    dets = np.array([[i * 100, 0, i * 100 + 10, 10, 0.9 - i * 0.01, 0] for i in range(20)], float)
    assert len(nms(dets, 0.5, False, 7)) == 7


def test_nms_iou_threshold_is_strict():
    a = [0, 0, 10, 10, 0.9, 0]
    b = [0, 0, 10, 5, 0.8, 0]   # IoU exactly 0.5
    assert len(nms(np.array([a, b], float), 0.5, False, 10)) == 2
    assert len(nms(np.array([a, b], float), 0.49, False, 10)) == 1


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("reg_max, agnostic", [(16, False), (16, True), (1, False)])
def test_detect_matches_ultralytics_nms(seed, reg_max, agnostic):
    pytest.importorskip("ultralytics")
    from ultralytics.utils.nms import non_max_suppression

    nc = 4
    meta = meta_for(nc, reg_max, agnostic=agnostic)
    outs, strides = make_outputs(nc, reg_max, seed=seed)
    from ultralytics.utils.ops import xyxy2xywh

    pred = ultralytics_decode(outs, strides, nc, reg_max)
    pred[:, :4] = xyxy2xywh(pred[:, :4].transpose(1, 2)).transpose(1, 2)   # NMS expects xywh (Detect output)
    ref = non_max_suppression(pred.float(), 0.25, 0.45, agnostic=agnostic, max_det=100)[0].numpy()
    ref[:, :4] = ref[:, :4].clip(0, 64)                                      # scale_boxes clips afterwards
    g = letterbox_geometry(64, 64, 64, 64)
    ours = np.array([d.as_list() for d in postprocess_detect(outs, meta, g)]).reshape(-1, 6)
    assert len(ours) == len(ref)
    np.testing.assert_allclose(ours, ref, rtol=1e-4, atol=1e-3)


# ---------------------------------------------------------------- e2e top-k
def ultralytics_e2e(pred, nc, max_det, agnostic, conf):
    """Ultralytics Detect.get_topk_index (k = max(max_det, 300) like the predictor) + conf filter + [:max_det]."""
    from ultralytics.nn.modules.head import Detect

    head = Detect(nc=nc, reg_max=1, end2end=True, ch=(8,))
    head.agnostic_nms = agnostic
    k = max(max_det, 300)
    boxes, scores = pred[:4].T[None], pred[4:].T[None]   # pred [4+nc, A] -> [1, A, 4], [1, A, nc]
    s, labels, idx = head.get_topk_index(scores, k)
    b = boxes[0][idx[0]]
    out = torch.cat([b, s[0], labels[0]], 1)
    out = out[out[:, 4] > conf][:max_det]
    return out.numpy()


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("agnostic", [False, True])
@pytest.mark.parametrize("max_det", [5, 100])
def test_e2e_topk_matches_ultralytics(seed, agnostic, max_det):
    pytest.importorskip("ultralytics")
    nc = 6
    outs, strides = make_outputs(nc, 1, seed=seed, n_obj=60)
    pred = ultralytics_decode(outs, strides, nc, 1)[0].double()
    ref = ultralytics_e2e(pred, nc, max_det, agnostic, 0.25)
    c = gather_candidates(outs, strides, nc, 1, 0.25)
    ours = e2e_topk(c, 0.25, max_det, agnostic)
    assert len(ours) == len(ref)
    # same detections; order may differ only between exactly-equal scores (torch.topk is not stable)
    key = lambda a: np.lexsort((a[:, 5], -a[:, 4]))  # noqa: E731
    np.testing.assert_allclose(ours[key(ours)], ref[key(ref)], rtol=1e-6, atol=1e-5)


def test_e2e_multi_class_same_anchor():
    """One anchor confident for two classes yields two detections (no NMS in one-to-one heads)."""
    c = Candidates(boxes=np.array([[0, 0, 10, 10.0]]), scores=np.array([[0.9, 0.8, 0.1]]),
                   best=np.array([0]), best_score=np.array([0.9]))
    out = e2e_topk(c, 0.25, 10, agnostic=False)
    assert out[:, 5].tolist() == [0, 1]
    assert len(e2e_topk(c, 0.25, 10, agnostic=True)) == 1
    assert len(e2e_topk(c, 0.85, 10, agnostic=False)) == 1


def test_e2e_empty():
    c = gather_candidates([np.full((6, 2, 2), -20, np.float32)], [8], 2, 1, 0.25)
    assert e2e_topk(c, 0.25, 10, False).shape == (0, 6)


# ---------------------------------------------------------------- to_source / classify
def test_to_source_clips_and_unletterboxes():
    g = letterbox_geometry(200, 100, 64, 64)    # r=0.32, resize 64x32, pad top 16
    dets = to_source(np.array([[-5, 16, 70, 48, 0.9, 1]], float), g)
    d = dets[0]
    assert (d.x1, d.y1) == (0.0, 0.0) and d.x2 == 200.0 and d.y2 == pytest.approx(100.0)
    assert d.cls == 1 and d.score == pytest.approx(0.9)


def test_postprocess_classify():
    p = PostSpec(method="topk", topk=3, softmax=True)
    r = postprocess_classify(np.array([0.1, 2.0, -1.0, 2.0]), p)
    assert [i for i, _ in r] == [1, 3, 0]       # stable on ties
    probs = postprocess_classify(np.array([0.1, 2.0, -1.0, 2.0]), PostSpec(method="topk", topk=10, softmax=True))
    assert sum(s for _, s in probs) == pytest.approx(1.0) and len(probs) == 4
    raw = postprocess_classify(np.array([0.2, 0.7, 0.1]), PostSpec(method="topk", topk=1, softmax=False))
    assert raw == [(1, pytest.approx(0.7))]
