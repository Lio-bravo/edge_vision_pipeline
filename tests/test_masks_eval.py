#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from evp.data.masks import mask_to_boxes
from evp.deploy.evaluate import (
    Box,
    average_precision,
    class_map,
    evaluate,
    iou_matrix,
    load_yolo_split,
    render_eval_markdown,
    run_eval,
)
from evp.errors import DataError
from tests.conftest import write_image

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- masks
def test_mask_to_boxes_merges_fragments_and_drops_specks():
    m = np.zeros((100, 200), np.uint8)
    m[10:20, 10:30] = 2          # fragment 1
    m[10:20, 35:50] = 2          # fragment 2, 5 px gap -> merged
    m[80:82, 180:182] = 2        # 2 px speck -> dropped
    m[50:90, 100:120] = 1        # another type
    boxes = mask_to_boxes(m, merge_px=4, min_side=5)
    assert [(b.value, b.x1, b.y1, b.x2, b.y2) for b in boxes] == [(1, 100, 50, 120, 90), (2, 10, 10, 50, 20)]
    assert boxes[1].pixels == 10 * 20 + 10 * 15


def test_mask_to_boxes_no_merge_keeps_fragments_separate():
    m = np.zeros((40, 80), np.uint8)
    m[5:15, 5:15] = 1
    m[5:15, 20:30] = 1
    assert len(mask_to_boxes(m, merge_px=0, min_side=1)) == 2
    assert len(mask_to_boxes(m, merge_px=4, min_side=1)) == 1


def test_mask_to_boxes_edge_cases():
    assert mask_to_boxes(np.zeros((10, 10), np.uint8)) == []
    full = np.ones((10, 12), np.uint8)
    (b,) = mask_to_boxes(full, merge_px=3, min_side=1)
    assert (b.x1, b.y1, b.x2, b.y2) == (0, 0, 12, 10)       # dilation never grows the box past the pixels
    m = np.zeros((10, 10), np.uint8)
    m[1:5, 1:5], m[6:9, 6:9] = 1, 3
    assert [b.value for b in mask_to_boxes(m, 0, 1, values={3: "x"})] == [3]
    with pytest.raises(DataError):
        mask_to_boxes(np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(DataError):
        mask_to_boxes(np.zeros((4, 4), np.uint8), merge_px=-1)


# ---------------------------------------------------------------- metric
def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0]])
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_average_precision_known_values():
    assert average_precision(np.array([True, True]), np.array([0.9, 0.8]), 2) == pytest.approx(1.0)
    assert average_precision(np.array([], bool), np.array([]), 3) == 0.0
    assert np.isnan(average_precision(np.array([False]), np.array([0.5]), 0))
    # TP, FP, TP over 2 GT: precision envelope 1.0 up to recall .5, then 2/3 up to recall 1
    ap = average_precision(np.array([True, False, True]), np.array([0.9, 0.8, 0.7]), 2)
    assert ap == pytest.approx((51 * 1.0 + 50 * 2 / 3) / 101)


def test_evaluate_perfect_and_empty():
    gts = {"a": [Box(0, 0, 0, 10, 10)], "b": [Box(1, 5, 5, 25, 25)], "c": []}
    perfect = {k: [Box(g.cls, g.x1, g.y1, g.x2, g.y2, 0.9) for g in v] for k, v in gts.items()}
    r = evaluate(perfect, gts, ["x", "y", "z"])
    assert r["mAP50"] == pytest.approx(1.0) and r["mAP50_95"] == pytest.approx(1.0)
    assert (r["precision"], r["recall"], r["f1"]) == (1.0, 1.0, 1.0)
    assert r["per_class"]["z"]["AP50"] is None and r["instances"] == 2
    r0 = evaluate({}, gts, ["x", "y", "z"])
    assert r0["mAP50"] == 0.0 and r0["recall"] == 0.0 and r0["precision"] == 0.0


def test_evaluate_conf_threshold_and_localization():
    gts = {"a": [Box(0, 0, 0, 100, 100)]}
    shifted = {"a": [Box(0, 20, 0, 120, 100, 0.9)]}            # IoU = 80/120 = 0.667
    r = evaluate(shifted, gts, ["x"])
    assert r["mAP50"] == pytest.approx(1.0)
    assert r["mAP50_95"] == pytest.approx(4 / 10)              # passes .50 .55 .60 .65 only
    low = {"a": [Box(0, 0, 0, 100, 100, 0.1)]}
    r = evaluate(low, gts, ["x"], conf=0.25)
    assert r["mAP50"] == pytest.approx(1.0) and r["recall"] == 0.0   # counted for mAP, not at the deploy conf


def test_evaluate_duplicate_detection_is_false_positive():
    gts = {"a": [Box(0, 0, 0, 10, 10)]}
    preds = {"a": [Box(0, 0, 0, 10, 10, 0.9), Box(0, 0, 0, 10, 10, 0.8)]}
    r = evaluate(preds, gts, ["x"])
    assert r["precision"] == 0.5 and r["recall"] == 1.0 and r["mAP50"] == pytest.approx(1.0)


def test_evaluate_rejects_bad_class_ids():
    with pytest.raises(DataError):
        evaluate({"a": [Box(5, 0, 0, 1, 1)]}, {"a": []}, ["x"])
    with pytest.raises(DataError):
        evaluate({}, {}, [])


def test_load_yolo_split_and_run_eval(tmp_path: Path):
    write_image(tmp_path / "images" / "val" / "a.jpg", 100, 50)
    write_image(tmp_path / "images" / "val" / "b.jpg", 100, 50, seed=1)
    (tmp_path / "labels" / "val").mkdir(parents=True)
    (tmp_path / "labels" / "val" / "a.txt").write_text("1 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    images, gts = load_yolo_split(tmp_path, "val")
    assert [p.name for p in images] == ["a.jpg", "b.jpg"] and gts["b"] == []
    g = gts["a"][0]
    assert (g.cls, g.x1, g.y1, g.x2, g.y2) == (1, 40.0, 15.0, 60.0, 35.0)

    def predict(p):  # model with class order [y, x]: id 0 is "y"
        return [Box(0, 40, 15, 60, 35, 0.8)] if p.stem == "a" else []

    r = run_eval(predict, ["y", "x"], images, gts, ["x", "y"], conf=0.25)
    assert r["mAP50"] == pytest.approx(1.0) and r["latency_ms"] >= 0
    md = render_eval_markdown({"ncnn-fp32": r}, ["x", "y"])
    assert "| y | 1 | 1.000 |" in md and "| x | 0 | n/a |" in md

    (tmp_path / "labels" / "val" / "a.txt").write_text("1 0.5 0.5\n", encoding="utf-8")
    with pytest.raises(DataError, match="5 fields"):
        load_yolo_split(tmp_path, "val")
    with pytest.raises(DataError, match="no images"):
        load_yolo_split(tmp_path, "test")


def test_class_map():
    assert class_map(["b", "a", "zz"], ["a", "b"]) == {0: 1, 1: 0}
    with pytest.raises(DataError):
        class_map(["q"], ["a"])


_SMOKE = ROOT / "scratch" / "e2e_smoke" / "workspace"


@pytest.mark.skipif(not (_SMOKE / "deploy" / "best" / "meta.json").is_file()
                    or not (_SMOKE / "corpus" / "images" / "val").is_dir(),
                    reason="needs the scratch/e2e_smoke bundle and corpus")
def test_cli_eval_on_smoke_bundle(tmp_path: Path, capsys):
    from evp.cli import main

    out = tmp_path / "ev"
    assert main(["eval", "--bundle", str(_SMOKE / "deploy" / "best"), "--data", str(_SMOKE / "corpus"),
                 "--limit", "4", "--threads", "2", "--out", str(out),
                 "--set", "classes=[person,car,chair]"]) == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert set(printed) == {"ncnn-fp32", "ncnn-int8"}
    data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["results"]["ncnn-fp32"]["images"] == 4
    assert "| model | mAP50 |" in out.with_suffix(".md").read_text(encoding="utf-8")


@pytest.mark.skipif(not (_SMOKE / "deploy" / "best" / "meta.json").is_file()
                    or not (_SMOKE / "corpus" / "images" / "val").is_dir(),
                    reason="needs the scratch/e2e_smoke bundle and corpus")
def test_cli_bench_on_smoke_bundle(tmp_path: Path, capsys):
    import shutil

    from evp.cli import main

    bundle = tmp_path / "b"
    shutil.copytree(_SMOKE / "deploy" / "best", bundle)
    assert main(["bench", "--bundle", str(bundle), "--images", str(_SMOKE / "corpus" / "images" / "val"),
                 "--count", "2", "--threads", "1", "2", "--runs", "2", "--warmup", "0"]) == 0
    rows = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert {(r["precision"], r["threads"]) for r in rows} == {("fp32", 1), ("fp32", 2), ("int8", 1), ("int8", 2)}
    assert "| fp32 | 1 |" in (bundle / "bench.md").read_text(encoding="utf-8")
    assert main(["bench", "--bundle", str(bundle), "--images", str(tmp_path / "empty_missing")]) == 2


# ---------------------------------------------------------------- VisA script on a synthetic mini dataset
def _load_script():
    spec = importlib.util.spec_from_file_location("prepare_visa", ROOT / "scripts" / "prepare_visa.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mini_visa(root: Path) -> None:
    import cv2

    rows = [("image", "label", "mask")]
    split = [("object", "split", "label", "image", "mask")]
    specs = [("000", "bent", {1: (10, 10, 40, 30)}, "train"), ("001", "melt", {2: (50, 20, 70, 60)}, "train"),
             ("002", "bent,dirt", {1: (5, 5, 25, 25), 3: (60, 60, 90, 90)}, "test")]
    for stem, label, regions, sp in specs:
        img = f"pcbx/Data/Images/Anomaly/{stem}.JPG"
        msk = f"pcbx/Data/Masks/Anomaly/{stem}.png"
        write_image(root / img, 100, 100, seed=int(stem))
        m = np.zeros((100, 100), np.uint8)
        for v, (x1, y1, x2, y2) in regions.items():
            m[y1:y2, x1:x2] = v
        (root / msk).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(root / msk), m)
        rows.append((img, label, msk))
        split.append(("pcbx", sp, "anomaly", img, msk))
    for i, sp in enumerate(["train", "train", "test"]):
        img = f"pcbx/Data/Images/Normal/{i:04d}.JPG"
        write_image(root / img, 100, 100, seed=100 + i)
        rows.append((img, "normal", ""))
        split.append(("pcbx", sp, "normal", img, ""))
    import csv

    with (root / "pcbx" / "image_anno.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    (root / "split_csv").mkdir()
    with (root / "split_csv" / "2cls_highshot.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(split)


def test_prepare_visa_end_to_end(tmp_path: Path):
    mod = _load_script()
    root = tmp_path / "visa"
    _mini_visa(root)
    res = mod.prepare(root, ["pcbx"], tmp_path / "lm", tmp_path / "test", normal_train=1, normal_test=5,
                      merge_px=2, min_side=4)
    assert res["names"] == ["bent", "dirt", "melt"]             # dirt never occurs alone: gets the free value 3
    assert res["train"]["images"] == 3 and res["test"]["images"] == 2
    data = json.loads((tmp_path / "lm" / "pcbx_a000.json").read_text(encoding="utf-8"))
    assert [(s["label"], s["points"]) for s in data["shapes"]] == [("bent", [[10, 10], [40, 30]])]
    assert json.loads((tmp_path / "lm" / "pcbx_n0000.json").read_text(encoding="utf-8"))["shapes"] == []
    images, gts = load_yolo_split(tmp_path / "test", "test")
    assert sorted(b.cls for b in gts["pcbx_a002"]) == [0, 1]
    assert (tmp_path / "test" / "VISA_ATTRIBUTION.txt").is_file()
    from evp.data.labelme import convert_dir

    st = convert_dir(tmp_path / "lm", tmp_path / "yolo", res["names"])
    assert (st.files, st.objects, st.unknown_labels) == (3, 2, {})

    single = mod.prepare(root, ["pcbx"], tmp_path / "lm1", tmp_path / "t1", single_class=True, normal_train=0,
                         normal_test=0, merge_px=2, min_side=4)
    assert single["names"] == ["defect"] and single["test"]["defect"] == 2


def test_prepare_visa_errors(tmp_path: Path):
    mod = _load_script()
    with pytest.raises(DataError, match="split_csv"):
        mod.prepare(tmp_path, ["pcbx"], tmp_path / "a", tmp_path / "b")
    (tmp_path / "split_csv").mkdir()
    (tmp_path / "split_csv" / "2cls_highshot.csv").write_text("object,split,label,image,mask\n", encoding="utf-8")
    with pytest.raises(DataError, match="image_anno.csv"):
        mod.prepare(tmp_path, ["pcbx"], tmp_path / "a", tmp_path / "b")
    assert mod.main(["--root", str(tmp_path / "nope"), "--labelme-out", "a", "--test-out", "b"]) == 2
