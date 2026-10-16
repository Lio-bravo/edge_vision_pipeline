#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""LabelMe -> YOLO conversion, split, merge, staging cleanup, CLI ingest."""

import json
from pathlib import Path

import pytest

from evp.data.labelme import build_labelme_payload, convert_dir, shape_to_yolo
from evp.data.merge import check_label_ids, merge_batch, unique_pair_dest
from evp.data.split import split_batch
from evp.errors import DataError
from tests.conftest import rect, write_image, write_labelme


# ---------------------------------------------------------------- shape_to_yolo
def test_shape_to_yolo_basic():
    cx, cy, w, h = shape_to_yolo([[10, 20], [30, 60]], 100, 100)
    assert (cx, cy, w, h) == pytest.approx((0.2, 0.4, 0.2, 0.4))


def test_shape_to_yolo_reversed_corners():
    assert shape_to_yolo([[30, 60], [10, 20]], 100, 100) == pytest.approx((0.2, 0.4, 0.2, 0.4))


def test_shape_to_yolo_clips_before_computing_center():
    # box half outside on the left: the old SOP computed center from unclipped corners
    cx, cy, w, h = shape_to_yolo([[-50, 0], [50, 100]], 100, 100)
    assert (cx, w) == pytest.approx((0.25, 0.5))
    assert 0 <= cx - w / 2 and cx + w / 2 <= 1


def test_shape_to_yolo_polygon_uses_bbox():
    assert shape_to_yolo([[10, 10], [50, 10], [30, 40]], 100, 100) == pytest.approx((0.3, 0.25, 0.4, 0.3))


@pytest.mark.parametrize("pts", [[[10, 10], [10.5, 50]], [[-20, -20], [-5, -5]], [[200, 200], [300, 300]], [[5, 5]]])
def test_shape_to_yolo_degenerate(pts):
    assert shape_to_yolo(pts, 100, 100) is None


# ---------------------------------------------------------------- convert_dir
def _labelme_dir(tmp_path: Path) -> Path:
    d = tmp_path / "labelme"
    write_image(d / "img1.jpg", 200, 100)
    write_labelme(d / "img1.json", "img1.jpg", 200, 100, [rect("a", 0, 0, 100, 50), rect("b", 100, 50, 200, 100),
                                                          rect("zzz", 1, 1, 50, 50)])
    write_image(d / "neg.png", 64, 64, seed=1)
    write_labelme(d / "neg.json", "neg.png", 64, 64, [])
    write_labelme(d / "orphan.json", "missing.jpg", 10, 10, [rect("a", 0, 0, 5, 5)])
    return d


def test_convert_dir(tmp_path: Path):
    d = _labelme_dir(tmp_path)
    stats = convert_dir(d, tmp_path / "batch", ["a", "b"], "train")
    assert stats.files == 2 and stats.objects == 2 and stats.empty == 1 and stats.missing_image == 1
    assert "zzz" in stats.unknown_labels
    lines = (tmp_path / "batch/labels/train/img1.txt").read_text().split("\n")
    assert lines[0].startswith("0 0.250000 0.250000 0.500000 0.500000")
    assert lines[1].startswith("1 0.750000 0.750000")
    assert (tmp_path / "batch/labels/train/neg.txt").read_text() == ""
    assert (tmp_path / "batch/images/train/neg.png").is_file()


def test_convert_dir_uses_real_image_size(tmp_path: Path):
    d = tmp_path / "lm"
    write_image(d / "x.jpg", 200, 100)
    write_labelme(d / "x.json", "x.jpg", 400, 200, [rect("a", 0, 0, 100, 50)])  # stale size in json
    convert_dir(d, tmp_path / "b", ["a"])
    assert (tmp_path / "b/labels/train/x.txt").read_text().startswith("0 0.250000 0.250000 0.500000 0.500000")


def test_convert_dir_errors(tmp_path: Path):
    with pytest.raises(DataError, match="not found"):
        convert_dir(tmp_path / "nope", tmp_path / "b", ["a"])
    (tmp_path / "empty").mkdir()
    with pytest.raises(DataError, match="no json"):
        convert_dir(tmp_path / "empty", tmp_path / "b", ["a"])
    with pytest.raises(DataError, match="class names"):
        convert_dir(tmp_path / "empty", tmp_path / "b", [])
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "x.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(DataError, match="broken LabelMe json"):
        convert_dir(bad, tmp_path / "b", ["a"])


def test_build_labelme_payload_roundtrip(tmp_path: Path):
    img = write_image(tmp_path / "p.jpg", 100, 80)
    payload = build_labelme_payload(img, 100, 80, [(10, 10, 50, 40, 1, 0.9), (0, 0, 5, 5, 7, 0.5)], ["a", "b"])
    assert len(payload["shapes"]) == 1 and payload["shapes"][0]["label"] == "b"
    assert payload["imageData"] is None
    (tmp_path / "p.json").write_text(json.dumps(payload), encoding="utf-8")
    stats = convert_dir(tmp_path, tmp_path / "out", ["a", "b"])
    assert stats.objects == 1
    embedded = build_labelme_payload(img, 100, 80, [], ["a"], embed_image_data=True)
    assert isinstance(embedded["imageData"], str) and len(embedded["imageData"]) > 10


# ---------------------------------------------------------------- split / merge
def _batch(root: Path, n: int, labeled: int | None = None) -> Path:
    labeled = n if labeled is None else labeled
    for i in range(n):
        write_image(root / "images/train" / f"im{i}.jpg", 32, 32, seed=i)
        if i < labeled:
            (root / "labels/train").mkdir(parents=True, exist_ok=True)
            (root / "labels/train" / f"im{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    return root


def test_split_moves_ratio_of_labeled_pairs(tmp_path: Path):
    b = _batch(tmp_path / "b", 10, labeled=8)
    n = split_batch(b, 0.25, seed=0)
    assert n == 2
    assert len(list((b / "images/val").iterdir())) == 2
    assert len(list((b / "labels/val").iterdir())) == 2
    assert len(list((b / "images/train").iterdir())) == 8  # 6 labeled + 2 unlabeled stay


def test_split_keeps_at_least_one_train(tmp_path: Path):
    b = _batch(tmp_path / "b", 2)
    assert split_batch(b, 0.9, seed=0) == 1


def test_split_deterministic(tmp_path: Path):
    a = _batch(tmp_path / "a", 12)
    b = _batch(tmp_path / "b", 12)
    split_batch(a, 0.25, seed=5)
    split_batch(b, 0.25, seed=5)
    assert sorted(p.name for p in (a / "images/val").iterdir()) == sorted(p.name for p in (b / "images/val").iterdir())


def test_unique_pair_dest(tmp_path: Path):
    (tmp_path / "i").mkdir()
    (tmp_path / "l").mkdir()
    (tmp_path / "i/x.png").write_bytes(b"1")  # same stem, other extension still collides
    img, lbl = unique_pair_dest(tmp_path / "i", tmp_path / "l", "x.jpg")
    assert img.name == "x__b1.jpg" and lbl.name == "x__b1.txt"


def test_check_label_ids(tmp_path: Path):
    f = tmp_path / "l.txt"
    f.write_text("0 0.5 0.5 0.1 0.1\n1 0.5 0.5 0.1 0.1\n")
    check_label_ids(f, 2)
    f.write_text("2 0.5 0.5 0.1 0.1\n")
    with pytest.raises(DataError):
        check_label_ids(f, 2)
    f.write_text("0 0.5 0.5\n")
    with pytest.raises(DataError):
        check_label_ids(f, 2)


def test_merge_batch_and_collisions(tmp_path: Path):
    b = _batch(tmp_path / "b", 4)
    split_batch(b, 0.25, seed=0)
    corpus = tmp_path / "corpus"
    r1 = merge_batch(b, corpus, ["a", "b"])
    assert r1["after"] == {"train": 3, "val": 1}
    assert (corpus / "data.yaml").is_file() and Path(r1["log"]).is_file()
    b2 = _batch(tmp_path / "b2", 4)  # same file names again
    split_batch(b2, 0.25, seed=0)
    r2 = merge_batch(b2, corpus, ["a", "b"])
    assert r2["after"] == {"train": 6, "val": 2}
    assert any("__b" in p.name for p in (corpus / "images/train").iterdir())


def test_merge_aborts_before_moving_on_bad_label(tmp_path: Path):
    b = _batch(tmp_path / "b", 3)
    (b / "labels/train/im2.txt").write_text("5 0.5 0.5 0.1 0.1\n")
    with pytest.raises(DataError):
        merge_batch(b, tmp_path / "corpus", ["a", "b"])
    assert len(list((b / "images/train").iterdir())) == 3  # nothing moved
    assert not (tmp_path / "corpus/images/train").exists() or not any((tmp_path / "corpus/images/train").iterdir())


def test_merge_empty_batch(tmp_path: Path):
    (tmp_path / "b/images/train").mkdir(parents=True)
    with pytest.raises(DataError, match="empty"):
        merge_batch(tmp_path / "b", tmp_path / "c", ["a"])


# ---------------------------------------------------------------- CLI ingest end-to-end
def test_cli_ingest_end_to_end(tmp_project: Path):
    from evp.cli import main

    lm = tmp_project / "ws/labelme"
    for i in range(8):
        write_image(lm / f"f{i}.jpg", 120, 90, seed=i)
        write_labelme(lm / f"f{i}.json", f"f{i}.jpg", 120, 90, [rect("a" if i % 2 else "b", 10, 10, 60, 50)])
    (tmp_project / "ws/raw").mkdir(parents=True)
    write_image(tmp_project / "ws/raw/new.jpg", 10, 10)
    rc = main(["ingest", "-c", str(tmp_project / "project.yaml")])
    assert rc == 0
    corpus = tmp_project / "ws/corpus"
    n_train = len(list((corpus / "images/train").iterdir()))
    n_val = len(list((corpus / "images/val").iterdir()))
    assert (n_train, n_val) == (6, 2)
    assert not any(lm.iterdir())                                  # staging cleared
    assert not any((tmp_project / "ws/raw").iterdir())
    data = (corpus / "data.yaml").read_text(encoding="utf-8")
    assert "nc: 2" in data


def test_cli_ingest_without_json_fails_cleanly(tmp_project: Path):
    from evp.cli import main

    (tmp_project / "ws/labelme").mkdir(parents=True)
    assert main(["ingest", "-c", str(tmp_project / "project.yaml")]) == 2


def test_cli_ingest_unknown_label_keeps_staging(tmp_project: Path):
    from evp.cli import main

    lm = tmp_project / "ws/labelme"
    for i in range(3):
        write_image(lm / f"f{i}.jpg", 50, 50, seed=i)
        write_labelme(lm / f"f{i}.json", f"f{i}.jpg", 50, 50, [rect("typo" if i == 1 else "a", 5, 5, 30, 30)])
    assert main(["ingest", "-c", str(tmp_project / "project.yaml")]) == 2
    assert len(list(lm.glob("*.json"))) == 3                    # reviewed work untouched
    assert not (tmp_project / "ws/corpus/data.yaml").exists()


def test_convert_dir_rejects_two_json_for_one_image(tmp_path: Path):
    write_image(tmp_path / "lm/x.jpg", 40, 40)
    write_labelme(tmp_path / "lm/a.json", "x.jpg", 40, 40, [rect("a", 1, 1, 20, 20)])
    write_labelme(tmp_path / "lm/b.json", "x.jpg", 40, 40, [rect("b", 1, 1, 20, 20)])
    with pytest.raises(DataError, match="same image name"):
        convert_dir(tmp_path / "lm", tmp_path / "out", ["a", "b"])
    assert not (tmp_path / "out/labels/train/x.txt").exists()


def test_convert_dir_resolves_relative_image_path(tmp_path: Path):
    write_image(tmp_path / "lm/sub/y.jpg", 40, 30)
    write_image(tmp_path / "lm/y.jpg", 80, 60, seed=3)           # same name, wrong image
    write_labelme(tmp_path / "lm/y.json", "sub/y.jpg", 40, 30, [rect("a", 0, 0, 20, 15)])
    convert_dir(tmp_path / "lm", tmp_path / "out", ["a"])
    line = (tmp_path / "out/labels/train/y.txt").read_text(encoding="utf-8").split()
    assert line[1:] == ["0.250000", "0.250000", "0.500000", "0.500000"]


def test_image_size_honours_exif_orientation(tmp_path: Path):
    from PIL import Image

    from evp.data.labelme import EXIF_ORIENTATION, image_size

    p = tmp_path / "rot.jpg"
    exif = Image.Exif()
    exif[EXIF_ORIENTATION] = 6                                     # rotate 90 CW on display
    Image.new("RGB", (40, 20)).save(p, exif=exif)
    assert image_size(p) == (20, 40)
    Image.new("RGB", (40, 20)).save(tmp_path / "plain.jpg")
    assert image_size(tmp_path / "plain.jpg") == (40, 20)


def test_map_class_ids_by_name():
    from evp.data.prelabel import map_class_ids

    assert map_class_ids(["a", "b"], ["a", "b"]) == {0: 0, 1: 1}
    assert map_class_ids(["cat", "dog", "car"], ["dog", "cat"]) == {0: 1, 1: 0}
    with pytest.raises(DataError, match="share no name"):
        map_class_ids(["person", "car"], ["tian", "chuan"])


def test_prelabel_rejects_duplicate_stems(tmp_path: Path):
    from evp.data.prelabel import prelabel

    write_image(tmp_path / "raw/a.jpg", 10, 10)
    write_image(tmp_path / "raw/a.png", 10, 10)
    with pytest.raises(DataError, match="share a file name stem"):
        prelabel(tmp_path / "raw", tmp_path / "lm", ["x"], model=tmp_path / "none.pt")


def test_train_check_corpus_val_hint(tmp_path: Path):
    from evp.train.trainer import check_corpus

    (tmp_path / "images/train").mkdir(parents=True)
    write_image(tmp_path / "images/train/a.jpg", 10, 10)
    (tmp_path / "images/val").mkdir(parents=True)
    (tmp_path / "data.yaml").write_text("nc: 1\n", encoding="utf-8")
    with pytest.raises(DataError) as ei:
        check_corpus(tmp_path)
    assert "val" in str(ei.value) and ">= 2 labeled images" in str(ei.value)


def test_cli_review_without_labelme(tmp_project: Path, monkeypatch):
    import evp.cli as cli

    monkeypatch.setattr(cli.shutil, "which", lambda n: None)
    assert cli.main(["review", "-c", str(tmp_project / "project.yaml")]) == 2
