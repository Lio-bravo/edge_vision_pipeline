#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
import json
import time
from pathlib import Path

import pytest

from evp.data.labelme import convert_dir
from evp.errors import ConfigError, DataError, EvpError
from evp.webui import services as S
from tests.conftest import rect, write_image, write_labelme


def _cfg(tmp_project: Path) -> dict:
    return S.load_project(tmp_project / "project.yaml")


def _staged(tmp_project: Path, n: int = 2) -> dict:
    cfg = _cfg(tmp_project)
    for i in range(n):
        write_image(Path(cfg["paths"]["raw_images"]) / f"img{i}.jpg", 80, 60, seed=i)
    return cfg


# ---------------------------------------------------------------- project
def test_load_project_errors(tmp_path: Path):
    with pytest.raises(ConfigError, match="no project"):
        S.load_project("  ")
    with pytest.raises(ConfigError, match="not found"):
        S.load_project(tmp_path / "nope.yaml")
    (tmp_path / "p.yaml").write_text("classes: []\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="classes is empty"):
        S.load_project(tmp_path / "p.yaml")


@pytest.mark.parametrize("text,expected", [("a, b", ["a", "b"]), ("划痕，缺件\n短路", ["划痕", "缺件", "短路"]),
                                           (" x ;y; ", ["x", "y"])])
def test_parse_classes(text, expected):
    assert S.parse_classes(text) == expected


@pytest.mark.parametrize("text", ["", " , ", "a, a", "a:b"])
def test_parse_classes_rejects(text):
    with pytest.raises(ConfigError):
        S.parse_classes(text)


def test_create_project_keeps_template_and_sets_classes(tmp_path: Path):
    p = S.create_project(tmp_path / "proj", ["划痕", "missing"])
    cfg = S.load_project(p)
    assert cfg["classes"] == ["划痕", "missing"]
    assert Path(cfg["paths"]["raw_images"]).is_dir()
    assert "#" in p.read_text(encoding="utf-8")  # template comments survive
    with pytest.raises(ConfigError, match="already exists"):
        S.create_project(tmp_path / "proj", ["a"])
    assert S.load_project(S.create_project(tmp_path / "proj", ["b"], force=True))["classes"] == ["b"]


def test_project_status_counts(tmp_project: Path):
    cfg = _staged(tmp_project, 3)
    s = S.project_status(cfg)
    assert (s["raw_images"], s["to_review"], s["reviewed"], s["corpus_train"]) == (3, 0, 0, 0)
    assert "类别" in S.status_markdown(cfg)


# ---------------------------------------------------------------- review
def test_stage_for_manual_labeling_never_overwrites(tmp_project: Path):
    cfg = _staged(tmp_project, 2)
    lm = Path(cfg["paths"]["labelme_dir"])
    write_image(lm / "img0.jpg", 80, 60, seed=0)
    write_labelme(lm / "img0.json", "img0.jpg", 80, 60, [rect("a", 1, 1, 20, 20)])
    assert S.stage_for_manual_labeling(cfg["paths"]["raw_images"], lm) == 1
    assert json.loads((lm / "img0.json").read_text(encoding="utf-8"))["shapes"]  # kept
    assert json.loads((lm / "img1.json").read_text(encoding="utf-8"))["shapes"] == []
    assert S.review_items(lm) == ["img0", "img1"]


def test_stage_for_manual_labeling_needs_images(tmp_project: Path):
    cfg = _cfg(tmp_project)
    Path(cfg["paths"]["raw_images"]).mkdir(parents=True, exist_ok=True)
    with pytest.raises(DataError, match="no images"):
        S.stage_for_manual_labeling(cfg["paths"]["raw_images"], cfg["paths"]["labelme_dir"])


def test_annotation_roundtrip_and_ingest_compatible(tmp_project: Path):
    cfg = _staged(tmp_project, 1)
    lm = Path(cfg["paths"]["labelme_dir"])
    S.stage_for_manual_labeling(cfg["paths"]["raw_images"], lm)
    v = S.load_annotation(lm, "img0", cfg["classes"])
    assert v["boxes"] == [] and Path(v["image"]).is_file()
    v["boxes"] = [{"xmin": 10, "ymin": 5, "xmax": 50, "ymax": 40, "label": "b"},
                  {"xmin": -5, "ymin": 30, "xmax": 200, "ymax": 90, "label": "a"}]  # clipped to 80x60
    assert S.save_annotation(lm, "img0", v, cfg["classes"])["boxes"] == 2
    assert "img0" in S.load_reviewed(lm)
    back = S.load_annotation(lm, "img0", cfg["classes"])["boxes"]
    assert [(b["xmin"], b["ymin"], b["xmax"], b["ymax"], b["label"]) for b in back] == [
        (10, 5, 50, 40, "b"), (0, 30, 80, 60, "a")]
    stats = convert_dir(lm, tmp_project / "out", cfg["classes"])
    assert (stats.files, stats.objects, stats.unknown_labels) == (1, 2, {})
    line = (tmp_project / "out" / "labels" / "train" / "img0.txt").read_text().splitlines()[0]
    assert line.startswith("1 0.375000 0.375000 0.500000 0.583333")


def test_save_annotation_rejects_bad_boxes_without_writing(tmp_project: Path):
    cfg = _staged(tmp_project, 1)
    lm = Path(cfg["paths"]["labelme_dir"])
    S.stage_for_manual_labeling(cfg["paths"]["raw_images"], lm)
    before = (lm / "img0.json").read_text(encoding="utf-8")
    for boxes, msg in [([{"xmin": 1, "ymin": 1, "xmax": 9, "ymax": 9, "label": "zzz"}], "not in classes"),
                       ([{"xmin": 1, "ymin": 1, "xmax": 1.5, "ymax": 9, "label": "a"}], "smaller than 1 px"),
                       ([{"xmin": 100, "ymin": 1, "xmax": 120, "ymax": 9, "label": "a"}], "smaller than 1 px")]:
        with pytest.raises(DataError, match=msg):
            S.save_annotation(lm, "img0", {"image": "x", "boxes": boxes}, cfg["classes"])
    with pytest.raises(DataError, match="rotated"):
        S.save_annotation(lm, "img0", {"image": "x", "boxes": [], "orientation": 1}, cfg["classes"])
    with pytest.raises(DataError, match="nothing to save"):
        S.save_annotation(lm, "img0", None, cfg["classes"])
    with pytest.raises(DataError, match="not found"):
        S.save_annotation(lm, "missing", {"boxes": []}, cfg["classes"])
    assert (lm / "img0.json").read_text(encoding="utf-8") == before
    assert S.load_reviewed(lm) == set()


def test_polygons_shown_as_boxes_and_other_shapes_kept(tmp_project: Path):
    cfg = _cfg(tmp_project)
    lm = Path(cfg["paths"]["labelme_dir"])
    write_image(lm / "p.jpg", 100, 100)
    poly = {"label": "a", "points": [[10, 20], [40, 10], [30, 50]], "shape_type": "polygon"}
    point = {"label": "a", "points": [[5, 5]], "shape_type": "point"}
    write_labelme(lm / "p.json", "p.jpg", 100, 100, [poly, point])
    v = S.load_annotation(lm, "p", cfg["classes"])
    assert [(b["xmin"], b["ymin"], b["xmax"], b["ymax"]) for b in v["boxes"]] == [(10, 10, 40, 50)]
    S.save_annotation(lm, "p", v, cfg["classes"])
    shapes = json.loads((lm / "p.json").read_text(encoding="utf-8"))["shapes"]
    assert [s["shape_type"] for s in shapes] == ["rectangle", "point"]


def test_load_annotation_errors(tmp_project: Path):
    cfg = _cfg(tmp_project)
    lm = Path(cfg["paths"]["labelme_dir"])
    lm.mkdir(parents=True)
    (lm / "bad.json").write_text("{oops", encoding="utf-8")
    with pytest.raises(DataError, match="broken"):
        S.load_annotation(lm, "bad", cfg["classes"])
    write_labelme(lm / "noimg.json", "noimg.jpg", 10, 10, [])
    with pytest.raises(DataError, match="not found"):
        S.load_annotation(lm, "noimg", cfg["classes"])


def test_delete_review_item(tmp_project: Path):
    cfg = _staged(tmp_project, 2)
    lm = Path(cfg["paths"]["labelme_dir"])
    S.stage_for_manual_labeling(cfg["paths"]["raw_images"], lm)
    S.save_annotation(lm, "img1", {"boxes": []}, cfg["classes"])
    S.delete_review_item(lm, "img1")
    assert S.review_items(lm) == ["img0"] and not (lm / "img1.jpg").exists()
    assert S.load_reviewed(lm) == set()
    assert convert_dir(lm, lm.parent / "out", cfg["classes"]).missing_image == 0  # review file is not a json
    with pytest.raises(DataError):
        S.delete_review_item(lm, "img1")


def test_broken_review_file_is_ignored(tmp_project: Path):
    lm = Path(_cfg(tmp_project)["paths"]["labelme_dir"])
    lm.mkdir(parents=True)
    (lm / S.REVIEW_FILE).write_text("[not json", encoding="utf-8")
    assert S.load_reviewed(lm) == set()


# ---------------------------------------------------------------- jobs
def test_job_runner_captures_logs_and_result():
    from evp.utils import log

    r = S.JobRunner()

    def work():
        log.info("hello from job")
        return 42

    r.start("t", work)
    job = r.wait(5)
    assert job.result == 42 and job.error is None and not job.running
    assert "hello from job" in job.log_text()


def test_job_runner_reports_errors_and_blocks_concurrent_jobs():
    r = S.JobRunner()

    def boom():
        raise DataError("bad data", "fix it")

    r.start("a", boom)
    assert "bad data" in r.wait(5).error

    r.start("slow", lambda: time.sleep(0.3))
    with pytest.raises(EvpError, match="still running"):
        r.start("other", lambda: None)
    r.wait(5)
    r.start("other", lambda: 1)
    assert r.wait(5).result == 1

    r.start("crash", lambda: 1 / 0)
    assert "ZeroDivisionError" in r.wait(5).error


# ---------------------------------------------------------------- infer / bench
def _smoke_bundle() -> Path | None:
    root = Path(__file__).resolve().parents[1] / "scratch" / "e2e_smoke" / "workspace" / "deploy"
    b = S.list_bundles(root)
    return b[0] if b else None


def test_list_bundles_and_weights(tmp_project: Path):
    cfg = _cfg(tmp_project)
    assert S.list_bundles(cfg["paths"]["deploy_out"]) == []
    b = Path(cfg["paths"]["deploy_out"]) / "m"
    b.mkdir(parents=True)
    (b / "meta.json").write_text("{}", encoding="utf-8")
    (Path(cfg["paths"]["deploy_out"]) / "_work").mkdir()
    assert S.list_bundles(cfg["paths"]["deploy_out"]) == [b]
    w = Path(cfg["paths"]["runs"]) / "train" / "weights" / "best.pt"
    w.parent.mkdir(parents=True)
    w.write_bytes(b"x")
    assert S.list_weights(cfg)[0] == w
    assert S.read_report(b).startswith("_")


def test_infer_and_benchmark_errors(tmp_path: Path):
    import numpy as np

    with pytest.raises(DataError, match="not a deploy bundle"):
        S.get_pipeline(tmp_path, "fp32", 1)
    with pytest.raises(DataError, match="no image"):
        S.infer_image(tmp_path, None)
    with pytest.raises(DataError, match="at least one image"):
        S.benchmark(tmp_path, [])
    with pytest.raises(DataError, match="runs"):
        S.benchmark(tmp_path, [np.zeros((8, 8, 3), np.uint8)], runs=0)


@pytest.mark.skipif(_smoke_bundle() is None, reason="no deploy bundle under scratch/e2e_smoke (run the smoke test)")
def test_infer_and_benchmark_on_real_bundle():
    import numpy as np

    b = _smoke_bundle()
    img = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    vis, rows, ms = S.infer_image(b, img, "fp32", 1, conf=0.01)
    assert vis.shape == img.shape and ms > 0 and isinstance(rows, list)
    pipe = S.get_pipeline(b, "fp32", 1)
    assert pipe.post.conf != 0.01  # conf override is not sticky
    table = S.benchmark(b, [img], ("fp32", "int8", "fp16"), (1, 2), runs=3, warmup=1)
    assert {(r["precision"], r["threads"]) for r in table} <= {("fp32", 1), ("fp32", 2), ("int8", 1), ("int8", 2)}
    assert all(r["mean_ms"] > 0 and r["p90_ms"] >= r["p50_ms"] for r in table)
    assert any(f["file"] == "meta.json" for f in S.bundle_files(b))
    with pytest.raises(DataError, match="none of the precisions"):
        S.benchmark(b, [img], ("fp16",), runs=1)
