#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
import json
from pathlib import Path

import numpy as np
import pytest

from evp.config import load_config
from evp.data.collect import (
    Collector,
    capture_camera,
    collector_from_config,
    dhash,
    extract_video,
    hamming,
    import_folder,
    save_frames,
    sharpness,
)
from evp.errors import ConfigError, DataError
from tests.conftest import write_image


def _noise(seed: int, h: int = 96, w: int = 128) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_dhash_identical_and_different():
    a = _noise(1)
    assert hamming(dhash(a), dhash(a.copy())) == 0
    assert hamming(dhash(a), dhash(_noise(2))) > 10
    # a mild brightness shift keeps the hash close
    assert hamming(dhash(a), dhash(np.clip(a.astype(int) + 10, 0, 255).astype(np.uint8))) <= 6


def test_dhash_rejects_empty():
    with pytest.raises(DataError):
        dhash(np.zeros((0, 0, 3), np.uint8))


def test_sharpness_blur_is_lower():
    import cv2

    a = _noise(3, 200, 200)
    assert sharpness(cv2.GaussianBlur(a, (15, 15), 5)) < sharpness(a) / 10


def test_import_folder_copies_bytes_and_drops_exact_duplicates(tmp_path: Path):
    src = tmp_path / "src"
    write_image(src / "a.png", 64, 48, seed=1)
    write_image(src / "sub" / "b.png", 64, 48, seed=2)
    (src / "sub" / "a_copy.png").write_bytes((src / "a.png").read_bytes())
    (src / "notes.txt").write_text("x")
    col = Collector(tmp_path / "raw")
    st = import_folder(src, col)
    assert (st.seen, st.saved, st.duplicates) == (3, 2, 1)
    assert sorted(p.name for p in (tmp_path / "raw").iterdir()) == ["a.png", "b.png"]
    assert (tmp_path / "raw" / "a.png").read_bytes() == (src / "a.png").read_bytes()


def test_import_folder_non_recursive(tmp_path: Path):
    src = tmp_path / "src"
    write_image(src / "a.jpg", 32, 32, seed=1)
    write_image(src / "sub" / "b.jpg", 32, 32, seed=2)
    st = import_folder(src, Collector(tmp_path / "raw"), recursive=False)
    assert st.saved == 1


def test_import_folder_renames_stem_collisions_case_insensitive(tmp_path: Path):
    raw = tmp_path / "raw"
    write_image(raw / "IMG.jpg", 32, 32, seed=1)
    src = tmp_path / "src"
    write_image(src / "img.png", 32, 32, seed=2)
    write_image(src / "x" / "img.png", 32, 32, seed=3)
    st = import_folder(src, Collector(raw))
    assert st.saved == 2
    stems = sorted(p.stem.lower() for p in raw.iterdir())
    assert stems == ["img", "img_1", "img_2"]


def test_dedup_against_other_folders(tmp_path: Path):
    corpus = tmp_path / "corpus" / "images" / "train"
    write_image(corpus / "old.png", 40, 40, seed=7)
    src = tmp_path / "src"
    (src).mkdir()
    (src / "again.png").write_bytes((corpus / "old.png").read_bytes())
    write_image(src / "new.png", 40, 40, seed=8)
    st = import_folder(src, Collector(tmp_path / "raw", against=[tmp_path / "corpus" / "images"]))
    assert (st.saved, st.duplicates) == (1, 1)


def test_near_duplicate_gate_off_by_default_on_when_set(tmp_path: Path):
    a = _noise(5)
    b = np.clip(a.astype(int) + 3, 0, 255).astype(np.uint8)  # same scene, slightly different pixels
    st = save_frames([a, b], Collector(tmp_path / "off"))
    assert st.saved == 2
    st = save_frames([a, b], Collector(tmp_path / "on", dedup_hamming=6))
    assert (st.saved, st.duplicates) == (1, 1)


def test_sharpness_gate(tmp_path: Path):
    import cv2

    a = _noise(9, 200, 200)
    blurred = cv2.GaussianBlur(a, (21, 21), 8)
    col = Collector(tmp_path / "raw", min_sharpness=sharpness(blurred) * 2)
    st = save_frames([a, blurred], col)
    assert (st.saved, st.blurry) == (1, 1)


def test_unreadable_files_are_counted(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "broken.jpg").write_bytes(b"not an image")
    write_image(src / "ok.jpg", 32, 32)
    st = import_folder(src, Collector(tmp_path / "raw"))
    assert (st.saved, st.unreadable) == (1, 1)


@pytest.mark.parametrize("kwargs", [{"jpeg_quality": 0}, {"dedup_hamming": 65}, {"min_sharpness": -1}])
def test_collector_rejects_bad_settings(tmp_path: Path, kwargs):
    with pytest.raises(DataError):
        Collector(tmp_path / "raw", **kwargs)


def test_import_folder_errors(tmp_path: Path):
    col = Collector(tmp_path / "raw")
    with pytest.raises(DataError, match="not found"):
        import_folder(tmp_path / "missing", col)
    (tmp_path / "empty").mkdir()
    with pytest.raises(DataError, match="no images"):
        import_folder(tmp_path / "empty", col)
    with pytest.raises(DataError, match="destination"):
        import_folder(tmp_path / "raw", col)


def _write_video(path: Path, n: int, fps: float = 10.0, size=(64, 48)) -> Path:
    import cv2

    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert vw.isOpened()
    for i in range(n):
        vw.write(_noise(100 + i, size[1], size[0]))
    vw.release()
    return path


def test_extract_video_samples_by_time(tmp_path: Path):
    v = _write_video(tmp_path / "clip.avi", 30, fps=10.0)
    st = extract_video(v, Collector(tmp_path / "raw"), every_sec=1.0)
    assert st.saved == 3  # frames 0, 10, 20
    names = sorted(p.name for p in (tmp_path / "raw").iterdir())
    assert names == ["clip_f000000.jpg", "clip_f000010.jpg", "clip_f000020.jpg"]


def test_extract_video_max_frames_and_errors(tmp_path: Path):
    v = _write_video(tmp_path / "clip.avi", 30, fps=10.0)
    assert extract_video(v, Collector(tmp_path / "raw"), every_sec=0.1, max_frames=4).saved == 4
    with pytest.raises(DataError, match="every_sec"):
        extract_video(v, Collector(tmp_path / "r2"), every_sec=0)
    with pytest.raises(DataError, match="not found"):
        extract_video(tmp_path / "nope.mp4", Collector(tmp_path / "r3"))
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"garbage")
    with pytest.raises(DataError):
        extract_video(bad, Collector(tmp_path / "r4"))


class FakeCap:
    def __init__(self, frames, opened=True):
        self.frames, self.opened, self.released = list(frames), opened, False

    def isOpened(self):  # noqa: N802 (cv2 API)
        return self.opened

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


def test_capture_camera_counts_only_saved_frames(tmp_path: Path):
    a = _noise(1)
    frames = [a] * 3 + [a, a, _noise(2), _noise(3)]  # 3 warmup, then a duplicate in the middle
    cap = FakeCap(frames)
    st = capture_camera(Collector(tmp_path / "raw"), count=3, interval=0.5, warmup=3, open_fn=lambda s: cap,
                        sleep_fn=lambda s: None)
    assert (st.saved, st.duplicates) == (3, 1)
    assert cap.released


def test_capture_camera_failures(tmp_path: Path):
    with pytest.raises(DataError, match="cannot open camera"):
        capture_camera(Collector(tmp_path / "raw"), open_fn=lambda s: FakeCap([], opened=False))
    cap = FakeCap([])
    with pytest.raises(DataError, match="stopped delivering"):
        capture_camera(Collector(tmp_path / "raw"), count=2, warmup=0, open_fn=lambda s: cap, sleep_fn=lambda s: None)
    assert cap.released
    with pytest.raises(DataError, match="count"):
        capture_camera(Collector(tmp_path / "raw"), count=0, open_fn=lambda s: FakeCap([]))


def test_collector_from_config_uses_paths(tmp_project: Path):
    cfg = load_config(tmp_project / "project.yaml")
    col = collector_from_config(cfg, dedup_hamming=3)
    assert col.dst == Path(cfg["paths"]["raw_images"])
    assert col.dedup_hamming == 3


@pytest.mark.parametrize("bad", [{"dedup_hamming": -2}, {"jpeg_quality": 101}, {"video_every_sec": 0},
                                 {"camera": True}, {"camera_count": 0}])
def test_collect_config_validation(tmp_path: Path, bad):
    import yaml

    (tmp_path / "p.yaml").write_text(yaml.safe_dump({"classes": ["a"], "collect": bad}), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(tmp_path / "p.yaml")


def test_cli_collect_folder(tmp_project: Path, capsys):
    from evp.cli import main

    src = tmp_project / "incoming"
    write_image(src / "a.jpg", 32, 32, seed=1)
    write_image(src / "b.jpg", 32, 32, seed=2)
    assert main(["collect", "-c", str(tmp_project / "project.yaml"), "--folder", str(src)]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["saved"] == 2
    # second run: everything is already there
    assert main(["collect", "-c", str(tmp_project / "project.yaml"), "--folder", str(src)]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (out["saved"], out["duplicates"]) == (0, 2)
