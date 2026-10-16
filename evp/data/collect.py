#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Image acquisition into paths.raw_images: folder import, video frame extraction, camera capture.

Every source goes through the same gate (`Collector.offer`):
  - unreadable / empty frames are counted and skipped,
  - optional sharpness gate (variance of Laplacian; 0 disables),
  - exact-duplicate gate (always on): identical decoded pixels are dropped,
  - optional near-duplicate gate: 64-bit dHash compared by Hamming distance (`dedup_hamming` >= 0; -1 = off).
    Off by default: on an inspection line good and defective parts look alike and would be dropped. Turn it on
    (3-6) for video / camera sampling of a changing scene.
  Both gates compare against the destination and any `against` folders (e.g. the labelme staging dir and the
  corpus), so an image already labeled is not collected again.
Saved names are unique case-insensitively by stem, which prelabel / ingest require.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from ..errors import DataError
from ..utils import IMAGE_EXTS, ensure_dir, list_images, log

_SAFE = re.compile(r"[^0-9A-Za-z_\-]+")


@dataclass
class CollectStats:
    seen: int = 0
    saved: int = 0
    duplicates: int = 0
    blurry: int = 0
    unreadable: int = 0
    files: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"seen": self.seen, "saved": self.saved, "duplicates": self.duplicates, "blurry": self.blurry,
                "unreadable": self.unreadable}


def dhash(img_bgr: np.ndarray, size: int = 8) -> int:
    """Difference hash: grayscale -> (size+1)x size -> compare horizontal neighbours -> size*size bits."""
    import cv2

    if img_bgr is None or img_bgr.size == 0:
        raise DataError("dhash: empty image")
    gray = img_bgr if img_bgr.ndim == 2 else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def sharpness(img_bgr: np.ndarray) -> float:
    """Variance of the Laplacian on a <=640 px grayscale copy (higher = sharper; blur gives small values)."""
    import cv2

    gray = img_bgr if img_bgr.ndim == 2 else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    s = 640.0 / max(h, w)
    if s < 1.0:
        gray = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _read(path: Path) -> np.ndarray | None:
    import cv2

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def _safe_stem(name: str, fallback: str = "img") -> str:
    s = _SAFE.sub("_", name).strip("_")
    return s[:80] or fallback


class Collector:
    """Shared save / dedup / sharpness gate for all sources (see module doc)."""

    def __init__(self, dst: Path, against: Sequence[Path] = (), dedup_hamming: int = -1, min_sharpness: float = 0.0,
                 jpeg_quality: int = 95):
        if not 1 <= int(jpeg_quality) <= 100:
            raise DataError(f"jpeg_quality must be in [1,100], got {jpeg_quality}")
        if dedup_hamming > 64:
            raise DataError(f"dedup_hamming must be <= 64, got {dedup_hamming}")
        if min_sharpness < 0:
            raise DataError(f"min_sharpness must be >= 0, got {min_sharpness}")
        self.dst = ensure_dir(Path(dst))
        self.dedup_hamming = int(dedup_hamming)
        self.min_sharpness = float(min_sharpness)
        self.jpeg_quality = int(jpeg_quality)
        self.stats = CollectStats()
        self._stems: set[str] = set()
        self._hashes: list[int] = []
        self._exact: set[bytes] = set()
        for d in [self.dst, *against]:
            d = Path(d)
            if not d.is_dir():
                continue
            for p in list_images(d, recursive=d != self.dst):
                if d == self.dst:
                    self._stems.add(p.stem.lower())
                img = _read(p)
                if img is not None:
                    self._remember(img)
        log.info("collect: %d reference images, %d existing names in %s", len(self._exact), len(self._stems),
                 self.dst)

    @staticmethod
    def _digest(img: np.ndarray) -> bytes:
        import hashlib

        return hashlib.blake2b(np.ascontiguousarray(img).tobytes() + repr(img.shape).encode(), digest_size=16).digest()

    def _remember(self, img: np.ndarray) -> None:
        self._exact.add(self._digest(img))
        if self.dedup_hamming >= 0:
            self._hashes.append(dhash(img))

    def is_duplicate(self, img: np.ndarray) -> bool:
        if self._digest(img) in self._exact:
            return True
        if self.dedup_hamming < 0:
            return False
        h = dhash(img)
        return any(hamming(h, o) <= self.dedup_hamming for o in self._hashes)

    def unique_stem(self, stem: str) -> str:
        base = _safe_stem(stem)
        s, n = base, 0
        while s.lower() in self._stems:
            n += 1
            s = f"{base}_{n}"
        return s

    def offer(self, img: np.ndarray | None, stem: str, src_file: Path | None = None) -> Path | None:
        """Gate one frame; returns the saved path or None (reason counted in stats). With `src_file`
        (folder import) the original bytes are copied when the extension is supported, else re-encoded."""
        import cv2

        self.stats.seen += 1
        if img is None or img.size == 0:
            self.stats.unreadable += 1
            return None
        if self.min_sharpness > 0 and sharpness(img) < self.min_sharpness:
            self.stats.blurry += 1
            return None
        if self.is_duplicate(img):
            self.stats.duplicates += 1
            return None
        s = self.unique_stem(stem)
        if src_file is not None and src_file.suffix.lower() in IMAGE_EXTS:
            out = self.dst / f"{s}{src_file.suffix.lower()}"
            out.write_bytes(Path(src_file).read_bytes())
        else:
            out = self.dst / f"{s}.jpg"
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                raise DataError(f"cannot encode frame {stem} as jpg")
            buf.tofile(str(out))
        self._stems.add(s.lower())
        self._remember(img)
        self.stats.saved += 1
        self.stats.files.append(out.name)
        return out


def collector_from_config(cfg: dict, dedup_hamming: int | None = None,
                          min_sharpness: float | None = None) -> Collector:
    """Collector writing into paths.raw_images; dedups against the labelme staging dir and (optionally) the
    corpus images. Explicit arguments override collect.* config values."""
    p, c = cfg["paths"], cfg["collect"]
    against = [Path(p["labelme_dir"])]
    if c.get("dedup_against_corpus", True):
        against.append(Path(p["corpus_root"]) / "images")
    return Collector(Path(p["raw_images"]), against,
                     dedup_hamming=int(c["dedup_hamming"] if dedup_hamming is None else dedup_hamming),
                     min_sharpness=float(c["min_sharpness"] if min_sharpness is None else min_sharpness),
                     jpeg_quality=int(c["jpeg_quality"]))


def import_folder(src: Path, collector: Collector, recursive: bool = True) -> CollectStats:
    """Copy images from `src` (recursively by default) through the gate. Original file bytes are kept."""
    src = Path(src)
    if not src.is_dir():
        raise DataError(f"source folder not found: {src}")
    if src.resolve() == collector.dst.resolve():
        raise DataError("source folder is the destination folder", "import from a different folder")
    files = list_images(src, recursive=recursive)
    if not files:
        raise DataError(f"no images ({', '.join(IMAGE_EXTS)}) under {src}")
    for p in files:
        collector.offer(_read(p), p.stem, src_file=p)
    log.info("import %s: %s", src, collector.stats.as_dict())
    return collector.stats


def extract_video(video: Path, collector: Collector, every_sec: float = 1.0, max_frames: int = 0,
                  prefix: str | None = None) -> CollectStats:
    """Sample one frame every `every_sec` seconds of video time (falls back to every 25th frame when the
    container reports no fps). `max_frames` > 0 caps the number of SAVED frames."""
    import cv2

    video = Path(video)
    if not video.is_file():
        raise DataError(f"video not found: {video}")
    if every_sec <= 0:
        raise DataError(f"every_sec must be > 0, got {every_sec}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise DataError(f"cannot open video: {video}", "is it a valid video file (mp4/avi/mkv)?")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    step = max(1, round(fps * every_sec)) if fps > 0 else 25
    stem = _safe_stem(prefix or video.stem, "video")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                collector.offer(frame, f"{stem}_f{idx:06d}")
                if max_frames and collector.stats.saved >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
    if idx == 0:
        raise DataError(f"no frames decoded from {video}", "the codec may be unsupported by OpenCV")
    log.info("video %s: %d frames read, step %d, %s", video.name, idx, step, collector.stats.as_dict())
    return collector.stats


def capture_camera(collector: Collector, source: int | str = 0, count: int = 10, interval: float = 1.0,
                   warmup: int = 5, timeout: float = 0.0, open_fn: Callable | None = None,
                   sleep_fn: Callable[[float], None] = time.sleep) -> CollectStats:
    """Grab frames from a camera index or stream URL until `count` are SAVED (duplicates do not count), one
    attempt every `interval` seconds. `timeout` > 0 stops early. `open_fn` is injectable for tests."""
    import cv2

    if count < 1:
        raise DataError(f"count must be >= 1, got {count}")
    if interval < 0:
        raise DataError(f"interval must be >= 0, got {interval}")
    cap = (open_fn or cv2.VideoCapture)(source)
    if not cap.isOpened():
        raise DataError(f"cannot open camera {source!r}", "check the index (0, 1, ...) or the stream URL, and that "
                        "no other program holds the camera")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    t0 = time.monotonic()
    misses = 0
    try:
        for _ in range(max(0, warmup)):
            cap.read()
        n = 0
        while collector.stats.saved < count:
            if timeout > 0 and time.monotonic() - t0 > timeout:
                log.warning("camera: timeout after %.0fs", timeout)
                break
            ok, frame = cap.read()
            if not ok:
                misses += 1
                if misses >= 10:
                    raise DataError(f"camera {source!r} stopped delivering frames")
                continue
            misses = 0
            collector.offer(frame, f"cam_{stamp}_{n:04d}")
            n += 1
            if interval and collector.stats.saved < count:
                sleep_fn(interval)
    finally:
        cap.release()
    log.info("camera %r: %s", source, collector.stats.as_dict())
    return collector.stats


def save_frames(frames: Iterable[np.ndarray], collector: Collector, prefix: str = "snap") -> CollectStats:
    """Save in-memory BGR frames (e.g. web UI webcam snapshots) through the gate."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, f in enumerate(frames):
        collector.offer(f, f"{_safe_stem(prefix, 'snap')}_{stamp}_{i:04d}")
    return collector.stats
