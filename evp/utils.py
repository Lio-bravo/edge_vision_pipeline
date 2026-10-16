#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Shared helpers: logging, filesystem, json, subprocess."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .errors import ConfigError, DataError, EvpError, ToolNotFoundError

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

_LOG_FORMAT = "[%(levelname)s] %(message)s"


def get_logger(name: str = "evp") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger("evp").handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root = logging.getLogger("evp")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    return logger


log = get_logger()


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(directory: Path, recursive: bool = False) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise DataError(f"image directory not found: {directory}")
    it: Iterable[Path] = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def clear_dir_contents(directory: Path) -> int:
    """Delete everything inside `directory`, keep the directory itself."""
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    removed = 0
    for child in directory.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
        removed += 1
    return removed


def write_json(path: Path, data: Any) -> Path:
    """Atomic json write (tmp file + replace) so a crash never leaves half a file."""
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def read_json(path: Path) -> Any:
    path = Path(path)
    if not path.is_file():
        raise EvpError(f"json not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise EvpError(f"invalid json {path}: {exc}") from exc


def find_executable(name: str, extra_dirs: Sequence[Path] = ()) -> Path:
    """Look up a tool in extra dirs, next to the python interpreter, then PATH."""
    candidates: list[Path] = []
    exe_names = [name] if os.name != "nt" else [name + ".exe", name]
    for d in list(extra_dirs) + [Path(sys.executable).parent, Path(sys.executable).parent / "Scripts"]:
        for n in exe_names:
            candidates.append(Path(d) / n)
    for c in candidates:
        if c.is_file():
            return c
    found = shutil.which(name)
    if found:
        return Path(found)
    raise ToolNotFoundError(
        f"executable not found: {name}",
        hint="install it or set its directory via env EVP_TOOLS_DIR / config deploy.tools_dir",
    )


def run_cmd(cmd: Sequence[str | os.PathLike], cwd: Path | None = None, timeout: float | None = 1800,
            check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, capture output, raise EvpError with the tail of stderr on failure."""
    cmd_s = [str(c) for c in cmd]
    log.debug("run: %s", " ".join(cmd_s))
    try:
        proc = subprocess.run(cmd_s, cwd=str(cwd) if cwd else None, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except FileNotFoundError as exc:
        raise ToolNotFoundError(f"cannot execute {cmd_s[0]}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise EvpError(f"command timed out after {timeout}s: {' '.join(cmd_s)}") from exc
    if check and proc.returncode != 0:
        tail = "\n".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-30:])
        raise EvpError(f"command failed (exit {proc.returncode}): {' '.join(cmd_s)}\n{tail}")
    return proc


def resolve_device(device: Any, key: str = "device", cuda_available: bool | None = None) -> str:
    """`auto` / empty -> "0" when CUDA is available, else "cpu". An explicit GPU without CUDA is an error
    (Ultralytics would fail later with a less helpful message); `key` names the config entry in that error."""
    d = "" if device is None else str(device).strip().lower()
    if cuda_available is None:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except ImportError:
            cuda_available = False
    if d in ("", "auto"):
        return "0" if cuda_available else "cpu"
    if d in ("cpu", "mps"):
        return d
    if not cuda_available:
        raise ConfigError(f"{key}={device!r} requests a GPU but CUDA is not available in this environment",
                          f"set {key}: auto (or cpu), or install a CUDA build of torch")
    return str(device).strip()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
