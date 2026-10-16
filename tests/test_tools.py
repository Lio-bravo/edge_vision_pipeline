#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
import subprocess
import sys
from pathlib import Path

import pytest

from evp.deploy import tools as T
from evp.errors import ToolNotFoundError


@pytest.mark.parametrize("src, want", [
    (r"E:\data\x\m.param", "/mnt/e/data/x/m.param"),
    (r"c:\a b\c", "/mnt/c/a b/c"),
    ("E:/data/y", "/mnt/e/data/y"),
    ("relative/dir", "relative/dir"),
    (r"rel\dir", "rel/dir"),
])
def test_win_to_wsl(src, want):
    assert T.win_to_wsl(src) == want


def test_tool_run_argv(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd, cwd=None, timeout=None, check=True):
        seen["cmd"], seen["cwd"] = list(cmd), cwd
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(T, "run_cmd", fake_run)
    wsl = T.Tool("ncnn2int8", "/home/u/bin/ncnn2int8", "Ubuntu")
    wsl.run(["a", "$HOME"], cwd=Path(r"E:\w"))
    assert seen["cmd"][:5] == ["wsl", "-d", "Ubuntu", "--cd", "/mnt/e/w"]
    assert seen["cmd"][5:] == ["--exec", "/home/u/bin/ncnn2int8", "a", "$HOME"]
    assert seen["cwd"] is None
    native = T.Tool("pnnx", "/usr/bin/pnnx")
    native.run(["x"], cwd=tmp_path)
    assert seen["cmd"] == ["/usr/bin/pnnx", "x"] and seen["cwd"] == tmp_path
    assert native.path_arg(tmp_path) == str(tmp_path.resolve())
    if T.IS_WINDOWS:
        assert wsl.path_arg(tmp_path).startswith("/mnt/")


def test_healthy_detects_crash_codes(monkeypatch):
    def fake(rc):
        return lambda *a, **k: subprocess.CompletedProcess(a, rc, b"", b"")

    for rc, ok in [(0, True), (1, True), (255, True), (-11, False), (139, False), (134, False), (0xC0000005, False)]:
        monkeypatch.setattr(T.subprocess, "run", fake(rc))
        assert T._healthy(["x"]) is ok, rc

    def boom(*a, **k):
        raise OSError("nope")

    monkeypatch.setattr(T.subprocess, "run", boom)
    assert T._healthy(["x"]) is False


def test_healthy_real_process():
    assert T._healthy([sys.executable, "-c", "import sys; sys.exit(2)"])


def test_runner_prefers_tools_dir(tmp_path, monkeypatch):
    exe = tmp_path / ("ncnn2int8" + (".exe" if T.IS_WINDOWS else ""))
    exe.write_bytes(b"")
    monkeypatch.setattr(T, "_healthy", lambda cmd: True)
    r = T.ToolRunner(tools_dir=tmp_path, allow_wsl=False)
    t = r.get("ncnn2int8")
    assert Path(t.exe) == exe and not t.via_wsl
    assert r.get("ncnn2int8") is t   # cached


def test_runner_missing_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("EVP_TOOLS_HOME", str(tmp_path / "none"))
    monkeypatch.delenv("EVP_NCNN_TOOLS_DIR", raising=False)
    monkeypatch.setattr(T.shutil, "which", lambda n: None)
    r = T.ToolRunner(tools_dir=tmp_path, allow_wsl=False)
    with pytest.raises(ToolNotFoundError, match="ncnnoptimize not available"):
        r.get("ncnnoptimize")
    assert r.describe()["ncnnoptimize"].startswith("MISSING")


def test_runner_skips_crashing_native(tmp_path, monkeypatch):
    exe = tmp_path / ("ncnn2table" + (".exe" if T.IS_WINDOWS else ""))
    exe.write_bytes(b"")
    monkeypatch.setattr(T, "_healthy", lambda cmd: False)
    monkeypatch.setenv("EVP_TOOLS_HOME", str(tmp_path / "none"))
    monkeypatch.setattr(T.shutil, "which", lambda n: None)
    r = T.ToolRunner(tools_dir=tmp_path, allow_wsl=False)
    with pytest.raises(ToolNotFoundError):
        r.get("ncnn2table")
