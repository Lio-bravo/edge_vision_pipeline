#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Locate and run native converter tools (pnnx, ncnn2table, ncnn2int8, ncnnoptimize).

Resolution order per tool:
  1. explicit path from env (EVP_PNNX, EVP_NCNN_TOOLS_DIR) or config deploy.tools_dir
  2. native binary (pnnx pip package / PATH), accepted only if a health check passes
  3. on Windows: the same tools inside WSL ($EVP_TOOLS_HOME, default ~/evp_deps, built by
     scripts/setup_tools_linux.sh). Paths are translated (E:\\a -> /mnt/e/a).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from ..errors import ToolNotFoundError
from ..utils import log, run_cmd

IS_WINDOWS = os.name == "nt"
NCNN_TOOLS = ("ncnn2table", "ncnn2int8", "ncnnoptimize")


def win_to_wsl(path: str | Path) -> str:
    """E:\\Project\\x -> /mnt/e/Project/x. Non-drive paths are returned with forward slashes."""
    p = PureWindowsPath(str(path))
    if p.drive and re.fullmatch(r"[A-Za-z]:", p.drive):
        rest = "/".join(p.parts[1:])
        return f"/mnt/{p.drive[0].lower()}/{rest}"
    return str(path).replace("\\", "/")


@dataclass
class Tool:
    """A resolved executable. With `wsl_distro` set it runs inside WSL: use `path_arg` for every file
    argument so Windows paths are translated to /mnt/<drive>/..."""
    name: str
    exe: str                      # native path or linux path inside WSL
    wsl_distro: str | None = None

    @property
    def via_wsl(self) -> bool:
        return self.wsl_distro is not None

    def path_arg(self, p: str | Path) -> str:
        return win_to_wsl(Path(p).resolve()) if self.via_wsl else str(Path(p).resolve())

    def run(self, args: Sequence[str], cwd: Path | None = None, timeout: float = 3600):
        if self.via_wsl:
            cmd = ["wsl", "-d", self.wsl_distro]
            if cwd is not None:
                cmd += ["--cd", win_to_wsl(Path(cwd).resolve())]
            # --exec skips the login shell, so `$`, `[...]` etc. in args are passed verbatim
            cmd += ["--exec", self.exe, *args]
            return run_cmd(cmd, timeout=timeout)
        return run_cmd([self.exe, *args], cwd=cwd, timeout=timeout)


def _healthy(cmd: list[str]) -> bool:
    """A tool is usable if it starts and exits without crashing (usage text exits non-zero, which is fine)."""
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    rc = p.returncode
    crashed = rc < 0 or rc >= 0xC0000000 or rc in (134, 139)  # signals, NTSTATUS crashes, abort/segv
    return not crashed


def _wsl_distros() -> list[str]:
    try:
        p = subprocess.run(["wsl", "-l", "-q"], capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    text = p.stdout.decode("utf-16-le", errors="ignore") if b"\x00" in p.stdout else p.stdout.decode(errors="ignore")
    return [ln.strip().strip("\x00") for ln in text.splitlines() if ln.strip().strip("\x00")]


class ToolRunner:
    """Locates pnnx / ncnn2table / ncnn2int8. Native candidates in order: EVP_PNNX (pnnx only), tools_dir,
    EVP_NCNN_TOOLS_DIR, the pip pnnx package, PATH, $EVP_TOOLS_HOME (default ~/evp_deps)/ncnn-install/bin;
    on Windows it falls back to <WSL distro>:$HOME/evp_deps. Every candidate must pass a start-up health check."""

    def __init__(self, tools_dir: str | Path | None = None, wsl_distro: str | None = None,
                 tools_home_wsl: str | None = None, allow_wsl: bool = True):
        self.tools_dir = Path(tools_dir) if tools_dir else None
        self.wsl_distro = wsl_distro or os.environ.get("EVP_WSL_DISTRO")
        self.tools_home_wsl = tools_home_wsl or os.environ.get("EVP_TOOLS_HOME_WSL")
        self.allow_wsl = allow_wsl and IS_WINDOWS
        self._cache: dict[str, Tool] = {}

    # ---------- resolution ----------
    def _native_candidates(self, name: str) -> list[str]:
        exe = name + (".exe" if IS_WINDOWS else "")
        out: list[str] = []
        env_key = "EVP_PNNX" if name == "pnnx" else None
        if env_key and os.environ.get(env_key):
            out.append(os.environ[env_key])
        dirs = [self.tools_dir] if self.tools_dir else []
        if os.environ.get("EVP_NCNN_TOOLS_DIR"):
            dirs.append(Path(os.environ["EVP_NCNN_TOOLS_DIR"]))
        for d in dirs:
            out.append(str(Path(d) / exe))
        if name == "pnnx":
            try:
                import importlib.util

                spec = importlib.util.find_spec("pnnx")
                if spec and spec.origin:
                    out.append(str(Path(spec.origin).parent / exe))
            except Exception:  # noqa: BLE001
                pass
        found = shutil.which(name)
        if found:
            out.append(found)
        home = Path(os.environ.get("EVP_TOOLS_HOME", Path.home() / "evp_deps"))
        out.append(str(home / "ncnn-install" / "bin" / exe))
        return [c for c in out if c and Path(c).is_file()]

    def _wsl_tool(self, name: str) -> Tool | None:
        if not self.allow_wsl:
            return None
        distros = [self.wsl_distro] if self.wsl_distro else _wsl_distros()
        for d in distros:
            home = self.tools_home_wsl or "$HOME/evp_deps"
            if name == "pnnx":
                script = f'cat {home}/pnnx_path.txt 2>/dev/null'
            else:
                script = f'p={home}/ncnn-install/bin/{name}; test -x "$p" && echo "$p"'
            try:
                p = subprocess.run(["wsl", "-d", d, "--exec", "bash", "-c", script], capture_output=True,
                                   text=True, timeout=60)
            except (OSError, subprocess.TimeoutExpired):
                continue
            exe = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
            if exe and _healthy(["wsl", "-d", d, "--exec", exe]):
                return Tool(name, exe, d)
        return None

    def get(self, name: str) -> Tool:
        """Resolve `name` (cached): first healthy native candidate, else the WSL build, else ToolNotFoundError."""
        if name in self._cache:
            return self._cache[name]
        for c in self._native_candidates(name):
            if _healthy([c]):
                tool = Tool(name, c)
                break
            log.warning("native %s at %s crashes on start; trying alternatives", name, c)
        else:
            tool = self._wsl_tool(name)
            if tool is None:
                raise ToolNotFoundError(
                    f"{name} not available",
                    "Windows: scripts/setup_tools_wsl.ps1 (builds inside WSL); Linux: scripts/setup_tools_linux.sh; "
                    "or set EVP_PNNX / EVP_NCNN_TOOLS_DIR")
        log.info("tool %s -> %s%s", name, tool.exe, f" (WSL {tool.wsl_distro})" if tool.via_wsl else "")
        self._cache[name] = tool
        return tool

    def describe(self) -> dict[str, str]:
        out = {}
        for n in ("pnnx",) + NCNN_TOOLS:
            try:
                t = self.get(n)
                out[n] = f"{t.exe}" + (f" [wsl:{t.wsl_distro}]" if t.via_wsl else "")
            except ToolNotFoundError as exc:
                out[n] = f"MISSING ({exc.message})"
        return out


_default: ToolRunner | None = None


def default_runner(tools_dir: str | None = None) -> ToolRunner:
    global _default
    if _default is None or (tools_dir and _default.tools_dir != Path(tools_dir)):
        _default = ToolRunner(tools_dir=tools_dir or None)
    return _default


def python_exe() -> str:
    return sys.executable
