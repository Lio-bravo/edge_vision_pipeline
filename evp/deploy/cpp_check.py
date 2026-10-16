#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""L3c: run the C++ runtime (evp_demo) on the bundle and compare with the Python reference pipeline.

Images are decoded once in Python and written as PPM so both sides get identical pixels. Checks:
  * preprocessed input tensor: C++ (--dump-input) vs Python preprocess_ncnn (max abs diff)
  * detections / top-k: same count, class, boxes within `box_tol` px, scores within `score_tol`
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..errors import EvpError
from ..utils import ensure_dir, log
from .postprocess import Detection
from .runner import NcnnPipeline, read_image
from .tools import IS_WINDOWS, Tool, ToolRunner, win_to_wsl
from .verify import match_detections

REPO_ROOT = Path(__file__).resolve().parents[2]
_demo_cache: dict[str, Tool] = {}


def write_ppm(path: Path, img_bgr: np.ndarray) -> Path:
    h, w = img_bgr.shape[:2]
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(np.ascontiguousarray(img_bgr[:, :, ::-1]).tobytes())
    return path


def find_demo(runner: ToolRunner | None = None, build: bool = True) -> Tool | None:
    """EVP_DEMO env > previously built binary > build with scripts/build_runtime.sh (native or WSL)."""
    if "demo" in _demo_cache:
        return _demo_cache["demo"]
    env = os.environ.get("EVP_DEMO")
    if env:
        tool = Tool("evp_demo", env, os.environ.get("EVP_WSL_DISTRO") if env.startswith("/") and IS_WINDOWS else None)
        _demo_cache["demo"] = tool
        return tool
    if not build:
        return None
    script = REPO_ROOT / "scripts" / "build_runtime.sh"
    try:
        if IS_WINDOWS:
            distro = None
            if runner is not None:
                try:
                    distro = runner.get("ncnn2table").wsl_distro
                except EvpError:
                    distro = None
            distro = distro or os.environ.get("EVP_WSL_DISTRO")
            if not distro:
                log.warning("C++ check skipped: no WSL distro with the ncnn toolchain")
                return None
            p = subprocess.run(["wsl", "-d", distro, "--cd", win_to_wsl(REPO_ROOT), "--exec", "bash",
                                win_to_wsl(script)], capture_output=True, text=True, timeout=900)
        else:
            if not shutil.which("cmake"):
                log.warning("C++ check skipped: cmake not found")
                return None
            distro = None
            p = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=900, cwd=REPO_ROOT)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("C++ check skipped: build failed to run (%s)", exc)
        return None
    if p.returncode != 0 or not p.stdout.strip():
        log.warning("C++ check skipped: runtime build failed\n%s", (p.stdout + p.stderr)[-2000:])
        return None
    tool = Tool("evp_demo", p.stdout.strip().splitlines()[-1], distro)
    _demo_cache["demo"] = tool
    return tool


def _parse_lines(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            out.append(json.loads(line))
    return out


def compare_detections(py: list, cpp: list, box_tol: float, score_tol: float) -> tuple[bool, dict]:
    info = {"n_py": len(py), "n_cpp": len(cpp), "max_box_diff": 0.0, "max_score_diff": 0.0, "cls_mismatch": 0}
    if len(py) != len(cpp):
        return False, info
    for a, b in zip(py, cpp):
        info["max_box_diff"] = max(info["max_box_diff"], max(abs(a[i] - b[i]) for i in range(4)))
        info["max_score_diff"] = max(info["max_score_diff"], abs(a[4] - b[4]))
        info["cls_mismatch"] += int(int(a[5]) != int(b[5]))
    ok = info["cls_mismatch"] == 0 and info["max_box_diff"] <= box_tol and info["max_score_diff"] <= score_tol
    return ok, info


def cpp_check(bundle: Path, images: Sequence[Path], runner: ToolRunner | None = None, box_tol: float = 0.05,
              score_tol: float = 1e-3, input_tol: float = 1e-6, int8_min_f1: float = 0.9) -> dict | None:
    """Returns a result dict, or None if the C++ runtime cannot be built here (reported as skipped)."""
    demo = find_demo(runner)
    if demo is None:
        return None
    bundle = Path(bundle).resolve()
    work = ensure_dir(bundle.parent / "_work" / bundle.name / "cpp_check")
    for f in work.iterdir():
        if f.is_file():
            f.unlink()
    imgs, ppms = [], []
    for i, p in enumerate(images):
        try:
            img = read_image(p)
        except EvpError:
            continue
        imgs.append(img)
        ppms.append(write_ppm(work / f"img_{i:04d}.ppm", img))
    if not ppms:
        return {"passed": False, "summary": "no readable images"}

    result: dict = {"passed": True, "precisions": {}}
    from .meta import load_meta

    meta = load_meta(bundle)
    for prec in [p for p in ("fp32", "int8") if p in meta.files]:
        dump = ensure_dir(work / f"dump_{prec}")
        proc = demo.run([demo.path_arg(bundle), *[demo.path_arg(p) for p in ppms], "--precision", prec,
                         "--dump-input", demo.path_arg(dump)], timeout=1800)
        rows = _parse_lines(proc.stdout)
        pipe = NcnnPipeline.from_bundle(bundle, prec)
        from .preprocess import preprocess_ncnn

        stat = {"images": len(rows), "input_max_abs_diff": 0.0, "mismatched_images": 0, "max_box_diff": 0.0,
                "max_score_diff": 0.0, "f1_mean": 1.0}
        if len(rows) != len(imgs):
            stat["error"] = f"C++ returned {len(rows)} results for {len(imgs)} images"
        f1s = []
        for k, (img, row) in enumerate(zip(imgs, rows)):
            mat, _ = preprocess_ncnn(img, meta.input)
            ref_in = np.array(mat, dtype=np.float32).reshape(-1)
            dump_file = dump / f"input_{k:04d}.f32"
            if dump_file.is_file():
                cin = np.fromfile(dump_file, dtype=np.float32)
                diff = float(np.abs(cin - ref_in).max()) if cin.size == ref_in.size else float("inf")
                stat["input_max_abs_diff"] = max(stat["input_max_abs_diff"], diff)
            else:
                stat["input_max_abs_diff"] = float("inf")
            res = pipe.run(img)
            if meta.task == "detect":
                py = [d.as_list() for d in res.detections or []]
                cp = row.get("detections", [])
                ok, info = compare_detections(py, cp, box_tol, score_tol)
                f1s.append(match_detections([Detection(*d[:5], int(d[5])) for d in py],
                                            [Detection(*d[:5], int(d[5])) for d in cp], 0.5).f1)
            elif meta.family == "classify":
                py = [[c, s, 0, 0, s, c] for c, s in res.topk or []]
                cp = [[c, s, 0, 0, s, c] for c, s in row.get("topk", [])]
                ok, info = compare_detections(py, cp, box_tol, score_tol)
                f1s.append(float(bool(py and cp and py[0][0] == cp[0][0])))
            else:
                ok, info = True, {"max_box_diff": 0.0, "max_score_diff": 0.0}
            stat["max_box_diff"] = max(stat["max_box_diff"], info["max_box_diff"])
            stat["max_score_diff"] = max(stat["max_score_diff"], info["max_score_diff"])
            if not ok:
                stat["mismatched_images"] += 1
                stat.setdefault("first_mismatch", {"image": row.get("image"), **info})
        stat["f1_mean"] = round(float(np.mean(f1s)), 4) if f1s else 1.0
        # fp32: strict (proves C++ pre/post == Python). int8: kernels differ across ncnn builds/ISAs, so
        # only detection-level agreement is meaningful.
        if prec == "fp32":
            stat["passed"] = not stat.get("error") and stat["input_max_abs_diff"] <= input_tol \
                and stat["mismatched_images"] == 0
        else:
            stat["passed"] = not stat.get("error") and stat["input_max_abs_diff"] <= input_tol \
                and stat["f1_mean"] >= int8_min_f1
        result["precisions"][prec] = stat
    result["passed"] = all(s["passed"] for s in result["precisions"].values())
    parts = [f"{p}: {s['images']} imgs, input diff {s['input_max_abs_diff']:.2e}, "
             f"exact-mismatch imgs {s['mismatched_images']}, box diff {s['max_box_diff']:.3g}px, "
             f"score diff {s['max_score_diff']:.2e}, F1 {s['f1_mean']}" for p, s in result["precisions"].items()]
    result["summary"] = "; ".join(parts)
    log.info("C++ vs Python: %s -> %s", result["summary"], "PASS" if result["passed"] else "FAIL")
    return result


def add_cpp_checks(report: dict, cpp: dict) -> None:
    """Merge cpp_check results into a verify report (fp32 gated, int8 informational)."""
    report["cpp"] = cpp
    for prec, s in (cpp.get("precisions") or {}).items():
        strict = prec == "fp32"
        report.setdefault("checks", {})[f"L3c_cpp_vs_python_{prec}"] = {
            "value": (1.0 if s["passed"] else 0.0) if strict else s.get("f1_mean", 0.0),
            "threshold": 1.0 if strict else 0.9, "passed": bool(s["passed"]),
            "group": f"cpp_{prec}", "gate": strict, "detail": s}
