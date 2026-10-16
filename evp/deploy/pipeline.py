#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""`evp deploy`: weights -> ONNX -> rules -> ncnn fp32 -> int8 -> meta -> verify -> bundle.

Bundle layout (deploy_out/<name>/):
    <name>.fp32.param/.bin   <name>.int8.param/.bin (if quantized)
    meta.json  model.meta  labels.txt          paired pre/post-processing contract
    report.json  report.md                     verification + quantization + conversion details
    runtime/                                   C++ runtime sources (meta loader, pre/post, demo)
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import ConfigError, ConvertError, DataError, EvpError
from ..utils import ensure_dir, list_images, log, sha256_file
from .convert import ConvertResult, convert_onnx
from .meta import InputSpec, ModelFile, ModelMeta, OutputSpec, PostSpec, save_meta
from .tools import ToolRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_SRC = REPO_ROOT / "runtime" / "cpp"


def _safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not s:
        raise ConfigError(f"invalid bundle name {name!r}")
    return s


def resolve_weights(cfg: dict, weights: str | Path | None = None) -> Path:
    """CLI --weights > deploy.weights > `auto` (newest best.pt under paths.runs)."""
    w = weights or cfg["deploy"].get("weights") or "auto"
    if str(w) == "auto":
        from ..train.trainer import find_latest_best

        return find_latest_best(Path(cfg["paths"]["runs"]), cfg["train"].get("run_name"))
    p = Path(w)
    if not p.is_file():
        raise ConfigError(f"deploy weights not found: {p}", "set deploy.weights or pass --weights")
    return p


def resolve_images(cfg: dict, kind: str) -> list[Path]:
    """kind: calib | verify. `auto` picks corpus train/val images, then raw_images."""
    dep = cfg["deploy"]
    v = dep["quant"]["calib_images"] if kind == "calib" else dep["verify"]["images"]
    if v not in (None, "", "auto"):
        return list_images(Path(v), recursive=True)
    corpus = Path(cfg["paths"]["corpus_root"])
    order = ["train", "val"] if kind == "calib" else ["val", "train"]
    for split in order:
        d = corpus / "images" / split
        if d.is_dir():
            imgs = list_images(d, recursive=True)
            if imgs:
                return imgs
    raw = Path(cfg["paths"]["raw_images"])
    if raw.is_dir():
        imgs = list_images(raw, recursive=True)
        if imgs:
            return imgs
    raise DataError(f"no images found for {kind}",
                    f"set deploy.{'quant.calib_images' if kind == 'calib' else 'verify.images'} to an image folder")


def build_runner(cfg: dict) -> ToolRunner:
    dep = cfg["deploy"]
    return ToolRunner(tools_dir=dep.get("tools_dir") or None, wsl_distro=dep.get("wsl_distro") or None)


def compose_normalization(mean0, norm0, mean1, norm1) -> tuple[list[float], list[float]]:
    """pixel -> x=(p-m0)*n0 (configured) then graph-extracted y=(x-m1)*n1  ==  (p-(m0+m1/n0))*(n0*n1)."""
    mean = [float(a + b / n) for a, b, n in zip(mean0, mean1, norm0)]
    norm = [float(a * b) for a, b in zip(norm0, norm1)]
    return mean, norm


def _apply_input_overrides(spec: InputSpec, cfg_input: dict) -> InputSpec:
    for k in ("resize", "mean", "norm", "pad_value"):
        v = (cfg_input or {}).get(k)
        if v is not None:
            setattr(spec, k, list(map(float, v)) if k in ("mean", "norm") else v)
    return spec


def _raw_meta(name: str, conv: ConvertResult, cfg: dict) -> ModelMeta:
    """Meta for a generic ONNX model: outputs discovered from the ncnn graph, post-processing chosen by family."""
    from .runner import NcnnNet

    dep = cfg["deploy"]
    shape = conv.onnx_input_shape
    if len(shape) != 4 or shape[1] != 3:
        raise ConvertError(f"generic ONNX input must be 1x3xHxW after rules, got {shape}",
                           "single-image 3-channel inputs only; gray/multi-input models need a custom adapter")
    h, w = shape[2], shape[3]
    spec = InputSpec(w=w, h=h, blob=conv.input_name, color=dep["input_color"], resize="stretch", pad_value=114)
    user_in = dep.get("input") or {}
    notes: list[str] = []
    if "input.mean" in conv.meta_updates:
        # normalization lived in the graph, so the original graph input is raw 0..255 pixels unless the user
        # says otherwise via deploy.input.mean/norm (which always describe the ORIGINAL ONNX input)
        spec.mean, spec.norm = [0.0] * 3, [1.0] * 3
    elif user_in.get("mean") is None and user_in.get("norm") is None:
        notes.append("generic ONNX without in-graph normalization: assuming input = pixels / 255; "
                     "set deploy.input.mean/norm if the model expects something else")
    spec = _apply_input_overrides(spec, user_in)
    if "input.mean" in conv.meta_updates:
        spec.mean, spec.norm = compose_normalization(spec.mean, spec.norm, conv.meta_updates["input.mean"],
                                                     conv.meta_updates["input.norm"])
        log.info("folded graph normalization into meta: mean=%s norm=%s", spec.mean, spec.norm)
    for n in notes:
        log.warning("%s", n)
    net = NcnnNet(conv.param, conv.bin)
    outs = net.forward(np.zeros((3, h, w), np.float32), conv.input_name, conv.output_names)
    outputs = [OutputSpec(n, list(o.shape)) for n, o in zip(conv.output_names, outs)]
    family = dep["family"] if dep["family"] in ("classify", "raw") else "raw"
    labels = [str(x) for x in dep.get("labels") or []]
    post_cfg = dep["post"]
    if family == "classify":
        if len(outputs) != 1:
            raise ConvertError(f"classify ONNX must have 1 output, has {len(outputs)}")
        probe = net.forward(np.random.default_rng(0).uniform(0, 1, (3, h, w)).astype(np.float32),
                            conv.input_name, conv.output_names)[0].reshape(-1)
        already_prob = bool(np.all(probe >= 0) and abs(float(probe.sum()) - 1.0) < 1e-3)
        post = PostSpec(method="topk", topk=int(post_cfg.get("topk", 5)), softmax=not already_prob)
        outputs[0].shape = [int(np.prod(outputs[0].shape))]
        if labels and len(labels) != outputs[0].shape[0]:
            raise ConfigError(f"deploy.labels has {len(labels)} names but the model outputs {outputs[0].shape[0]}")
    else:
        post = PostSpec(method="none")
    return ModelMeta(name=name, family=family, input=spec, outputs=outputs, labels=labels, post=post,
                     source={"framework": "onnx", "onnx": str(conv.onnx), "notes": notes})


def deploy(cfg: dict, weights: str | Path | None = None, skip_quant: bool = False, skip_verify: bool = False,
           out_dir: str | Path | None = None) -> dict[str, Any]:
    """Build a verified bundle from `weights` (default: cfg deploy.weights / latest best.pt).

    Steps: export (.pt adapter) or take the .onnx -> convert_onnx -> meta -> quantize (unless skip_quant)
    -> save meta + runtime sources -> verify_bundle + cpp_check (unless skip_verify) -> report.
    Returns a summary dict: bundle, report, passed (None if verification was skipped), passed_int8,
    recommended_precision, convert, quant. Intermediate files go to <deploy_out>/_work/<name>/."""
    from .verify import finalize_verdict, onnx_forward_fn, torch_forward_fn, verify_bundle, write_report

    dep = cfg["deploy"]
    wpath = resolve_weights(cfg, weights)
    name = _safe_name(dep.get("name") or wpath.stem)
    root = Path(out_dir) if out_dir else Path(cfg["paths"]["deploy_out"])
    bundle = ensure_dir(root / name)
    work = ensure_dir(root / "_work" / name)
    runner = build_runner(cfg)
    h, w = dep["imgsz"]
    log.info("deploy %s -> %s (imgsz %dx%d)", wpath, bundle, h, w)

    module = None
    ul_weights = None
    if wpath.suffix.lower() == ".pt":
        from .export import export_ultralytics

        exp = export_ultralytics(wpath, (h, w), work, e2e=bool(dep["e2e"]), family=dep["family"],
                                 input_color=dep["input_color"], post_cfg=dep["post"], name=name)
        meta, module, ul_weights = exp.meta, exp.module, wpath
        meta.input = _apply_input_overrides(meta.input, dep.get("input"))
        conv = convert_onnx(exp.onnx, work, rules=dep["onnx"].get("rules", "default"),
                            cut=dep["onnx"].get("cut_outputs") or None, input_shape=[1, 3, h, w], runner=runner,
                            fp16_weights=bool(dep["weights_fp16"]))
        if conv.output_names != [o.name for o in meta.outputs]:
            raise ConvertError(f"ncnn outputs {conv.output_names} != exported {[o.name for o in meta.outputs]}")
        meta.input.blob = conv.input_name
    elif wpath.suffix.lower() == ".onnx":
        if dep["family"] == "yolo":
            raise ConfigError("YOLO detectors must be deployed from the Ultralytics .pt",
                              "the .pt adapter cuts the head at the raw tensors the runtime post-processing expects")
        conv = convert_onnx(wpath, work, rules=dep["onnx"].get("rules", "default"),
                            cut=dep["onnx"].get("cut_outputs") or None, input_shape=[1, 3, h, w], runner=runner,
                            fp16_weights=bool(dep["weights_fp16"]))
        meta = _raw_meta(name, conv, cfg)
    else:
        raise ConfigError(f"unsupported weights type {wpath.suffix}", "use an Ultralytics .pt or an .onnx file")

    # fp32 files
    files = {"fp32": ModelFile(f"{name}.fp32.param", f"{name}.fp32.bin")}
    shutil.copyfile(conv.param, bundle / files["fp32"].param)
    shutil.copyfile(conv.bin, bundle / files["fp32"].bin)
    # meta/report are rewritten at the end; removing them now means a deploy that fails half-way leaves a
    # bundle that refuses to load instead of new weights paired with an old contract
    for stale in (f"{name}.int8.param", f"{name}.int8.bin", f"{name}.int8.table", "meta.json", "model.meta",
                  "labels.txt", "report.json", "report.md"):
        if (bundle / stale).exists():
            (bundle / stale).unlink()

    summary: dict[str, Any] = {
        "convert": {"layer_count": len(conv.ncnn.layers), "layer_types": conv.ncnn.layer_types(),
                    "onnx_ops": conv.onnx_ops, "rules_applied": [r.name for r in conv.rules if r.changed],
                    "rule_notes": [n for r in conv.rules for n in r.notes]},
    }

    q = dep["quant"]
    if q.get("enable") and not skip_quant:
        from .quant import quantize

        calib = resolve_images(cfg, "calib")
        try:
            ver_imgs = resolve_images(cfg, "verify")
        except DataError:
            ver_imgs = None
        qr = quantize(conv.param, conv.bin, meta.input, [o.name for o in meta.outputs], calib, work / "quant",
                      method=q["method"], calib_count=int(q["calib_count"]), target_cos=float(q["target_cos"]),
                      keep_output_layers_fp32=bool(q["keep_output_layers_fp32"]),
                      auto_mixed_precision=bool(q["auto_mixed_precision"]),
                      max_fp32_layers=int(q["max_fp32_layers"]), eval_images=ver_imgs,
                      eval_count=int(q.get("eval_count", 16)), fp32_layers=q.get("fp32_layers") or [], runner=runner,
                      box_channels=meta.box_channels if meta.task == "detect" else None)
        files["int8"] = ModelFile(f"{name}.int8.param", f"{name}.int8.bin")
        shutil.copyfile(qr.param, bundle / files["int8"].param)
        shutil.copyfile(qr.bin, bundle / files["int8"].bin)
        shutil.copyfile(qr.table, bundle / f"{name}.int8.table")
        meta.quant = qr.to_dict()
        summary["quant"] = meta.quant
    for f in files.values():
        f.sha256_bin = sha256_file(bundle / f.bin)
    meta.files = files
    save_meta(meta, bundle)
    _copy_runtime(bundle)

    report: dict[str, Any] = {"model": meta.name, "family": meta.family, "passed": None, "checks": {}, "notes": []}
    if not skip_verify:
        imgs = resolve_images(cfg, "verify")
        n = int(dep["verify"]["count"])
        imgs = imgs[:: max(1, len(imgs) // n)][:n] if len(imgs) > n else imgs
        ref_fn = torch_forward_fn(module) if module is not None else onnx_forward_fn(conv.onnx)
        v = dep["verify"]
        report = verify_bundle(bundle, imgs, ref_fn, ul_weights, float(v["min_fp32_cos"]), float(v["min_int8_cos"]),
                               float(v["min_e2e_match"]))
        if dep.get("cpp_runtime"):
            from .cpp_check import add_cpp_checks, cpp_check

            cpp = cpp_check(bundle, imgs[: min(len(imgs), 8)], runner)
            if cpp is None:
                report["notes"].append("C++ runtime check skipped (no build toolchain found)")
            else:
                add_cpp_checks(report, cpp)
                finalize_verdict(report, has_int8="int8" in meta.files)
    jp, mp = write_report(report, bundle, summary)
    log.info("bundle ready: %s (report %s)", bundle, mp)
    if report.get("passed") is False:
        log.warning("verification FAILED, see %s", mp)
    elif "int8" in (report.get("unverified") or []):
        log.warning("int8 could not be verified (nothing to compare on the verification images); ship fp32 or "
                    "re-verify with representative images (see %s)", mp)
    elif report.get("passed_int8") is False:
        log.warning("int8 did not pass verification; ship fp32 (see %s)", mp)
    return {"bundle": str(bundle), "report": str(jp), "passed": report.get("passed"),
            "passed_int8": report.get("passed_int8"), "recommended_precision": report.get("recommended_precision"),
            **summary}


def _copy_runtime(bundle: Path) -> None:
    if not RUNTIME_SRC.is_dir():
        log.warning("runtime/cpp not found next to the package; bundle will not include C++ sources")
        return
    dst = bundle / "runtime"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(RUNTIME_SRC, dst, ignore=shutil.ignore_patterns("build*", "*.o", "*.a", "__pycache__"))


def load_bundle_meta(bundle: Path) -> ModelMeta:
    from .meta import load_meta

    try:
        return load_meta(bundle)
    except EvpError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise EvpError(f"cannot read bundle meta in {bundle}: {exc}") from exc
