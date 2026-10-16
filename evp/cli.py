#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""evp command line.

Data loop  : prelabel -> review (LabelMe, human) -> ingest (json2yolo + split + merge) -> train
Deploy     : deploy (export -> ONNX rules -> ncnn -> int8 -> meta -> verify -> bundle), verify, infer
Utilities  : init, inspect-onnx, tools, all
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .config import load_config
from .errors import ConfigError, EvpError, ToolNotFoundError
from .utils import ensure_dir, log

TEMPLATE = Path(__file__).resolve().parent.parent / "configs" / "project.example.yaml"


def parse_overrides(items: list[str] | None) -> dict:
    """--set a.b=1 --set classes=[x,y]  ->  nested dict (values parsed as YAML)."""
    out: dict[str, Any] = {}
    for it in items or []:
        if "=" not in it:
            raise ConfigError(f"--set expects key=value, got {it!r}")
        key, val = it.split("=", 1)
        parts = [p for p in key.strip().split(".") if p]
        if not parts:
            raise ConfigError(f"--set has an empty key: {it!r}")
        try:
            v = yaml.safe_load(val)
        except yaml.YAMLError as exc:
            raise ConfigError(f"--set {key}: cannot parse value {val!r}: {exc}") from exc
        d = out
        for p in parts[:-1]:
            d = d.setdefault(p, {})
            if not isinstance(d, dict):
                raise ConfigError(f"--set {key}: conflicting keys")
        d[parts[-1]] = v
    return out


def _cfg(args) -> dict:
    return load_config(args.config, parse_overrides(getattr(args, "set", None)))


def _names(cfg: dict) -> list[str]:
    names = [str(c) for c in cfg.get("classes") or []]
    if not names:
        raise ConfigError("config.classes is empty", "list your class names under `classes:`")
    return names


# ------------------------------------------------------------------ commands
def cmd_init(args) -> int:
    root = Path(args.dir).resolve()
    ensure_dir(root)
    dst = root / "project.yaml"
    if dst.exists() and not args.force:
        raise ConfigError(f"{dst} exists", "use --force to overwrite")
    if not TEMPLATE.is_file():
        raise ConfigError(f"template missing: {TEMPLATE}")
    shutil.copyfile(TEMPLATE, dst)
    cfg = load_config(dst)
    for k in ("raw_images", "labelme_dir", "batch_dataset", "corpus_root", "runs", "deploy_out"):
        ensure_dir(Path(cfg["paths"][k]))
    ensure_dir(root / "models")
    log.info("project created: %s (edit classes/paths, put images into %s)", dst, cfg["paths"]["raw_images"])
    return 0


def cmd_collect(args) -> int:
    """Folder / video / camera -> paths.raw_images (exact-duplicate, optional near-duplicate and blur gates)."""
    from .data.collect import capture_camera, collector_from_config, extract_video, import_folder

    cfg = _cfg(args)
    c = cfg["collect"]
    col = collector_from_config(cfg, args.dedup, args.min_sharpness)
    if args.folder:
        stats = import_folder(Path(args.folder), col, recursive=not args.no_recursive)
    elif args.video:
        stats = extract_video(Path(args.video), col, every_sec=float(args.every or c["video_every_sec"]),
                              max_frames=int(args.max_frames if args.max_frames is not None else c["max_frames"]))
    else:
        src = c["camera"] if args.camera is None else args.camera
        src = int(src) if isinstance(src, str) and src.isdigit() else src
        stats = capture_camera(col, src, count=int(args.count or c["camera_count"]),
                               interval=float(c["camera_interval"] if args.interval is None else args.interval),
                               timeout=float(args.timeout))
    out = stats.as_dict()
    out["raw_images"] = cfg["paths"]["raw_images"]
    print(json.dumps(out, ensure_ascii=False))
    if stats.saved == 0:
        log.warning("nothing saved (all frames were duplicates, blurry or unreadable)")
    return 0


def cmd_prelabel(args) -> int:
    from .data.prelabel import prelabel

    cfg = _cfg(args)
    p, pl = cfg["paths"], cfg["prelabel"]
    n = prelabel(Path(p["raw_images"]), Path(p["labelme_dir"]), _names(cfg),
                 model=None if args.bundle else Path(args.model or p["prelabel_model"]),
                 bundle=Path(args.bundle) if args.bundle else None, conf=float(pl["conf"]), iou=float(pl["iou"]),
                 imgsz=int(pl["imgsz"]), device=str(pl["device"]), embed_image_data=bool(pl["embed_image_data"]),
                 precision=args.precision, overwrite=args.overwrite)
    print(json.dumps({"written": n, "labelme_dir": p["labelme_dir"]}))
    return 0


def cmd_review(args) -> int:
    cfg = _cfg(args)
    d = cfg["paths"]["labelme_dir"]
    exe = shutil.which("labelme")
    if not exe:
        raise ToolNotFoundError("labelme not found on PATH",
                                f"pip install labelme, or open {d} in your LabelMe app and then run `evp ingest`")
    log.info("opening LabelMe on %s (save each image, then run `evp ingest`)", d)
    labels = ",".join(_names(cfg))
    return subprocess.call([exe, d, "--labels", labels, "--nodata", "--autosave"])


def cmd_ingest(args) -> int:
    """Reviewed LabelMe json -> YOLO batch -> split -> merge into corpus -> clear staging.
    Any failure (broken json, unknown label, bad id) stops before merge/clear, so reviewed work is kept."""
    from .data.ingest import ingest

    res = ingest(_cfg(args), keep_staging=args.keep_staging)
    print(json.dumps(res, ensure_ascii=False))
    return 0


def cmd_train(args) -> int:
    from .train.trainer import train

    cfg = _cfg(args)
    best = train(Path(cfg["paths"]["corpus_root"]), Path(cfg["paths"]["runs"]), cfg["train"])
    print(json.dumps({"best": str(best)}))
    return 0


def cmd_deploy(args) -> int:
    from .deploy.pipeline import deploy

    cfg = _cfg(args)
    res = deploy(cfg, weights=args.weights, skip_quant=args.no_quant, skip_verify=args.no_verify, out_dir=args.out)
    if res.get("passed") is None:
        log.warning("verification was skipped: the bundle is UNVERIFIED (run `evp verify --bundle %s`)", res["bundle"])
    print(json.dumps({k: res[k] for k in ("bundle", "report", "passed", "passed_int8", "recommended_precision")}))
    return 0 if res.get("passed") is not False else 3


def cmd_verify(args) -> int:
    """Re-verify an existing bundle (L1 needs _work/<name>/model.prepared.onnx; L3 needs the .pt)."""
    from .deploy.cpp_check import add_cpp_checks, cpp_check
    from .deploy.meta import load_meta
    from .deploy.pipeline import build_runner, resolve_images
    from .deploy.verify import finalize_verdict, onnx_forward_fn, verify_bundle, write_report
    from .utils import list_images

    cfg = _cfg(args)
    bundle = Path(args.bundle)
    meta = load_meta(bundle)
    imgs = list_images(Path(args.images), recursive=True) if args.images else resolve_images(cfg, "verify")
    imgs = imgs[: args.count] if args.count else imgs
    ref_fn = None
    onnx = bundle.parent / "_work" / bundle.name / "model.prepared.onnx"
    if onnx.is_file():
        ref_fn = onnx_forward_fn(onnx)
    v = cfg["deploy"]["verify"]
    weights = Path(args.weights) if args.weights else (Path(meta.source["weights"]) if meta.source.get("weights")
                                                       and Path(meta.source["weights"]).is_file() else None)
    report = verify_bundle(bundle, imgs, ref_fn, weights, float(v["min_fp32_cos"]), float(v["min_int8_cos"]),
                           float(v["min_e2e_match"]))
    if not args.no_cpp:
        cpp = cpp_check(bundle, imgs[:8], build_runner(cfg))
        if cpp is not None:
            add_cpp_checks(report, cpp)
            finalize_verdict(report, has_int8="int8" in meta.files)
    write_report(report, bundle)
    print(json.dumps({k: report.get(k) for k in ("passed", "passed_int8", "recommended_precision")}))
    return 0 if report.get("passed") else 3


def cmd_eval(args) -> int:
    """mAP of the bundle (each precision) and optionally the .pt on a labeled YOLO split; writes eval_<split>.*"""
    from .deploy.evaluate import (
        load_yolo_split,
        log_eval,
        predict_bundle,
        predict_ultralytics,
        render_eval_markdown,
        run_eval,
    )
    from .deploy.meta import load_meta
    from .utils import resolve_device, write_json

    cfg = _cfg(args)
    bundle = Path(args.bundle)
    meta = load_meta(bundle)
    root = Path(args.data) if args.data else Path(cfg["paths"]["corpus_root"])
    names = _names(cfg)
    dy = root / "data.yaml"
    if dy.is_file():
        dn = (yaml.safe_load(dy.read_text(encoding="utf-8")) or {}).get("names")
        names = [dn[k] for k in sorted(dn)] if isinstance(dn, dict) else list(dn or names)
    images, gts = load_yolo_split(root, args.split)
    if args.limit:
        images = images[: args.limit]
        gts = {p.stem: gts[p.stem] for p in images}
    conf = float(args.conf if args.conf is not None else meta.post.conf)
    results: dict[str, dict] = {}
    if args.weights:
        imgsz = int(args.pt_imgsz or max(meta.input.w, meta.input.h))
        fn, mn = predict_ultralytics(Path(args.weights), imgsz, device=resolve_device("auto", "eval.device"),
                                     iou=meta.post.iou)
        results[f"pt@{imgsz}"] = run_eval(fn, mn, images, gts, names, conf)
        log_eval(f"pt@{imgsz}", results[f"pt@{imgsz}"])
    for prec in args.precision:
        if prec not in meta.files:
            log.warning("bundle has no %s model, skipped", prec)
            continue
        fn, mn = predict_bundle(bundle, prec, args.threads)
        results[f"ncnn-{prec}"] = run_eval(fn, mn, images, gts, names, conf)
        log_eval(f"ncnn-{prec}", results[f"ncnn-{prec}"])
    if not results:
        raise ConfigError("nothing evaluated", f"bundle precisions: {sorted(meta.files)}")
    out = Path(args.out) if args.out else bundle / f"eval_{args.split}"
    write_json(out.with_suffix(".json"), {"data": str(root), "split": args.split, "names": names, "results": results})
    md = (f"# Accuracy on `{root}` ({args.split}, {len(images)} images)\n\nbundle `{bundle.name}`, input "
          f"{meta.input.w}x{meta.input.h}\n\n" + render_eval_markdown(results, names))
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(json.dumps({k: {m: (round(v[m], 4) if isinstance(v[m], float) else v[m])
                          for m in ("mAP50", "mAP50_95", "precision", "recall")} for k, v in results.items()}))
    return 0


def cmd_bench(args) -> int:
    """Host latency table (fp32/int8 x threads) of the Python reference runtime; writes <bundle>/bench.{json,md}."""
    from .deploy.runner import read_image
    from .utils import list_images, write_json
    from .webui.services import benchmark

    bundle = Path(args.bundle)
    imgs = [read_image(p) for p in list_images(Path(args.images), recursive=True)[: args.count]]
    if not imgs:
        raise ConfigError(f"no images under {args.images}")
    rows = benchmark(bundle, imgs, args.precision, args.threads, runs=args.runs, warmup=args.warmup)
    import platform

    host = f"{platform.processor() or platform.machine()} / {platform.system()} {platform.release()}"
    write_json(bundle / "bench.json", {"host": host, "runs": args.runs, "rows": rows})
    head = "| precision | threads | input | weights MB | mean ms | p50 ms | p90 ms | FPS |"
    lines = [f"# Host latency: `{bundle.name}`", "", f"host: {host}; {args.runs} runs per row; pre + forward + post "
             "in the Python reference runtime (board numbers come from the C++ runtime)", "", head,
             "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['precision']} | {r['threads']} | {r['input']} | {r['weights_MB']} | {r['mean_ms']} | "
              f"{r['p50_ms']} | {r['p90_ms']} | {r['fps']} |" for r in rows]
    (bundle / "bench.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(rows))
    return 0


def cmd_infer(args) -> int:
    from .deploy.runner import NcnnPipeline, read_image

    pipe = NcnnPipeline.from_bundle(Path(args.bundle), args.precision, num_threads=args.threads)
    if args.conf is not None:
        pipe.post.conf = args.conf
    labels = pipe.meta.labels
    vis_dir = ensure_dir(Path(args.save_vis)) if args.save_vis else None
    for p in args.images:
        img = read_image(Path(p))
        res = pipe.run(img)
        row: dict[str, Any] = {"image": str(p)}
        if res.detections is not None:
            row["detections"] = [{"box": [round(d.x1, 2), round(d.y1, 2), round(d.x2, 2), round(d.y2, 2)],
                                  "score": round(d.score, 4), "cls": d.cls,
                                  "label": labels[d.cls] if d.cls < len(labels) else str(d.cls)}
                                 for d in res.detections]
            if vis_dir is not None:
                _draw(img, res.detections, labels, vis_dir / (Path(p).stem + "_det.jpg"))
        elif res.topk is not None:
            row["topk"] = [{"cls": c, "score": round(s, 5), "label": labels[c] if c < len(labels) else str(c)}
                           for c, s in res.topk]
        else:
            row["outputs"] = [list(o.shape) for o in res.outputs]
        print(json.dumps(row, ensure_ascii=False))
    return 0


def _draw(img, dets, labels, path: Path) -> None:
    import cv2

    vis = img.copy()
    for d in dets:
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)
        name = labels[d.cls] if d.cls < len(labels) else str(d.cls)
        cv2.putText(vis, f"{name} {d.score:.2f}", (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1)
    ok, buf = cv2.imencode(".jpg", vis)
    if ok:
        buf.tofile(str(path))


def cmd_inspect_onnx(args) -> int:
    import onnx

    from .deploy.convert import prepare_onnx
    from .deploy.onnx_rules import op_inventory, risky_ops, suggest_cut

    src = Path(args.onnx)
    if not src.is_file():
        raise ConfigError(f"ONNX not found: {src}")
    m = onnx.load(str(src))
    info: dict[str, Any] = {"ops": op_inventory(m), "risky": risky_ops(m), "suggested_cut": suggest_cut(m),
                            "outputs": [o.name for o in m.graph.output]}
    if args.out:
        cut = args.cut if args.cut in (None, "auto") else [c for c in args.cut.split(",") if c]
        _, results, meta_updates, ops, shape = prepare_onnx(src, Path(args.out), rules=args.rules, cut=cut,
                                                            fail_on_risky=False)
        info.update({"written": args.out, "ops_after": ops, "input_shape": shape, "meta_updates": meta_updates,
                     "rules": [{"name": r.name, "changed": r.changed, "notes": r.notes} for r in results]})
    print(json.dumps(info, indent=1, ensure_ascii=False))
    return 0


def cmd_tools(args) -> int:
    from .deploy.pipeline import build_runner

    cfg = _cfg(args)
    print(json.dumps(build_runner(cfg).describe(), indent=1))
    return 0


def cmd_webui(args) -> int:
    try:
        from .webui.app import launch
    except ImportError as exc:
        raise ToolNotFoundError(f"web UI dependencies missing: {exc}", "pip install -e .[webui]") from exc
    if args.config and not Path(args.config).is_file():
        raise ConfigError(f"config not found: {args.config}", "omit -c and create a project in the UI")
    if args.host not in ("127.0.0.1", "localhost"):
        log.warning("binding to %s: anyone who can reach this port can read/write files as you", args.host)
    launch(args.config, host=args.host, port=args.port, inbrowser=args.open)
    return 0


def cmd_all(args) -> int:
    """ingest (if a reviewed batch exists) -> train -> deploy."""
    cfg = _cfg(args)
    lm = Path(cfg["paths"]["labelme_dir"])
    if lm.is_dir() and any(lm.glob("*.json")):
        cmd_ingest(args)
    else:
        log.info("no reviewed LabelMe json in %s; skipping ingest", lm)
    if not args.skip_train:
        cmd_train(args)
    args.weights, args.no_quant, args.no_verify, args.out = None, False, False, None
    return cmd_deploy(args)


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="evp", description="edge vision pipeline: label -> train -> ncnn deploy")
    ap.add_argument("--version", action="version", version=f"evp {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def with_cfg(p: argparse.ArgumentParser, required: bool = False) -> argparse.ArgumentParser:
        p.add_argument("-c", "--config", required=required, help="project YAML (default: built-in defaults)")
        p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override config, e.g. deploy.e2e=true")
        return p

    p = sub.add_parser("init", help="create a project folder from the template")
    p.add_argument("dir")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_init)

    p = with_cfg(sub.add_parser("collect", help="folder / video / camera -> raw images (dedup + blur gate)"), True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--folder", help="import images from this folder")
    src.add_argument("--video", help="extract frames from this video file")
    src.add_argument("--camera", nargs="?", const=None, default=argparse.SUPPRESS,
                     help="capture from a camera index / stream URL (default collect.camera)")
    p.add_argument("--no-recursive", action="store_true", help="folder: do not descend into subfolders")
    p.add_argument("--every", type=float, help="video: seconds between frames (default collect.video_every_sec)")
    p.add_argument("--max-frames", type=int, help="video: cap on saved frames (0 = none)")
    p.add_argument("--count", type=int, help="camera: frames to save (default collect.camera_count)")
    p.add_argument("--interval", type=float, help="camera: seconds between grabs")
    p.add_argument("--timeout", type=float, default=0.0, help="camera: give up after N seconds (0 = never)")
    p.add_argument("--dedup", type=int, help="near-duplicate dHash distance, -1 = off (default collect.dedup_hamming)")
    p.add_argument("--min-sharpness", type=float, help="blur gate, 0 = off (default collect.min_sharpness)")
    p.set_defaults(fn=cmd_collect)

    p = with_cfg(sub.add_parser("prelabel", help="raw images -> LabelMe json (model predictions)"), True)
    p.add_argument("--model", help="Ultralytics .pt (default paths.prelabel_model)")
    p.add_argument("--bundle", help="use an ncnn deploy bundle instead of a .pt")
    p.add_argument("--precision", default="fp32", choices=["fp32", "int8"])
    p.add_argument("--overwrite", action="store_true", help="replace existing (possibly reviewed) json")
    p.set_defaults(fn=cmd_prelabel)

    p = with_cfg(sub.add_parser("review", help="open LabelMe on the prelabel folder"), True)
    p.set_defaults(fn=cmd_review)

    p = with_cfg(sub.add_parser("ingest", help="reviewed json -> yolo -> split -> merge into corpus"), True)
    p.add_argument("--keep-staging", action="store_true", help="do not clear raw/labelme/batch folders")
    p.set_defaults(fn=cmd_ingest)

    p = with_cfg(sub.add_parser("train", help="train on the corpus"), True)
    p.set_defaults(fn=cmd_train)

    p = with_cfg(sub.add_parser("deploy", help="weights -> verified ncnn bundle"))
    p.add_argument("--weights", help=".pt (Ultralytics) or .onnx; default deploy.weights / latest best.pt")
    p.add_argument("--no-quant", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--out", help="output root (default paths.deploy_out)")
    p.set_defaults(fn=cmd_deploy)

    p = with_cfg(sub.add_parser("verify", help="re-run verification on an existing bundle"))
    p.add_argument("--bundle", required=True)
    p.add_argument("--images")
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--weights", help="framework weights for the L3 reference (default: from meta)")
    p.add_argument("--no-cpp", action="store_true")
    p.set_defaults(fn=cmd_verify)

    p = with_cfg(sub.add_parser("eval", help="mAP of a bundle (fp32/int8) and optionally the .pt on labeled data"))
    p.add_argument("--bundle", required=True)
    p.add_argument("--data", help="YOLO dataset root with images/<split>, labels/<split> (default paths.corpus_root)")
    p.add_argument("--split", default="val")
    p.add_argument("--precision", nargs="+", default=["fp32", "int8"], choices=["fp32", "int8"])
    p.add_argument("--weights", help="also evaluate this .pt (Ultralytics) with the same metric")
    p.add_argument("--pt-imgsz", type=int, help="inference size for --weights (default: bundle input size)")
    p.add_argument("--conf", type=float, help="confidence for P/R/F1 (default: bundle post.conf)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N images")
    p.add_argument("--out", help="output path stem (default <bundle>/eval_<split>)")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("bench", help="host latency of a bundle: fp32/int8 x threads (Python reference runtime)")
    p.add_argument("--bundle", required=True)
    p.add_argument("--images", required=True, help="folder of sample images")
    p.add_argument("--count", type=int, default=8, help="images to cycle through")
    p.add_argument("--precision", nargs="+", default=["fp32", "int8"], choices=["fp32", "int8"])
    p.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--warmup", type=int, default=3)
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("infer", help="run a bundle on images (Python reference runtime)")
    p.add_argument("--bundle", required=True)
    p.add_argument("images", nargs="+")
    p.add_argument("--precision", default="fp32", choices=["fp32", "int8"])
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--conf", type=float)
    p.add_argument("--save-vis", help="write images with boxes to this folder")
    p.set_defaults(fn=cmd_infer)

    p = sub.add_parser("inspect-onnx", help="operator inventory / risky ops / suggested cut; optional rewrite")
    p.add_argument("onnx")
    p.add_argument("--out", help="write the processed ONNX here")
    p.add_argument("--cut", help="'auto' or comma-separated tensor names")
    p.add_argument("--rules", default="default", help="default | none")
    p.set_defaults(fn=cmd_inspect_onnx)

    p = with_cfg(sub.add_parser("tools", help="show resolved converter tools (native / WSL)"))
    p.set_defaults(fn=cmd_tools)

    p = sub.add_parser("webui", help="low-code web workbench (collect / label / train / deploy / benchmark)")
    p.add_argument("-c", "--config", help="project YAML to open (optional; projects can be created in the UI)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--open", action="store_true", help="open the browser")
    p.set_defaults(fn=cmd_webui)

    p = with_cfg(sub.add_parser("all", help="ingest -> train -> deploy"), True)
    p.add_argument("--keep-staging", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.set_defaults(fn=cmd_all)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except EvpError as exc:
        log.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
