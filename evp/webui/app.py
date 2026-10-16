#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""All-in-one low-code workbench (Gradio): collect -> label/review -> ingest/train -> deploy -> infer/benchmark.

Thin UI layer over `services` and the evp library; every handler reloads the project YAML. Errors from the
library (EvpError) are shown as toasts with their hint instead of tracebacks.
"""

from __future__ import annotations

import functools
import inspect
import time
from pathlib import Path

import gradio as gr
import numpy as np
from gradio_image_annotation import image_annotator

from ..errors import EvpError
from ..utils import list_images
from . import services as S

RUNNER = S.JobRunner()
POLL_S = 0.5
CSS = ".evp-log textarea {font-family: Consolas, monospace; font-size: 12px}"


def _ui_errors(fn):
    """EvpError / ValueError -> gr.Error toast (message + hint). Works for plain and generator handlers."""
    if inspect.isgeneratorfunction(fn):
        @functools.wraps(fn)
        def gen_wrapper(*a, **k):
            try:
                yield from fn(*a, **k)
            except (EvpError, ValueError) as exc:
                raise gr.Error(str(exc), duration=12) from exc
        return gen_wrapper

    @functools.wraps(fn)
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except (EvpError, ValueError) as exc:
            raise gr.Error(str(exc), duration=12) from exc
    return wrapper


def _rgb(img_bgr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img_bgr[..., ::-1])


def _thumbs(folder: str | Path, limit: int = 24, side: int = 160) -> list[tuple[np.ndarray, str]]:
    import cv2

    from ..deploy.runner import read_image

    d = Path(folder)
    if not d.is_dir():
        return []
    files = sorted(list_images(d), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    out = []
    for p in files:
        try:
            img = read_image(p)
        except EvpError:
            continue
        s = side / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
        out.append((_rgb(img), p.name))
    return out


def _annotation(cfg: dict, stem: str) -> dict:
    """Annotator value with the image decoded to RGB numpy (no file serving / allowed_paths needed)."""
    from ..deploy.runner import read_image

    v = S.load_annotation(cfg["paths"]["labelme_dir"], stem, cfg["classes"])
    v["image"] = _rgb(read_image(Path(v["image"])))
    return v


def _stream(job: S.Job, extra=None):
    """Yield the log (or (log, *extra(job))) every POLL_S until the job ends; raise gr.Error if it failed."""
    def snap():
        return (job.log_text(), *extra(job)) if extra else job.log_text()

    while job.running:
        yield snap()
        time.sleep(POLL_S)
    yield snap()
    if job.error:
        raise gr.Error(f"{job.name} 失败: {job.error}", duration=15)


def build_app(config_path: str | None = None) -> gr.Blocks:
    with gr.Blocks(title="EVP 边缘视觉工作台") as demo:
        gr.Markdown("## EVP 边缘视觉工作台　·　采集 → 标注 → 训练 → 一键部署（ncnn）→ 推理与性能")
        with gr.Row():
            cfg_box = gr.Textbox(value=config_path or "", label="项目配置 project.yaml", scale=5)
            load_btn = gr.Button("加载 / 刷新", variant="primary", scale=1)
        status = gr.Markdown("_未加载项目_")
        with gr.Accordion("新建项目", open=not config_path):
            with gr.Row():
                new_dir = gr.Textbox(label="项目目录", placeholder="my_line", scale=3)
                new_cls = gr.Textbox(label="类别（逗号分隔）", placeholder="scratch, missing, short", scale=3)
                new_btn = gr.Button("创建", scale=1)

        with gr.Tabs():
            # ---------------------------------------------------------- collect
            with gr.Tab("① 图像采集"):
                with gr.Row():
                    dedup = gr.Slider(-1, 16, value=-1, step=1,
                                      label="近重复过滤 dHash 距离（-1 关闭；视频/相机建议 4）")
                    sharp = gr.Number(value=0, label="清晰度阈值（Laplacian 方差，0 关闭）")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**文件夹导入**")
                        folder = gr.Textbox(label="图片文件夹（含子目录）")
                        folder_btn = gr.Button("导入")
                    with gr.Column():
                        gr.Markdown("**视频抽帧**")
                        video = gr.File(label="视频文件", file_types=["video"], type="filepath")
                        with gr.Row():
                            every = gr.Number(value=1.0, label="间隔（秒）")
                            maxf = gr.Number(value=0, label="最多帧数（0 不限）", precision=0)
                        video_btn = gr.Button("抽帧")
                    with gr.Column():
                        gr.Markdown("**相机采集**")
                        cam_src = gr.Textbox(value="0", label="本机相机序号或 RTSP 地址")
                        with gr.Row():
                            cam_n = gr.Number(value=10, label="张数", precision=0)
                            cam_dt = gr.Number(value=1.0, label="间隔（秒）")
                        cam_btn = gr.Button("开始采集")
                with gr.Accordion("浏览器摄像头快照", open=False):
                    snap = gr.Image(sources=["webcam"], type="numpy", label="拍照后点保存")
                    snap_btn = gr.Button("保存快照")
                collect_out = gr.JSON(label="采集结果")
                raw_gallery = gr.Gallery(label="待标注图片（最新 24 张）", columns=8, height=260)

            # ---------------------------------------------------------- label
            with gr.Tab("② 预标注与审核"):
                with gr.Row():
                    pl_model = gr.Dropdown(label="预标注模型（.pt 或部署包）", choices=[], allow_custom_value=True,
                                           scale=4)
                    pl_conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="置信度", scale=2)
                    pl_btn = gr.Button("模型预标注", variant="primary", scale=1)
                    blank_btn = gr.Button("无模型：空白标注", scale=1)
                pl_log = gr.Textbox(label="预标注日志", lines=4, max_lines=8, autoscroll=True, elem_classes="evp-log")
                with gr.Row():
                    item = gr.Dropdown(label="图片", choices=[], allow_custom_value=True, scale=4)
                    prev_btn = gr.Button("◀ 上一张", scale=1)
                    next_btn = gr.Button("下一张 ▶", scale=1)
                progress = gr.Markdown("")
                annot = image_annotator(label_list=["object"], label="拖拽画框；点击框可改类别；Delete 删除框",
                                        sources=[], height=560, use_default_label=True)
                with gr.Row():
                    save_btn = gr.Button("保存", variant="primary")
                    save_next_btn = gr.Button("保存并下一张", variant="primary")
                    del_btn = gr.Button("从本批删除此图", variant="stop")
                gr.Markdown("审核也可在 LabelMe 中完成（`evp review`），两边读写同一份 LabelMe json。")

            # ---------------------------------------------------------- train
            with gr.Tab("③ 入库与训练"):
                with gr.Row():
                    allow_unrev = gr.Checkbox(label="允许包含未在工作台审核的图片（例如已在 LabelMe 中审核）")
                    keep_stage = gr.Checkbox(label="入库后保留暂存目录")
                    ingest_btn = gr.Button("入库（json→YOLO→划分→合并）", variant="primary")
                ingest_out = gr.JSON(label="入库结果")
                with gr.Row():
                    base_w = gr.Textbox(value="", label="基础权重（空 = 配置 train.base_weights）")
                    epochs = gr.Number(value=50, label="epochs", precision=0)
                    t_imgsz = gr.Number(value=640, label="imgsz", precision=0)
                    t_batch = gr.Number(value=16, label="batch", precision=0)
                    t_dev = gr.Dropdown(["auto", "cpu", "0"], value="auto", label="设备")
                train_btn = gr.Button("开始训练", variant="primary")
                train_log = gr.Textbox(label="训练日志", lines=16, max_lines=16, autoscroll=True,
                                       elem_classes="evp-log")

            # ---------------------------------------------------------- deploy
            with gr.Tab("④ 一键部署"):
                with gr.Row():
                    d_weights = gr.Dropdown(label="权重（auto = 最近一次训练 best.pt）", choices=["auto"], value="auto",
                                            allow_custom_value=True, scale=4)
                    d_imgsz = gr.Number(value=320, label="部署输入尺寸（32 的倍数）", precision=0, scale=1)
                    d_name = gr.Textbox(label="部署包名（空 = 权重名）", scale=2)
                with gr.Row():
                    d_quant = gr.Checkbox(value=True, label="int8 量化（KL/ACIQ 自动择优 + 自动混合精度）")
                    d_verify = gr.Checkbox(value=True, label="三级验证（L1 张量 / L2 int8 / L3 端到端）")
                    d_e2e = gr.Checkbox(value=False, label="YOLO26 端到端头（免 NMS）")
                deploy_btn = gr.Button("一键部署", variant="primary")
                deploy_log = gr.Textbox(label="部署日志", lines=14, max_lines=14, autoscroll=True,
                                        elem_classes="evp-log")
                verdict = gr.Markdown("")
                with gr.Accordion("验证报告 report.md", open=False):
                    report_md = gr.Markdown("")

            # ---------------------------------------------------------- infer / bench
            with gr.Tab("⑤ 推理与性能"):
                with gr.Row():
                    bundle = gr.Dropdown(label="部署包", choices=[], allow_custom_value=True, scale=3)
                    prec = gr.Radio(["fp32", "int8"], value="int8", label="精度", scale=1)
                    threads = gr.Slider(1, 8, value=1, step=1, label="线程数", scale=1)
                    i_conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="置信度", scale=1)
                with gr.Row():
                    in_img = gr.Image(type="numpy", label="输入图片", sources=["upload", "clipboard", "webcam"])
                    out_img = gr.Image(type="numpy", label="检测结果")
                with gr.Row():
                    infer_btn = gr.Button("推理", variant="primary")
                    latency = gr.Markdown("")
                dets = gr.Dataframe(label="检测框", interactive=False)
                gr.Markdown("**性能测试**（Python ncnn 参考运行时，在本机测量；板端数据以 C++ runtime 实测为准）")
                with gr.Row():
                    b_threads = gr.CheckboxGroup(["1", "2", "3", "4"], value=["1", "2"], label="线程数")
                    b_prec = gr.CheckboxGroup(["fp32", "int8"], value=["fp32", "int8"], label="精度")
                    b_runs = gr.Number(value=20, label="每组次数", precision=0)
                    bench_btn = gr.Button("运行性能测试")
                bench_tab = gr.Dataframe(label="延迟（毫秒，含前后处理）", interactive=False)
                files_tab = gr.Dataframe(label="部署包文件", interactive=False)

        # ============================================================== handlers
        @_ui_errors
        def on_load(path):
            cfg = S.load_project(path)
            names = list(cfg["classes"])
            weights = [str(w) for w in S.list_weights(cfg)]
            bundles = [str(b) for b in S.list_bundles(cfg["paths"]["deploy_out"])]
            items = S.review_items(cfg["paths"]["labelme_dir"])
            first = _annotation(cfg, items[0]) if items else None
            return (S.status_markdown(cfg),
                    gr.update(choices=weights + bundles, value=(weights + bundles or [None])[0]),
                    gr.update(choices=items, value=items[0] if items else None),
                    image_annotator(value=first, label_list=names, label_colors=S.class_colors(names),
                                    use_default_label=True),
                    gr.update(choices=["auto"] + weights),
                    gr.update(choices=bundles, value=bundles[0] if bundles else None),
                    _thumbs(cfg["paths"]["raw_images"]))

        load_outputs = [status, pl_model, item, annot, d_weights, bundle, raw_gallery]
        load_btn.click(on_load, cfg_box, load_outputs)
        if config_path:
            demo.load(on_load, cfg_box, load_outputs)

        @_ui_errors
        def on_create(d, classes):
            if not str(d or "").strip():
                raise gr.Error("请填写项目目录")
            path = S.create_project(d, S.parse_classes(classes))
            gr.Info(f"已创建 {path}")
            return str(path)

        new_btn.click(on_create, [new_dir, new_cls], cfg_box).success(on_load, cfg_box, load_outputs)

        # ---- collect
        def _collector(path, dd, sh):
            from ..data.collect import collector_from_config

            cfg = S.load_project(path)
            return cfg, collector_from_config(cfg, int(dd), float(sh or 0))

        def _after_collect(cfg, stats):
            return stats.as_dict(), _thumbs(cfg["paths"]["raw_images"]), S.status_markdown(cfg)

        @_ui_errors
        def on_folder(path, dd, sh, src):
            from ..data.collect import import_folder

            if not str(src or "").strip():
                raise gr.Error("请填写图片文件夹")
            cfg, col = _collector(path, dd, sh)
            return _after_collect(cfg, import_folder(Path(src.strip().strip('"')), col))

        @_ui_errors
        def on_video(path, dd, sh, f, ev, mx):
            from ..data.collect import extract_video

            if not f:
                raise gr.Error("请先上传视频")
            cfg, col = _collector(path, dd, sh)
            return _after_collect(cfg, extract_video(Path(f), col, float(ev or 1.0), int(mx or 0)))

        @_ui_errors
        def on_camera(path, dd, sh, src, n, dt):
            from ..data.collect import capture_camera

            cfg, col = _collector(path, dd, sh)
            s = str(src).strip()
            return _after_collect(cfg, capture_camera(col, int(s) if s.isdigit() else s, int(n or 1), float(dt or 0),
                                                      timeout=max(30.0, float(n or 1) * (float(dt or 0) + 2) * 3)))

        @_ui_errors
        def on_snap(path, dd, sh, img):
            from ..data.collect import save_frames

            if img is None:
                raise gr.Error("请先拍照")
            cfg, col = _collector(path, dd, sh)
            return _after_collect(cfg, save_frames([np.ascontiguousarray(img[..., ::-1])], col, "webcam"))

        col_out = [collect_out, raw_gallery, status]
        folder_btn.click(on_folder, [cfg_box, dedup, sharp, folder], col_out)
        video_btn.click(on_video, [cfg_box, dedup, sharp, video, every, maxf], col_out)
        cam_btn.click(on_camera, [cfg_box, dedup, sharp, cam_src, cam_n, cam_dt], col_out)
        snap_btn.click(on_snap, [cfg_box, dedup, sharp, snap], col_out)

        # ---- prelabel / review
        def _progress(cfg, stem):
            items = S.review_items(cfg["paths"]["labelme_dir"])
            done = S.load_reviewed(Path(cfg["paths"]["labelme_dir"]))
            idx = items.index(stem) + 1 if stem in items else 0
            mark = "✅ 已审核" if stem in done else "⏳ 未审核"
            return f"第 {idx}/{len(items)} 张　{mark}　|　已审核 {sum(1 for i in items if i in done)}/{len(items)}"

        @_ui_errors
        def on_prelabel(path, model, conf):
            from ..data.prelabel import prelabel

            cfg = S.load_project(path)
            if not model:
                raise gr.Error("请选择预标注模型（还没有模型时用“空白标注”）")
            p, pl = cfg["paths"], cfg["prelabel"]
            m = Path(str(model).strip().strip('"'))
            is_bundle = m.is_dir()

            def work():
                return prelabel(Path(p["raw_images"]), Path(p["labelme_dir"]), list(cfg["classes"]),
                                model=None if is_bundle else m, bundle=m if is_bundle else None, conf=float(conf),
                                iou=float(pl["iou"]), imgsz=int(pl["imgsz"]), device=str(pl["device"]),
                                precision="fp32")

            yield from _stream(RUNNER.start("prelabel", work))

        @_ui_errors
        def on_blank(path):
            cfg = S.load_project(path)
            n = S.stage_for_manual_labeling(cfg["paths"]["raw_images"], cfg["paths"]["labelme_dir"])
            return f"已为 {n} 张图片创建空白标注，请在下方逐张画框并保存"

        def _refresh_items(path):
            cfg = S.load_project(path)
            items = S.review_items(cfg["paths"]["labelme_dir"])
            return gr.update(choices=items, value=items[0] if items else None), S.status_markdown(cfg)

        pl_btn.click(on_prelabel, [cfg_box, pl_model, pl_conf], [pl_log]).then(
            _ui_errors(_refresh_items), cfg_box, [item, status])
        blank_btn.click(on_blank, cfg_box, pl_log).success(_ui_errors(_refresh_items), cfg_box, [item, status])

        @_ui_errors
        def on_item(path, stem):
            if not stem:
                return None, ""
            cfg = S.load_project(path)
            return _annotation(cfg, stem), _progress(cfg, stem)

        item.change(on_item, [cfg_box, item], [annot, progress])

        def _step(path, stem, delta):
            cfg = S.load_project(path)
            items = S.review_items(cfg["paths"]["labelme_dir"])
            if not items:
                return gr.update(choices=[], value=None)
            i = items.index(stem) if stem in items else -delta
            return gr.update(choices=items, value=items[max(0, min(len(items) - 1, i + delta))])

        prev_btn.click(_ui_errors(lambda p, s: _step(p, s, -1)), [cfg_box, item], item)
        next_btn.click(_ui_errors(lambda p, s: _step(p, s, 1)), [cfg_box, item], item)

        @_ui_errors
        def on_save(path, stem, value):
            if not stem:
                raise gr.Error("没有选中的图片")
            cfg = S.load_project(path)
            r = S.save_annotation(cfg["paths"]["labelme_dir"], stem, value, cfg["classes"])
            gr.Info(f"已保存 {stem}：{r['boxes']} 个框")
            return _progress(cfg, stem), S.status_markdown(cfg)

        save_btn.click(on_save, [cfg_box, item, annot], [progress, status])
        save_next_btn.click(on_save, [cfg_box, item, annot], [progress, status]).success(
            _ui_errors(lambda p, s: _step(p, s, 1)), [cfg_box, item], item)

        @_ui_errors
        def on_delete(path, stem):
            if not stem:
                raise gr.Error("没有选中的图片")
            cfg = S.load_project(path)
            S.delete_review_item(cfg["paths"]["labelme_dir"], stem)
            items = S.review_items(cfg["paths"]["labelme_dir"])
            return gr.update(choices=items, value=items[0] if items else None), S.status_markdown(cfg)

        del_btn.click(on_delete, [cfg_box, item], [item, status])

        # ---- ingest / train
        @_ui_errors
        def on_ingest(path, allow, keep):
            from ..data.ingest import ingest

            cfg = S.load_project(path)
            st = S.project_status(cfg)
            unrev = st["to_review"] - st["reviewed"]
            if unrev and not allow:
                raise gr.Error(f"还有 {unrev} 张未在工作台审核。请先审核，或勾选“允许包含未审核图片”。")
            res = ingest(cfg, keep_staging=bool(keep))
            return res, S.status_markdown(S.load_project(path)), gr.update(choices=[], value=None)

        ingest_btn.click(on_ingest, [cfg_box, allow_unrev, keep_stage], [ingest_out, status, item])

        @_ui_errors
        def on_train(path, bw, ep, sz, bs, dev):
            from ..train.trainer import train

            over = {"train": {"epochs": int(ep), "imgsz": int(sz), "batch": int(bs), "device": dev}}
            if str(bw or "").strip():
                over["train"]["base_weights"] = str(bw).strip().strip('"')
            cfg = S.load_project(path, over)

            def work():
                return train(Path(cfg["paths"]["corpus_root"]), Path(cfg["paths"]["runs"]), cfg["train"])

            yield from _stream(RUNNER.start("train", work))

        train_btn.click(on_train, [cfg_box, base_w, epochs, t_imgsz, t_batch, t_dev], train_log).then(
            _ui_errors(on_load), cfg_box, load_outputs)

        # ---- deploy
        def _verdict(job: S.Job) -> tuple:
            if job.running or not job.result:
                return ("_部署中…_" if job.running else "", "")
            r = job.result
            flag = {True: "✅ 通过", False: "❌ 未通过", None: "⚠️ 未验证"}
            md = (f"**部署包** `{r['bundle']}`\n\n**fp32** {flag[r.get('passed')]}　|　"
                  f"**int8** {flag[r.get('passed_int8')]}　|　**推荐精度** `{r.get('recommended_precision')}`")
            return md, S.read_report(r["bundle"])

        @_ui_errors
        def on_deploy(path, w, sz, name, q, v, e2e):
            from ..deploy.pipeline import deploy

            sz = int(sz)
            if sz <= 0 or sz % 32:
                raise gr.Error("部署输入尺寸需为 32 的正整数倍，例如 256 / 320 / 416")
            over = {"deploy": {"imgsz": [sz, sz], "e2e": bool(e2e), "name": str(name or "").strip()}}
            cfg = S.load_project(path, over)
            weights = None if str(w or "auto").strip() in ("", "auto") else str(w).strip().strip('"')

            def work():
                return deploy(cfg, weights=weights, skip_quant=not q, skip_verify=not v)

            yield from _stream(RUNNER.start("deploy", work), _verdict)

        deploy_btn.click(on_deploy, [cfg_box, d_weights, d_imgsz, d_name, d_quant, d_verify, d_e2e],
                         [deploy_log, verdict, report_md]).then(_ui_errors(on_load), cfg_box, load_outputs)

        # ---- infer / benchmark
        @_ui_errors
        def on_infer(b, p, t, c, img):
            if not b:
                raise gr.Error("请选择部署包")
            if img is None:
                raise gr.Error("请上传图片")
            vis, rows, ms = S.infer_image(b, np.ascontiguousarray(img[..., ::-1]), p, int(t), float(c))
            return _rgb(vis), rows, f"**{p} / {int(t)} 线程：{ms:.1f} ms**（含前后处理，{len(rows)} 个结果）"

        infer_btn.click(on_infer, [bundle, prec, threads, i_conf, in_img], [out_img, dets, latency])

        @_ui_errors
        def on_bench(path, b, th, pr, runs):
            from ..deploy.runner import read_image

            if not b:
                raise gr.Error("请选择部署包")
            if not th or not pr:
                raise gr.Error("请至少选择一个线程数和一种精度")
            cfg = S.load_project(path)
            imgs = [read_image(p) for p in S.sample_images(cfg)]
            if not imgs:
                raise gr.Error("项目里没有可用于测试的图片（训练集/验证集/待标注目录都为空）")
            rows = S.benchmark(b, imgs, list(pr), [int(x) for x in th], runs=int(runs or 20))
            return rows, S.bundle_files(b)

        bench_btn.click(on_bench, [cfg_box, bundle, b_threads, b_prec, b_runs], [bench_tab, files_tab])
        bundle.change(_ui_errors(lambda b: S.bundle_files(b) if b else []), bundle, files_tab)
    return demo


def bypass_proxy_for_localhost() -> None:
    """Gradio probes its own server over HTTP at startup; a system proxy (e.g. Windows registry proxy picked up by
    httpx) turns that into a 502. Adds localhost to NO_PROXY for this process only."""
    import os

    hosts = ["localhost", "127.0.0.1", "::1"]
    for key in ("NO_PROXY", "no_proxy"):
        cur = [h.strip() for h in os.environ.get(key, "").split(",") if h.strip()]
        os.environ[key] = ",".join(cur + [h for h in hosts if h not in cur])


def launch(config_path: str | None = None, host: str = "127.0.0.1", port: int = 7860, inbrowser: bool = False,
           prevent_thread_lock: bool = False) -> gr.Blocks:
    """Start the workbench. Binds to localhost by default: the UI can read and write any path the user enters."""
    bypass_proxy_for_localhost()
    demo = build_app(config_path)
    demo.queue(default_concurrency_limit=4)
    demo.launch(server_name=host, server_port=int(port), inbrowser=inbrowser, theme=gr.themes.Soft(), css=CSS,
                prevent_thread_lock=prevent_thread_lock)
    return demo
