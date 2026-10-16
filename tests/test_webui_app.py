#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Workbench UI wiring: a real Gradio server driven through gradio_client (same endpoints the browser uses)."""

import json
import shutil
import socket
from pathlib import Path

import pytest

pytest.importorskip("gradio")
pytest.importorskip("gradio_image_annotation")

from gradio_client import Client, handle_file  # noqa: E402

from tests.conftest import PRETRAINED, needs_weights, write_image  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def client():
    from evp.webui.app import build_app, bypass_proxy_for_localhost

    bypass_proxy_for_localhost()
    port = _free_port()
    demo = build_app(None)
    demo.queue()
    demo.launch(server_port=port, prevent_thread_lock=True, quiet=True)
    try:
        yield Client(f"http://127.0.0.1:{port}/", verbose=False)
    finally:
        demo.close()


def test_label_flow_through_ui(client, tmp_path: Path):
    for i in range(3):
        write_image(tmp_path / "src" / f"a{i}.jpg", 120, 90, seed=i)
    p = client.predict(str(tmp_path / "proj"), "scratch, dent", api_name="/on_create")
    assert Path(p).is_file()

    res = client.predict(p, -1, 0, str(tmp_path / "src"), api_name="/on_folder")
    assert res[0]["saved"] == 3
    res = client.predict(p, -1, 0, str(tmp_path / "src"), api_name="/on_folder")
    assert (res[0]["saved"], res[0]["duplicates"]) == (0, 3)        # re-import is a no-op
    assert "3" in client.predict(p, api_name="/on_blank")

    value, progress = client.predict(p, "a0", api_name="/on_item")
    assert value["boxes"] == [] and "1/3" in progress
    value["image"] = handle_file(value["image"])
    value["boxes"] = [{"xmin": 10, "ymin": 10, "xmax": 60, "ymax": 50, "label": "dent", "color": "rgb(255, 0, 0)"}]
    client.predict(p, "a0", value, api_name="/on_save")
    shapes = json.loads((tmp_path / "proj" / "workspace" / "labelme" / "a0.json").read_text(encoding="utf-8"))["shapes"]
    assert [(s["label"], s["points"]) for s in shapes] == [("dent", [[10.0, 10.0], [60.0, 50.0]])]

    value["boxes"][0]["label"] = "not-a-class"
    with pytest.raises(Exception, match="not in classes"):
        client.predict(p, "a0", value, api_name="/on_save")

    with pytest.raises(Exception, match="2"):                     # two images still unreviewed
        client.predict(p, False, False, api_name="/on_ingest")
    merged = client.predict(p, True, False, api_name="/on_ingest")[0]
    assert merged["after"]["train"] + merged["after"]["val"] == 3


def test_ui_errors_are_toasts_not_crashes(client, tmp_path: Path):
    with pytest.raises(Exception, match="请填写项目目录"):
        client.predict("", "a", api_name="/on_create")
    with pytest.raises(Exception, match="duplicate"):
        client.predict(str(tmp_path / "p"), "a, a", api_name="/on_create")
    with pytest.raises(Exception, match="no project"):
        client.predict("", api_name="/on_load")
    p = client.predict(str(tmp_path / "p"), "a", api_name="/on_create")
    with pytest.raises(Exception, match="not found"):
        client.predict(p, "nope", api_name="/on_item")
    with pytest.raises(Exception, match="32"):
        client.predict(p, "auto", 300, "", True, True, False, api_name="/on_deploy")
    with pytest.raises(Exception, match="not found"):
        client.predict(p, -1, 0, str(tmp_path / "missing"), api_name="/on_folder")
    with pytest.raises(Exception, match="请选择部署包"):
        client.predict(p, "", ["1"], ["fp32"], 5, api_name="/on_bench")


@pytest.mark.integration
@pytest.mark.slow
@needs_weights("yolo11n.pt")
def test_full_pipeline_through_ui(client, tmp_path: Path, toolchain):
    """collect -> prelabel (.pt) -> ingest -> train (1 CPU epoch) -> deploy -> infer -> benchmark, all via the UI."""
    import yaml

    from tests.test_integration import IMAGES

    if IMAGES is None:
        pytest.skip("no test images")
    from evp.utils import list_images

    imgs = list_images(IMAGES, recursive=True)[:12]
    if len(imgs) < 12:
        pytest.skip("need >= 12 test images")
    src = tmp_path / "src"
    src.mkdir()
    for x in imgs:
        shutil.copy2(x, src / x.name)
    p = client.predict(str(tmp_path / "proj"), "person, car, chair", api_name="/on_create")
    cfg = yaml.safe_load(Path(p).read_text(encoding="utf-8"))
    cfg["prelabel"]["device"] = "cpu"
    cfg["train"].update({"workers": 0, "extra": {"plots": False}})
    cfg["deploy"].update({"cpp_runtime": False, "quant": {"calib_count": 8, "eval_count": 4}, "verify": {"count": 4}})
    Path(p).write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    assert client.predict(p, -1, 0, str(src), api_name="/on_folder")[0]["saved"] == 12
    log = client.predict(p, str(PRETRAINED / "yolo11n.pt"), 0.25, api_name="/on_prelabel")
    assert "wrote 12 json" in log
    merged = client.predict(p, True, False, api_name="/on_ingest")[0]
    assert merged["after"]["val"] >= 1

    log = client.predict(p, str(PRETRAINED / "yolo11n.pt"), 1, 160, 8, "auto", api_name="/on_train")
    assert "train done" in log
    log, verdict, report = client.predict(p, "auto", 160, "ui_demo", True, True, False, api_name="/on_deploy")
    assert "bundle ready" in log and "ui_demo" in verdict and "#" in report
    bundle = Path(cfg["paths"]["deploy_out"]) if Path(cfg["paths"]["deploy_out"]).is_absolute() else \
        Path(p).parent / cfg["paths"]["deploy_out"]
    bundle = bundle / "ui_demo"
    assert (bundle / "meta.json").is_file()

    out_img, dets, latency = client.predict(str(bundle), "fp32", 1, 0.25, handle_file(str(imgs[0])),
                                            api_name="/on_infer")
    assert Path(out_img).is_file() and "ms" in latency
    bench, files = client.predict(p, str(bundle), ["1", "2"], ["fp32", "int8"], 3, api_name="/on_bench")
    assert len(bench["data"]) >= 2 and files["data"]
