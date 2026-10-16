#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
import copy
from pathlib import Path

import pytest

from evp.deploy.meta import (
    InputSpec,
    MetaError,
    ModelFile,
    ModelMeta,
    OutputSpec,
    PostSpec,
    load_meta,
    parse_flat,
    save_meta,
    to_flat,
)


def det_meta(nc: int = 2, reg_max: int = 16, method: str = "nms") -> ModelMeta:
    c = 4 * reg_max + nc
    return ModelMeta(name="m", family="yolo", input=InputSpec(w=64, h=32),
                     outputs=[OutputSpec("out0", [c, 4, 8], 8), OutputSpec("out1", [c, 2, 4], 16)],
                     labels=[f"c{i}" for i in range(nc)], post=PostSpec(method=method), reg_max=reg_max,
                     files={"fp32": ModelFile("m.fp32.param", "m.fp32.bin")})


def cls_meta() -> ModelMeta:
    return ModelMeta(name="c", family="classify", input=InputSpec(w=224, h=224, resize="center_crop"),
                     outputs=[OutputSpec("out0", [3])], labels=["x", "y", "z"], post=PostSpec(method="topk"))


def test_valid_metas():
    det_meta().validate()
    det_meta(reg_max=1, method="e2e_topk").validate()
    cls_meta().validate()


@pytest.mark.parametrize("mutate, fragment", [
    (lambda m: setattr(m, "family", "seg"), "family"),
    (lambda m: setattr(m.input, "color", "GRAY"), "color"),
    (lambda m: setattr(m.input, "resize", "crop"), "resize"),
    (lambda m: setattr(m.input, "pad_value", 256), "pad_value"),
    (lambda m: setattr(m.input, "mean", [0, 0]), "mean"),
    (lambda m: setattr(m, "outputs", []), "no outputs"),
    (lambda m: setattr(m.outputs[1], "name", "out0"), "duplicate"),
    (lambda m: setattr(m, "labels", ["a\nb", "c"]), "label"),
    (lambda m: setattr(m.post, "method", "topk"), "nms|e2e_topk"),
    (lambda m: setattr(m.outputs[0], "shape", [65, 4, 8]), "channels"),
    (lambda m: setattr(m.outputs[0], "stride", 16), "grid"),
    (lambda m: setattr(m.outputs[0], "stride", 0), "stride"),
    (lambda m: setattr(m.post, "conf", 1.5), "post params"),
    (lambda m: setattr(m, "schema_version", 99), "schema_version"),
    (lambda m: setattr(m, "labels", []), "labels"),
])
def test_invalid_detect_meta(mutate, fragment):
    m = det_meta()
    mutate(m)
    with pytest.raises(MetaError) as ei:
        m.validate()
    assert fragment in str(ei.value)


def test_invalid_classify_meta():
    m = cls_meta()
    m.outputs[0].shape = [4]
    with pytest.raises(MetaError, match="labels"):
        m.validate()
    m = cls_meta()
    m.post.method = "nms"
    with pytest.raises(MetaError, match="topk"):
        m.validate()


def test_dict_roundtrip():
    m = det_meta()
    m.quant = {"cos_final": 0.97}
    m2 = ModelMeta.from_dict(copy.deepcopy(m.to_dict()) | {})
    assert m2 == m


def test_from_dict_malformed():
    with pytest.raises(MetaError, match="malformed"):
        ModelMeta.from_dict({"name": "x"})


def test_flat_contains_contract_keys():
    flat = parse_flat(to_flat(det_meta()))
    assert flat["family"] == "yolo" and flat["reg_max"] == "16" and flat["num_classes"] == "2"
    assert flat["output.count"] == "2" and flat["output.1.stride"] == "16" and flat["output.0.shape"] == "66,4,8"
    assert flat["file.fp32.param"] == "m.fp32.param" and flat["label.1"] == "c1"
    assert float(flat["input.norm"].split(",")[0]) == 1 / 255.0   # repr() keeps full double precision


def test_flat_labels_with_spaces_and_unicode():
    m = det_meta()
    m.labels = ["托盘 tian", "a=b"]
    flat = parse_flat(to_flat(m))
    assert flat["label.0"] == "托盘 tian" and flat["label.1"] == "a=b"


def test_parse_flat_errors():
    with pytest.raises(MetaError, match="missing '='"):
        parse_flat("a=1\nbroken\n")
    with pytest.raises(MetaError, match="duplicate"):
        parse_flat("a=1\na=2\n")
    assert parse_flat("# c\n\r\na=1\r\n") == {"a": "1"}


def test_save_and_load(tmp_path: Path):
    m = det_meta()
    jp, fp = save_meta(m, tmp_path)
    assert jp.is_file() and fp.is_file()
    assert b"\r\n" not in fp.read_bytes()
    assert (tmp_path / "labels.txt").read_text(encoding="utf-8") == "c0\nc1\n"
    assert load_meta(tmp_path) == m
    assert load_meta(jp) == m


def test_save_invalid_meta_refuses(tmp_path: Path):
    m = det_meta()
    m.post.method = "bogus"
    with pytest.raises(MetaError):
        save_meta(m, tmp_path)
    assert not (tmp_path / "meta.json").exists()
