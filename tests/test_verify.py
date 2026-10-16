#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
import pytest

from evp.deploy.postprocess import Detection
from evp.deploy.verify import box_iou, cosine, finalize_verdict, match_detections, render_markdown, write_report


def D(x1, y1, x2, y2, s=0.9, c=0):
    return Detection(x1, y1, x2, y2, s, c)


def test_box_iou():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert box_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    assert box_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_cosine_edge_cases():
    assert cosine([1, 0], [1, 0]) == 1.0
    assert cosine([1, 0], [1, 0, 0]) == -1.0
    assert cosine([0, 0], [0, 0]) == 1.0
    assert cosine([0, 0], [1, 0]) == 0.0


def test_match_detections():
    a = [D(0, 0, 10, 10, 0.9, 0), D(20, 20, 30, 30, 0.8, 1), D(50, 50, 60, 60, 0.5, 0)]
    b = [D(0, 0, 10, 11, 0.85, 0), D(20, 20, 30, 30, 0.8, 0)]   # second has wrong class
    m = match_detections(a, b, 0.5)
    assert (m.matched, m.n_a, m.n_b) == (1, 3, 2)
    assert m.f1 == pytest.approx(2 / 5)
    assert m.max_score_diff == pytest.approx(0.05)
    assert match_detections([], []).f1 == 1.0
    assert match_detections(a, []).f1 == 0.0


def test_match_is_one_to_one_highest_iou_first():
    a = [D(0, 0, 10, 10)]
    b = [D(0, 0, 10, 9), D(0, 0, 10, 10)]
    m = match_detections(a, b, 0.5)
    assert m.matched == 1 and m.mean_iou == 1.0


def chk(passed, group="fp32", gate=True):
    value = None if passed is None else (1.0 if passed else 0.0)
    return {"value": value, "threshold": 0.5, "passed": passed, "group": group, "gate": gate}


@pytest.mark.parametrize("checks, has_int8, want", [
    ({"a": chk(True), "b": chk(True, "int8")}, True, (True, True, "int8")),
    ({"a": chk(True), "b": chk(False, "int8")}, True, (True, False, "fp32")),
    # int8 needs at least one measured gated check; informational ones alone do not verify it
    ({"a": chk(True), "b": chk(False, "int8", gate=False)}, True, (True, False, "fp32")),
    ({"a": chk(True), "b": chk(True, "int8"), "i": chk(False, "int8", gate=False)}, True, (True, True, "int8")),
    ({"a": chk(False), "b": chk(True, "int8")}, True, (False, False, "fp32")),
    ({"a": chk(True), "c": chk(False, "cpp_fp32")}, True, (False, False, "fp32")),
    ({"a": chk(True), "c": chk(False, "cpp_int8")}, True, (True, False, "fp32")),
    ({"a": chk(True)}, False, (True, None, "fp32")),
    ({"b": chk(True, "int8")}, True, (False, False, "fp32")),           # no fp32 gate -> unverified
    ({"a": chk(True, gate=False)}, False, (False, None, "fp32")),
])
def test_finalize_verdict(checks, has_int8, want):
    r = finalize_verdict({"checks": checks, "images_used": 3}, has_int8)
    assert (r["passed_fp32"], r["passed_int8"], r["recommended_precision"]) == want
    assert r["passed"] == r["passed_fp32"]
    if not any(c["group"] == "fp32" and c["gate"] for c in checks.values()):
        assert any("unverified" in n for n in r["notes"])


def test_finalize_unmeasured_checks_never_pass():
    # the smoke-test case: weak model, nothing detected -> L2 and L3 unmeasured, only L1 measured
    checks = {"L1": chk(True), "L3": chk(None), "L2": chk(None, "int8"), "L3i": chk(None, "int8", gate=False)}
    r = finalize_verdict({"checks": checks, "images_used": 4}, True)
    assert (r["passed_fp32"], r["passed_int8"], r["recommended_precision"]) == (True, False, "fp32")
    assert r["unverified"] == ["int8"] and any("int8 could not be verified" in n for n in r["notes"])
    md = render_markdown(r)
    assert "int8: **UNVERIFIED**" in md and "| L2 | n/a | 0.5 | UNMEASURED |" in md
    finalize_verdict(r, True)
    assert sum("int8 could not be verified" in n for n in r["notes"]) == 1


def test_finalize_unmeasured_does_not_hide_failures():
    r = finalize_verdict({"checks": {"L1": chk(True), "L3": chk(None), "L2": chk(False, "int8")},
                          "images_used": 2}, True)
    assert (r["passed_fp32"], r["passed_int8"], r["unverified"]) == (True, False, [])
    r = finalize_verdict({"checks": {"L1": chk(False), "L3": chk(None)}, "images_used": 2}, False)
    assert r["passed_fp32"] is False


def test_finalize_only_unmeasured_fp32_is_unverified():
    r = finalize_verdict({"checks": {"L3": chk(None)}, "images_used": 2}, False)
    assert r["passed_fp32"] is False and any("fp32 conversion is unverified" in n for n in r["notes"])


def test_finalize_without_images_fails():
    r = finalize_verdict({"checks": {"a": chk(True)}, "images_used": 0}, True)
    assert r["passed_fp32"] is False and r["passed_int8"] is False


def test_finalize_is_idempotent_on_notes():
    r = {"checks": {}, "images_used": 1}
    finalize_verdict(r, False)
    finalize_verdict(r, False)
    assert len(r["notes"]) == 1


def test_render_and_write_report(tmp_path):
    r = finalize_verdict({"model": "m", "family": "yolo", "images_used": 2,
                          "checks": {"L1": chk(True), "L2": chk(True, "int8"),
                                     "L3_int8": chk(False, "int8", gate=False)},
                          "host_latency_ms": {"fp32": 12.3}, "notes": ["hello"]}, has_int8=True)
    md = render_markdown({**r, "quant": {"method": "aciq", "calib_count": 64, "cos_full_int8": 0.9, "cos_final": 0.98,
                                         "target_cos": 0.97, "target_met": True, "fp32_layers": ["conv_9"]},
                          "convert": {"layer_count": 10, "layer_types": {"Convolution": 5}, "rules_applied": []},
                          "cpp": {"passed": True, "summary": "ok"}})
    for s in ("# Deploy report: m", "fp32: **PASS**", "int8: **PASS**", "FAIL (info)", "conv_9", "Convolutionx5",
              "## C++ runtime", "- hello", "fp32 12.3 ms"):
        assert s in md, s
    assert "int8: **not built**" in render_markdown({"passed_fp32": True, "passed_int8": None})
    jp, mp = write_report(r, tmp_path, {"extra": 1})
    assert jp.is_file() and mp.read_text(encoding="utf-8").startswith("# Deploy report")
