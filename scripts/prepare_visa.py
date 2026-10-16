#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""VisA (CC BY 4.0) -> evp industrial defect-detection demo data.

  python scripts/prepare_visa.py --root <visa-extracted> --categories pcb1 pcb2 pcb3 \
      --labelme-out <project>/workspace/labelme --test-out <visa-test-out>

VisA ships pixel masks whose values are per-category defect-type ids. The value -> type table is inferred from the
single-label rows of <category>/image_anno.csv (e.g. pcb1: 1 bent, 2 melt, 3 scratch, 4 missing). Masks are turned
into boxes by evp.data.masks.mask_to_boxes. An official split is kept, by default split_csv/2cls_highshot.csv
(the supervised split: 60 defect images per category in train, 40 in test; 1cls.csv puts every defect in test and
only suits unsupervised anomaly detection):
  train -> LabelMe json + images in --labelme-out (review them in the workbench, then ingest),
  test  -> a YOLO dataset (images/test, labels/test, data.yaml) in --test-out for `evp eval --split test`.
A sample of normal images is added to both splits as negatives. Dataset: Zou et al., "SPot-the-Difference
Self-Supervised Pre-training for Anomaly Detection and Segmentation", ECCV 2022; https://github.com/amazon-science/spot-diff
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evp.data.labelme import LABELME_VERSION  # noqa: E402
from evp.data.masks import mask_to_boxes  # noqa: E402
from evp.data.merge import write_data_yaml  # noqa: E402
from evp.errors import DataError  # noqa: E402
from evp.utils import ensure_dir, log  # noqa: E402

ATTRIBUTION = """Images and masks derived from the VisA dataset (Amazon), licensed CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). Source: https://github.com/amazon-science/spot-diff
Zou, Y., Jeong, J., Pemula, L., Zhang, D., Dabeer, O. "SPot-the-Difference Self-Supervised Pre-training for
Anomaly Detection and Segmentation." ECCV 2022.
Changes: pixel masks converted to bounding boxes (scripts/prepare_visa.py in edge_vision_pipeline); a subset of
normal images sampled; files renamed.
"""


def read_rows(root: Path, category: str) -> list[dict]:
    f = root / category / "image_anno.csv"
    if not f.is_file():
        raise DataError(f"{f} not found", "--root must be the extracted VisA folder (contains pcb1/, split_csv/)")
    with f.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_split(root: Path, split_file: str = "2cls_highshot") -> dict[str, str]:
    f = root / "split_csv" / f"{split_file}.csv"
    if not f.is_file():
        raise DataError(f"{f} not found", "extract split_csv/ from the VisA tar as well")
    with f.open(encoding="utf-8", newline="") as fh:
        return {r["image"]: r["split"] for r in csv.DictReader(fh)}


def infer_value_table(root: Path, rows: list[dict]) -> dict[int, str]:
    """Mask value -> defect type, by majority over single-label anomaly rows."""
    import cv2
    import numpy as np

    votes: dict[int, Counter] = defaultdict(Counter)
    for r in rows:
        labs = [x.strip() for x in r["label"].split(",") if x.strip()]
        if not r["mask"] or len(labs) != 1:
            continue
        m = cv2.imread(str(root / r["mask"]), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise DataError(f"cannot read mask {root / r['mask']}")
        for v in np.unique(m):
            if v:
                votes[int(v)][labs[0]] += 1
    table = {v: c.most_common(1)[0][0] for v, c in votes.items()}
    if len(set(table.values())) != len(table):
        raise DataError(f"ambiguous mask value table {table}")
    return table


def complete_table(root: Path, rows: list[dict], table: dict[int, str]) -> dict[int, str]:
    """A type that never occurs alone (pcb4 'dirt') gets the one value still unassigned, if that is unambiguous."""
    import cv2
    import numpy as np

    all_types = {x.strip() for r in rows if r["mask"] for x in r["label"].split(",") if x.strip()}
    missing_types = sorted(all_types - set(table.values()))
    seen_values: set[int] = set()
    for r in rows:
        if r["mask"]:
            m = cv2.imread(str(root / r["mask"]), cv2.IMREAD_UNCHANGED)
            seen_values |= {int(v) for v in np.unique(m) if v}
    free = sorted(seen_values - set(table))
    if len(missing_types) == 1 and len(free) == 1:
        return {**table, free[0]: missing_types[0]}
    if free:
        log.warning("mask values %s have no defect type (types without a single-label example: %s); their boxes "
                    "are dropped", free, missing_types)
    return table


def labelme_json(image_name: str, w: int, h: int, boxes: list[tuple[str, int, int, int, int]]) -> dict:
    return {"version": LABELME_VERSION, "flags": {}, "imagePath": image_name, "imageData": None,
            "imageHeight": h, "imageWidth": w,
            "shapes": [{"label": lab, "points": [[x1, y1], [x2, y2]], "group_id": None, "description": "visa-mask",
                        "shape_type": "rectangle", "flags": {}} for lab, x1, y1, x2, y2 in boxes]}


def prepare(root: Path, categories: list[str], labelme_out: Path, test_out: Path, single_class: bool = False,
            normal_train: int = 60, normal_test: int = 20, merge_px: int = 15, min_side: int = 8,
            seed: int = 0, split_file: str = "2cls_highshot") -> dict:
    import cv2

    split_of = read_split(root, split_file)
    rng = random.Random(seed)
    items: list[tuple[str, dict, dict[int, str]]] = []
    for cat in categories:
        rows = read_rows(root, cat)
        table = complete_table(root, rows, infer_value_table(root, rows))
        log.info("%s: mask values %s", cat, table)
        anomalies = [r for r in rows if r["mask"]]
        normals = [r for r in rows if not r["mask"]]
        n_tr = [r for r in normals if split_of.get(r["image"]) == "train"]
        n_te = [r for r in normals if split_of.get(r["image"]) == "test"]
        rng.shuffle(n_tr)
        rng.shuffle(n_te)
        for r in anomalies + n_tr[:normal_train] + n_te[:normal_test]:
            items.append((cat, r, table))
    types = sorted({t for _, _, tb in items for t in tb.values()})
    names = ["defect"] if single_class else types
    stats = {"train": Counter(), "test": Counter(), "names": names, "dropped_boxes": 0}
    ensure_dir(labelme_out)
    test_img = ensure_dir(test_out / "images" / "test")
    test_lbl = ensure_dir(test_out / "labels" / "test")
    for cat, r, table in items:
        split = split_of.get(r["image"])
        if split not in ("train", "test"):
            raise DataError(f"{r['image']} has no train/test entry in split_csv/{split_file}.csv")
        src = root / r["image"]
        img = cv2.imread(str(src))
        if img is None:
            raise DataError(f"cannot read image {src}")
        h, w = img.shape[:2]
        boxes = []
        if r["mask"]:
            m = cv2.imread(str(root / r["mask"]), cv2.IMREAD_UNCHANGED)
            if m is None or m.shape[:2] != (h, w):
                raise DataError(f"mask missing or size mismatch for {src}")
            for b in mask_to_boxes(m, merge_px, min_side):
                if b.value not in table:
                    stats["dropped_boxes"] += 1
                    continue
                boxes.append(("defect" if single_class else table[b.value], b.x1, b.y1, b.x2, b.y2))
        kind = "a" if r["mask"] else "n"
        name = f"{cat}_{kind}{Path(r['image']).stem}.jpg"
        if split == "train":
            (labelme_out / name).write_bytes(src.read_bytes())
            (labelme_out / name).with_suffix(".json").write_text(
                json.dumps(labelme_json(name, w, h, boxes), indent=1, ensure_ascii=False), encoding="utf-8")
        else:
            (test_img / name).write_bytes(src.read_bytes())
            lines = [f"{names.index(lab)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} "
                     f"{(y2 - y1) / h:.6f}" for lab, x1, y1, x2, y2 in boxes]
            (test_lbl / f"{Path(name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                             encoding="utf-8")
        stats[split]["images"] += 1
        stats[split]["defect_images" if r["mask"] else "normal_images"] += 1
        for lab, *_ in boxes:
            stats[split][lab] += 1
    dy = write_data_yaml(test_out, names)
    text = dy.read_text(encoding="utf-8").replace("val: images/val", "val: images/test")
    dy.write_text(text + "test: images/test\n", encoding="utf-8")
    for d in (labelme_out, test_out):
        (d / "VISA_ATTRIBUTION.txt").write_text(ATTRIBUTION, encoding="utf-8")
    return {"names": names, "train": dict(stats["train"]), "test": dict(stats["test"]),
            "dropped_boxes": stats["dropped_boxes"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, help="extracted VisA folder (contains pcb1/, split_csv/)")
    ap.add_argument("--categories", nargs="+", default=["pcb1", "pcb2", "pcb3"])
    ap.add_argument("--labelme-out", required=True, help="train split as LabelMe json (e.g. the project labelme_dir)")
    ap.add_argument("--test-out", required=True, help="held-out test split as a YOLO dataset")
    ap.add_argument("--single-class", action="store_true", help="one class 'defect' instead of the defect types")
    ap.add_argument("--normal-train", type=int, default=60, help="normal images per category in train")
    ap.add_argument("--normal-test", type=int, default=20, help="normal images per category in test")
    ap.add_argument("--merge-px", type=int, default=15, help="join mask fragments closer than this (px)")
    ap.add_argument("--min-side", type=int, default=8, help="drop boxes whose longer side is below this (px)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-file", default="2cls_highshot", help="split_csv/<name>.csv to follow")
    a = ap.parse_args(argv)
    try:
        res = prepare(Path(a.root), a.categories, Path(a.labelme_out), Path(a.test_out), a.single_class,
                      a.normal_train, a.normal_test, a.merge_px, a.min_side, a.seed, a.split_file)
    except DataError as exc:
        log.error("%s", exc)
        return 2
    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
