#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Normalize line endings and encoding of the repo's text files (mirrors .gitattributes).

LF everywhere except *.ps1 (CRLF); UTF-8 without BOM. Editors on Windows tend to write CRLF, and a CRLF
shell script breaks bash in WSL, so run this after editing:

    python scripts/normalize_eol.py          # fix in place
    python scripts/normalize_eol.py --check  # exit 1 if anything would change (CI / pre-commit)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_EXT = {".py", ".cpp", ".h", ".sh", ".md", ".yaml", ".yml", ".toml", ".txt", ".cfg", ".ps1", ".meta"}
TEXT_NAMES = {"CMakeLists.txt", ".gitattributes", ".gitignore", "LICENSE", "NOTICE"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "scratch", "models", "workspace", "build"}
BOM = b"\xef\xbb\xbf"


def wanted(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
        return False
    return path.suffix.lower() in TEXT_EXT or path.name in TEXT_NAMES


def normalize(data: bytes, crlf: bool) -> bytes:
    if data.startswith(BOM):
        data = data[len(BOM):]
    data = data.replace(b"\r\n", b"\n")
    return data.replace(b"\n", b"\r\n") if crlf else data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report only, exit 1 if changes are needed")
    args = ap.parse_args()
    changed = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or not wanted(p):
            continue
        data = p.read_bytes()
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            print(f"[WARN] not UTF-8, skipped: {p.relative_to(ROOT)}", file=sys.stderr)
            continue
        new = normalize(data, crlf=p.suffix.lower() == ".ps1")
        if new != data:
            changed.append(p.relative_to(ROOT))
            if not args.check:
                p.write_bytes(new)
    for c in changed:
        print(("needs fix: " if args.check else "fixed: ") + str(c))
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
