#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
"""Add SPDX license headers (idempotent): AGPL-3.0-only for the Python toolchain and scripts,
Apache-2.0 for the C++ runtime under runtime/cpp (it is copied into deploy bundles and onto boards).

    python scripts/add_spdx.py          # add missing headers
    python scripts/add_spdx.py --check  # exit 1 if any file lacks one (CI / pre-commit)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "SPDX-License-Identifier"
COPYRIGHT = "Copyright (c) 2026 edge_vision_pipeline contributors"

# (glob, license, comment prefix)
GROUPS = [
    ("evp/**/*.py", "AGPL-3.0-only", "#"),
    ("tests/**/*.py", "AGPL-3.0-only", "#"),
    ("scripts/*.py", "AGPL-3.0-only", "#"),
    ("scripts/*.sh", "AGPL-3.0-only", "#"),
    ("scripts/*.ps1", "AGPL-3.0-only", "#"),
    ("runtime/cpp/**/*.cpp", "Apache-2.0", "//"),
    ("runtime/cpp/**/*.h", "Apache-2.0", "//"),
    ("runtime/cpp/CMakeLists.txt", "Apache-2.0", "#"),
]


def _header_end(lines: list[str]) -> int:
    """Index after a shebang / coding cookie, which must stay on the first two lines."""
    i = 0
    if lines and lines[0].startswith("#!"):
        i = 1
    if len(lines) > i and lines[i].startswith("#") and "coding" in lines[i]:
        i += 1
    return i


def process(path: Path, lic: str, prefix: str, check: bool) -> bool:
    """True if the file has (or now has) a header."""
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    if TAG in "".join(text.splitlines(keepends=True)[:6]):
        return True
    if check:
        return False
    nl = "\r\n" if b"\r\n" in raw else "\n"
    lines = text.splitlines(keepends=True)
    at = _header_end(lines)
    hdr = [f"{prefix} {TAG}: {lic}{nl}", f"{prefix} {COPYRIGHT}{nl}"]
    path.write_bytes("".join(lines[:at] + hdr + lines[at:]).encode("utf-8"))
    print(f"added {lic}: {path.relative_to(ROOT)}")
    return True


def main() -> int:
    check = "--check" in sys.argv
    missing = []
    for pattern, lic, prefix in GROUPS:
        for p in sorted(ROOT.glob(pattern)):
            if "__pycache__" in p.parts or not p.is_file():
                continue
            if not process(p, lic, prefix, check):
                missing.append(str(p.relative_to(ROOT)))
    if missing:
        print("missing SPDX header:\n  " + "\n  ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
