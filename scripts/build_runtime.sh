#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
# Build the C++ runtime (evp_demo + evp_unit_tests) against an ncnn install.
#
#   scripts/build_runtime.sh                 # host build, ncnn from $EVP_TOOLS_HOME/ncnn-install
#   NCNN_DIR=/path/ncnn-install BUILD_DIR=/tmp/b scripts/build_runtime.sh
#   TOOLCHAIN=/path/arm-linux-gnueabihf.toolchain.cmake NCNN_DIR=/path/ncnn-armv7 scripts/build_runtime.sh
#
# Prints the path of evp_demo on the last line.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/../runtime/cpp"
EVP_TOOLS_HOME="${EVP_TOOLS_HOME:-$HOME/evp_deps}"
NCNN_DIR="${NCNN_DIR:-$EVP_TOOLS_HOME/ncnn-install}"
BUILD_DIR="${BUILD_DIR:-$EVP_TOOLS_HOME/evp_runtime_build}"

if [ ! -d "$NCNN_DIR/lib/cmake/ncnn" ]; then
    echo "ncnn cmake package not found under $NCNN_DIR/lib/cmake/ncnn" >&2
    echo "run scripts/setup_tools_linux.sh first or set NCNN_DIR" >&2
    exit 1
fi

ARGS=(-S "$SRC" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -Dncnn_DIR="$NCNN_DIR/lib/cmake/ncnn")
if [ -n "${TOOLCHAIN:-}" ]; then
    ARGS+=(-DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" -DEVP_BUILD_TESTS=OFF)
fi
cmake "${ARGS[@]}" >/dev/null
cmake --build "$BUILD_DIR" -j"$(nproc)" >/dev/null
if [ -z "${TOOLCHAIN:-}" ]; then
    "$BUILD_DIR/evp_unit_tests" >&2
fi
echo "$BUILD_DIR/evp_demo"
