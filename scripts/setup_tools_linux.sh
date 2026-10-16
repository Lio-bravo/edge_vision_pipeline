#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
# Build/install the native tools used by the deploy toolchain on Linux (or WSL):
#   - ncnn tools: ncnn2table, ncnn2int8, ncnnoptimize   -> $EVP_TOOLS_HOME/ncnn-install/bin
#   - pnnx (pip wheel) + ncnn python                      -> $EVP_TOOLS_HOME/venv
# Re-runnable: skips steps whose outputs already exist.
set -euo pipefail

EVP_TOOLS_HOME="${EVP_TOOLS_HOME:-$HOME/evp_deps}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
NCNN_REF="${NCNN_REF:-master}"
JOBS="${JOBS:-$(nproc)}"

mkdir -p "$EVP_TOOLS_HOME"
cd "$EVP_TOOLS_HOME"

if [ ! -x "$EVP_TOOLS_HOME/ncnn-install/bin/ncnn2int8" ]; then
  command -v cmake >/dev/null || { echo "[ERROR] cmake missing: sudo apt install cmake g++ git" >&2; exit 1; }
  [ -d ncnn ] || git clone --depth 1 --branch "$NCNN_REF" https://github.com/Tencent/ncnn.git
  mkdir -p ncnn/build && cd ncnn/build
  cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_TOOLS=ON \
        -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_SIMPLEOCV=ON -DCMAKE_INSTALL_PREFIX="$EVP_TOOLS_HOME/ncnn-install" .. > cmake.log
  make -j"$JOBS" > make.log
  make install > install.log
  cd "$EVP_TOOLS_HOME"
fi
echo "[OK] ncnn tools: $EVP_TOOLS_HOME/ncnn-install/bin"

VENV="$EVP_TOOLS_HOME/venv"
if [ ! -x "$VENV/bin/pip" ]; then
  rm -rf "$VENV"
  if python3 -m venv "$VENV" 2>/dev/null; then
    :
  else
    # no python3-venv/ensurepip (and maybe no sudo): pip-less venv, installed into by the user-level pip
    rm -rf "$VENV"
    python3 -m venv --without-pip "$VENV"
  fi
fi
if [ -x "$VENV/bin/pip" ]; then
  PIP=("$VENV/bin/pip")
else
  python3 -m pip --version >/dev/null 2>&1 || { echo "[ERROR] no pip: sudo apt install python3-venv python3-pip" >&2; exit 1; }
  PIP=(python3 -m pip --python "$VENV/bin/python")
fi
# --no-deps: the pnnx wheel declares torch as a dependency, but its converter binary is self-contained
"${PIP[@]}" install -q -i "$PIP_INDEX" --timeout 60 --no-deps pnnx
# locate the binary on disk: `import pnnx` would require torch
PNNX_EXE="$(ls "$VENV"/lib/python3*/site-packages/pnnx/pnnx 2>/dev/null | head -1)"
[ -x "$PNNX_EXE" ] || { echo "[ERROR] pnnx binary not found under $VENV" >&2; exit 1; }
"$PNNX_EXE" >/dev/null 2>&1 || true
echo "[OK] pnnx: $PNNX_EXE"
echo "$PNNX_EXE" > "$EVP_TOOLS_HOME/pnnx_path.txt"
