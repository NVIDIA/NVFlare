#!/usr/bin/env bash
# Install dependencies so flash_attn can build (it needs torch at build time).
# Run from repo root with your venv activated, or pass path to requirements.txt.
#
# If your project lives on a different filesystem than $HOME or /tmp (e.g. /scratch vs /home),
# building flash_attn can hit "Invalid cross-device link" when it moves the downloaded wheel
# from a temp dir into the pip cache. Use cache and temp dirs in the current tree so both
# live on the same filesystem.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRIPT_DIR/.pip_cache}"
BUILD_TMP="$SCRIPT_DIR/.tmp_build"
mkdir -p "$PIP_CACHE_DIR" "$BUILD_TMP"
export TMPDIR="$BUILD_TMP"
export TEMP="$BUILD_TMP"
export TMP="$BUILD_TMP"

pip install -U pip

echo "==> Installing PyTorch first (required for building flash_attn)..."
pip install torch==2.6.0 torchvision==0.21.0

echo "==> Installing build-time deps for flash_attn (e.g. psutil)..."
pip install psutil packaging ninja

echo "==> Installing remaining requirements (flash_attn built with --no-build-isolation)..."
pip install --no-build-isolation -r requirements.txt

echo "==> Done."
