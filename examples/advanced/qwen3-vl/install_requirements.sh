#!/usr/bin/env bash
# Install dependencies so flash_attn can build (it needs torch at build time).
# Run with your venv activated from any directory; by default this script uses
# requirements.txt next to itself (or pass a custom path as the first argument).
#
# If your project lives on a different filesystem than $HOME or /tmp (e.g. /scratch vs /home),
# building flash_attn can hit "Invalid cross-device link" when it moves the downloaded wheel
# from a temp dir into the pip cache. Use cache and temp dirs in the current tree so both
# live on the same filesystem.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="${1:-$SCRIPT_DIR/requirements.txt}"
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    echo "requirements.txt not found: $REQUIREMENTS_FILE" >&2
    exit 1
fi
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRIPT_DIR/.pip_cache}"
BUILD_TMP="$SCRIPT_DIR/.tmp_build"
mkdir -p "$PIP_CACHE_DIR" "$BUILD_TMP"
export TMPDIR="$BUILD_TMP"
export TEMP="$BUILD_TMP"
export TMP="$BUILD_TMP"

PYTHON_BIN="${PYTHON:-python}"
"$PYTHON_BIN" -m pip install -U pip

echo "==> Installing PyTorch first (required for building flash_attn)..."
"$PYTHON_BIN" -m pip install torch==2.6.0 torchvision==0.21.0

echo "==> Installing build-time deps for flash_attn (e.g. psutil)..."
"$PYTHON_BIN" -m pip install psutil packaging ninja

echo "==> Installing remaining requirements (flash_attn built with --no-build-isolation)..."
"$PYTHON_BIN" -m pip install --no-build-isolation -r "$REQUIREMENTS_FILE"

echo "==> Done."
