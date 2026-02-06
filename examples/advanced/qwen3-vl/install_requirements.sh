#!/usr/bin/env bash
# Install dependencies so flash_attn can build (it needs torch at build time).
# Run from repo root with your venv activated, or pass path to requirements.txt.

set -e
pip install -U pip

echo "==> Installing PyTorch first (required for building flash_attn)..."
pip install torch==2.6.0 torchvision==0.21.0

echo "==> Installing build-time deps for flash_attn (e.g. psutil)..."
pip install psutil packaging ninja

echo "==> Installing remaining requirements (flash_attn built with --no-build-isolation)..."
pip install --no-build-isolation -r requirements.txt

echo "==> Done."
