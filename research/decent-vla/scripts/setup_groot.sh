#!/usr/bin/env bash
# Build the GR00T N1.7 environment: a Python 3.10 venv with NVIDIA Isaac-GR00T, the native FLARE example and NVFlare.
# Isaac-GR00T requires Python 3.10 and pins its own torch/transformers stack, so GR00T runs in this
# separate env rather than in the lerobot one.
#
# Usage: bash scripts/setup_groot.sh            (run from this project directory; needs git, uv and a CUDA 12.8 driver,
#                                                and git-lfs on aarch64)

set -euo pipefail
cd "$(dirname "$0")/.."

GROOT_REPO=https://github.com/NVIDIA/Isaac-GR00T.git
GROOT_SHA=65cc4a192e6d084650d97747308b6a8deb790722   # the Isaac-GR00T commit the GR00T adapter is written against
GROOT_DIR=third_party/Isaac-GR00T
VENV=.venv-groot
TORCH_INDEX=https://download.pytorch.org/whl/cu128

if [ ! -d "$GROOT_DIR/.git" ]; then
    git clone "$GROOT_REPO" "$GROOT_DIR"
fi
git -C "$GROOT_DIR" fetch --quiet origin "$GROOT_SHA"
git -C "$GROOT_DIR" checkout --quiet "$GROOT_SHA"
# On aarch64, Isaac-GR00T installs flash-attn and torchcodec from wheels it stores with Git LFS; a clone made
# without git-lfs holds pointer files instead, which uv cannot install.
if [ "$(uname -m)" = aarch64 ] && grep -qs "git-lfs.github.com/spec" "$GROOT_DIR"/scripts/deployment/dgpu/wheels/*.whl; then
    git -C "$GROOT_DIR" lfs pull --include "scripts/deployment/dgpu/wheels/*.whl"
fi
# PyAV video decoding for the AV1-encoded LIBERO videos (see the patch header); skipped when already applied.
PATCH="$PWD/scripts/isaac_gr00t_pyav.patch"
if ! git -C "$GROOT_DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
    git -C "$GROOT_DIR" apply "$PATCH"
fi

uv venv "$VENV" --python 3.10
# torch first, matching Isaac-GR00T's pin, from the CUDA 12.8 index.
uv pip install --python "$VENV" --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match \
    "torch==2.7.1" "torchvision==0.22.1"
"$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), 'torch cannot use the GPU'; print(torch.cuda.get_device_name(0))"
uv pip install --python "$VENV" --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match \
    -e "$GROOT_DIR"
# the native example without its extras (the lerobot stack is not needed for GR00T), then peft and NVFlare.
uv pip install --python "$VENV" --no-deps -e .
uv pip install --python "$VENV" "pyyaml>=6.0" "peft==0.17.1" "nvflare==2.8.1"

"$VENV/bin/python" - <<'PY'
import sys
import torch
import gr00t
import nvflare
import src
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda)
print("gr00t", gr00t.__file__)
print("nvflare", nvflare.__version__, "| native src import OK")
PY
echo "GR00T env ready: source $VENV/bin/activate"
