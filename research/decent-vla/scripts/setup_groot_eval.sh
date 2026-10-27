#!/usr/bin/env bash
# Build the GR00T LIBERO evaluation environment (.venv-groot-eval): Python 3.10 with the LIBERO simulator,
# robosuite, MuJoCo and decent-vla. The simulator needs package versions that the GR00T training environment
# cannot share, so evaluation has its own environment. gr00t itself comes from the Isaac-GR00T clone that
# scripts/setup_groot.sh made, through PYTHONPATH, so evaluation runs the model code that training ran.
# The package list (scripts/groot_eval_requirements.txt) is for aarch64 machines such as GH200 nodes.
#
# Usage: bash scripts/setup_groot_eval.sh   (after scripts/setup_groot.sh, on a GPU node; needs uv and a
#                                            CUDA 12.8 driver)
set -euo pipefail
cd "$(dirname "$0")/.."

GROOT_DIR=third_party/Isaac-GR00T
VENV=.venv-groot-eval
TORCH_INDEX=https://download.pytorch.org/whl/cu128

if [ ! -d "$GROOT_DIR/.git" ]; then
    echo "run scripts/setup_groot.sh first: it clones $GROOT_DIR" >&2
    exit 2
fi

uv venv "$VENV" --python 3.10
uv pip install --python "$VENV" --index-url "$TORCH_INDEX" "torch==2.7.1" "torchvision==0.22.1"
"$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), 'torch cannot use the GPU'; print(torch.cuda.get_device_name(0))"
uv pip install --python "$VENV" --no-deps -r scripts/groot_eval_requirements.txt
uv pip install --python "$VENV" --no-deps -e .
# robosuite writes a log file to /tmp by default, which fails where /tmp is read-only.
sed -i 's/^FILE_LOGGING_LEVEL\s*=.*/FILE_LOGGING_LEVEL = None/' "$VENV"/lib/python3.10/site-packages/robosuite/macros.py

PYTHONPATH="$PWD/$GROOT_DIR" "$VENV/bin/python" -c "import libero, robosuite, mujoco, gr00t, decent_vla; print('imports OK')"
echo "GR00T eval env ready: source $VENV/bin/activate && export PYTHONPATH=\$PWD/$GROOT_DIR MUJOCO_GL=egl"
