#!/usr/bin/env bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# One-click setup: creates both conda environments and installs
# the DeepSeek Janus package for the JanusPro backend.
#
# Usage:
#   bash scripts/setup_envs.sh          # create both envs
#   bash scripts/setup_envs.sh blip     # BLIP only
#   bash scripts/setup_envs.sh januspro # JanusPro only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
TARGET="${1:-all}"

echo "============================================"
echo " NVFlare VLM-FL Environment Setup"
echo "============================================"

# ---- BLIP ----
if [[ "${TARGET}" == "all" || "${TARGET}" == "blip" ]]; then
    echo ""
    echo ">>> Creating BLIP environment (nvflare_blip) ..."
    conda env remove -n nvflare_blip -y 2>/dev/null || true
    conda env create -f "${PROJECT_DIR}/envs/env_blip.yml"
    echo ">>> BLIP environment ready."
fi

# ---- JanusPro ----
if [[ "${TARGET}" == "all" || "${TARGET}" == "januspro" ]]; then
    echo ""
    echo ">>> Creating JanusPro environment (nvflare_januspro) ..."
    conda env remove -n nvflare_januspro -y 2>/dev/null || true
    conda env create -f "${PROJECT_DIR}/envs/env_januspro.yml"

    echo ">>> Installing DeepSeek Janus package ..."
    JANUS_DIR="/tmp/Janus"
    if [ ! -d "${JANUS_DIR}" ]; then
        git clone https://github.com/deepseek-ai/Janus.git "${JANUS_DIR}"
    else
        cd "${JANUS_DIR}" && git pull && cd -
    fi

    # Activate env and install janus in editable mode
    eval "$(conda shell.bash hook)"
    conda activate nvflare_januspro
    pip install -e "${JANUS_DIR}"
    conda deactivate
    echo ">>> JanusPro environment ready."
fi

echo ""
echo "============================================"
echo " Setup complete!  Quick check:"
echo "   conda activate nvflare_blip"
echo "   python -c \"import src; from src.model_registry import list_backends; print(list_backends())\""
echo ""
echo "   conda activate nvflare_januspro"
echo "   python -c \"import src; from src.model_registry import list_backends; print(list_backends())\""
echo "============================================"
