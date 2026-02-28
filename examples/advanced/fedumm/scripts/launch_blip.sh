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

# Wrapper that activates the BLIP conda environment before running the
# FL client script.  Used by NVFlare's SubprocessLauncher / ScriptRunner
# with launch_external_process=True.
#
# Usage (called automatically by NVFlare):
#   bash scripts/launch_blip.sh src/fl_client.py --model_backend blip_vqa ...

set -euo pipefail

# ---- Locate conda ----
CONDA_BASE="${CONDA_PREFIX:-${HOME}/miniconda3}"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    echo "[launch_blip] ERROR: conda not found.  Set CONDA_PREFIX or install conda." >&2
    exit 1
fi

# ---- Activate environment ----
ENV_NAME="${NVFLARE_BLIP_ENV:-nvflare_blip}"
conda activate "${ENV_NAME}"
echo "[launch_blip] Activated env: ${ENV_NAME}  (python=$(which python))"

# ---- Run the script (all args forwarded) ----
exec python -u "$@"
