#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NVFLARE_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
IMAGE="${NVFLARE_EVO2_IMAGE:-nvflare-evo2}"
GPU="${NVFLARE_EVO2_GPU:-0}"
CACHE_DIR="${NVFLARE_EVO2_CACHE:-${SCRIPT_DIR}/.evo2_cache}"
WORKSPACE_DIR="${NVFLARE_EVO2_WORKSPACE:-${SCRIPT_DIR}/.evo2_workspace}"

mkdir -p "${CACHE_DIR}/huggingface" "${CACHE_DIR}/bionemo" "${WORKSPACE_DIR}"

docker run --rm -it \
    --gpus "device=${GPU}" \
    --ipc=host \
    --network=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${NVFLARE_ROOT}:/workspace/nvflare" \
    -v "${CACHE_DIR}/huggingface:/root/.cache/huggingface" \
    -v "${CACHE_DIR}/bionemo:/root/.cache/bionemo" \
    -v "${WORKSPACE_DIR}:/tmp/nvflare" \
    -w /workspace/nvflare/examples/advanced/bionemo/evo2 \
    "${IMAGE}" \
    bash -lc 'uv pip install -e /workspace/nvflare && python verify_environment.py && exec bash'
