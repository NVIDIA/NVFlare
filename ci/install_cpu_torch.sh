#!/bin/bash
#
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

packages=("$@")
if [[ ${#packages[@]} -eq 0 ]]; then
    packages=(torch torchvision)
fi

if command -v uv >/dev/null 2>&1; then
    echo "Downloading and installing CPU-only PyTorch packages with uv: ${packages[*]}"
    uv_system_flag=()
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        uv_system_flag=(--system)
    fi
    UV_TORCH_BACKEND=cpu uv pip install "${uv_system_flag[@]}" --torch-backend=cpu "${packages[@]}"
else
    echo "Downloading and installing CPU-only PyTorch packages with pip: ${packages[*]}"
    python3 -m pip install "${packages[@]}" --index-url https://download.pytorch.org/whl/cpu
fi

python3 - <<'PY'
import importlib

torch = importlib.import_module("torch")
if torch.version.cuda is not None or "+cu" in torch.__version__:
    raise SystemExit(f"Expected CPU-only PyTorch, got torch {torch.__version__} with CUDA {torch.version.cuda}")

print(f"Installed CPU-only torch {torch.__version__}")

try:
    torchvision = importlib.import_module("torchvision")
except ModuleNotFoundError:
    torchvision = None

if torchvision is not None:
    print(f"Installed torchvision {torchvision.__version__}")
PY
