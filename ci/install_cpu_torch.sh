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

allow_pypi_fallback="${NVFLARE_ALLOW_PYPI_TORCH_FALLBACK:-}"
if [[ -z "${allow_pypi_fallback}" && "${GITHUB_ACTIONS:-}" == "true" ]]; then
    # GitHub-hosted runners have seen intermittent TLS handshake failures from
    # download-r2.pytorch.org. Keep the CPU-only index as the first choice, but
    # let pre-merge CI fall back to PyPI torch packages so an external mirror
    # outage does not block unrelated validation.
    allow_pypi_fallback=true
fi
used_pypi_fallback=false

function uv_pip_install {
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        uv pip install --system "$@"
    else
        uv pip install "$@"
    fi
}

function install_from_pypi {
    used_pypi_fallback=true
    if command -v uv >/dev/null 2>&1; then
        echo "Falling back to PyPI PyTorch packages with uv: ${packages[*]}"
        uv_pip_install "${packages[@]}"
    else
        echo "Falling back to PyPI PyTorch packages with pip: ${packages[*]}"
        python3 -m pip install "${packages[@]}"
    fi
}

if command -v uv >/dev/null 2>&1; then
    echo "Downloading and installing CPU-only PyTorch packages with uv: ${packages[*]}"
    if ! UV_TORCH_BACKEND=cpu uv_pip_install --torch-backend=cpu "${packages[@]}"; then
        if [[ "${allow_pypi_fallback}" == "true" ]]; then
            install_from_pypi
        else
            exit 1
        fi
    fi
else
    echo "Downloading and installing CPU-only PyTorch packages with pip: ${packages[*]}"
    if ! python3 -m pip install "${packages[@]}" --index-url https://download.pytorch.org/whl/cpu; then
        if [[ "${allow_pypi_fallback}" == "true" ]]; then
            install_from_pypi
        else
            exit 1
        fi
    fi
fi

NVFLARE_USED_PYPI_TORCH_FALLBACK="${used_pypi_fallback}" python3 - <<'PY'
import importlib
import os

torch = importlib.import_module("torch")
used_pypi_fallback = os.environ.get("NVFLARE_USED_PYPI_TORCH_FALLBACK") == "true"
is_cpu_only = torch.version.cuda is None and "+cu" not in torch.__version__

if not used_pypi_fallback and not is_cpu_only:
    raise SystemExit(f"Expected CPU-only PyTorch, got torch {torch.__version__} with CUDA {torch.version.cuda}")

if used_pypi_fallback:
    print(
        "Installed PyPI fallback torch "
        f"{torch.__version__} with CUDA runtime metadata {torch.version.cuda}; "
        "CI is running on CPU-only hosts."
    )
else:
    print(f"Installed CPU-only torch {torch.__version__}")

try:
    torchvision = importlib.import_module("torchvision")
except ModuleNotFoundError:
    torchvision = None

if torchvision is not None:
    print(f"Installed torchvision {torchvision.__version__}")
PY
