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
"""Fail fast if the Evo2 native CUDA extension is incompatible with PyTorch."""

import importlib
from collections.abc import Callable
from typing import Any


def _load_dependencies() -> tuple[Any, Callable[..., Any]]:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError("PyTorch could not be imported inside the Evo2 container.") from exc

    try:
        causal_conv1d = importlib.import_module("causal_conv1d")
        causal_conv1d_fn = causal_conv1d.causal_conv1d_fn
    except Exception as exc:
        raise RuntimeError(
            "causal-conv1d could not be imported inside the Evo2 container. "
            "Rebuild the pinned image so the extension matches its PyTorch/CUDA runtime."
        ) from exc
    return torch, causal_conv1d_fn


def _run_cuda_smoke(torch: Any, causal_conv1d_fn: Callable[..., Any]) -> None:
    try:
        cuda_available = torch.cuda.is_available()
    except Exception as exc:
        raise RuntimeError("Could not query CUDA availability inside the Evo2 container.") from exc

    if not cuda_available:
        raise RuntimeError("CUDA is unavailable inside the Evo2 container.")

    try:
        input_tensor = torch.randn(2, 32, 600, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(32, 4, device="cuda", dtype=torch.float32, requires_grad=True)
        output = causal_conv1d_fn(input_tensor, weight)
        output.float().sum().backward()
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(
            "causal-conv1d CUDA forward/backward smoke failed. "
            "Rebuild the pinned image so the extension matches its PyTorch/CUDA runtime."
        ) from exc


def main() -> None:
    torch, causal_conv1d_fn = _load_dependencies()
    _run_cuda_smoke(torch, causal_conv1d_fn)
    print("causal-conv1d CUDA forward/backward smoke passed")


if __name__ == "__main__":
    main()
