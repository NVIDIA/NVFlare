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

from __future__ import annotations

import gc
import os

import torch


def abs_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def params_size_mb(params) -> float:
    if not params:
        return 0.0
    nbytes = 0
    for value in params.values():
        if isinstance(value, torch.Tensor):
            nbytes += value.numel() * value.element_size()
        elif hasattr(value, "nbytes"):
            nbytes += value.nbytes
    return nbytes / (1024.0 * 1024.0)


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_cuda_memory_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_cuda_memory_usage_mb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated() / (1024.0 * 1024.0),
        torch.cuda.memory_reserved() / (1024.0 * 1024.0),
    )


def get_peak_cuda_memory_usage_mb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
        torch.cuda.max_memory_reserved() / (1024.0 * 1024.0),
    )


def require_supported_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA GPU because MedGemma QLoRA uses bitsandbytes 4-bit loading.")
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("MedGemma fine-tuning requires a GPU with bfloat16 support (compute capability >= 8.0).")
