# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import os
import subprocess
from shutil import which
from typing import List, Optional

CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
_NVIDIA_SMI_TIMEOUT_SECONDS = 30.0


def has_nvidia_smi() -> bool:
    return which("nvidia-smi") is not None


def use_nvidia_smi(query: str, report_format: str = "csv", *, timeout: float = _NVIDIA_SMI_TIMEOUT_SECONDS):
    if has_nvidia_smi():
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}", f"--format={report_format}"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError, UnicodeError) as e:
            raise RuntimeError(f"Failed to call nvidia-smi with query {query}") from e
        rc = result.returncode
        if rc != 0:
            raise RuntimeError(f"Failed to call nvidia-smi with query {query}", result.stderr)
        else:
            return result.stdout.splitlines()
    return None


def _parse_gpu_mem(result: str = None, unit: str = "MiB") -> List:
    gpu_memory = []
    if result:
        for i in result[1:]:
            mem, mem_unit = i.split(" ")
            if mem_unit != unit:
                raise RuntimeError("Memory unit does not match.")
            gpu_memory.append(int(mem))
    return gpu_memory


def get_host_gpu_memory_total(unit="MiB") -> List:
    result = use_nvidia_smi("memory.total")
    return _parse_gpu_mem(result, unit)


def get_host_gpu_memory_free(unit="MiB") -> List:
    result = use_nvidia_smi("memory.free")
    return _parse_gpu_mem(result, unit)


def get_host_gpu_ids(*, timeout: float = _NVIDIA_SMI_TIMEOUT_SECONDS) -> List[int]:
    """Gets GPU IDs.

    Note:
        Only supports nvidia-smi now.
    """
    result = use_nvidia_smi("index", timeout=timeout)
    if not result or len(result) < 2 or result[0].strip() != "index":
        raise RuntimeError("nvidia-smi did not return a GPU inventory")
    try:
        gpu_ids = [int(value.strip()) for value in result[1:]]
    except ValueError as e:
        raise RuntimeError("nvidia-smi returned invalid GPU indices") from e
    if any(gpu_id < 0 for gpu_id in gpu_ids) or len(gpu_ids) != len(set(gpu_ids)):
        raise RuntimeError("nvidia-smi returned invalid GPU indices")
    return gpu_ids


def get_visible_gpu_ids(host_gpu_ids: Optional[list[int]] = None) -> Optional[list[int]]:
    """Select nvidia-smi integer IDs using the process visibility mask.

    Without a mask, return the supplied inventory. None means an unsupported
    selector, or an unspecified pool when no inventory was supplied. Negative,
    missing, or out-of-inventory indices stop the integer prefix; duplicates
    retain their first position. This function never queries the host.
    """
    value = os.environ.get(CUDA_VISIBLE_DEVICES)
    if value is None:
        return list(host_gpu_ids) if host_gpu_ids is not None else None
    value = value.strip()
    if not value:
        return []

    gpu_ids = []
    seen = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            break
        try:
            gpu_id = int(item)
        except ValueError:
            return None
        if gpu_id < 0 or (host_gpu_ids is not None and gpu_id not in host_gpu_ids):
            break
        if gpu_id not in seen:
            gpu_ids.append(gpu_id)
            seen.add(gpu_id)
    return gpu_ids
