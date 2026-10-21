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
import subprocess
import uuid
from typing import List


def has_nvidia_smi() -> bool:
    from shutil import which

    return which("nvidia-smi") is not None


def use_nvidia_smi(query: str, report_format: str = "csv", timeout=None):
    if has_nvidia_smi():
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", f"--format={report_format}"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        rc = result.returncode
        if rc > 0:
            raise Exception(f"Failed to call nvidia-smi with query {query}", result.stderr)
        else:
            return result.stdout.splitlines()
    return None


def get_cuda_device_environment(devices) -> dict[str, str]:
    """Build CUDA visibility settings; these do not enforce device access."""
    return {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": ",".join(str(device) for device in devices)}


def normalize_gpu_uuid(value: str) -> str:
    """Validate and canonicalize a full physical NVIDIA GPU UUID."""
    if not isinstance(value, str) or not value.startswith("GPU-"):
        raise ValueError("devices must be full physical NVIDIA GPU UUIDs")
    try:
        canonical = "GPU-" + str(uuid.UUID(value[4:]))
    except ValueError as error:
        raise ValueError("devices must be full physical NVIDIA GPU UUIDs") from error
    if canonical.lower() != value.lower():
        raise ValueError("devices must be full hyphenated GPU UUIDs")
    return canonical


def get_host_gpu_uuid_modes() -> dict[str, str]:
    """Return host physical GPU UUIDs and their reported current MIG modes."""
    rows = use_nvidia_smi("uuid,mig.mode.current", report_format="csv,noheader", timeout=10)
    if rows is None:
        raise RuntimeError("nvidia-smi is required to discover local GPU UUIDs")
    devices = {}
    for row in rows:
        device, mode = (item.strip() for item in row.split(",", 1))
        devices[normalize_gpu_uuid(device)] = mode
    return devices


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


def get_host_gpu_ids() -> List:
    """Gets GPU IDs.

    Note:
        Only supports nvidia-smi now.
    """
    result = use_nvidia_smi("index")
    gpu_ids = []
    if result:
        for i in result[1:]:
            gpu_ids.append(int(i))
    return gpu_ids
