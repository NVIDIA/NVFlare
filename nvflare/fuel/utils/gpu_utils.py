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
from typing import List


def has_nvidia_smi() -> bool:
    from shutil import which

    return which("nvidia-smi") is not None


def use_nvidia_smi(query: str, report_format: str = "csv"):
    if has_nvidia_smi():
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", f"--format={report_format}"],
            capture_output=True,
            text=True,
        )
        rc = result.returncode
        if rc > 0:
            raise Exception(f"Failed to call nvidia-smi with query {query}", result.stderr)
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
