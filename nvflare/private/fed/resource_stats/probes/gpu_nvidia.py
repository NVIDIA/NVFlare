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

"""Measure CUDA devices available to this process with the system Driver API.

The Driver count is numeric authority. Driver metadata and NVML classification
are optional: neither can add, remove, or silently relabel a counted ordinal.
No CUDA Runtime, framework package, command, or environment-string parser is
used. UUIDs remain private to this call and never enter a report. This module is
also a standalone helper: keep its imports limited to the Python standard library.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import re
import signal
import sys
import time
from collections import Counter
from typing import Any, Optional

CUDA_SUCCESS = 0
CUDA_ERROR_NO_DEVICE = 100
MAX_CUDA_DEVICES = 4096
_MAX_OUTPUT_BYTES = 64 * 1024
_MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._()+/@-]{0,127}$")
_UUID_IN_MODEL = re.compile(r"(?:GPU|MIG|MIG-GPU)-[0-9a-f-]{16,}", re.IGNORECASE)


class _CudaUuid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]


def _load_library(candidates: tuple[str, ...]):
    """Load only well-known system library names; never inspect job packages."""

    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except (AttributeError, OSError, TypeError, ValueError):
            continue
    return None


def _load_driver():
    system = platform.system()
    if system == "Linux":
        # The versioned Driver SONAME avoids a toolkit's unversioned stub.
        return _load_library(("libcuda.so.1",))
    if system == "Darwin":
        # Legacy macOS drivers only. A missing library remains unavailable.
        return _load_library(("libcuda.dylib",))
    return None


def _load_nvml():
    system = platform.system()
    if system == "Linux":
        return _load_library(("libnvidia-ml.so.1",))
    return None


def _bind(library: Any, name: str, argtypes: list[Any]):
    """Bind one optional ctypes symbol without allowing a Python-side error out."""

    try:
        function = getattr(library, name)
        function.argtypes = argtypes
        function.restype = ctypes.c_int
        return function
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _result(function: Any, *args: Any) -> Optional[int]:
    if function is None:
        return None
    try:
        return function(*args)
    except Exception:
        # Native wrappers and test doubles can raise after a successful count.
        # An optional metadata/classification call must not lose that count.
        return None


def _driver_count(driver: Any) -> Optional[int]:
    cu_init = _bind(driver, "cuInit", [ctypes.c_uint])
    cu_get_count = _bind(driver, "cuDeviceGetCount", [ctypes.POINTER(ctypes.c_int)])
    if cu_init is None or cu_get_count is None:
        return None
    initialized = _result(cu_init, 0)
    if initialized == CUDA_ERROR_NO_DEVICE:
        return 0
    if initialized != CUDA_SUCCESS:
        return None
    count = ctypes.c_int(-1)
    enumerated = _result(cu_get_count, ctypes.byref(count))
    if enumerated == CUDA_ERROR_NO_DEVICE:
        return 0
    if enumerated != CUDA_SUCCESS or count.value < 0 or count.value > MAX_CUDA_DEVICES:
        return None
    return count.value


def _uuid_text(value: _CudaUuid) -> str:
    raw = bytes(value.bytes).hex()
    return f"{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}"


def _driver_devices(driver: Any, count: int) -> list[dict[str, Any]]:
    cu_get_device = _bind(driver, "cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int])
    cu_get_name = _bind(driver, "cuDeviceGetName", [ctypes.c_char_p, ctypes.c_int, ctypes.c_int])
    # cuda.h maps size_t* cuDeviceTotalMem to the exported _v2 entry point.
    cu_total_mem = _bind(driver, "cuDeviceTotalMem_v2", [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int])
    # The _v2 UUID identifies MIG compute instances correctly where supported.
    cu_get_uuid = _bind(driver, "cuDeviceGetUuid_v2", [ctypes.POINTER(_CudaUuid), ctypes.c_int])

    devices: list[dict[str, Any]] = []
    for ordinal in range(count):
        device_info: dict[str, Any] = {"kind": "cuda_device_unclassified"}
        devices.append(device_info)
        device = ctypes.c_int(-1)
        if _result(cu_get_device, ctypes.byref(device), ordinal) != CUDA_SUCCESS or device.value < 0:
            continue
        if cu_get_name is not None:
            name_buffer = ctypes.create_string_buffer(128)
            if _result(cu_get_name, name_buffer, len(name_buffer), device.value) == CUDA_SUCCESS:
                model = " ".join(name_buffer.value.decode("ascii", errors="ignore").split())
                if _MODEL_PATTERN.fullmatch(model) and not _UUID_IN_MODEL.search(model):
                    device_info["model"] = model
        if cu_total_mem is not None:
            memory = ctypes.c_size_t()
            if _result(cu_total_mem, ctypes.byref(memory), device.value) == CUDA_SUCCESS and memory.value > 0:
                device_info["memory_bytes"] = str(memory.value)
        if cu_get_uuid is not None:
            uuid = _CudaUuid()
            if _result(cu_get_uuid, ctypes.byref(uuid), device.value) == CUDA_SUCCESS and any(uuid.bytes):
                device_info["_uuid"] = _uuid_text(uuid)
    return devices


def _classify_with_nvml(devices: list[dict[str, Any]]) -> None:
    identities = Counter(device.get("_uuid") for device in devices if device.get("_uuid"))
    if not identities:
        return
    nvml = _load_nvml()
    if nvml is None:
        return
    nvml_init = _bind(nvml, "nvmlInit_v2", [])
    nvml_shutdown = _bind(nvml, "nvmlShutdown", [])
    get_handle = _bind(nvml, "nvmlDeviceGetHandleByUUID", [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)])
    is_mig = _bind(nvml, "nvmlDeviceIsMigDeviceHandle", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)])
    if any(function is None for function in (nvml_init, nvml_shutdown, get_handle, is_mig)):
        return
    if _result(nvml_init) != CUDA_SUCCESS:
        return
    try:
        for device in devices:
            identity = device.get("_uuid")
            if not identity or identities[identity] != 1:
                continue
            # UUID prefixes are lookup encodings only, never a type test.
            for prefix in ("GPU-", "MIG-"):
                handle = ctypes.c_void_p()
                if _result(get_handle, (prefix + identity).encode("ascii"), ctypes.byref(handle)) != 0:
                    continue
                if not handle.value:
                    continue
                mig = ctypes.c_uint(2)
                if _result(is_mig, handle, ctypes.byref(mig)) == 0 and mig.value in (0, 1):
                    device["kind"] = "mig_compute_instance" if mig.value else "full_gpu"
                break
    finally:
        _result(nvml_shutdown)


def probe_gpu() -> Optional[dict[str, list[dict[str, Any]]]]:
    """Return one capacity group per kind/model/memory set, or unknown.

    A successful Driver count of zero and explicit CUDA_ERROR_NO_DEVICE are
    observed zero. No loadable Driver, other Driver errors, or a count outside
    0..4096 are unavailable. Classification failures never change a known
    positive count. Each Driver ordinal contributes to one group exactly once.
    """

    driver = _load_driver()
    if driver is None:
        return None
    count = _driver_count(driver)
    if count is None:
        return None
    if count == 0:
        return {"groups": []}

    devices = _driver_devices(driver, count)
    try:
        _classify_with_nvml(devices)
    except Exception:
        # NVML is optional and must never prevent report finalization.
        pass
    groups: dict[tuple[str, Optional[str], Optional[str]], dict[str, Any]] = {}
    for device in devices:
        key = (device["kind"], device.get("model"), device.get("memory_bytes"))
        group = groups.setdefault(
            key,
            {
                "kind": device["kind"],
                "count": 0,
                **({"model": device["model"]} if "model" in device else {}),
                **({"memory_bytes": device["memory_bytes"]} if "memory_bytes" in device else {}),
            },
        )
        group["count"] += 1
    return {"groups": [groups[key] for key in sorted(groups, key=lambda item: tuple(value or "" for value in item))]}


def _protect_owner(parent_pid: int, deadline: float) -> None:
    # EOF on stdin releases the hold on every platform. Linux also terminates
    # a helper whose owner dies while a native discovery call is still running.
    if platform.system() == "Linux":
        libc = ctypes.CDLL(None)
        prctl = _bind(libc, "prctl", [ctypes.c_int] + [ctypes.c_ulong] * 4)
        if _result(prctl, 1, signal.SIGKILL, 0, 0, 0) != 0:
            raise OSError("cannot bind GPU discovery to its owner")
    if os.getppid() != parent_pid:
        raise SystemExit(1)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise SystemExit(1)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, remaining)


def main(*, parent_pid: Optional[int] = None, deadline: Optional[float] = None):
    """Run with ordinary Python startup without importing the NVFlare package."""

    if parent_pid is not None:
        _protect_owner(parent_pid, deadline)
    value = probe_gpu()
    encoded = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) + 1 > _MAX_OUTPUT_BYTES:
        # Optional descriptions cannot erase an already enumerated CUDA count.
        counts = Counter()
        for group in value["groups"]:
            counts[group["kind"]] += int(group["count"])
        value = {"groups": [{"kind": kind, "count": count} for kind, count in sorted(counts.items())]}
        encoded = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if parent_pid is not None:
        signal.setitimer(signal.ITIMER_REAL, 0)
    sys.stdout.buffer.write(encoded + b"\n")
    sys.stdout.buffer.flush()
    if parent_pid is not None:
        # Keeping the CUDA client alive avoids a second cold initialization
        # when application code first uses its GPU. Parent closure releases it.
        sys.stdin.buffer.read(1)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--hold":
        main(parent_pid=int(sys.argv[2]), deadline=float(sys.argv[3]))
    else:
        main()
