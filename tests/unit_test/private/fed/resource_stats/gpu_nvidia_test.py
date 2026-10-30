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

"""Controlled Driver/NVML fakes: no GPU, CUDA Runtime, or NVML is required."""

import ctypes
from types import SimpleNamespace

import pytest

from nvflare.private.fed.resource_stats.probes import gpu_nvidia


class _FakeFunction:
    def __init__(self, implementation):
        self.implementation = implementation

    def __call__(self, *args):
        return self.implementation(*args)


def _write(pointer, c_type, value):
    ctypes.cast(pointer, ctypes.POINTER(c_type))[0] = value


def _driver(
    *,
    count=1,
    init_result=0,
    count_result=0,
    device_results=None,
    models=None,
    memories=None,
    uuids=None,
    missing_symbols=(),
):
    """Provide just the Driver ABI; no cudaGetDeviceCount symbol exists."""

    device_results = device_results or {}
    models = models if models is not None else ["NVIDIA H100 80GB HBM3"] * max(count, 0)
    memories = memories if memories is not None else [80 * 2**30] * max(count, 0)
    uuids = uuids if uuids is not None else [bytes([(index % 255) + 1]) * 16 for index in range(max(count, 0))]
    calls = []

    def init(flags):
        calls.append(("cuInit", flags))
        return init_result

    def get_count(pointer):
        calls.append(("cuDeviceGetCount",))
        if count_result == 0:
            _write(pointer, ctypes.c_int, count)
        return count_result

    def get_device(pointer, ordinal):
        calls.append(("cuDeviceGet", ordinal))
        result = device_results.get(ordinal, 0)
        if result == 0:
            _write(pointer, ctypes.c_int, 100 + ordinal)
        return result

    def get_name(buffer, _length, device):
        calls.append(("cuDeviceGetName", device))
        model = models[device - 100]
        if model is None:
            return 1
        buffer.value = model.encode("ascii")
        return 0

    def total_mem(pointer, device):
        calls.append(("cuDeviceTotalMem_v2", device))
        memory = memories[device - 100]
        if memory is None:
            return 1
        _write(pointer, ctypes.c_size_t, memory)
        return 0

    def get_uuid(pointer, device):
        calls.append(("cuDeviceGetUuid_v2", device))
        raw = uuids[device - 100]
        if raw is None:
            return 1
        value = ctypes.cast(pointer, ctypes.POINTER(gpu_nvidia._CudaUuid)).contents
        for index, byte in enumerate(raw):
            value.bytes[index] = byte
        return 0

    symbols = {
        "cuInit": init,
        "cuDeviceGetCount": get_count,
        "cuDeviceGet": get_device,
        "cuDeviceGetName": get_name,
        "cuDeviceTotalMem_v2": total_mem,
        "cuDeviceGetUuid_v2": get_uuid,
    }
    return SimpleNamespace(
        **{name: _FakeFunction(function) for name, function in symbols.items() if name not in missing_symbols},
        calls=calls,
    )


def _uuid_text(raw):
    value = gpu_nvidia._CudaUuid((ctypes.c_ubyte * 16)(*raw))
    return gpu_nvidia._uuid_text(value)


def _nvml(matches=(), *, init_result=0, missing_symbols=()):
    """matches are (prefixed UUID bytes, handle integer, is_mig result)."""

    by_uuid = {identity: (handle, mig) for identity, handle, mig in matches}
    by_handle = {handle: mig for _identity, handle, mig in matches}
    calls = []

    def init():
        calls.append(("nvmlInit_v2",))
        return init_result

    def shutdown():
        calls.append(("nvmlShutdown",))
        return 0

    def get_handle(identity, pointer):
        calls.append(("nvmlDeviceGetHandleByUUID", identity))
        matched = by_uuid.get(identity)
        if matched is None:
            return 6
        _write(pointer, ctypes.c_void_p, matched[0])
        return 0

    def is_mig(handle, pointer):
        calls.append(("nvmlDeviceIsMigDeviceHandle", handle.value))
        result = by_handle[handle.value]
        if isinstance(result, int):
            _write(pointer, ctypes.c_uint, result)
            return 0
        return result[0]

    symbols = {
        "nvmlInit_v2": init,
        "nvmlShutdown": shutdown,
        "nvmlDeviceGetHandleByUUID": get_handle,
        "nvmlDeviceIsMigDeviceHandle": is_mig,
    }
    return SimpleNamespace(
        **{name: _FakeFunction(function) for name, function in symbols.items() if name not in missing_symbols},
        calls=calls,
    )


def _install(monkeypatch, driver, nvml=None):
    monkeypatch.setattr(gpu_nvidia, "_load_driver", lambda: driver)
    monkeypatch.setattr(gpu_nvidia, "_load_nvml", lambda: nvml)


def test_driver_positive_count_needs_no_cuda_runtime_or_nvml(monkeypatch):
    driver = _driver(count=2)
    _install(monkeypatch, driver)

    assert gpu_nvidia.probe_gpu() == {
        "groups": [
            {
                "kind": "cuda_device_unclassified",
                "count": 2,
                "model": "NVIDIA H100 80GB HBM3",
                "memory_bytes": str(80 * 2**30),
            }
        ]
    }
    assert driver.calls[:2] == [("cuInit", 0), ("cuDeviceGetCount",)]
    assert not hasattr(driver, "cudaGetDeviceCount")


@pytest.mark.parametrize("init_result,count_result", [(0, 0), (100, 0), (0, 100)])
def test_zero_and_explicit_no_device_results_are_observed_zero(monkeypatch, init_result, count_result):
    driver = _driver(count=0, init_result=init_result, count_result=count_result)
    _install(monkeypatch, driver)

    assert gpu_nvidia.probe_gpu() == {"groups": []}
    if init_result == 100:
        assert driver.calls == [("cuInit", 0)]


@pytest.mark.parametrize("init_result,count_result", [(34, 0), (999, 0), (0, 34), (0, 999)])
def test_driver_init_and_count_errors_are_unavailable(monkeypatch, init_result, count_result):
    _install(monkeypatch, _driver(init_result=init_result, count_result=count_result))

    assert gpu_nvidia.probe_gpu() is None


def test_missing_driver_is_unavailable_even_when_environment_looks_empty(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    _install(monkeypatch, None)

    assert gpu_nvidia.probe_gpu() is None


@pytest.mark.parametrize("mask", [None, "", "0", "1,0", "0,1,2,3"])
def test_raw_visible_devices_never_sets_or_overrides_driver_count(monkeypatch, mask):
    if mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    _install(monkeypatch, _driver(count=0))
    assert gpu_nvidia.probe_gpu() == {"groups": []}

    _install(monkeypatch, _driver(count=1))
    assert sum(group["count"] for group in gpu_nvidia.probe_gpu()["groups"]) == 1


@pytest.mark.parametrize("count", [-1, gpu_nvidia.MAX_CUDA_DEVICES + 1])
def test_invalid_or_excessive_driver_count_is_unavailable_without_metadata_loop(monkeypatch, count):
    driver = _driver(count=count)
    _install(monkeypatch, driver)

    assert gpu_nvidia.probe_gpu() is None
    assert driver.calls == [("cuInit", 0), ("cuDeviceGetCount",)]


@pytest.mark.parametrize("missing", ["cuInit", "cuDeviceGetCount"])
def test_missing_required_driver_symbol_is_unavailable(monkeypatch, missing):
    _install(monkeypatch, _driver(missing_symbols=(missing,)))

    assert gpu_nvidia.probe_gpu() is None


def test_nvml_init_or_symbol_failure_retains_count_as_unclassified(monkeypatch):
    raw = bytes(range(16))
    identity = ("GPU-" + _uuid_text(raw)).encode("ascii")
    driver = _driver(uuids=[raw])
    for nvml in (
        _nvml([(identity, 101, 0)], init_result=9),
        _nvml([(identity, 101, 0)], missing_symbols=("nvmlDeviceIsMigDeviceHandle",)),
    ):
        _install(monkeypatch, driver, nvml)
        assert gpu_nvidia.probe_gpu()["groups"][0]["kind"] == "cuda_device_unclassified"


@pytest.mark.parametrize("failure", ["lookup", "classification", "invalid_classification"])
def test_nvml_lookup_and_classification_failures_leave_device_unclassified(monkeypatch, failure):
    raw = bytes(range(16))
    identity = ("GPU-" + _uuid_text(raw)).encode("ascii")
    nvml = _nvml([] if failure == "lookup" else [(identity, 101, (9,) if failure == "classification" else 2)])
    _install(monkeypatch, _driver(models=["NVIDIA H100 MIG 1g.10gb"], uuids=[raw]), nvml)

    result = gpu_nvidia.probe_gpu()

    assert result["groups"][0]["kind"] == "cuda_device_unclassified"
    assert result["groups"][0]["model"] == "NVIDIA H100 MIG 1g.10gb"
    assert nvml.calls[-1] == ("nvmlShutdown",)


def test_nvml_exception_cannot_block_a_known_driver_count(monkeypatch):
    raw = bytes([1]) * 16
    nvml = _nvml([(("GPU-" + _uuid_text(raw)).encode(), 1000, 0)])
    nvml.nvmlDeviceGetHandleByUUID = _FakeFunction(lambda _identity, _pointer: 1 / 0)
    _install(monkeypatch, _driver(uuids=[raw]), nvml)

    result = gpu_nvidia.probe_gpu()

    assert result["groups"][0]["kind"] == "cuda_device_unclassified"
    assert result["groups"][0]["count"] == 1
    assert nvml.calls[-1] == ("nvmlShutdown",)


@pytest.mark.parametrize("prefix,is_mig,expected", [("GPU-", 1, "mig_compute_instance"), ("MIG-", 0, "full_gpu")])
def test_uuid_prefix_is_lookup_only_not_device_type(monkeypatch, prefix, is_mig, expected):
    raw = bytes([1]) * 16
    nvml = _nvml([((prefix + _uuid_text(raw)).encode(), 1000, is_mig)])
    _install(monkeypatch, _driver(uuids=[raw]), nvml)

    assert gpu_nvidia.probe_gpu()["groups"][0]["kind"] == expected


def test_full_mig_and_unclassified_devices_are_exclusive_groups(monkeypatch):
    raws = [bytes([index + 1]) * 16 for index in range(3)]
    nvml = _nvml(
        [
            (("GPU-" + _uuid_text(raws[0])).encode(), 1000, 0),
            (("MIG-" + _uuid_text(raws[1])).encode(), 1001, 1),
        ]
    )
    driver = _driver(
        count=3,
        models=["NVIDIA H100 80GB HBM3", "NVIDIA H100 MIG 1g.10gb", "NVIDIA H100 80GB HBM3"],
        memories=[80 * 2**30, 10 * 2**30, 80 * 2**30],
        uuids=raws,
    )
    _install(monkeypatch, driver, nvml)

    groups = gpu_nvidia.probe_gpu()["groups"]

    assert {group["kind"] for group in groups} == {
        "full_gpu",
        "mig_compute_instance",
        "cuda_device_unclassified",
    }
    assert sum(group["count"] for group in groups) == 3
    assert {group["kind"]: group["memory_bytes"] for group in groups} == {
        "full_gpu": str(80 * 2**30),
        "mig_compute_instance": str(10 * 2**30),
        "cuda_device_unclassified": str(80 * 2**30),
    }
    assert nvml.calls[-1] == ("nvmlShutdown",)
    assert not any("uuid" in str(key).lower() for group in groups for key in group)


def test_missing_uuid_name_and_memory_do_not_erase_driver_count(monkeypatch):
    driver = _driver(models=[None], memories=[None], uuids=[None])
    _install(monkeypatch, driver, _nvml())

    assert gpu_nvidia.probe_gpu() == {"groups": [{"kind": "cuda_device_unclassified", "count": 1}]}


def test_invalid_optional_model_does_not_erase_count(monkeypatch):
    _install(monkeypatch, _driver(models=["NVIDIA <private-description>"]))
    group = gpu_nvidia.probe_gpu()["groups"][0]
    assert group["count"] == 1
    assert "model" not in group


def test_missing_v2_uuid_does_not_fall_back_to_legacy_uuid(monkeypatch):
    driver = _driver(missing_symbols=("cuDeviceGetUuid_v2",))
    driver.cuDeviceGetUuid = _FakeFunction(lambda _pointer, _device: pytest.fail("legacy UUID is unsafe for MIG"))
    _install(monkeypatch, driver, _nvml())

    assert gpu_nvidia.probe_gpu()["groups"][0]["kind"] == "cuda_device_unclassified"


def test_device_get_race_keeps_the_counted_ordinal(monkeypatch):
    raw = bytes([1]) * 16
    nvml = _nvml([(("GPU-" + _uuid_text(raw)).encode(), 1000, 0)])
    driver = _driver(count=2, device_results={1: 101})
    _install(monkeypatch, driver, nvml)

    groups = gpu_nvidia.probe_gpu()["groups"]

    assert sum(group["count"] for group in groups) == 2
    assert {group["kind"] for group in groups} == {"full_gpu", "cuda_device_unclassified"}
    assert next(group for group in groups if group["kind"] == "cuda_device_unclassified") == {
        "kind": "cuda_device_unclassified",
        "count": 1,
    }


def test_duplicate_driver_uuids_are_not_used_for_nvml_classification(monkeypatch):
    raw = bytes([1]) * 16
    identity = ("GPU-" + _uuid_text(raw)).encode()
    nvml = _nvml([(identity, 1000, 0)])
    _install(monkeypatch, _driver(count=2, uuids=[raw, raw]), nvml)

    result = gpu_nvidia.probe_gpu()

    assert result["groups"][0]["kind"] == "cuda_device_unclassified"
    assert result["groups"][0]["count"] == 2
    assert nvml.calls == [("nvmlInit_v2",), ("nvmlShutdown",)]


def test_driver_model_containing_uuid_is_not_persisted(monkeypatch):
    raw = bytes([1]) * 16
    model = "NVIDIA H100 GPU-" + _uuid_text(raw)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "private-value")
    _install(monkeypatch, _driver(models=[model], uuids=[raw]))

    result = gpu_nvidia.probe_gpu()

    assert "model" not in result["groups"][0]
    assert "GPU-" not in str(result)
    assert "private-value" not in str(result)


def test_driver_uses_v2_memory_symbol_not_legacy_abi(monkeypatch):
    driver = _driver()
    driver.cuDeviceTotalMem = _FakeFunction(lambda _pointer, _device: pytest.fail("legacy size_t ABI"))
    _install(monkeypatch, driver)

    gpu_nvidia.probe_gpu()

    assert ("cuDeviceTotalMem_v2", 100) in driver.calls


def test_darwin_driver_loader_uses_legacy_library_without_nvml(monkeypatch):
    loads = []

    def cdll(name):
        loads.append(name)
        return "driver"

    monkeypatch.setattr(gpu_nvidia.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(gpu_nvidia.ctypes, "CDLL", cdll)

    assert gpu_nvidia._load_driver() == "driver"
    assert gpu_nvidia._load_nvml() is None
    assert loads == ["libcuda.dylib"]


def test_unrecognized_platform_does_not_attempt_gpu_libraries(monkeypatch):
    monkeypatch.setattr(gpu_nvidia.platform, "system", lambda: "Other")
    monkeypatch.setattr(gpu_nvidia, "_load_library", lambda _candidates: pytest.fail("unexpected library load"))

    assert gpu_nvidia._load_driver() is None
    assert gpu_nvidia._load_nvml() is None


def test_linux_system_loader_uses_versioned_driver_soname_only(monkeypatch):
    driver = _driver(count=0)
    calls = []

    def load_library(candidates):
        calls.append(candidates)
        return driver

    monkeypatch.setattr(gpu_nvidia.platform, "system", lambda: "Linux")
    monkeypatch.setattr(gpu_nvidia, "_load_library", load_library)
    monkeypatch.setattr(gpu_nvidia, "_load_nvml", lambda: None)

    assert gpu_nvidia.probe_gpu() == {"groups": []}
    assert calls == [("libcuda.so.1",)]
