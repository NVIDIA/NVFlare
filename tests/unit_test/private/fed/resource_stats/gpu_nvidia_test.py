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


"""Controlled CUDA Driver count calls, without a GPU or process launch."""

import ctypes
from types import SimpleNamespace

import pytest

from nvflare.private.fed.resource_stats.probes import gpu_nvidia as probe


@pytest.fixture(autouse=True)
def no_visibility_override(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


class _Function:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


def _driver(monkeypatch, *, count=2, init_result=0, count_result=0, initialized_only=False):
    calls = []

    def initialize(flags):
        calls.append(("cuInit", flags))
        return init_result

    def get_count(pointer):
        calls.append(("cuDeviceGetCount",))
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = count
        return count_result

    driver = SimpleNamespace(cuDeviceGetCount=_Function(get_count))
    if not initialized_only:
        driver.cuInit = _Function(initialize)
    monkeypatch.setattr(probe, "_load_driver", lambda: driver)
    return calls


def test_probe_queries_only_driver_count(monkeypatch):
    calls = _driver(monkeypatch)
    assert probe.probe_gpu() == {"count": 2}
    assert calls == [("cuInit", 0), ("cuDeviceGetCount",)]


@pytest.mark.parametrize("count,init_result,count_result", [(0, 0, 0), (2, 100, 0), (2, 0, 100)])
def test_explicit_no_device_is_observed_zero(monkeypatch, count, init_result, count_result):
    _driver(monkeypatch, count=count, init_result=init_result, count_result=count_result)
    assert probe.probe_gpu() == {"count": 0}


@pytest.mark.parametrize("count,init_result,count_result", [(-1, 0, 0), (4097, 0, 0), (2, 34, 0), (2, 0, 999)])
def test_failed_or_out_of_range_discovery_is_unavailable(monkeypatch, count, init_result, count_result):
    _driver(monkeypatch, count=count, init_result=init_result, count_result=count_result)
    assert probe.probe_gpu() is None


def test_empty_mask_avoids_initialization_with_a_loadable_driver(monkeypatch):
    calls = _driver(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert probe.probe_gpu() == {"count": 0}
    assert calls == []
    monkeypatch.setattr(probe, "_load_driver", lambda: None)
    assert probe.probe_gpu() is None


@pytest.mark.parametrize("mask", ["0,2", "GPU-private-identity", "MIG-private-identity", "not-a-mask"])
def test_nonempty_visibility_uses_driver_authority_without_rewriting_environment(monkeypatch, mask):
    calls = _driver(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    assert probe.probe_gpu() == {"count": 2}
    assert probe.os.environ["CUDA_VISIBLE_DEVICES"] == mask
    assert calls == [("cuInit", 0), ("cuDeviceGetCount",)]


@pytest.mark.parametrize(
    "count,status,mask,expected",
    [
        (2, 0, "", {"count": 2}),
        (0, 0, None, {"count": 0}),
        (2, 100, None, {"count": 0}),
        (2, 3, "", {"count": 0}),
        (2, 3, "0", None),
        (2, 999, None, None),
        (-1, 0, None, None),
        (4097, 0, None, None),
    ],
)
def test_initialized_probe_never_initializes_and_prefers_live_state(monkeypatch, count, status, mask, expected):
    calls = _driver(monkeypatch, count=count, count_result=status, initialized_only=True)
    if mask is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    assert probe.probe_initialized_gpu() == expected
    assert calls == [("cuDeviceGetCount",)]


def test_missing_driver_or_required_symbol_is_unavailable(monkeypatch):
    monkeypatch.setattr(probe, "_load_driver", lambda: None)
    assert probe.probe_gpu() is None
    assert probe.probe_initialized_gpu() is None
    monkeypatch.setattr(probe, "_load_driver", lambda: SimpleNamespace())
    assert probe.probe_gpu() is None
    assert probe.probe_initialized_gpu() is None


def test_driver_library_load_failure_is_unavailable(monkeypatch):
    def unavailable(name):
        assert name == "libcuda.so.1"
        raise OSError("driver unavailable")

    monkeypatch.setattr(probe.platform, "system", lambda: "Linux")
    monkeypatch.setattr(probe.ctypes, "CDLL", unavailable)
    assert probe.probe_gpu() is None
