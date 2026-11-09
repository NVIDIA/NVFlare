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

import subprocess
from unittest.mock import Mock

import pytest

from nvflare.fuel.utils import gpu_utils


@pytest.mark.parametrize("returncode", [1, -9, -15])
def test_nvidia_smi_nonzero_exit(monkeypatch, returncode):
    monkeypatch.setattr(gpu_utils, "has_nvidia_smi", lambda: True)
    monkeypatch.setattr(
        gpu_utils.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], returncode, stdout="", stderr="failed"),
    )

    with pytest.raises(Exception, match="Failed to call nvidia-smi"):
        gpu_utils.get_host_gpu_ids()


@pytest.mark.parametrize("timeout", [None, 2.0])
def test_gpu_inventory_preserves_host_indices_and_applies_query_timeout(monkeypatch, timeout):
    monkeypatch.setattr(gpu_utils, "has_nvidia_smi", lambda: True)
    run = Mock(return_value=subprocess.CompletedProcess([], 0, stdout="index\n4\n1\n", stderr=""))
    monkeypatch.setattr(gpu_utils.subprocess, "run", run)

    assert gpu_utils.get_host_gpu_ids(**({"timeout": timeout} if timeout is not None else {})) == [4, 1]

    assert run.call_args.args[0] == ["nvidia-smi", "--query-gpu=index", "--format=csv"]
    assert run.call_args.kwargs["timeout"] == (30.0 if timeout is None else timeout)


@pytest.mark.parametrize(
    "failure",
    [
        OSError("cannot execute nvidia-smi"),
        subprocess.TimeoutExpired("nvidia-smi", 2.0),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid output"),
    ],
)
def test_nvidia_smi_execution_failures_are_reported_as_unavailable_inventory(monkeypatch, failure):
    monkeypatch.setattr(gpu_utils, "has_nvidia_smi", lambda: True)
    monkeypatch.setattr(gpu_utils.subprocess, "run", Mock(side_effect=failure))

    with pytest.raises(RuntimeError, match="Failed to call nvidia-smi") as error:
        gpu_utils.get_host_gpu_ids(timeout=2.0)
    assert error.value.__cause__ is failure


@pytest.mark.parametrize(
    "output",
    ["", "index\n", "wrong header\n0\n", "index\nbad\n", "index\n-1\n", "index\n0\n0\n", "index\n0\n\n"],
)
def test_missing_or_malformed_inventory_never_fabricates_an_empty_pool(monkeypatch, output):
    monkeypatch.setattr(gpu_utils, "has_nvidia_smi", lambda: True)
    monkeypatch.setattr(
        gpu_utils.subprocess, "run", Mock(return_value=subprocess.CompletedProcess([], 0, stdout=output, stderr=""))
    )

    with pytest.raises(RuntimeError, match="nvidia-smi"):
        gpu_utils.get_host_gpu_ids()


def test_missing_binary_preserves_legacy_empty_inventory_and_memory_fallback(monkeypatch):
    monkeypatch.setattr(gpu_utils, "has_nvidia_smi", lambda: False)
    run = Mock(side_effect=AssertionError("missing binary must not be executed"))
    monkeypatch.setattr(gpu_utils.subprocess, "run", run)

    assert gpu_utils.get_host_gpu_memory_total() == []
    assert gpu_utils.get_host_gpu_memory_free() == []
    assert gpu_utils.get_host_gpu_ids() == []
    run.assert_not_called()


@pytest.mark.parametrize("getter", [gpu_utils.get_host_gpu_memory_total, gpu_utils.get_host_gpu_memory_free])
def test_memory_queries_preserve_mib_units(monkeypatch, getter):
    monkeypatch.setattr(gpu_utils, "use_nvidia_smi", lambda _query: ["memory [MiB]", "1024 MiB", "2048 MiB"])

    assert getter() == [1024, 2048]
    with pytest.raises(RuntimeError, match="unit"):
        getter(unit="GiB")


@pytest.mark.parametrize(
    "mask,host_ids,expected",
    [
        (None, [0, 2, 5], [0, 2, 5]),
        (None, None, None),
        ("", None, []),
        ("  ", [0, 2, 5], []),
        ("2,0,2", [0, 2, 5], [2, 0]),
        ("2,9,0", [0, 2, 5], [2]),
        ("2,-1,0", [0, 2, 5], [2]),
        ("2,,0", [0, 2, 5], [2]),
        ("-1,0", [0, 2, 5], []),
        ("GPU-selected-device", [0, 2, 5], None),
        ("0,unsupported", [0, 2, 5], None),
        ("5,7", None, [5, 7]),
    ],
)
def test_integer_visibility_is_shared_without_host_discovery(monkeypatch, mask, host_ids, expected):
    if mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    query = Mock(side_effect=AssertionError("visibility parsing must not query the host"))
    monkeypatch.setattr(gpu_utils, "get_host_gpu_ids", query)

    assert gpu_utils.get_visible_gpu_ids(host_ids) == expected
    query.assert_not_called()
