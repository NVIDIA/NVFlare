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


"""GPU count policy with controlled inventory, without a GPU or subprocess."""

import subprocess
from unittest.mock import Mock

import pytest

from nvflare.fuel.utils import gpu_utils
from nvflare.private.fed.resource_stats.probes import gpu_nvidia as probe


@pytest.mark.parametrize(
    "mask,expected",
    [
        (None, 3),
        ("4,0", 2),
        ("4,4,0", 2),
        ("2,9,0", 1),
        ("9,0", 0),
        ("0,-1,4", 1),
        ("0,,4", 1),
        ("GPU-selected-device", None),
        ("0,GPU-selected-device", None),
        ("not-an-index", None),
    ],
)
def test_count_matches_shared_gpu_selection(monkeypatch, mask, expected):
    if mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    queries = []

    def inventory(*, timeout):
        queries.append(timeout)
        return [0, 2, 4]

    monkeypatch.setattr(probe, "get_host_gpu_ids", inventory)
    assert probe.probe_gpu() == (None if expected is None else {"count": expected})
    assert queries == [2.0]


@pytest.mark.parametrize(
    "mask,expected",
    [
        ("", {"count": 0}),
        ("  ", {"count": 0}),
        ("-1", {"count": 0}),
        ("-1,unsupported", {"count": 0}),
    ],
)
def test_empty_selections_do_not_query_inventory(monkeypatch, mask, expected):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)

    queries = []

    def unexpected_inventory(**_kwargs):
        queries.append(True)
        return [0, 2, 4]

    monkeypatch.setattr(probe, "get_host_gpu_ids", unexpected_inventory)
    assert probe.probe_gpu() == expected
    assert not queries


@pytest.mark.parametrize(
    "host_ids,mask,expected",
    [
        ([0, 1], "0,99,unsupported", 1),
        ([0, 1], "99,unsupported", 0),
        ([0, 1, 99], "0,99,unsupported", None),
    ],
)
def test_inventory_controls_selector_stop_point(monkeypatch, host_ids, mask, expected):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    queries = []

    def inventory(*, timeout):
        queries.append(timeout)
        return host_ids

    monkeypatch.setattr(probe, "get_host_gpu_ids", inventory)
    assert probe.probe_gpu() == (None if expected is None else {"count": expected})
    assert queries == [2.0]


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("query failed"),
        FileNotFoundError(),
        subprocess.TimeoutExpired("nvidia-smi", 2.0),
        ValueError("bad inventory"),
    ],
)
def test_inventory_failures_are_unavailable(monkeypatch, error):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def failed_inventory(**_kwargs):
        raise error

    monkeypatch.setattr(probe, "get_host_gpu_ids", failed_inventory)
    assert probe.probe_gpu() is None


def test_empty_unverified_inventory_is_unavailable(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(probe, "get_host_gpu_ids", lambda **_kwargs: [])
    assert probe.probe_gpu() is None


def test_missing_nvidia_smi_is_unavailable_not_zero(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gpu_utils, "has_nvidia_smi", lambda: False)
    run = Mock()
    monkeypatch.setattr(gpu_utils.subprocess, "run", run)

    assert probe.probe_gpu() is None
    run.assert_not_called()
