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

"""CPU quota formatting and bounded GPU helper input, without native calls."""

from copy import deepcopy

import pytest

from nvflare.private.fed.resource_stats.contract import ContractError, normalize_gpu_capacity, normalize_quota_units


@pytest.mark.parametrize(
    "quota,period,expected",
    [
        (200000, 100000, "2"),
        (1, 3, "0.333333333"),
        (1, 1000000000, "0.000000001"),
        (1048576, 1, "1048576"),
    ],
)
def test_quota_ratio_is_bounded_and_floored(quota, period, expected):
    assert normalize_quota_units(quota, period) == expected


@pytest.mark.parametrize(
    "quota,period",
    [(True, 1), (1, False), ("1", 1), (0, 1), (1, -1), (2**64, 1), (1, 2**64)],
)
def test_quota_requires_positive_u64_integers(quota, period):
    with pytest.raises(ValueError):
        normalize_quota_units(quota, period)


@pytest.mark.parametrize("quota,period", [(1048577, 1), (1, 1000000001)])
def test_quota_outside_visible_or_precision_bounds_is_unavailable(quota, period):
    with pytest.raises(ValueError):
        normalize_quota_units(quota, period)


def test_gpu_capacity_normalizes_quantities_and_order_without_mutating_input():
    source = {
        "groups": [
            {
                "kind": "mig_compute_instance",
                "count": "01.000",
                "model": "NVIDIA H100",
                "memory_bytes": "010.0",
                "mig_profile": "1g.10gb",
            },
            {"kind": "full_gpu", "count": "0002", "memory_bytes": str(2**64 - 1)},
        ]
    }
    original = deepcopy(source)
    assert normalize_gpu_capacity(source) == {
        "groups": [
            {"kind": "full_gpu", "count": "2", "memory_bytes": str(2**64 - 1)},
            {
                "kind": "mig_compute_instance",
                "count": "1",
                "model": "NVIDIA H100",
                "memory_bytes": "10",
                "mig_profile": "1g.10gb",
            },
        ]
    }
    assert source == original
    assert normalize_gpu_capacity({"groups": []}) == {"groups": []}


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"groups": {}},
        {"groups": [], "uuid": "private"},
        {"groups": [{}]},
        {"groups": [[]]},
    ],
)
def test_gpu_helper_shape_is_fixed(value):
    with pytest.raises(ContractError):
        normalize_gpu_capacity(value)


@pytest.mark.parametrize(
    "changes",
    [
        {"count": 1},
        {"count": True},
        {"count": "0"},
        {"count": "-1"},
        {"count": "1.5"},
        {"count": "1e1"},
        {"count": "4097"},
        {"kind": "physical_host_gpu"},
        {"kind": []},
        {"uuid": "private"},
        {"memory_bytes": "0"},
        {"memory_bytes": str(2**64)},
        {"model": "x" * 129},
        {"model": {}},
        {"model": "GPU-01234567-89ab-cdef-0123-456789abcdef"},
        {"model": "host uuid label"},
        {"mig_profile": "1g.10gb"},
        {"kind": "mig_compute_instance", "mig_profile": "bad"},
    ],
)
def test_gpu_helper_rejects_bad_counts_or_private_metadata(changes):
    with pytest.raises(ContractError):
        normalize_gpu_capacity({"groups": [{"kind": "full_gpu", "count": "1", **changes}]})


@pytest.mark.parametrize(
    "groups",
    [
        [{"kind": "full_gpu", "count": "1"}] * 4097,
        [
            {"kind": "full_gpu", "count": "4096"},
            {"kind": "cuda_device_unclassified", "count": "1"},
        ],
        [
            {"kind": "full_gpu", "count": "1", "memory_bytes": "10"},
            {"kind": "full_gpu", "count": "2", "memory_bytes": "010.0"},
        ],
    ],
)
def test_gpu_helper_rejects_excessive_or_duplicate_groups(groups):
    with pytest.raises(ContractError):
        normalize_gpu_capacity({"groups": groups})
