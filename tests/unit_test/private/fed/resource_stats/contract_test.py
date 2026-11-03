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

"""Numeric measurements retain units, coverage, and independent arithmetic."""

from copy import deepcopy

import pytest

from nvflare.private.fed.resource_stats.contract import (
    ContractError,
    aggregate_cpu_consumed,
    aggregate_message_traffic,
    aggregate_resource_time,
    aggregate_scope_issues,
    capacity_to_resource_time,
    normalize_duration,
    normalize_measurement,
    normalize_scope_issues,
    normalize_worker_record,
    scoped_measurement,
)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cpu_consumed", {"cpu_seconds": 0.125}),
        ("retained_content", {"bytes": 10**50}),
        ("message_traffic", {"sent_bytes": 3, "received_bytes": 7}),
        ("workspace_filesystem", {"capacity_bytes": 4096}),
        ("resource_capacity", {"cpu": {"units": 1.5}, "memory": {"bytes": 1024}, "gpu": {"count": 0}}),
        ("resource_capacity", {"cpu": {"units": 1.5}, "memory": {"bytes": 0}, "gpu": {"count": 0}}),
    ],
)
def test_measurements_copy_native_numbers_without_mutating_input(field, value):
    original = deepcopy(value)
    result = normalize_measurement(field, value)
    assert result == original == value
    assert result is not value


@pytest.mark.parametrize("value", [True, "1", -1, float("nan"), float("inf"), None])
def test_duration_requires_a_finite_nonnegative_number(value):
    with pytest.raises(ContractError):
        normalize_duration(value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cpu_consumed", {"cpu_seconds": "2"}),
        ("cpu_consumed", {"cpu_seconds": float("nan")}),
        ("cpu_consumed", {"cpu_seconds": True}),
        ("retained_content", {"bytes": 1.0}),
        ("message_traffic", {"sent_bytes": 1, "received_bytes": False}),
        ("cpu_consumed", {"cpu_seconds": 1, "duration_seconds": 2}),
        ("cpu_consumed", {"status": "unavailable", "issues": ["unsupported"], "cpu_seconds": 1}),
        ("retained_content", {"status": "partial", "bytes": 1, "issues": ["unsupported"]}),
        ("retained_content", {"status": "unavailable", "issues": ["unsupported", "unsupported"]}),
    ],
)
def test_measurements_reject_invalid_numbers_and_quality_combinations(field, value):
    with pytest.raises(ContractError):
        normalize_measurement(field, value)


@pytest.mark.parametrize(
    "dimension,value",
    [
        ("cpu", {"units": 0}),
        ("cpu", {"units": float("inf")}),
        ("memory", {"bytes": 1.0}),
        ("memory", {"bytes": -1}),
        ("gpu", {"count": 0.5}),
        ("gpu", {"count": True}),
        ("gpu", {"count": -1}),
        ("gpu", {"groups": []}),
        ("gpu", {"count": 1, "model": "private metadata"}),
    ],
)
def test_capacity_has_flat_numeric_dimensions(dimension, value):
    with pytest.raises(ContractError):
        normalize_measurement("resource_capacity", {"status": "partial", dimension: value})


@pytest.mark.parametrize("memory_bytes", [0, 1024])
def test_capacity_time_uses_one_duration_and_preserves_zero_capacity(memory_bytes):
    capacity = {"cpu": {"units": 1.5}, "memory": {"bytes": memory_bytes}, "gpu": {"count": 0}}
    assert capacity_to_resource_time(capacity, 0.5) == {
        "cpu": {"unit_seconds": 0.75},
        "memory": {"byte_seconds": memory_bytes * 0.5},
        "gpu": {"instance_seconds": 0},
    }
    assert capacity_to_resource_time(capacity, 0) == {
        "cpu": {"unit_seconds": 0},
        "memory": {"byte_seconds": 0},
        "gpu": {"instance_seconds": 0},
    }
    assert capacity["cpu"] == {"units": 1.5}


def test_product_overflow_preserves_other_capacity_time_dimensions():
    capacity = {"cpu": {"units": 1e308}, "memory": {"bytes": 1024}, "gpu": {"count": 2}}
    issues = []
    assert capacity_to_resource_time(capacity, 2.0, issues) == {
        "status": "partial",
        "memory": {"byte_seconds": 2048},
        "gpu": {"instance_seconds": 4},
    }
    assert issues == [{"metric": "resource_time", "code": "observation_incomplete"}]


def test_aggregation_retains_missing_coverage_and_exact_integer_bytes():
    missing = {"status": "unavailable"}
    assert aggregate_cpu_consumed([{"cpu_seconds": 2.5}, missing]) == {
        "status": "partial",
        "cpu_seconds": 2.5,
    }
    assert aggregate_message_traffic(
        [{"sent_bytes": 10**50, "received_bytes": 7}, {"sent_bytes": 2, "received_bytes": 4}]
    ) == {"sent_bytes": 10**50 + 2, "received_bytes": 11}
    capacity = {"cpu": {"units": 2}, "memory": {"bytes": 8}, "gpu": {"count": 1}}
    values = [capacity_to_resource_time(capacity, duration) for duration in (2, 3)]
    assert aggregate_resource_time(values) == {
        "cpu": {"unit_seconds": 10},
        "memory": {"byte_seconds": 40},
        "gpu": {"instance_seconds": 5},
    }
    with pytest.raises(ContractError):
        aggregate_cpu_consumed([{"cpu_seconds": 1e308}, {"cpu_seconds": 1e308}])


def test_worker_record_binds_numeric_measurements_to_the_owned_attempt():
    identity = dict(job_id="job-1", site_name="site-1", task_id="task-1", task_name="train", attempt_id="attempt-1")
    record = {
        "schema_version": "1.0",
        "identity": identity,
        "scope": "worker",
        "issues": [],
        "duration_seconds": 0.5,
        "resource_capacity": {"cpu": {"units": 2}, "memory": {"bytes": 1024}, "gpu": {"count": 1}},
        "cpu_consumed": {"cpu_seconds": 0.25},
        "message_traffic": {"sent_bytes": 3, "received_bytes": 5},
    }
    copied = normalize_worker_record(record, identity)
    assert copied == record
    copied["identity"]["attempt_id"] = "another-attempt"
    assert record["identity"] == identity
    with pytest.raises(ContractError, match="owned execution"):
        normalize_worker_record(copied, identity)


def test_shared_issues_keep_independent_causes_and_partial_numeric_values():
    measurements = {
        "cpu_consumed": {"status": "partial", "cpu_seconds": 0.25},
        "resource_capacity": {"status": "partial", "cpu": {"units": 2}},
        "message_traffic": {"sent_bytes": 0, "received_bytes": 0},
    }
    issues = [
        {"metric": "resource_capacity", "code": "observation_incomplete"},
        {"metric": "cpu_consumed", "code": "attribution_incomplete"},
        {"metric": "cpu_consumed", "code": "attribution_incomplete"},
    ]
    assert normalize_scope_issues(issues, measurements) == [
        {"metric": "cpu_consumed", "code": "attribution_incomplete"},
        {"metric": "resource_capacity", "code": "observation_incomplete"},
    ]
    assert measurements["cpu_consumed"]["cpu_seconds"] == 0.25
    assert normalize_scope_issues([], {"cpu_consumed": {"cpu_seconds": 0}}) == []
    assert normalize_scope_issues(
        [{"metric": "cpu_consumed", "code": "unsupported"}],
        {"cpu_consumed": {"status": "unavailable"}},
    ) == [{"metric": "cpu_consumed", "code": "unsupported"}]


@pytest.mark.parametrize(
    "issues,measurements",
    [
        ({}, {}),
        ([{"metric": "cpu_consumed", "code": "invented"}], {}),
        ([{"metric": "invented", "code": "unsupported"}], {}),
        ([{"metric": "cpu_consumed", "code": "unsupported", "detail": "private"}], {}),
        ([{"metric": "cpu_consumed", "code": "unsupported"}], {}),
        ([{"metric": "cpu_consumed", "code": "unsupported"}], {"cpu_consumed": {"cpu_seconds": 0}}),
        ([], {"cpu_consumed": {"status": "unavailable"}}),
        (
            [{"metric": "cpu_consumed", "code": "unsupported"}],
            {"cpu_consumed": {"status": "partial", "cpu_seconds": 1}},
        ),
        (
            [{"metric": "message_traffic", "code": "attribution_incomplete"}],
            {"message_traffic": {"status": "unavailable"}},
        ),
    ],
)
def test_shared_issues_reject_unknown_or_inconsistent_causes(issues, measurements):
    with pytest.raises(ContractError):
        normalize_scope_issues(issues, measurements)


def test_scope_assembly_moves_private_codes_once_without_mutating_the_producer():
    private = {"status": "partial", "cpu_seconds": 0.5, "issues": ["attribution_incomplete"]}
    issues = []
    assert scoped_measurement("cpu_consumed", private, issues) == {"status": "partial", "cpu_seconds": 0.5}
    assert issues == [{"metric": "cpu_consumed", "code": "attribution_incomplete"}]
    assert private["issues"] == ["attribution_incomplete"]


def test_capacity_time_shares_causes_and_never_invents_zero_when_duration_is_missing():
    capacity = {"status": "partial", "cpu": {"units": 2}}
    issues = [{"metric": "resource_capacity", "code": "attribution_incomplete"}]
    assert capacity_to_resource_time(capacity, 3, scope_issues=issues) == {
        "status": "partial",
        "cpu": {"unit_seconds": 6},
    }
    assert issues[-1] == {"metric": "resource_time", "code": "attribution_incomplete"}
    issues = []
    assert capacity_to_resource_time(capacity, None, issues) == {"status": "unavailable"}
    assert issues == [{"metric": "resource_time", "code": "observation_incomplete"}]


def test_aggregate_scope_issues_map_source_failures_and_retain_attribution():
    assert aggregate_scope_issues(
        [
            {"metric": "cpu_consumed", "code": "unsupported"},
            {"metric": "cpu_consumed", "code": "attribution_incomplete"},
            {"metric": "message_traffic", "code": "malformed_source"},
        ],
        {
            "cpu_consumed": {"status": "partial", "cpu_seconds": 1},
            "message_traffic": {"status": "partial", "sent_bytes": 2, "received_bytes": 3},
        },
    ) == [
        {"metric": "cpu_consumed", "code": "attribution_incomplete"},
        {"metric": "cpu_consumed", "code": "observation_incomplete"},
        {"metric": "message_traffic", "code": "observation_incomplete"},
    ]
