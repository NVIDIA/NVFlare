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

"""Acceptance is based on validated values, not sender presentation."""

import json
from copy import deepcopy
from unittest.mock import patch

import pytest

from nvflare.private.fed.resource_stats import coordinator as coordinator_module
from nvflare.private.fed.resource_stats.coordinator import (
    RESOURCE_REPORT_ACCEPTED,
    RESOURCE_REPORT_CONFLICT,
    RESOURCE_REPORT_DUPLICATE,
    RESOURCE_REPORT_INVALID,
    ResourceStatsCoordinator,
)
from nvflare.private.fed.resource_stats.measurements import canonical_json_bytes, capacity_to_resource_time
from nvflare.private.fed.resource_stats.report_reader import read_resource_report
from tests.resource_stats_report_test_utils import _accepted_entry, _participant_summary


def _participant():
    record = _participant_summary(reported_at="2026-10-08T12:00:00Z")
    record.update(
        duration_seconds=10,
        resource_capacity={"status": "partial", "cpu": {"units": 3}},
        resource_utilization={
            "cpu_consumed": {"cpu_seconds": 2},
            "message_traffic": {"sent_bytes": 2, "received_bytes": 3},
        },
        issues=[
            {"metric": "resource_capacity", "code": "attribution_incomplete"},
            {"metric": "resource_capacity", "code": "observation_incomplete"},
        ],
    )
    record["resource_time"] = capacity_to_resource_time(
        record["resource_capacity"], record["duration_seconds"], scope_issues=record["issues"]
    )
    return record


def _alternate(record):
    alternate = deepcopy(record)
    alternate["issues"].reverse()
    alternate["issues"].append(dict(alternate["issues"][0]))
    alternate["resource_capacity"]["cpu"]["units"] = 3.0
    alternate["duration_seconds"] = 10.0
    alternate["resource_utilization"]["cpu_consumed"]["cpu_seconds"] = 2.0
    return dict(reversed(list(alternate.items())))


@pytest.mark.parametrize("compact_first", [False, True])
def test_equivalent_input_is_normalized_and_retry_is_duplicate_even_after_finalization(tmp_path, compact_first):
    coordinator = ResourceStatsCoordinator()
    coordinator.start_job("job-1", ["site-1"], tmp_path)
    original = _participant()
    pretty = json.dumps(original, indent=4).encode()
    compact = json.dumps(_alternate(original), separators=(",", ":")).encode()
    first, retry = (compact, pretty) if compact_first else (pretty, compact)

    assert (
        coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": first})
        == RESOURCE_REPORT_ACCEPTED
    )
    assert (
        coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": retry})
        == RESOURCE_REPORT_DUPLICATE
    )
    summary = coordinator.finalize_job("job-1")
    saved = tmp_path / "resource_stats" / "resource_summary.json"
    assert saved.read_bytes() == canonical_json_bytes(summary)
    entry = summary["participants"][0]
    assert entry == _accepted_entry(received_at=entry["received_at"], record=read_resource_report(first))
    assert (
        coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": retry})
        == RESOURCE_REPORT_DUPLICATE
    )


def test_changed_value_is_still_a_conflicting_report(tmp_path):
    coordinator = ResourceStatsCoordinator()
    coordinator.start_job("job-1", ["site-1"], tmp_path)
    original = _participant()
    assert (
        coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": canonical_json_bytes(original)})
        == RESOURCE_REPORT_ACCEPTED
    )
    changed = _alternate(original)
    changed["resource_utilization"]["cpu_consumed"]["cpu_seconds"] = 3.0
    assert (
        coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": json.dumps(changed).encode()})
        == RESOURCE_REPORT_CONFLICT
    )


def test_normalized_record_must_also_fit_the_per_report_limit(tmp_path):
    coordinator = ResourceStatsCoordinator()
    coordinator.start_job("job-1", ["site-1"], tmp_path)
    compact = json.dumps(_participant(), separators=(",", ":")).encode()
    assert len(canonical_json_bytes(_participant())) > len(compact) + 1
    with patch.object(coordinator_module, "MAX_PARTICIPANT_SUMMARY_BYTES", len(compact) + 1):
        assert (
            coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": compact})
            == RESOURCE_REPORT_INVALID
        )
    invalid = next(
        item for item in coordinator.finalize_job("job-1")["participants"] if item["participant_name"] == "site-1"
    )
    assert invalid["status"] == "invalid"
    assert "issues" not in invalid


@pytest.mark.parametrize("field,value", [("job_id", "another-job"), ("participant_name", "another-site")])
def test_presentation_flexibility_does_not_relax_identity(tmp_path, field, value):
    coordinator = ResourceStatsCoordinator()
    coordinator.start_job("job-1", ["site-1"], tmp_path)
    record = _alternate(_participant())
    record[field] = value
    assert (
        coordinator.accept_resource_report("job-1", "site-1", {"participant_summary": json.dumps(record).encode()})
        == RESOURCE_REPORT_INVALID
    )
