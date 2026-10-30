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
from nvflare.private.fed.resource_stats.contract import (
    SCHEMA_VERSION,
    canonical_json_bytes,
    capacity_to_resource_time,
    load_and_validate,
)
from nvflare.private.fed.resource_stats.coordinator import (
    RESOURCE_REPORT_ACCEPTED,
    RESOURCE_REPORT_CONFLICT,
    RESOURCE_REPORT_DUPLICATE,
    RESOURCE_REPORT_INVALID,
    ResourceStatsCoordinator,
)


def _participant():
    record = {
        "duration_seconds": "10",
        "schema_version": SCHEMA_VERSION,
        "kind": "nvflare.resource_stats.participant_summary",
        "job_id": "job-1",
        "participant_name": "site-1",
        "reported_at": "2026-10-08T12:00:00Z",
        "resource_capacity": {
            "status": "partial",
            "issues": ["attribution_incomplete", "observation_incomplete"],
            "cpu": {"groups": [{"units": "2", "model": "CPU A"}, {"units": "1", "model": "CPU B"}]},
        },
        "workspace_filesystem": {"capacity_bytes": "1024"},
        "resource_utilization": {
            "cpu_consumed": {"cpu_seconds": "2"},
            "retained_content": {"bytes": "2"},
            "message_traffic": {"sent_bytes": "2", "received_bytes": "3"},
        },
    }
    record["resource_time"] = capacity_to_resource_time(record["resource_capacity"], record["duration_seconds"])
    return record


def _alternate(record):
    alternate = deepcopy(record)
    alternate["resource_capacity"]["issues"].reverse()
    alternate["resource_capacity"]["cpu"]["groups"].reverse()
    alternate["resource_capacity"]["cpu"]["groups"][0]["units"] = "1.000"
    alternate["duration_seconds"] = "10.000"
    alternate["resource_utilization"]["cpu_consumed"]["cpu_seconds"] = "2.0"
    alternate["resource_utilization"]["retained_content"]["bytes"] = "2.000"
    alternate["resource_utilization"]["message_traffic"]["sent_bytes"] = "2.0"
    alternate["resource_utilization"]["message_traffic"]["received_bytes"] = "3.0"
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
    saved = tmp_path / "resource_stats" / "participants" / "site-1.json"
    assert saved.read_bytes() == canonical_json_bytes(load_and_validate(first))
    assert summary["participants"][0]["resource_utilization"]["cpu_consumed"]["cpu_seconds"] == "2"
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
    changed["resource_utilization"]["cpu_consumed"]["cpu_seconds"] = "3.0"
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
