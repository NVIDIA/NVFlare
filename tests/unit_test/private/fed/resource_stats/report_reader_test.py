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

"""Numeric participant reports are bound and checked when accepted or loaded."""

import json
from copy import deepcopy

import pytest

from nvflare.private.fed.resource_stats import report_reader
from nvflare.private.fed.resource_stats.measurements import (
    canonical_json_bytes,
    participant_measurements,
    sum_job_totals,
)
from nvflare.private.fed.resource_stats.report_reader import (
    KIND_PARTICIPANT_SUMMARY,
    KIND_RESOURCE_SUMMARY,
    read_resource_report,
)
from tests.resource_stats_report_test_utils import _accepted_entry, _participant_summary, _resource_summary


def test_participant_round_trip_preserves_json_numbers():
    record = _participant_summary()
    assert read_resource_report(canonical_json_bytes(record), KIND_PARTICIPANT_SUMMARY) == record
    encoded = json.loads(canonical_json_bytes(record))
    assert encoded["duration_seconds"] == 2
    assert encoded["resource_capacity"]["gpu"] == {"count": 1}
    assert encoded["resource_time"]["gpu"] == {"instance_seconds": 2}
    assert encoded["resource_utilization"]["cpu_consumed"] == {"cpu_seconds": 1.5}


@pytest.mark.parametrize("change", ["version", "extra", "numeric_string", "missing_measurement", "unsafe_identity"])
def test_report_boundary_rejects_invalid_record_contents(change):
    record = _participant_summary()
    if change == "version":
        record["schema_version"] = "2.0"
    elif change == "extra":
        record["resource_capacity"]["gpu"]["model"] = "private metadata"
    elif change == "numeric_string":
        record["duration_seconds"] = "2"
    elif change == "missing_measurement":
        del record["resource_utilization"]["cpu_consumed"]
    else:
        record["participant_name"] = "../another-site"
    with pytest.raises(ValueError):
        read_resource_report(canonical_json_bytes(record), KIND_PARTICIPANT_SUMMARY)


@pytest.mark.parametrize("data", [b'{"kind":1,"kind":2}', b'{"value":NaN}', b"\xff", b"not JSON"])
def test_decoder_rejects_ambiguous_or_invalid_json(data):
    with pytest.raises(ValueError):
        read_resource_report(data)


def test_reader_enforces_size_kind_and_depth_limits(monkeypatch):
    data = canonical_json_bytes(_participant_summary())
    with pytest.raises(ValueError, match="expected record kind"):
        read_resource_report(data, KIND_RESOURCE_SUMMARY)
    with monkeypatch.context() as context:
        context.setitem(report_reader.MAX_RECORD_BYTES, KIND_PARTICIPANT_SUMMARY, len(data) - 1)
        with pytest.raises(ValueError, match="byte limit"):
            read_resource_report(data, KIND_PARTICIPANT_SUMMARY)
    record = _participant_summary()
    nested = record
    for _ in range(report_reader.MAX_JSON_DEPTH + 1):
        nested["extra"] = {}
        nested = nested["extra"]
    with pytest.raises(ValueError, match="depth"):
        read_resource_report(canonical_json_bytes(record))


def test_reader_sorts_known_collections_without_mutating_original():
    first, second = _accepted_entry("site-1"), _accepted_entry("site-2")
    first["resource_time"]["status"] = "partial"
    first["issues"] = [
        {"metric": "resource_time", "code": "observation_incomplete"},
        {"metric": "resource_time", "code": "attribution_incomplete"},
    ]
    record = _resource_summary(participants=[second, first])
    original = deepcopy(record)
    normalized = read_resource_report(canonical_json_bytes(record))
    assert record == original
    assert normalized["participants"][0]["participant_name"] == "site-1"
    assert normalized["participants"][0]["issues"] == [
        {"metric": "resource_time", "code": "attribution_incomplete"},
        {"metric": "resource_time", "code": "observation_incomplete"},
    ]
    copied = participant_measurements(normalized["participants"][0])
    copied["resource_capacity"]["gpu"]["count"] = 99
    assert record["participants"][1]["resource_capacity"]["gpu"] == {"count": 1}


def test_job_totals_sum_explicit_time_and_keep_missing_report_coverage():
    first, second = _accepted_entry("site-1"), _accepted_entry("site-2")
    second["duration_seconds"] = 999
    totals = sum_job_totals([first, second, {"participant_name": "site-3", "role": "client", "status": "missing"}])
    assert totals["resource_time"] == {
        "status": "partial",
        "cpu": {"unit_seconds": 8},
        "memory": {"byte_seconds": 16384},
        "gpu": {"instance_seconds": 4},
    }
    assert totals["resource_utilization"]["cpu_consumed"] == {"status": "partial", "cpu_seconds": 3}
    assert "duration_seconds" not in totals
    assert totals["issues"] == [
        {"metric": metric, "code": "observation_incomplete"}
        for metric in ("cpu_consumed", "message_traffic", "resource_time", "retained_content")
    ]
    with pytest.raises(ValueError, match="unique"):
        read_resource_report(canonical_json_bytes(_resource_summary(participants=[first, first])))


def test_workspace_issues_stay_in_detail_and_do_not_leak_into_job_rows():
    record = _participant_summary()
    record["workspace_filesystem"] = {"status": "unavailable"}
    record["issues"] = [{"metric": "workspace_filesystem", "code": "unsupported"}]
    read_resource_report(canonical_json_bytes(record))
    values = participant_measurements(record)
    assert values["issues"] == []
    assert "workspace_filesystem" not in values
    assert record["issues"] == [{"metric": "workspace_filesystem", "code": "unsupported"}]


@pytest.mark.parametrize("change", ["missing", "nested", "wrong_metric", "complete"])
def test_participant_report_requires_one_consistent_scope_issue_list(change):
    record = _participant_summary()
    if change == "missing":
        del record["issues"]
    elif change == "nested":
        record["resource_utilization"]["cpu_consumed"]["issues"] = []
    elif change == "wrong_metric":
        record["resource_utilization"]["cpu_consumed"] = {"status": "unavailable"}
        record["issues"] = [{"metric": "resource_time", "code": "unsupported"}]
    else:
        record["issues"] = [{"metric": "cpu_consumed", "code": "observation_incomplete"}]
    with pytest.raises(ValueError):
        read_resource_report(canonical_json_bytes(record))


@pytest.mark.parametrize(
    "causes,expected",
    [
        (["attribution_incomplete"], "attribution_incomplete"),
        (["unsupported"], "observation_incomplete"),
        (["attribution_incomplete", "observation_incomplete"], "observation_incomplete"),
    ],
)
def test_job_capacity_time_maps_unavailable_source_causes_without_losing_valid_subtotals(causes, expected):
    first, second = _accepted_entry("site-1"), _accepted_entry("site-2")
    second["resource_time"] = {"status": "unavailable"}
    second["issues"] = [{"metric": "resource_time", "code": code} for code in causes]
    totals = sum_job_totals([first, second])
    assert totals["resource_time"]["cpu"] == {"unit_seconds": 4}
    assert totals["resource_time"]["status"] == "partial"
    assert totals["issues"] == [{"metric": "resource_time", "code": expected}]


def test_all_unavailable_time_derives_observation_issue_and_missing_rows_have_no_scope():
    entry = _accepted_entry()
    entry["resource_time"] = {"status": "unavailable"}
    entry["issues"] = [{"metric": "resource_time", "code": "attribution_incomplete"}]
    totals = sum_job_totals([entry])
    assert totals["resource_time"] == {"status": "unavailable"}
    assert totals["issues"] == [{"metric": "resource_time", "code": "observation_incomplete"}]
    missing = {"participant_name": "site-2", "role": "client", "status": "missing"}
    read_resource_report(canonical_json_bytes(_resource_summary(participants=[missing])))
    missing["issues"] = []
    with pytest.raises(ValueError):
        read_resource_report(canonical_json_bytes(_resource_summary(participants=[missing])))


@pytest.mark.parametrize("field", ["reported_at", "received_at"])
@pytest.mark.parametrize(
    "timestamp", ["2026-11-03T00:00:00Z", "2024-02-29T23:59:59.1Z", "2026-11-03T00:00:00.123456789Z"]
)
def test_report_timestamps_preserve_supported_utc_precision(field, timestamp):
    record = _participant_summary() if field == "reported_at" else _resource_summary()
    target = record if field == "reported_at" else record["participants"][0]
    target[field] = timestamp
    decoded = read_resource_report(canonical_json_bytes(record))
    assert (decoded if field == "reported_at" else decoded["participants"][0])[field] == timestamp


@pytest.mark.parametrize("field", ["reported_at", "received_at"])
@pytest.mark.parametrize(
    "timestamp",
    [
        "20261103T00:00:00Z",
        "2026-11-03T000000Z",
        "2026-11-03 00:00:00Z",
        "2026-11-03T00:00Z",
        "2026-11-03T00:00:00.1234567890Z",
        "2026-11-03T00:00:00+00:00",
        "2026-02-29T00:00:00Z",
        "2026-02-29T00:00:00.1Z",
        "2026-11-03T24:00:00Z",
        "2026-11-03T24:00:00.123456789Z",
        "2026-11-03T23:59:60.1Z",
        "2026-11-03T00:00:00Z\n",
        True,
    ],
)
def test_report_timestamps_reject_other_iso_forms_and_invalid_dates(field, timestamp):
    record = _participant_summary() if field == "reported_at" else _resource_summary()
    target = record if field == "reported_at" else record["participants"][0]
    target[field] = timestamp
    with pytest.raises(ValueError, match="timestamp"):
        read_resource_report(canonical_json_bytes(record))
