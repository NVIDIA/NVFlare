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

"""Study reports sum included jobs and keep selection coverage separate."""

from copy import deepcopy

import pytest

from nvflare.private.fed.resource_stats.contract import (
    KIND_STUDY_SUMMARY,
    SCHEMA_VERSION,
    ContractError,
    canonical_json_bytes,
    derive_job_totals,
    derive_study_totals,
    load_and_validate,
    validate_record,
)
from tests.resource_stats_contract_test_utils import _accepted_entry


def _study():
    rows = [
        {
            "job_id": "job-1",
            "job_name": "An ordinary display name é",
            "job_status": "FINISHED:COMPLETED",
            "resource_data": "included",
            "totals": derive_job_totals([_accepted_entry()]),
        },
        {"job_id": "job-2", "job_name": "Unavailable", "job_status": "FINISHED:FAILED", "resource_data": "unavailable"},
        {"job_id": "job-3", "job_name": "Running", "job_status": "RUNNING", "resource_data": "nonterminal"},
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND_STUDY_SUMMARY,
        "selection": {"study_name": "default"},
        "generated_at": "2026-09-17T12:00:00Z",
        "coverage": {"selected_jobs": 3, "included_jobs": 1, "unavailable_jobs": 1, "nonterminal_jobs": 1},
        "jobs": rows,
        "totals": derive_study_totals(rows),
    }


def test_study_totals_preserve_numeric_units_and_distinct_coverage():
    record = _study()
    assert load_and_validate(canonical_json_bytes(record), KIND_STUDY_SUMMARY) == record
    assert record["totals"]["resource_time"]["gpu"] == {"instance_seconds": 2}
    assert record["totals"]["resource_utilization"]["message_traffic"] == {"sent_bytes": 3, "received_bytes": 4}
    assert "status" not in record["totals"]["resource_time"]
    assert record["coverage"] == {"selected_jobs": 3, "included_jobs": 1, "unavailable_jobs": 1, "nonterminal_jobs": 1}


@pytest.mark.parametrize("change", ["coverage", "total", "duplicate", "status", "unknown_field"])
def test_study_reader_checks_received_rows_and_totals(change):
    record = deepcopy(_study())
    if change == "coverage":
        record["coverage"]["included_jobs"] = 2
    elif change == "total":
        record["totals"]["resource_time"]["gpu"]["instance_seconds"] = 99
    elif change == "duplicate":
        record["jobs"][1]["job_id"] = "job-1"
    elif change == "status":
        record["jobs"][0]["job_status"] = "RUNNING"
    else:
        record["new_field"] = "future value"
    with pytest.raises(ContractError):
        validate_record(record)


def test_empty_study_has_unavailable_totals_without_invented_zeros():
    assert derive_study_totals([]) == {
        "resource_time": {"status": "unavailable"},
        "resource_utilization": {
            field: {"status": "unavailable"} for field in ("cpu_consumed", "retained_content", "message_traffic")
        },
        "issues": [
            {"metric": metric, "code": "observation_incomplete"}
            for metric in ("cpu_consumed", "message_traffic", "resource_time", "retained_content")
        ],
    }


def test_study_accepts_equivalent_issue_order_without_changing_coverage():
    record = _study()
    record["jobs"][0]["totals"]["resource_time"]["status"] = "partial"
    record["jobs"][0]["totals"]["issues"] = [
        {"metric": "resource_time", "code": "observation_incomplete"},
        {"metric": "resource_time", "code": "attribution_incomplete"},
    ]
    record["totals"] = derive_study_totals(record["jobs"])
    record["totals"]["issues"].reverse()
    normalized = load_and_validate(canonical_json_bytes(record), KIND_STUDY_SUMMARY)
    assert normalized["totals"]["issues"] == [
        {"metric": "resource_time", "code": "attribution_incomplete"},
        {"metric": "resource_time", "code": "observation_incomplete"},
    ]
    assert "issues" not in normalized
    assert normalized["coverage"] == record["coverage"]


def test_study_has_issues_only_in_its_measurement_scopes():
    record = _study()
    assert record["totals"]["issues"] == []
    assert record["jobs"][0]["totals"]["issues"] == []
    record["issues"] = []
    with pytest.raises(ContractError):
        validate_record(record)
