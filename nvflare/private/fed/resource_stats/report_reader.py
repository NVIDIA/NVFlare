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

"""Read bounded participant, job and study reports at their receiving boundaries."""

import json
import re
from datetime import datetime

from .handoff import _read_capacity, _read_fields, _read_issues, _read_number, _read_usage, _read_workspace
from .measurements import SCHEMA_VERSION, sum_job_totals, sum_study_totals

KIND_PARTICIPANT_SUMMARY = "nvflare.resource_stats.participant_summary"
KIND_RESOURCE_SUMMARY = "nvflare.resource_stats.resource_summary"
KIND_STUDY_SUMMARY = "nvflare.resource_stats.study_summary"
MAX_STUDY_SUMMARY_BYTES = 64 * 1024 * 1024
MAX_STUDY_JOBS = 10_000
MAX_JOB_NAME_LENGTH = 255
JOB_STATUS_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._:/+-]{0,63}$")
STUDY_NAME_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?$")
MAX_PARTICIPANT_SUMMARY_BYTES = 1024 * 1024
MAX_RESOURCE_SUMMARY_BYTES = 64 * 1024 * 1024
MAX_RECORD_BYTES = {
    KIND_PARTICIPANT_SUMMARY: MAX_PARTICIPANT_SUMMARY_BYTES,
    KIND_RESOURCE_SUMMARY: MAX_RESOURCE_SUMMARY_BYTES,
    KIND_STUDY_SUMMARY: MAX_STUDY_SUMMARY_BYTES,
}
MAX_PARTICIPANTS = 10_000
MAX_JSON_DEPTH = 32
PARTICIPANT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}$")
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _read_identifier(value, pattern):
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError("invalid resource report identity")
    return value


def _read_timestamp(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,9})?Z", value
    ):
        raise ValueError("resource report requires a UTC timestamp")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("resource report requires a valid UTC timestamp") from error
    return value


def _read_measurements(record):
    utilization = _read_fields(record["resource_utilization"], {"cpu_consumed", "retained_content", "message_traffic"})
    values = {
        "resource_capacity": _read_capacity(record["resource_capacity"]),
        "resource_time": _read_capacity(record["resource_time"], elapsed=True),
        "resource_utilization": {
            "cpu_consumed": _read_usage(utilization["cpu_consumed"], {"cpu_seconds"}, integer=False),
            "retained_content": _read_usage(utilization["retained_content"], {"bytes"}),
            "message_traffic": _read_usage(utilization["message_traffic"], {"sent_bytes", "received_bytes"}),
        },
    }
    if "workspace_filesystem" in record:
        values["workspace_filesystem"] = _read_workspace(record["workspace_filesystem"])
    scope = {name: value for name, value in values.items() if name != "resource_utilization"}
    values["issues"] = _read_issues(record["issues"], {**scope, **values["resource_utilization"]})
    if "duration_seconds" in record:
        values["duration_seconds"] = _read_number(record["duration_seconds"])
    return values


def _read_participants(participants):
    if not isinstance(participants, list) or not 1 <= len(participants) <= MAX_PARTICIPANTS:
        raise ValueError("resource report requires a bounded nonempty participant list")
    names = set()
    result = []
    for entry in participants:
        if not isinstance(entry, dict):
            raise ValueError("resource participant must be an object")
        base = {"participant_name", "role"}
        status = entry.get("status")
        if "status" not in entry:
            entry = _read_fields(
                entry,
                base | {"received_at", "resource_capacity", "resource_time", "resource_utilization", "issues"},
                {"duration_seconds"},
            )
            entry = {**entry, **_read_measurements(entry)}
        elif status in ("missing", "disabled", "invalid"):
            entry = _read_fields(entry, base | {"status"} | ({"received_at"} if status == "invalid" else set()))
        else:
            raise ValueError("invalid resource participant status")
        name = _read_identifier(entry["participant_name"], PARTICIPANT_NAME_PATTERN)
        if name in names:
            raise ValueError("resource participant names must be unique")
        names.add(name)
        if entry["role"] not in ("client", "server"):
            raise ValueError("invalid resource participant role")
        if "received_at" in entry:
            _read_timestamp(entry["received_at"])
        result.append(entry)
    return sorted(result, key=lambda entry: (entry["role"], entry["participant_name"]))


def _read_participant_summary(record):
    record = _read_fields(
        record,
        {
            "schema_version",
            "kind",
            "job_id",
            "participant_name",
            "reported_at",
            "resource_capacity",
            "resource_time",
            "resource_utilization",
            "workspace_filesystem",
            "issues",
        },
        {"duration_seconds"},
    )
    _read_identifier(record["job_id"], JOB_ID_PATTERN)
    _read_identifier(record["participant_name"], PARTICIPANT_NAME_PATTERN)
    _read_timestamp(record["reported_at"])
    return {**record, **_read_measurements(record)}


def _read_resource_summary(record):
    record = _read_fields(record, {"schema_version", "kind", "job_id", "participants"})
    _read_identifier(record["job_id"], JOB_ID_PATTERN)
    participants = _read_participants(record["participants"])
    # Independently valid observations can still overflow when added together.
    sum_job_totals(participants)
    return {**record, "participants": participants}


def _read_totals(value):
    value = _read_fields(value, {"resource_time", "resource_utilization", "issues"})
    utilization = _read_fields(value["resource_utilization"], {"cpu_consumed", "retained_content", "message_traffic"})
    resource_time = _read_capacity(value["resource_time"], elapsed=True)
    usage = {
        "cpu_consumed": _read_usage(utilization["cpu_consumed"], {"cpu_seconds"}, integer=False),
        "retained_content": _read_usage(utilization["retained_content"], {"bytes"}),
        "message_traffic": _read_usage(utilization["message_traffic"], {"sent_bytes", "received_bytes"}),
    }
    issues = _read_issues(value["issues"], {"resource_time": resource_time, **usage})
    if any(item["code"] not in {"observation_incomplete", "attribution_incomplete"} for item in issues):
        raise ValueError("aggregate issues must describe derived quality")
    return {"resource_time": resource_time, "resource_utilization": usage, "issues": issues}


def _read_study_jobs(jobs):
    if not isinstance(jobs, list) or len(jobs) > MAX_STUDY_JOBS:
        raise ValueError("study requires a bounded job list")
    result = []
    identities = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("study job must be an object")
        state = job.get("resource_data")
        if state not in ("included", "unavailable", "nonterminal"):
            raise ValueError("invalid study resource-data state")
        job = _read_fields(
            job,
            {"job_id", "job_name", "job_status", "resource_data"} | ({"totals"} if state == "included" else set()),
        )
        identity = _read_identifier(job["job_id"], JOB_ID_PATTERN)
        if identity in identities:
            raise ValueError("study job IDs must be unique")
        identities.add(identity)
        if not isinstance(job["job_name"], str) or not 1 <= len(job["job_name"]) <= MAX_JOB_NAME_LENGTH:
            raise ValueError("invalid study job display name")
        status = _read_identifier(job["job_status"], JOB_STATUS_PATTERN)
        terminal = status.startswith("FINISHED:") or status in {
            "FINISHED_OK",
            "FINISHED_EXCEPTION",
            "ABORTED",
            "ABANDONED",
            "FAILED",
        }
        if terminal == (state == "nonterminal"):
            raise ValueError("study job status and resource-data state disagree")
        if state == "included":
            job = {**job, "totals": _read_totals(job["totals"])}
        result.append(job)
    return sorted(result, key=lambda job: job["job_id"])


def _read_study_summary(record):
    record = _read_fields(record, {"schema_version", "kind", "selection", "generated_at", "coverage", "jobs", "totals"})
    selection = _read_fields(record["selection"], {"study_name"})
    _read_identifier(selection["study_name"], STUDY_NAME_PATTERN)
    _read_timestamp(record["generated_at"])
    jobs = _read_study_jobs(record["jobs"])
    expected = {
        "selected_jobs": len(jobs),
        **{
            name: sum(job["resource_data"] == state for job in jobs)
            for name, state in (
                ("included_jobs", "included"),
                ("unavailable_jobs", "unavailable"),
                ("nonterminal_jobs", "nonterminal"),
            )
        },
    }
    coverage = _read_fields(record["coverage"], set(expected))
    for name, count in expected.items():
        if _read_number(coverage[name], integer=True) != count:
            raise ValueError("study coverage does not match selected jobs")
    totals = _read_totals(record["totals"])
    if totals != sum_study_totals(jobs):
        raise ValueError("study totals must equal included-job sums")
    return {**record, "jobs": jobs, "totals": totals}


def _json_depth(value):
    pending = [(value, 1)]
    while pending:
        item, depth = pending.pop()
        if depth > MAX_JSON_DEPTH:
            return depth
        if isinstance(item, dict):
            pending.extend((nested, depth + 1) for nested in item.values())
        elif isinstance(item, list):
            pending.extend((nested, depth + 1) for nested in item)
    return 0


def _json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("resource report contains duplicate JSON object keys")
        value[key] = item
    return value


def _reject_nonfinite(value):
    raise ValueError(f"resource report contains a non-finite JSON number: {value}")


def read_resource_report(data, expected_kind=None):
    """Decode one incoming report, checking its shape, bounds and arithmetic."""
    if not isinstance(data, bytes):
        raise ValueError("resource report input must be bytes")
    if expected_kind is not None and (not isinstance(expected_kind, str) or expected_kind not in MAX_RECORD_BYTES):
        raise ValueError("unsupported expected resource report kind")
    if len(data) > MAX_RECORD_BYTES.get(expected_kind, max(MAX_RECORD_BYTES.values())):
        raise ValueError("resource report exceeds the JSON byte limit")
    try:
        record = json.loads(data.decode("utf-8"), object_pairs_hook=_json_object, parse_constant=_reject_nonfinite)
    except (ValueError, RecursionError) as error:
        raise ValueError(f"invalid resource JSON: {error}") from error
    if not isinstance(record, dict):
        raise ValueError("resource report must be an object")
    kind = record.get("kind")
    if not isinstance(kind, str) or kind not in MAX_RECORD_BYTES or record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported resource report kind or schema_version")
    if expected_kind is not None and kind != expected_kind:
        raise ValueError("resource report does not match the expected record kind")
    if len(data) > MAX_RECORD_BYTES[kind]:
        raise ValueError("resource report exceeds its serialized size limit")
    if _json_depth(record) > MAX_JSON_DEPTH:
        raise ValueError("resource report exceeds maximum JSON depth")
    readers = {
        KIND_PARTICIPANT_SUMMARY: _read_participant_summary,
        KIND_RESOURCE_SUMMARY: _read_resource_summary,
        KIND_STUDY_SUMMARY: _read_study_summary,
    }
    return readers[kind](record)
