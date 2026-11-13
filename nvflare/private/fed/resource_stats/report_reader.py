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

"""Read bounded participant and job reports at their receiving boundaries."""

import json
import math
import re
from datetime import datetime

from .measurements import SCHEMA_VERSION, ordered_issues

KIND_PARTICIPANT_SUMMARY = "nvflare.resource_stats.participant_summary"
KIND_RESOURCE_SUMMARY = "nvflare.resource_stats.resource_summary"
KIND_STUDY_SUMMARY = "nvflare.resource_stats.study_summary"
MAX_STUDY_SUMMARY_BYTES = 64 * 1024 * 1024
MAX_STUDY_JOBS = 10_000
MAX_JOB_NAME_LENGTH = 255
JOB_STATUS_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._:/+-]{0,63}$")
MAX_PARTICIPANT_SUMMARY_BYTES = 1024 * 1024
MAX_RESOURCE_SUMMARY_BYTES = 64 * 1024 * 1024
MAX_RECORD_BYTES = {
    KIND_PARTICIPANT_SUMMARY: MAX_PARTICIPANT_SUMMARY_BYTES,
    KIND_RESOURCE_SUMMARY: MAX_RESOURCE_SUMMARY_BYTES,
}
MAX_PARTICIPANTS = 10_000
PARTICIPANT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}$")
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _read_fields(value, required, optional=()):
    if not isinstance(value, dict):
        raise ValueError("measurement must be an object")
    required, optional = set(required), set(optional)
    if not required <= value.keys() or value.keys() - required - optional:
        raise ValueError("measurement has missing or unexpected fields")
    return value


def _read_number(value, *, integer=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
        raise ValueError("measurement must be an integer" if integer else "measurement must be a number")
    if value < 0 or (positive and value == 0) or (isinstance(value, float) and not math.isfinite(value)):
        raise ValueError(
            "measurement must be finite and positive" if positive else "measurement must be finite and non-negative"
        )
    return value


def _read_status(value, allowed=("partial", "unavailable")):
    if not isinstance(value, dict):
        raise ValueError("measurement must be an object")
    status = value.get("status", "reported")
    if "status" in value and status not in allowed:
        raise ValueError("measurement has an invalid status")
    return status


def _read_capacity(value, *, elapsed=False):
    status = _read_status(value)
    dimensions = {"cpu", "memory", "gpu"}
    _read_fields(value, dimensions if status == "reported" else {"status"}, dimensions if status == "partial" else ())
    if status == "partial" and not dimensions & value.keys():
        raise ValueError("partial capacity needs an observed dimension")
    fields = (
        {"cpu": "unit_seconds", "memory": "byte_seconds", "gpu": "instance_seconds"}
        if elapsed
        else {"cpu": "units", "memory": "bytes", "gpu": "count"}
    )
    for dimension, field in fields.items():
        if dimension in value:
            item = _read_fields(value[dimension], {field})
            _read_number(
                item[field], integer=not elapsed and dimension != "cpu", positive=not elapsed and dimension == "cpu"
            )
    return value


def _read_usage(value, fields, *, integer=True):
    status = _read_status(value)
    required = set(fields) if status != "unavailable" else set()
    if status != "reported":
        required.add("status")
    _read_fields(value, required)
    if status != "unavailable":
        for field in fields:
            _read_number(value[field], integer=integer)
    return value


def _read_issues(issues, measurements):
    """Read bounded report diagnostics and order distinct metric/code pairs."""
    if not isinstance(issues, list):
        raise ValueError("scope issues must be a list")
    allowed_codes = {"observation_incomplete", "attribution_incomplete", "unsupported", "malformed_source"}
    for item in issues:
        _read_fields(item, {"metric", "code"})
        if not isinstance(item["metric"], str) or item["metric"] not in measurements:
            raise ValueError("scope issue names an unknown metric")
        if not isinstance(item["code"], str) or item["code"] not in allowed_codes:
            raise ValueError("scope issue has an unknown cause")
    return ordered_issues(issues)


def _read_identifier(value, pattern):
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError("invalid resource report identity")
    return value


def _read_timestamp(value):
    # Newer ISO parsers accept 24:00:00; report hours must stay within 00-23.
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T(?:[01][0-9]|2[0-3]):[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,9})?Z", value
    ):
        raise ValueError("resource report requires a UTC timestamp")
    try:
        # The regex checks fractional syntax; validate calendar/time without
        # Python 3.10's restriction to three or six fractional digits.
        datetime.fromisoformat(value[:19])
    except ValueError as error:
        raise ValueError("resource report requires a valid UTC timestamp") from error
    return value


def _read_measurements(record):
    utilization = _read_fields(record["resource_utilization"], {"cpu_consumed", "message_traffic"})
    _read_capacity(record["resource_capacity"])
    _read_capacity(record["resource_time"], elapsed=True)
    _read_usage(utilization["cpu_consumed"], {"cpu_seconds"}, integer=False)
    _read_usage(utilization["message_traffic"], {"sent_bytes", "received_bytes"})
    record["issues"] = _read_issues(
        record["issues"], {"resource_capacity", "resource_time", "cpu_consumed", "message_traffic"}
    )
    if "duration_seconds" in record:
        _read_number(record["duration_seconds"])
    return record


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
            _read_measurements(entry)
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
            "issues",
        },
        {"duration_seconds"},
    )
    _read_identifier(record["job_id"], JOB_ID_PATTERN)
    _read_identifier(record["participant_name"], PARTICIPANT_NAME_PATTERN)
    _read_timestamp(record["reported_at"])
    return _read_measurements(record)


def _read_resource_summary(record):
    record = _read_fields(record, {"schema_version", "kind", "job_id", "participants"})
    _read_identifier(record["job_id"], JOB_ID_PATTERN)
    record["participants"] = _read_participants(record["participants"])
    return record


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
    readers = {
        KIND_PARTICIPANT_SUMMARY: _read_participant_summary,
        KIND_RESOURCE_SUMMARY: _read_resource_summary,
    }
    return readers[kind](record)
