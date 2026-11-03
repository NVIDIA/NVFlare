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


"""Numeric measurements with one metric-tagged diagnostics list per scope."""


import json
import math
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping

SCHEMA_VERSION = "1.0"

KIND_PARTICIPANT_SUMMARY = "nvflare.resource_stats.participant_summary"

KIND_RESOURCE_SUMMARY = "nvflare.resource_stats.resource_summary"

RECORD_KINDS = {KIND_PARTICIPANT_SUMMARY, KIND_RESOURCE_SUMMARY}

MAX_PARTICIPANT_SUMMARY_BYTES = 1024 * 1024

MAX_RESOURCE_SUMMARY_BYTES = 64 * 1024 * 1024

MAX_RECORD_BYTES = {
    KIND_PARTICIPANT_SUMMARY: MAX_PARTICIPANT_SUMMARY_BYTES,
    KIND_RESOURCE_SUMMARY: MAX_RESOURCE_SUMMARY_BYTES,
}

MAX_PARTICIPANTS = 10_000

MAX_JSON_DEPTH = 32

PARTICIPANT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}$")

JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

ISSUE_CODES = {"observation_incomplete", "attribution_incomplete", "unsupported", "malformed_source"}

_PARTIAL_ISSUES = {"observation_incomplete", "attribution_incomplete"}

_DIMENSIONS = {"cpu", "memory", "gpu"}

_UTILIZATION = {"cpu_consumed", "retained_content", "message_traffic"}

_METRICS = _UTILIZATION | {"resource_capacity", "resource_time", "workspace_filesystem"}


class ContractError(ValueError):
    """An incoming resource record or arithmetic result is invalid."""


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _fail(path, message):
    raise ContractError(f"{path}: {message}")


def _mapping(value, path):
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    return value


def _exact_keys(value, required, optional, path):
    if any(not isinstance(key, str) for key in value):
        _fail(path, "object keys must be strings")
    missing, extra = required - value.keys(), value.keys() - required - optional
    if missing:
        _fail(path, f"missing required fields: {', '.join(sorted(missing))}")
    if extra:
        _fail(path, f"unexpected fields: {', '.join(sorted(extra))}")


def _number(value, path, *, integer=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
        _fail(path, "must be an integer" if integer else "must be a number")
    if (isinstance(value, float) and not math.isfinite(value)) or value < 0 or (positive and value == 0):
        _fail(path, "must be finite and positive" if positive else "must be finite and non-negative")
    return value


def _identifier(value, pattern, path):
    if not isinstance(value, str) or not pattern.fullmatch(value):
        _fail(path, "has invalid syntax")


def _timestamp(value, path):
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,9})?Z", value
    ):
        _fail(path, "must be a UTC timestamp")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        _fail(path, "must be a valid UTC timestamp")


def _status(value, path, allowed=("partial", "unavailable")):
    status = value.get("status", "reported")
    if "status" in value and status not in allowed:
        _fail(path, "has an invalid status")
    return status


def _ordered_issues(issues):
    return [
        {"metric": metric, "code": code} for metric, code in sorted({(item["metric"], item["code"]) for item in issues})
    ]


def normalize_scope_issues(issues, measurements=None):
    """Validate flat metric/code tags, deduplicate, and order them deterministically.

    With a measurement map, require causes for each exceptional measurement and
    reject tags for absent or complete metrics. No measurement data is converted.
    """
    if not isinstance(issues, list):
        _fail("issues", "must be a list")
    for item in issues:
        item = _mapping(item, "issues")
        _exact_keys(item, {"metric", "code"}, set(), "issues")
        if not isinstance(item["metric"], str) or item["metric"] not in _METRICS:
            _fail("issues", "has an unknown metric")
        if not isinstance(item["code"], str) or item["code"] not in ISSUE_CODES:
            _fail("issues", "has an unknown code")
    result = _ordered_issues(issues)
    if measurements is not None:
        if any(item["metric"] not in measurements for item in result):
            _fail("issues", "refers to a metric outside this scope")
        for metric, value in measurements.items():
            codes = {item["code"] for item in result if item["metric"] == metric}
            status = _mapping(value, metric).get("status", "reported")
            if status == "reported":
                if codes:
                    _fail("issues", f"complete {metric} must not have issues")
                continue
            allowed = _PARTIAL_ISSUES if status == "partial" else ISSUE_CODES
            if metric == "message_traffic":
                allowed = (
                    {"observation_incomplete"}
                    if status == "partial"
                    else {"observation_incomplete", "malformed_source"}
                )
            if not codes or codes - allowed:
                _fail("issues", f"{metric} requires supported causes for its status")
    return result


def _append_issues(issues, metric, codes):
    if issues is not None:
        issues.extend({"metric": metric, "code": code} for code in sorted(set(codes)))


def scoped_measurement(metric, value, scope_issues):
    """Move a private producer's codes into its containing scope exactly once."""
    result = dict(value)
    _append_issues(scope_issues, metric, result.pop("issues", ()))
    return result


def aggregate_scope_issues(source_issues, measurements):
    """Derive causes for status-only subtotals without copying source identities."""
    sources = normalize_scope_issues(source_issues)
    result = []
    for metric, value in measurements.items():
        status = value.get("status", "reported")
        if status == "reported":
            continue
        codes = {item["code"] for item in sources if item["metric"] == metric}
        if status == "partial":
            codes = {code if code == "attribution_incomplete" else "observation_incomplete" for code in codes}
        _append_issues(result, metric, codes or {"observation_incomplete"})
    return normalize_scope_issues(result, measurements)


def _scope_measurements(record):
    return {
        **{
            name: record[name]
            for name in ("resource_capacity", "resource_time", "workspace_filesystem")
            if name in record
        },
        **record.get("resource_utilization", {}),
    }


def _validate_capacity(value, path, *, elapsed=False):
    value = _mapping(value, path)
    status = _status(value, path)
    if status == "unavailable":
        _exact_keys(value, {"status"}, set(), path)
        return
    if status == "partial":
        _exact_keys(value, {"status"}, _DIMENSIONS, path)
        if not value.keys() & _DIMENSIONS:
            _fail(path, "partial capacity must contain a known dimension")
    else:
        _exact_keys(value, _DIMENSIONS, set(), path)
    fields = (
        {"cpu": "unit_seconds", "memory": "byte_seconds", "gpu": "instance_seconds"}
        if elapsed
        else {"cpu": "units", "memory": "bytes", "gpu": "count"}
    )
    for dimension, field in fields.items():
        if dimension in value:
            item = _mapping(value[dimension], f"{path}.{dimension}")
            _exact_keys(item, {field}, set(), f"{path}.{dimension}")
            _number(
                item[field],
                f"{path}.{dimension}.{field}",
                integer=not elapsed and dimension != "cpu",
                positive=not elapsed and dimension == "cpu",
            )


def _validate_usage(value, path, fields):
    value = _mapping(value, path)
    status = _status(value, path)
    required = set(fields) if status != "unavailable" else set()
    if status != "reported":
        required.add("status")
    _exact_keys(value, required, set(), path)
    if status != "unavailable":
        for field in fields:
            _number(value[field], f"{path}.{field}", integer=field != "cpu_seconds")


def _validate_utilization(value, path):
    value = _mapping(value, path)
    _exact_keys(value, _UTILIZATION, set(), path)
    for field, fields in (
        ("cpu_consumed", {"cpu_seconds"}),
        ("retained_content", {"bytes"}),
        ("message_traffic", {"sent_bytes", "received_bytes"}),
    ):
        _validate_usage(value[field], f"{path}.{field}", fields)


def _validate_workspace(value, path):
    value = _mapping(value, path)
    if _status(value, path, ("unavailable",)) == "reported":
        _exact_keys(value, {"capacity_bytes"}, set(), path)
        _number(value["capacity_bytes"], path, integer=True, positive=True)
    else:
        _exact_keys(value, {"status"}, set(), path)


def normalize_measurement(field, value):
    """Check one status/numeric measurement; diagnostics belong to its scope."""
    if not isinstance(field, str):
        _fail("measurement", "is not a supported measurement")
    if field in {"resource_capacity", "resource_time"}:
        _validate_capacity(value, field, elapsed=field == "resource_time")
    elif field == "workspace_filesystem":
        _validate_workspace(value, field)
    elif field == "cpu_consumed":
        _validate_usage(value, field, {"cpu_seconds"})
    elif field == "retained_content":
        _validate_usage(value, field, {"bytes"})
    elif field == "message_traffic":
        _validate_usage(value, field, {"sent_bytes", "received_bytes"})
    else:
        _fail("measurement", "is not a supported measurement")
    return deepcopy(value)


def normalize_duration(value):
    return _number(value, "duration_seconds")


def normalize_worker_record(value, expected_identity):
    value = _mapping(value, "worker record")
    fields = {"schema_version", "identity", "scope", "resource_capacity", "cpu_consumed", "message_traffic", "issues"}
    _exact_keys(value, fields, {"duration_seconds"}, "worker record")
    if value["schema_version"] != SCHEMA_VERSION or value["scope"] != "worker":
        _fail("worker record", "has an unsupported version or scope")
    identity = _mapping(value["identity"], "worker identity")
    _exact_keys(identity, {"job_id", "site_name", "task_id", "task_name", "attempt_id"}, set(), "worker identity")
    if any(not isinstance(item, str) for item in identity.values()) or identity != expected_identity:
        _fail("worker identity", "does not match its owned execution")
    result = {"schema_version": SCHEMA_VERSION, "identity": dict(expected_identity), "scope": "worker"}
    for field in ("resource_capacity", "cpu_consumed", "message_traffic"):
        result[field] = normalize_measurement(field, value[field])
    if "duration_seconds" in value:
        result["duration_seconds"] = normalize_duration(value["duration_seconds"])
    measurements = {name: result[name] for name in ("resource_capacity", "cpu_consumed", "message_traffic")}
    result["issues"] = normalize_scope_issues(value["issues"], measurements)
    return result


def _sum_numbers(values, path):
    try:
        result = sum(values) if all(isinstance(value, int) for value in values) else math.fsum(values)
    except (OverflowError, ValueError) as error:
        raise ContractError(f"{path}: numeric sum overflow") from error
    return _number(result, path)


def _capacity_time_from_validated(capacity, duration_seconds, scope_issues=None):
    """Derive status-only capacity-time, appending its causes to the scope sink.

    The optional list carries the source resource_capacity tags and receives
    resource_time tags; its caller normalizes/deduplicates the completed scope.
    """
    codes = {item["code"] for item in scope_issues or () if item["metric"] == "resource_capacity"}
    if capacity.get("status") == "unavailable":
        _append_issues(scope_issues, "resource_time", codes or {"observation_incomplete"})
        return {"status": "unavailable"}
    if duration_seconds is None:
        _append_issues(scope_issues, "resource_time", {"observation_incomplete"})
        return {"status": "unavailable"}
    result = {}
    for dimension, source, target in (
        ("cpu", "units", "unit_seconds"),
        ("memory", "bytes", "byte_seconds"),
        ("gpu", "count", "instance_seconds"),
    ):
        if dimension not in capacity:
            continue
        try:
            result[dimension] = {target: _number(capacity[dimension][source] * duration_seconds, target)}
        except (OverflowError, ContractError):
            codes.add("observation_incomplete")
    if not result:
        _append_issues(scope_issues, "resource_time", {"observation_incomplete"})
        return {"status": "unavailable"}
    if capacity.get("status") == "partial" or set(result) != _DIMENSIONS or codes:
        result["status"] = "partial"
        _append_issues(scope_issues, "resource_time", codes or {"observation_incomplete"})
    return result


def capacity_to_resource_time(capacity, duration_seconds, scope_issues=None):
    _validate_capacity(capacity, "resource_capacity")
    if duration_seconds is not None:
        normalize_duration(duration_seconds)
    return _capacity_time_from_validated(capacity, duration_seconds, scope_issues)


def _resource_time_aggregate(values, *, complete=True, path="resource_time"):
    result = {}
    for dimension, field in (("cpu", "unit_seconds"), ("memory", "byte_seconds"), ("gpu", "instance_seconds")):
        amounts = [value[dimension][field] for value in values if dimension in value]
        if amounts:
            result[dimension] = {field: _sum_numbers(amounts, f"{path}.{dimension}")}
    if not result:
        return {"status": "unavailable"}
    if not complete or any("status" in value for value in values):
        result["status"] = "partial"
    return result


def aggregate_resource_time(values, *, complete=True):
    values = [normalize_measurement("resource_time", value) for value in values]
    return _resource_time_aggregate(values, complete=complete)


def _aggregate_usage(values, fields, *, complete, path):
    numeric = [value for value in values if value.get("status") != "unavailable"]
    if not numeric:
        return {"status": "unavailable"}
    result = {field: _sum_numbers([value[field] for value in numeric], f"{path}.{field}") for field in fields}
    if not complete or len(numeric) != len(values) or any("status" in value for value in numeric):
        result["status"] = "partial"
    return result


def _aggregate_execution(field, values, fields, complete):
    values = [normalize_measurement(field, value) for value in values]
    return _aggregate_usage(values, fields, complete=complete, path=field)


def aggregate_cpu_consumed(values, *, complete=True):
    return _aggregate_execution("cpu_consumed", values, {"cpu_seconds"}, complete)


def aggregate_message_traffic(values, *, complete=True):
    return _aggregate_execution("message_traffic", values, {"sent_bytes", "received_bytes"}, complete)


def _sum_totals(entries, *, complete):
    totals = {
        "resource_time": _resource_time_aggregate([entry["resource_time"] for entry in entries], complete=complete),
        "resource_utilization": {
            field: _aggregate_usage(
                [entry["resource_utilization"][field] for entry in entries], fields, complete=complete, path=field
            )
            for field, fields in (
                ("cpu_consumed", {"cpu_seconds"}),
                ("retained_content", {"bytes"}),
                ("message_traffic", {"sent_bytes", "received_bytes"}),
            )
        },
    }
    issues = []
    if totals["resource_time"].get("status") == "partial":
        codes = set() if complete else {"observation_incomplete"}
        for entry in entries:
            source = entry["resource_time"]
            causes = {item["code"] for item in entry["issues"] if item["metric"] == "resource_time"}
            if source.get("status") == "partial":
                codes.update(causes)
            elif source.get("status") == "unavailable":
                codes.add(
                    "attribution_incomplete" if causes == {"attribution_incomplete"} else "observation_incomplete"
                )
        _append_issues(issues, "resource_time", codes or {"observation_incomplete"})
    elif totals["resource_time"].get("status") == "unavailable":
        _append_issues(issues, "resource_time", {"observation_incomplete"})
    for metric, value in totals["resource_utilization"].items():
        if "status" in value:
            _append_issues(issues, metric, {"observation_incomplete"})
    totals["issues"] = normalize_scope_issues(issues)
    return totals


def participant_measurements(record):
    result = {
        field: deepcopy(record[field])
        for field in ("resource_capacity", "resource_time", "resource_utilization", "duration_seconds")
        if field in record
    }
    result["issues"] = [dict(item) for item in record["issues"] if item["metric"] != "workspace_filesystem"]
    return result


def _validate_measurements(record, path):
    _validate_capacity(record["resource_capacity"], f"{path}.resource_capacity")
    _validate_capacity(record["resource_time"], f"{path}.resource_time", elapsed=True)
    _validate_utilization(record["resource_utilization"], f"{path}.resource_utilization")
    normalize_scope_issues(record["issues"], _scope_measurements(record))
    if "duration_seconds" in record:
        _number(record["duration_seconds"], f"{path}.duration_seconds")


def _validate_participants(participants, path):
    if not isinstance(participants, list) or not 1 <= len(participants) <= MAX_PARTICIPANTS:
        _fail(path, "must contain a bounded nonempty participant list")
    names = set()
    for entry in participants:
        entry = _mapping(entry, path)
        _identifier(entry.get("participant_name"), PARTICIPANT_NAME_PATTERN, path)
        if entry["participant_name"] in names:
            _fail(path, "participant names must be unique")
        names.add(entry["participant_name"])
        if entry.get("role") not in ("client", "server"):
            _fail(path, "has an invalid participant role")
        base = {"participant_name", "role"}
        if "status" not in entry:
            _exact_keys(
                entry,
                base | {"received_at", "resource_capacity", "resource_time", "resource_utilization", "issues"},
                {"duration_seconds"},
                path,
            )
            _timestamp(entry["received_at"], path)
            _validate_measurements(entry, path)
        elif entry["status"] in ("missing", "disabled", "invalid"):
            _exact_keys(
                entry, base | {"status"} | ({"received_at"} if entry["status"] == "invalid" else set()), set(), path
            )
            if entry["status"] == "invalid":
                _timestamp(entry["received_at"], path)
        else:
            _fail(path, "has an invalid participant status")


def sum_job_totals(participants):
    accepted = [entry for entry in participants if "status" not in entry]
    return _sum_totals(accepted, complete=len(accepted) == len(participants))


def derive_job_totals(participants):
    _validate_participants(participants, "participants")
    return sum_job_totals(participants)


def _json_depth(value):
    pending = [(value, 1)]
    while pending:
        item, depth = pending.pop()
        if depth > MAX_JSON_DEPTH:
            return depth
        if isinstance(item, Mapping):
            pending.extend((nested, depth + 1) for nested in item.values())
        elif isinstance(item, list):
            pending.extend((nested, depth + 1) for nested in item)
    return 0


def _validate_participant_summary(record):
    common = {"schema_version", "kind"}
    _exact_keys(
        record,
        common
        | {
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
        "$",
    )
    _identifier(record["job_id"], JOB_ID_PATTERN, "job_id")
    _identifier(record["participant_name"], PARTICIPANT_NAME_PATTERN, "participant_name")
    _timestamp(record["reported_at"], "reported_at")
    _validate_measurements(record, "$")
    _validate_workspace(record["workspace_filesystem"], "workspace_filesystem")


def _validate_resource_summary(record):
    common = {"schema_version", "kind"}
    _exact_keys(record, common | {"job_id", "participants"}, set(), "$")
    _identifier(record["job_id"], JOB_ID_PATTERN, "job_id")
    _validate_participants(record["participants"], "participants")
    sum_job_totals(record["participants"])


def validate_record(record, *, serialized_size=None):
    record = _mapping(record, "$")
    kind = record.get("kind")
    if not isinstance(kind, str) or kind not in RECORD_KINDS or record.get("schema_version") != SCHEMA_VERSION:
        _fail("$", "has an unsupported kind or schema_version")
    if serialized_size is not None and (
        type(serialized_size) is not int or not 0 <= serialized_size <= MAX_RECORD_BYTES[kind]
    ):
        _fail("$", "exceeds its serialized size limit")
    if _json_depth(record) > MAX_JSON_DEPTH:
        _fail("$", "exceeds maximum JSON depth")
    validators = {
        KIND_PARTICIPANT_SUMMARY: _validate_participant_summary,
        KIND_RESOURCE_SUMMARY: _validate_resource_summary,
    }
    validators[kind](record)


def normalize_record(record, *, serialized_size=None):
    """Validate once, then copy/order participant rows and their shared issues."""
    validate_record(record, serialized_size=serialized_size)
    result = deepcopy(record)
    if "participants" in result:
        result["participants"].sort(key=lambda entry: (entry["role"], entry["participant_name"]))
    for scope in [result, *result.get("participants", [])]:
        if "issues" in scope:
            scope["issues"] = _ordered_issues(scope["issues"])
    return result


def _json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            _fail("$", "contains duplicate JSON object keys")
        value[key] = item
    return value


def _reject_nonfinite(value):
    _fail("$", f"contains a non-finite JSON number: {value}")


def load_and_validate(data, expected_kind=None):
    if not isinstance(data, bytes):
        _fail("$", "input must be bytes")
    if expected_kind is not None and expected_kind not in RECORD_KINDS:
        _fail("$", "unsupported expected kind")
    if len(data) > MAX_RECORD_BYTES.get(expected_kind, max(MAX_RECORD_BYTES.values())):
        _fail("$", "exceeds the JSON byte limit")
    try:
        record = json.loads(data.decode("utf-8"), object_pairs_hook=_json_object, parse_constant=_reject_nonfinite)
    except (ValueError, UnicodeDecodeError, RecursionError) as error:
        raise ContractError(f"invalid resource JSON: {error}") from error
    record = normalize_record(record, serialized_size=len(data))
    if expected_kind is not None and record["kind"] != expected_kind:
        _fail("kind", "does not match the expected record kind")
    return record
