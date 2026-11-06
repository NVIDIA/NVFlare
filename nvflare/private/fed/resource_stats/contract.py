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
from copy import deepcopy
from typing import Any, Mapping

SCHEMA_VERSION = "1.0"

MAX_PARTICIPANT_SUMMARY_BYTES = 1024 * 1024

ISSUE_CODES = {"observation_incomplete", "attribution_incomplete", "unsupported", "malformed_source"}

_PARTIAL_ISSUES = {"observation_incomplete", "attribution_incomplete"}

_DIMENSIONS = {"cpu", "memory", "gpu"}

_UTILIZATION = {"cpu_consumed", "retained_content", "message_traffic"}

_METRICS = _UTILIZATION | {"resource_capacity", "resource_time", "workspace_filesystem"}


class ContractError(ValueError):
    """An incoming resource record or arithmetic result is invalid."""


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
