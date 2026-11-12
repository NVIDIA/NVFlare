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


"""Arithmetic and diagnostics composition for measurements collected by NVFlare."""

import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping

SCHEMA_VERSION = "1.0"
_DIMENSIONS = {"cpu", "memory", "gpu"}


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def ordered_issues(issues):
    return [
        {"metric": metric, "code": code} for metric, code in sorted({(item["metric"], item["code"]) for item in issues})
    ]


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
    sources = source_issues
    result = []
    for metric, value in measurements.items():
        status = value.get("status", "reported")
        if status == "reported":
            continue
        codes = {item["code"] for item in sources if item["metric"] == metric}
        if status == "partial":
            codes = {code if code == "attribution_incomplete" else "observation_incomplete" for code in codes}
        _append_issues(result, metric, codes or {"observation_incomplete"})
    return ordered_issues(result)


def _finite_result(value):
    if value < 0 or (isinstance(value, float) and not math.isfinite(value)):
        raise ValueError("measurement arithmetic must remain finite and non-negative")
    return value


def _sum_numbers(values):
    try:
        return _finite_result(sum(values) if all(isinstance(value, int) for value in values) else math.fsum(values))
    except (OverflowError, ValueError) as error:
        raise ValueError("measurement sum overflow") from error


def capacity_to_resource_time(capacity, duration_seconds, scope_issues=None):
    """Multiply accepted capacity by its window, preserving independent products.

    The optional list carries the source resource_capacity tags and receives
    resource_time tags; its caller deduplicates the completed scope.
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
            result[dimension] = {target: _finite_result(capacity[dimension][source] * duration_seconds)}
        except (OverflowError, ValueError):
            codes.add("observation_incomplete")
    if not result:
        _append_issues(scope_issues, "resource_time", {"observation_incomplete"})
        return {"status": "unavailable"}
    if capacity.get("status") == "partial" or set(result) != _DIMENSIONS or codes:
        result["status"] = "partial"
        _append_issues(scope_issues, "resource_time", codes or {"observation_incomplete"})
    return result


def aggregate_resource_time(values, *, complete=True):
    result = {}
    for dimension, field in (("cpu", "unit_seconds"), ("memory", "byte_seconds"), ("gpu", "instance_seconds")):
        amounts = [value[dimension][field] for value in values if dimension in value]
        if amounts:
            result[dimension] = {field: _sum_numbers(amounts)}
    if not result:
        return {"status": "unavailable"}
    if not complete or any("status" in value for value in values):
        result["status"] = "partial"
    return result


def aggregate_usage(values, fields, *, complete=True):
    numeric = [value for value in values if value.get("status") != "unavailable"]
    if not numeric:
        return {"status": "unavailable"}
    result = {field: _sum_numbers([value[field] for value in numeric]) for field in fields}
    if not complete or len(numeric) != len(values) or any("status" in value for value in numeric):
        result["status"] = "partial"
    return result


def aggregate_cpu_consumed(values, *, complete=True):
    return aggregate_usage(values, ("cpu_seconds",), complete=complete)


def aggregate_message_traffic(values, *, complete=True):
    return aggregate_usage(values, ("sent_bytes", "received_bytes"), complete=complete)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sum_totals(entries, *, complete):
    totals = {
        "resource_time": aggregate_resource_time([entry["resource_time"] for entry in entries], complete=complete),
        "resource_utilization": {
            field: aggregate_usage(
                [entry["resource_utilization"][field] for entry in entries], fields, complete=complete
            )
            for field, fields in (
                ("cpu_consumed", {"cpu_seconds"}),
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
    totals["issues"] = ordered_issues(issues)
    return totals


def participant_measurements(record):
    result = {
        field: deepcopy(record[field])
        for field in ("resource_capacity", "resource_time", "resource_utilization", "duration_seconds")
        if field in record
    }
    result["issues"] = deepcopy(record["issues"])
    return result


def sum_job_totals(participants):
    accepted = [entry for entry in participants if "status" not in entry]
    return _sum_totals(accepted, complete=len(accepted) == len(participants))


def sum_study_totals(jobs):
    """Sum already accepted job observations; selection coverage stays separate."""
    return _sum_totals([job["totals"] for job in jobs if job["resource_data"] == "included"], complete=True)
