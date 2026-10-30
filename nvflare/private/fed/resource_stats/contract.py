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

"""Production validation for the final-only resource-statistics contract.

Quantities are strings containing unsigned, non-exponent decimal values with
at most nine fractional digits. Equivalent spellings and collection orders are
accepted; writers normalize values and sort collections for repeatable output.

A participant report separates startup capacity, accumulated capacity-time,
and grouped utilization, with one containing elapsed duration. Each execution
contributes its own capacity-time once. All readers require the current
schema version.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "1.0"
U128_MAX = 2**128 - 1
PARTICIPANT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}$")

KIND_PARTICIPANT_SUMMARY = "nvflare.resource_stats.participant_summary"
KIND_RESOURCE_SUMMARY = "nvflare.resource_stats.resource_summary"
KIND_STUDY_SUMMARY = "nvflare.resource_stats.study_summary"
RECORD_KINDS = frozenset(
    {
        KIND_PARTICIPANT_SUMMARY,
        KIND_RESOURCE_SUMMARY,
        KIND_STUDY_SUMMARY,
    }
)

ISSUE_CODES = frozenset(
    {
        "observation_incomplete",
        "attribution_incomplete",
        "unsupported",
        "malformed_source",
    }
)
RESOURCE_TIME_PARTIAL_ISSUES = frozenset({"attribution_incomplete", "observation_incomplete"})
RESOURCE_TIME_UNAVAILABLE_ISSUES = frozenset(
    {
        "attribution_incomplete",
        "malformed_source",
        "observation_incomplete",
        "unsupported",
    }
)
SOURCE_UNAVAILABLE_ISSUES = frozenset(
    {"attribution_incomplete", "malformed_source", "observation_incomplete", "unsupported"}
)
CAPACITY_UNAVAILABLE_ISSUES = SOURCE_UNAVAILABLE_ISSUES

RESOURCE_TIME_STATUSES = frozenset({"partial", "unavailable"})
MEASUREMENT_STATUSES = frozenset({"partial", "unavailable"})
TOTAL_STATUSES = frozenset({"partial", "unavailable"})
PARTICIPANT_STATUSES = frozenset({"missing", "invalid", "disabled"})
ROLES = frozenset({"client", "server"})
GPU_KINDS = frozenset({"cuda_device_unclassified", "full_gpu", "mig_compute_instance"})
RESOURCE_DATA_STATES = frozenset({"included", "unavailable", "nonterminal"})
LEGACY_TERMINAL_JOB_STATES = frozenset({"FINISHED_OK", "FINISHED_EXCEPTION", "ABORTED", "ABANDONED", "FAILED"})

MAX_ISSUES = 4
MAX_PARTICIPANTS = 10_000
MAX_STUDY_JOBS = 10_000
MAX_GPU_GROUPS = 4_096
MAX_JSON_DEPTH = 32
MAX_RECORD_BYTES = {
    # Participant data travels in the existing terminal-outcome request.  Keep
    # that critical-path payload bounded well below the server-local rollups.
    KIND_PARTICIPANT_SUMMARY: 1024 * 1024,
    KIND_RESOURCE_SUMMARY: 64 * 1024 * 1024,
    KIND_STUDY_SUMMARY: 64 * 1024 * 1024,
}
MAX_PARTICIPANT_SUMMARY_BYTES = MAX_RECORD_BYTES[KIND_PARTICIPANT_SUMMARY]
MAX_RESOURCE_SUMMARY_BYTES = MAX_RECORD_BYTES[KIND_RESOURCE_SUMMARY]
MAX_STUDY_SUMMARY_BYTES = MAX_RECORD_BYTES[KIND_STUDY_SUMMARY]

U32_MAX = 2**32 - 1
U64_MAX = 2**64 - 1
MAX_CPU_UNITS = Decimal("1048576")

INTEGER_PATTERN = re.compile(r"^[0-9]{1,39}(?:\.0{1,9})?$")
DECIMAL_PATTERN = re.compile(r"^[0-9]{1,39}(?:\.[0-9]{1,9})?$")
TIMESTAMP_PATTERN = re.compile(r"^([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2})(?:\.([0-9]{1,9}))?Z$")
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
MAX_JOB_NAME_LENGTH = 255
MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._()+/@-]{0,127}$")
ARCHITECTURE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")
MIG_PROFILE_PATTERN = re.compile(r"^[1-9][0-9]*g\.[1-9][0-9]*gb(?:\+me)?$")
STUDY_NAME_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?$")
JOB_STATUS_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._:/+-]{0,63}$")

_FORBIDDEN_MODEL_PATTERNS = (
    re.compile(r"(?:GPU|MIG|MIG-GPU)-[0-9A-Fa-f-]{16,}", re.IGNORECASE),
    re.compile(r"(?:[0-9A-Fa-f]{4}:)?[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]"),
    re.compile(r"\b(?:serial|uuid|cuda_visible_devices|pci_bus_id|hostname)\b", re.IGNORECASE),
)
_FORBIDDEN_KEYS = frozenset(
    {
        "hostname",
        "host_id",
        "ip_address",
        "mac_address",
        "serial",
        "serial_number",
        "uuid",
        "gpu_uuid",
        "pci_bus_id",
        "bdf",
        "cuda_visible_devices",
        "raw_cpuinfo",
        "cpu_flags",
        "cpu_topology",
    }
)


class ContractError(ValueError):
    """Raised when a record violates the versioned resource-statistics contract."""


def utc_timestamp() -> str:
    """Return a schema-compatible UTC timestamp for production records."""

    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize a validated record deterministically and readably."""

    return (
        json.dumps(_normalize_tree(value), ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _fail(path: str, message: str) -> None:
    raise ContractError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], required: set[str], optional: set[str], path: str) -> None:
    keys = set(value)
    missing = required - keys
    extra = keys - required - optional
    if missing:
        _fail(path, f"missing required fields: {', '.join(sorted(missing))}")
    if extra:
        _fail(path, f"unexpected fields: {', '.join(sorted(extra))}")


def _enum(value: Any, allowed: frozenset[str] | set[str], path: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        _fail(path, f"must be one of {', '.join(sorted(allowed))}")
    return value


def _optional_status(value: Mapping[str, Any], default: str, allowed: frozenset[str] | set[str], path: str) -> str:
    """An omitted status is the successful case; explicit statuses describe exceptions."""

    return _enum(value["status"], allowed, path) if "status" in value else default


def _identifier(value: Any, pattern: re.Pattern[str], path: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        _fail(path, "has invalid syntax")
    return value


def _timestamp(value: Any, path: str) -> str:
    if not isinstance(value, str):
        _fail(path, "must be a UTC timestamp string")
    match = TIMESTAMP_PATTERN.fullmatch(value)
    if not match:
        _fail(path, "must be RFC 3339 UTC with at most nanosecond precision")
    try:
        datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        _fail(path, "is not a valid calendar timestamp")
    return value


def _decimal(value: Any, path: str, *, maximum: int | Decimal = U128_MAX, positive: bool = False) -> Decimal:
    if not isinstance(value, str) or not DECIMAL_PATTERN.fullmatch(value):
        _fail(path, "must be a non-negative plain decimal string with at most 9 fractional digits")
    try:
        number = Decimal(value)
    except InvalidOperation:
        _fail(path, "must be a decimal")
    if number > Decimal(maximum) or (positive and number <= 0):
        qualifier = "positive and " if positive else ""
        _fail(path, f"must be {qualifier}no greater than {maximum}")
    return number


def _integer(value: Any, path: str, *, maximum: int = U128_MAX, positive: bool = False) -> int:
    if not isinstance(value, str) or not INTEGER_PATTERN.fullmatch(value):
        _fail(path, "must be a non-negative integer-valued plain decimal string")
    number = int(Decimal(value))
    if number > maximum or (positive and number == 0):
        qualifier = "positive and " if positive else ""
        _fail(path, f"must be {qualifier}no greater than {maximum}")
    return number


def _model(value: Any, path: str) -> str:
    if not isinstance(value, str) or not MODEL_PATTERN.fullmatch(value):
        _fail(path, "must be a controlled hardware-model label of at most 128 ASCII characters")
    if any(pattern.search(value) for pattern in _FORBIDDEN_MODEL_PATTERNS):
        _fail(path, "must not contain a serial, UUID, PCI address, host identifier, or raw mask")
    return value


def _issues(value: Any, allowed: frozenset[str] | set[str], path: str) -> list[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_ISSUES:
        _fail(path, f"must contain 1..{MAX_ISSUES} issue codes")
    if any(not isinstance(item, str) for item in value):
        _fail(path, "must contain strings")
    if len(value) != len(set(value)):
        _fail(path, "must be unique")
    unknown = set(value) - allowed
    if unknown:
        _fail(path, f"contains unsupported issues: {', '.join(sorted(unknown))}")
    return value


def _privacy_walk(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                _fail(path, "object keys must be strings")
            if key.lower() in _FORBIDDEN_KEYS:
                _fail(f"{path}.{key}", "field is forbidden by the resource-statistics privacy policy")
            _privacy_walk(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _privacy_walk(nested, f"{path}[{index}]")


def _json_depth(value: Any) -> int:
    """Return enough depth information without using the Python call stack."""

    stack: list[tuple[Any, int]] = [(value, 1)]
    maximum = 0
    while stack:
        item, depth = stack.pop()
        maximum = max(maximum, depth)
        # Validation only needs to know that the fixed bound was exceeded.
        # Stopping here also makes cyclic already-decoded inputs terminate.
        if depth > MAX_JSON_DEPTH:
            return depth
        if isinstance(item, Mapping):
            stack.extend((nested, depth + 1) for nested in item.values())
        elif isinstance(item, list):
            stack.extend((nested, depth + 1) for nested in item)
    return maximum


def _validate_cpu_time_group(value: Any, path: str) -> None:
    group = _mapping(value, path)
    _exact_keys(group, {"unit_seconds"}, {"model", "architecture"}, path)
    _decimal(group["unit_seconds"], f"{path}.unit_seconds")
    if "model" in group:
        _model(group["model"], f"{path}.model")
    if "architecture" in group:
        _identifier(group["architecture"], ARCHITECTURE_PATTERN, f"{path}.architecture")


def _gpu_group_key(group: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        group["kind"],
        group.get("model", ""),
        group.get("memory_bytes", ""),
        group.get("mig_profile", ""),
    )


def _validate_gpu_metadata(group: Mapping[str, Any], path: str) -> None:
    """Check device classification and optional metadata in either representation."""

    kind = _enum(group["kind"], GPU_KINDS, f"{path}.kind")
    if "model" in group:
        _model(group["model"], f"{path}.model")
    if "memory_bytes" in group:
        _integer(group["memory_bytes"], f"{path}.memory_bytes", maximum=U64_MAX, positive=True)
    if "mig_profile" in group:
        _identifier(group["mig_profile"], MIG_PROFILE_PATTERN, f"{path}.mig_profile")
    if kind != "mig_compute_instance" and "mig_profile" in group:
        _fail(f"{path}.mig_profile", "is valid only for a MIG compute-instance group")


def _validate_gpu_time_group(value: Any, path: str) -> None:
    group = _mapping(value, path)
    _exact_keys(group, {"kind", "instance_seconds"}, {"model", "memory_bytes", "mig_profile"}, path)
    _validate_gpu_metadata(group, path)
    _decimal(group["instance_seconds"], f"{path}.instance_seconds", positive=True)


def _validate_group_list(value: Any, path: str, *, gpu: bool) -> None:
    if not isinstance(value, list) or len(value) > MAX_GPU_GROUPS or (not gpu and not value):
        minimum = "1" if not gpu else "0"
        _fail(path, f"must contain {minimum}..{MAX_GPU_GROUPS} consolidated groups")
    keys: list[tuple[str, ...]] = []
    for index, group in enumerate(value):
        item_path = f"{path}[{index}]"
        if gpu:
            _validate_gpu_time_group(group, item_path)
            keys.append(_gpu_group_key(group))
        else:
            _validate_cpu_time_group(group, item_path)
            keys.append((group.get("model", ""), group.get("architecture", "")))
    if len(keys) != len(set(keys)):
        _fail(path, "groups must be consolidated and unique by identifying fields")


def _validate_cpu_time(value: Any, path: str) -> None:
    cpu = _mapping(value, path)
    _exact_keys(cpu, {"groups"}, set(), path)
    _validate_group_list(cpu["groups"], f"{path}.groups", gpu=False)


def _validate_memory_time(value: Any, path: str) -> None:
    memory = _mapping(value, path)
    _exact_keys(memory, {"byte_seconds"}, set(), path)
    _decimal(memory["byte_seconds"], f"{path}.byte_seconds")


def _validate_gpu_time(value: Any, path: str) -> None:
    gpu = _mapping(value, path)
    _exact_keys(gpu, {"groups"}, set(), path)
    _validate_group_list(gpu["groups"], f"{path}.groups", gpu=True)


_RESOURCE_TIME_NUMERIC_FIELDS = frozenset({"cpu", "memory", "gpu"})


def _validate_resource_time(value: Any, path: str) -> None:
    resource_time = _mapping(value, path)
    status = _optional_status(resource_time, "reported", RESOURCE_TIME_STATUSES, f"{path}.status")
    if status == "reported":
        _exact_keys(resource_time, set(_RESOURCE_TIME_NUMERIC_FIELDS), set(), path)
    elif status == "partial":
        _exact_keys(resource_time, {"status", "issues"}, set(_RESOURCE_TIME_NUMERIC_FIELDS), path)
        _issues(resource_time["issues"], RESOURCE_TIME_PARTIAL_ISSUES, f"{path}.issues")
        if not set(resource_time) & _RESOURCE_TIME_NUMERIC_FIELDS:
            _fail(path, "partial resource_time must contain at least one numeric member")
    else:
        _exact_keys(resource_time, {"status", "issues"}, set(), path)
        _issues(resource_time["issues"], RESOURCE_TIME_UNAVAILABLE_ISSUES, f"{path}.issues")

    if "cpu" in resource_time:
        _validate_cpu_time(resource_time["cpu"], f"{path}.cpu")
    if "memory" in resource_time:
        _validate_memory_time(resource_time["memory"], f"{path}.memory")
    if "gpu" in resource_time:
        _validate_gpu_time(resource_time["gpu"], f"{path}.gpu")


def _validate_capacity_dimensions(value: Any, path: str, *, complete: bool) -> None:
    capacity = _mapping(value, path)
    required = {"cpu", "memory", "gpu"} if complete else set()
    _exact_keys(capacity, required, {"cpu", "memory", "gpu"} - required, path)
    if "cpu" in capacity:
        cpu = _mapping(capacity["cpu"], f"{path}.cpu")
        _exact_keys(cpu, {"groups"}, set(), f"{path}.cpu")
        groups = cpu["groups"]
        if not isinstance(groups, list) or not 1 <= len(groups) <= MAX_GPU_GROUPS:
            _fail(f"{path}.cpu.groups", "must contain 1..4096 consolidated groups")
        keys = []
        units = []
        for index, raw in enumerate(groups):
            group_path = f"{path}.cpu.groups[{index}]"
            group = _mapping(raw, group_path)
            _exact_keys(group, {"units"}, {"model", "architecture"}, group_path)
            units.append(_decimal(group["units"], f"{group_path}.units", maximum=MAX_CPU_UNITS, positive=True))
            if "model" in group:
                _model(group["model"], f"{group_path}.model")
            if "architecture" in group:
                _identifier(group["architecture"], ARCHITECTURE_PATTERN, f"{group_path}.architecture")
            keys.append((group.get("model", ""), group.get("architecture", "")))
        if len(keys) != len(set(keys)) or _sum_decimals(units, path) > MAX_CPU_UNITS:
            _fail(f"{path}.cpu.groups", "groups must be unique and total at most 1048576 CPU units")
    if "memory" in capacity:
        memory = _mapping(capacity["memory"], f"{path}.memory")
        _exact_keys(memory, {"bytes"}, set(), f"{path}.memory")
        _integer(memory["bytes"], f"{path}.memory.bytes", maximum=U64_MAX, positive=True)
    if "gpu" in capacity:
        gpu = _mapping(capacity["gpu"], f"{path}.gpu")
        _exact_keys(gpu, {"groups"}, set(), f"{path}.gpu")
        groups = gpu["groups"]
        if not isinstance(groups, list) or len(groups) > MAX_GPU_GROUPS:
            _fail(f"{path}.gpu.groups", "must contain 0..4096 consolidated groups")
        keys = []
        count = 0
        for index, raw in enumerate(groups):
            group_path = f"{path}.gpu.groups[{index}]"
            group = _mapping(raw, group_path)
            _exact_keys(group, {"kind", "count"}, {"model", "memory_bytes", "mig_profile"}, group_path)
            count += _integer(group["count"], f"{group_path}.count", maximum=MAX_GPU_GROUPS, positive=True)
            _validate_gpu_metadata(group, group_path)
            keys.append(_gpu_group_key(group))
        if len(keys) != len(set(keys)) or count > MAX_GPU_GROUPS:
            _fail(f"{path}.gpu.groups", "groups must be unique and total at most 4096 CUDA devices")


def _validate_resource_capacity(value: Any, path: str) -> dict[str, Any]:
    """Validate a startup capacity snapshot without measurement timing."""

    capacity = _mapping(value, path)
    status = _optional_status(capacity, "reported", RESOURCE_TIME_STATUSES, f"{path}.status")
    if status == "unavailable":
        _exact_keys(capacity, {"status", "issues"}, set(), path)
        _issues(capacity["issues"], RESOURCE_TIME_UNAVAILABLE_ISSUES, f"{path}.issues")
        return dict(capacity)
    numeric = {key: item for key, item in capacity.items() if key not in {"status", "issues"}}
    if status == "partial":
        _issues(capacity.get("issues"), RESOURCE_TIME_PARTIAL_ISSUES, f"{path}.issues")
    elif "issues" in capacity:
        _fail(path, "unexpected fields: complete capacity must omit status and issues")
    _validate_capacity_dimensions(numeric, path, complete=status == "reported")
    if not numeric:
        _fail(path, "partial capacity must contain a known dimension")
    return dict(capacity)


def _capacity_time_from_validated(capacity: Mapping[str, Any], duration_seconds: str | None) -> dict[str, Any]:
    if capacity.get("status") == "unavailable":
        return deepcopy(capacity)
    if duration_seconds is None:
        return {"status": "unavailable", "issues": ["observation_incomplete"]}
    duration = _decimal(duration_seconds, "duration_seconds")
    value: dict[str, Any] = {}

    def product(number: str) -> str:
        with localcontext() as context:
            context.prec = 100
            result = Decimal(number) * duration
            if result.as_tuple().exponent < -9:
                result = result.quantize(Decimal("0.000000001"), rounding=ROUND_HALF_EVEN)
        if result > U128_MAX:
            _fail("resource_capacity", "derived capacity-time exceeds the unsigned 128-bit bound")
        return _canonical_decimal(result)

    for resource in ("cpu", "memory", "gpu"):
        if resource not in capacity:
            continue
        try:
            if resource == "memory":
                value[resource] = {"byte_seconds": product(capacity[resource]["bytes"])}
            else:
                source_field, target_field = (
                    ("units", "unit_seconds") if resource == "cpu" else ("count", "instance_seconds")
                )
                value[resource] = {
                    "groups": [
                        {
                            **{key: item for key, item in group.items() if key != source_field},
                            target_field: product(group[source_field]),
                        }
                        for group in capacity[resource]["groups"]
                        if resource != "gpu" or duration > 0
                    ]
                }
        except ContractError:
            # Keep independent capacity-time when this dimension overflows.
            continue
    if not value:
        return {"status": "unavailable", "issues": ["observation_incomplete"]}
    if not all(name in value for name in ("cpu", "memory", "gpu")):
        value.update(status="partial", issues=["observation_incomplete"])
    result = value
    if capacity.get("status") == "partial":
        result["status"] = "partial"
        result["issues"] = sorted(set(result.get("issues", [])) | set(capacity["issues"]))
    return result


def capacity_to_resource_time(capacity: Mapping[str, Any], duration_seconds: str | None) -> dict[str, Any]:
    """Calculate one execution's capacity-time over its own measurement window."""

    capacity = _normalize_tree(capacity)
    _validate_resource_capacity(capacity, "resource_capacity")
    return _capacity_time_from_validated(capacity, duration_seconds)


def _validate_workspace_filesystem(value: Any, path: str) -> None:
    workspace = _mapping(value, path)
    status = _optional_status(workspace, "reported", frozenset({"unavailable"}), f"{path}.status")
    if status == "reported":
        _exact_keys(workspace, {"capacity_bytes"}, set(), path)
        _integer(workspace["capacity_bytes"], f"{path}.capacity_bytes", maximum=U64_MAX, positive=True)
    else:
        _exact_keys(workspace, {"status", "issues"}, set(), path)
        _issues(workspace["issues"], CAPACITY_UNAVAILABLE_ISSUES, f"{path}.issues")


def _validate_cpu_consumed(value: Any, path: str, *, aggregate: bool = False) -> None:
    consumed = _mapping(value, path)
    status = _optional_status(
        consumed, "reported", TOTAL_STATUSES if aggregate else MEASUREMENT_STATUSES, f"{path}.status"
    )
    if status in {"reported", "partial"}:
        required = {"cpu_seconds"} if status == "reported" else {"status", "cpu_seconds"}
        if status == "partial" and not aggregate:
            required.add("issues")
        _exact_keys(consumed, required, set(), path)
        _decimal(consumed["cpu_seconds"], f"{path}.cpu_seconds")
        if status == "partial" and not aggregate:
            _issues(
                consumed["issues"],
                frozenset({"attribution_incomplete", "observation_incomplete"}),
                f"{path}.issues",
            )
    else:
        required = {"status"} if aggregate else {"status", "issues"}
        _exact_keys(consumed, required, set(), path)
        if not aggregate:
            _issues(consumed["issues"], SOURCE_UNAVAILABLE_ISSUES, f"{path}.issues")


def _validate_retained_content(value: Any, path: str, *, aggregate: bool = False) -> None:
    retained = _mapping(value, path)
    statuses = TOTAL_STATUSES if aggregate else MEASUREMENT_STATUSES
    status = _optional_status(retained, "reported", statuses, f"{path}.status")
    if status == "reported":
        _exact_keys(retained, {"bytes"}, set(), path)
        _integer(retained["bytes"], f"{path}.bytes")
    elif status == "partial":
        if aggregate:
            _exact_keys(retained, {"status", "bytes"}, set(), path)
        else:
            _exact_keys(retained, {"status", "issues", "bytes"}, set(), path)
            _issues(
                retained["issues"],
                frozenset({"attribution_incomplete", "observation_incomplete"}),
                f"{path}.issues",
            )
        _integer(retained["bytes"], f"{path}.bytes")
    else:
        required = {"status"} if aggregate else {"status", "issues"}
        _exact_keys(retained, required, set(), path)
        if not aggregate:
            _issues(retained["issues"], SOURCE_UNAVAILABLE_ISSUES, f"{path}.issues")


def _validate_message_traffic(value: Any, path: str, *, aggregate: bool = False) -> None:
    traffic = _mapping(value, path)
    statuses = TOTAL_STATUSES if aggregate else MEASUREMENT_STATUSES
    status = _optional_status(traffic, "reported", statuses, f"{path}.status")
    if status in {"reported", "partial"}:
        required = {"sent_bytes", "received_bytes"}
        if status == "partial":
            required.add("status")
            if not aggregate:
                required.add("issues")
        _exact_keys(traffic, required, set(), path)
        if status == "partial" and not aggregate:
            _issues(traffic["issues"], frozenset({"observation_incomplete"}), f"{path}.issues")
        for field in ("sent_bytes", "received_bytes"):
            _integer(traffic[field], f"{path}.{field}")
    else:
        _exact_keys(traffic, {"status"} if aggregate else {"status", "issues"}, set(), path)
        if not aggregate:
            _issues(traffic["issues"], frozenset({"malformed_source", "observation_incomplete"}), f"{path}.issues")


def _validate_resource_utilization(value: Any, path: str, *, aggregate: bool = False) -> None:
    utilization = _mapping(value, path)
    _exact_keys(utilization, {"cpu_consumed", "retained_content", "message_traffic"}, set(), path)
    _validate_cpu_consumed(utilization["cpu_consumed"], f"{path}.cpu_consumed", aggregate=aggregate)
    _validate_retained_content(utilization["retained_content"], f"{path}.retained_content", aggregate=aggregate)
    _validate_message_traffic(utilization["message_traffic"], f"{path}.message_traffic", aggregate=aggregate)


def _validate_totals(value: Any, path: str) -> None:
    totals = _mapping(value, path)
    _exact_keys(totals, {"resource_time", "resource_utilization"}, set(), path)
    _validate_resource_time(totals["resource_time"], f"{path}.resource_time")
    _validate_resource_utilization(totals["resource_utilization"], f"{path}.resource_utilization", aggregate=True)


def _canonical_decimal(value: Decimal) -> str:
    if value == value.to_integral():
        # Decimal.quantize(Decimal(1)) uses the ambient precision and raises
        # InvalidOperation for otherwise valid 29..39 digit totals. Fixed-point
        # rendering is exact and independent of that context.
        return format(value, "f").partition(".")[0]
    return format(value, "f").rstrip("0").rstrip(".") or "0"


_QUANTITY_FIELDS = frozenset(
    {
        "duration_seconds",
        "unit_seconds",
        "units",
        "count",
        "instance_seconds",
        "byte_seconds",
        "cpu_seconds",
        "capacity_bytes",
        "memory_bytes",
        "bytes",
        "sent_bytes",
        "received_bytes",
        "selected_jobs",
        "included_jobs",
        "unavailable_jobs",
        "nonterminal_jobs",
    }
)


def _normalize_tree(value: Any) -> Any:
    """Copy values into writer order without accepting unknown fields or values.

    Validation remains responsible for types, bounds, uniqueness and contents.
    In particular, sorting does not deduplicate input collections.
    """

    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if key in _QUANTITY_FIELDS and isinstance(item, str) and DECIMAL_PATTERN.fullmatch(item):
                result[key] = _canonical_decimal(Decimal(item))
            else:
                result[key] = _normalize_tree(item)
        for key in ("issues", "groups", "participants", "jobs"):
            items = result.get(key)
            if not isinstance(items, list):
                continue
            if key == "issues" and all(isinstance(item, str) for item in items):
                result[key] = sorted(items)
            elif items and all(isinstance(item, Mapping) for item in items):
                if key == "groups":
                    fields = ("kind", "model", "memory_bytes", "mig_profile", "architecture")
                elif key == "participants":
                    fields = ("role", "participant_name")
                else:
                    fields = ("job_id",)
                if all(isinstance(item.get(field, ""), str) for item in items for field in fields):
                    result[key] = sorted(items, key=lambda item: tuple(item.get(field, "") for field in fields))
        return result
    if isinstance(value, list):
        return [_normalize_tree(item) for item in value]
    return deepcopy(value)


def normalize_quota_units(quota_us: int, period_us: int) -> str:
    """Return an exact quota ratio conservatively floored to nine digits."""

    for name, value in (("quota_us", quota_us), ("period_us", period_us)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > U64_MAX:
            raise ValueError(f"{name} must be a positive unsigned 64-bit integer")
    with localcontext() as context:
        context.prec = 100
        normalized = (Decimal(quota_us) / Decimal(period_us)).quantize(Decimal("0.000000001"), rounding=ROUND_FLOOR)
    if normalized <= 0:
        raise ValueError("CPU quota ratio is below the nine-decimal precision")
    if normalized > MAX_CPU_UNITS:
        raise ValueError("normalized CPU quota exceeds the visible-unit bound")
    return _canonical_decimal(normalized)


def normalize_message_traffic(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one current participant traffic measurement."""

    return normalize_measurement("message_traffic", value)


def normalize_duration(value: Any) -> str:
    """Validate and normalize the participant measurement duration."""

    return _canonical_decimal(_decimal(value, "duration_seconds"))


def normalize_measurement(field: str, value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and independently normalize one named participant measurement.

    This is the same validation used inside a participant report. A malformed
    measurement can be rejected without discarding unrelated measurements.
    Identity and envelope validation remain the caller's responsibility.
    """

    validators = {
        "resource_capacity": _validate_resource_capacity,
        "resource_time": _validate_resource_time,
        "cpu_consumed": _validate_cpu_consumed,
        "workspace_filesystem": _validate_workspace_filesystem,
        "retained_content": _validate_retained_content,
        "message_traffic": _validate_message_traffic,
    }
    if not isinstance(field, str) or field not in validators:
        _fail("measurement", "is not a supported participant measurement")
    value = _mapping(value, field)
    if _json_depth(value) > MAX_JSON_DEPTH:
        _fail(field, f"exceeds maximum JSON depth {MAX_JSON_DEPTH}")
    _privacy_walk(value, field)
    normalized = _normalize_tree(value)
    validators[field](normalized, field)
    return normalized


def _sum_decimals(values: Sequence[Decimal], path: str) -> Decimal:
    with localcontext() as context:
        context.prec = 100
        result = sum(values, Decimal(0))
    if result > Decimal(U128_MAX):
        _fail(path, "aggregate exceeds the unsigned 128-bit bound")
    return result


def _resource_time_aggregate(
    values: Sequence[Mapping[str, Any]],
    *,
    complete: bool,
    path: str,
) -> dict[str, Any]:
    numeric_seen = {name: any(name in value for value in values) for name in _RESOURCE_TIME_NUMERIC_FIELDS}
    fully_reported = complete and bool(values) and all("status" not in value for value in values)

    result: dict[str, Any] = {}
    if numeric_seen["cpu"]:
        sums: dict[tuple[str, str], list[Decimal]] = defaultdict(list)
        templates: dict[tuple[str, str], dict[str, Any]] = {}
        for value in values:
            if "cpu" not in value:
                continue
            for group in value["cpu"]["groups"]:
                key = (group.get("model", ""), group.get("architecture", ""))
                templates[key] = {name: item for name, item in group.items() if name != "unit_seconds"}
                sums[key].append(Decimal(group["unit_seconds"]))
        result["cpu"] = {
            "groups": [
                {
                    **templates[key],
                    "unit_seconds": _canonical_decimal(_sum_decimals(sums[key], f"{path}.cpu.groups")),
                }
                for key in sorted(sums)
            ]
        }

    if numeric_seen["memory"]:
        values_to_sum = [Decimal(value["memory"]["byte_seconds"]) for value in values if "memory" in value]
        result["memory"] = {
            "byte_seconds": _canonical_decimal(_sum_decimals(values_to_sum, f"{path}.memory.byte_seconds"))
        }

    if numeric_seen["gpu"]:
        gpu_sums: dict[tuple[str, str, str, str], list[Decimal]] = defaultdict(list)
        gpu_templates: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for value in values:
            if "gpu" not in value:
                continue
            for group in value["gpu"]["groups"]:
                key = _gpu_group_key(group)
                gpu_templates[key] = {name: item for name, item in group.items() if name != "instance_seconds"}
                gpu_sums[key].append(Decimal(group["instance_seconds"]))
        result["gpu"] = {
            "groups": [
                {
                    **gpu_templates[key],
                    "instance_seconds": _canonical_decimal(_sum_decimals(gpu_sums[key], f"{path}.gpu.groups")),
                }
                for key in sorted(gpu_sums)
            ]
        }

    if not fully_reported and any(numeric_seen.values()):
        issues: set[str] = set()
        if not complete:
            issues.add("observation_incomplete")
        for value in values:
            if value.get("status") == "partial":
                issues.update(value["issues"])
            elif value.get("status") == "unavailable":
                if value["issues"] == ["attribution_incomplete"]:
                    issues.add("attribution_incomplete")
                else:
                    issues.add("observation_incomplete")
        if not issues:
            issues.add("observation_incomplete")
        result["status"] = "partial"
        result["issues"] = sorted(issues)
    elif not fully_reported:
        result = {"status": "unavailable", "issues": ["observation_incomplete"]}

    ordered = {}
    if "status" in result:
        ordered["status"] = result.pop("status")
    if "issues" in result:
        ordered["issues"] = result.pop("issues")
    ordered.update(result)
    _validate_resource_time(ordered, path)
    return ordered


def aggregate_resource_time(values: Sequence[Mapping[str, Any]], *, complete: bool = True) -> dict[str, Any]:
    """Sum explicit execution capacity-time, retaining independent dimensions.

    A malformed input is rejected. Callers bound their input inventory and keep
    their existing subtotal if adding another contribution would overflow.
    """

    normalized = [normalize_measurement("resource_time", value) for value in values]
    return _resource_time_aggregate(normalized, complete=complete, path="resource_time")


def _aggregate_retained(
    values: Sequence[Mapping[str, Any]],
    *,
    complete: bool,
    path: str,
) -> dict[str, Any]:
    numeric = [value for value in values if value.get("status", "reported") in {"reported", "partial"}]
    if not numeric:
        return {"status": "unavailable"}
    amount = sum(int(value["bytes"]) for value in numeric)
    if amount > U128_MAX:
        _fail(f"{path}.bytes", "aggregate exceeds the unsigned 128-bit bound")
    reported = complete and len(numeric) == len(values) and all("status" not in value for value in values)
    return {"bytes": str(amount)} if reported else {"status": "partial", "bytes": str(amount)}


def _aggregate_cpu_consumed(
    values: Sequence[Mapping[str, Any]],
    *,
    complete: bool,
    path: str,
) -> dict[str, Any]:
    numeric = [value for value in values if value.get("status", "reported") in {"reported", "partial"}]
    if not numeric:
        return {"status": "unavailable"}
    seconds = _sum_decimals([Decimal(value["cpu_seconds"]) for value in numeric], f"{path}.cpu_seconds")
    reported = complete and len(numeric) == len(values) and all("status" not in value for value in values)
    result = {"cpu_seconds": _canonical_decimal(seconds)}
    return result if reported else {"status": "partial", **result}


def _aggregate_message_traffic(
    values: Sequence[Mapping[str, Any]],
    *,
    complete: bool,
    path: str,
) -> dict[str, Any]:
    numeric = [value for value in values if value.get("status", "reported") in {"reported", "partial"}]
    if not numeric:
        return {"status": "unavailable"}
    result = {}
    for field in ("sent_bytes", "received_bytes"):
        amount = sum(int(value[field]) for value in numeric)
        if amount > U128_MAX:
            _fail(f"{path}.{field}", "aggregate exceeds the unsigned 128-bit bound")
        result[field] = str(amount)
    reported = complete and len(numeric) == len(values) and all("status" not in value for value in values)
    return result if reported else {"status": "partial", **result}


def _aggregate_execution_measurement(field, values, complete, aggregate):
    normalized = [normalize_measurement(field, value) for value in values]
    result = aggregate(normalized, complete=complete, path=field)
    if "status" in result:
        issues = set() if complete else {"observation_incomplete"}
        for value in normalized:
            issues.update(value.get("issues", []))
        if result["status"] == "partial":
            issues = {issue if issue == "attribution_incomplete" else "observation_incomplete" for issue in issues}
        result["issues"] = sorted(issues or {"observation_incomplete"})
    return normalize_measurement(field, result)


def aggregate_cpu_consumed(values: Sequence[Mapping[str, Any]], *, complete: bool = True) -> dict[str, Any]:
    """Sum disjoint CPU work into a participant-valid measurement."""

    return _aggregate_execution_measurement("cpu_consumed", values, complete, _aggregate_cpu_consumed)


def aggregate_message_traffic(values: Sequence[Mapping[str, Any]], *, complete: bool = True) -> dict[str, Any]:
    """Sum owned endpoint counters once, retaining incomplete-source quality."""

    return _aggregate_execution_measurement("message_traffic", values, complete, _aggregate_message_traffic)


def _aggregate_totals(
    resource_times: Sequence[Mapping[str, Any]],
    cpu_consumed_values: Sequence[Mapping[str, Any]],
    retained_values: Sequence[Mapping[str, Any]],
    message_traffic_values: Sequence[Mapping[str, Any]],
    *,
    complete: bool,
    path: str,
) -> dict[str, Any]:
    totals = {
        "resource_time": _resource_time_aggregate(resource_times, complete=complete, path=f"{path}.resource_time"),
        "resource_utilization": {
            "cpu_consumed": _aggregate_cpu_consumed(
                cpu_consumed_values, complete=complete, path=f"{path}.resource_utilization.cpu_consumed"
            ),
            "retained_content": _aggregate_retained(
                retained_values, complete=complete, path=f"{path}.resource_utilization.retained_content"
            ),
            "message_traffic": _aggregate_message_traffic(
                message_traffic_values, complete=complete, path=f"{path}.resource_utilization.message_traffic"
            ),
        },
    }
    _validate_totals(totals, path)
    return totals


def _validate_participant_summary(record: Mapping[str, Any], path: str) -> None:
    _exact_keys(
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
        },
        {"duration_seconds"},
        path,
    )
    _identifier(record["job_id"], JOB_ID_PATTERN, f"{path}.job_id")
    _identifier(record["participant_name"], PARTICIPANT_NAME_PATTERN, f"{path}.participant_name")
    _timestamp(record["reported_at"], f"{path}.reported_at")
    _validate_resource_capacity(record["resource_capacity"], f"{path}.resource_capacity")
    _validate_resource_time(record["resource_time"], f"{path}.resource_time")
    if "duration_seconds" in record:
        _decimal(record["duration_seconds"], f"{path}.duration_seconds")
    _validate_resource_utilization(record["resource_utilization"], f"{path}.resource_utilization")
    _validate_workspace_filesystem(record["workspace_filesystem"], f"{path}.workspace_filesystem")


def participant_measurements(participant_record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the validated terminal measurement objects into the server entry."""

    participant_record = normalize_record(participant_record)
    if participant_record["kind"] != KIND_PARTICIPANT_SUMMARY:
        _fail("$.kind", f"must equal {KIND_PARTICIPANT_SUMMARY}")
    return {
        **(
            {"duration_seconds": participant_record["duration_seconds"]}
            if "duration_seconds" in participant_record
            else {}
        ),
        "resource_capacity": participant_record["resource_capacity"],
        "resource_time": participant_record["resource_time"],
        "resource_utilization": participant_record["resource_utilization"],
    }


def _validate_participant_entry(value: Any, path: str) -> dict[str, Any] | None:
    entry = _mapping(value, path)
    status = _optional_status(entry, "accepted", PARTICIPANT_STATUSES, f"{path}.status")
    base = {"participant_name", "role"}
    resource_time = None
    if status == "accepted":
        _exact_keys(
            entry,
            base
            | {
                "received_at",
                "resource_capacity",
                "resource_time",
                "resource_utilization",
            },
            {"duration_seconds"},
            path,
        )
        _timestamp(entry["received_at"], f"{path}.received_at")
        if "duration_seconds" in entry:
            _decimal(entry["duration_seconds"], f"{path}.duration_seconds")
        _validate_resource_capacity(entry["resource_capacity"], f"{path}.resource_capacity")
        _validate_resource_time(entry["resource_time"], f"{path}.resource_time")
        resource_time = dict(entry["resource_time"])
        _validate_resource_utilization(entry["resource_utilization"], f"{path}.resource_utilization")
    elif status == "invalid":
        _exact_keys(entry, base | {"status", "received_at"}, set(), path)
        _timestamp(entry["received_at"], f"{path}.received_at")
    else:
        _exact_keys(entry, base | {"status"}, set(), path)
    _identifier(entry["participant_name"], PARTICIPANT_NAME_PATTERN, f"{path}.participant_name")
    _enum(entry["role"], ROLES, f"{path}.role")
    return resource_time


def _validate_participant_list(
    participants: Sequence[Mapping[str, Any]],
    *,
    path: str,
) -> list[dict[str, Any]]:
    if not isinstance(participants, list) or not 1 <= len(participants) <= MAX_PARTICIPANTS:
        _fail(path, f"must contain 1..{MAX_PARTICIPANTS} entries")
    ordering: list[tuple[str, str]] = []
    resource_times = []
    for index, entry in enumerate(participants):
        resource_time = _validate_participant_entry(entry, f"{path}[{index}]")
        if resource_time is not None:
            resource_times.append(resource_time)
        ordering.append((entry["role"], entry["participant_name"]))
    if len({item[1] for item in ordering}) != len(ordering):
        _fail(path, "participant_name values must be unique")
    return resource_times


def derive_job_totals(participants: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum recorded participant work/time from the expected-participant list."""

    participants = _normalize_tree(participants)
    resource_times = _validate_participant_list(participants, path="participants")
    accepted = [entry for entry in participants if "status" not in entry]
    return _aggregate_totals(
        resource_times,
        [entry["resource_utilization"]["cpu_consumed"] for entry in accepted],
        [entry["resource_utilization"]["retained_content"] for entry in accepted],
        [entry["resource_utilization"]["message_traffic"] for entry in accepted],
        complete=len(accepted) == len(participants),
        path="totals",
    )


def _validate_resource_summary(record: Mapping[str, Any], path: str) -> None:
    _exact_keys(
        record,
        {
            "schema_version",
            "kind",
            "job_id",
            "participants",
        },
        set(),
        path,
    )
    _identifier(record["job_id"], JOB_ID_PATTERN, f"{path}.job_id")
    # Derivation validates the participant list and aggregate numeric bounds.
    # The resulting totals are not stored in the summary.
    derive_job_totals(record["participants"])


def _validate_study_job(value: Any, path: str) -> None:
    job = _mapping(value, path)
    resource_data = _enum(job.get("resource_data"), RESOURCE_DATA_STATES, f"{path}.resource_data")
    if resource_data == "included":
        _exact_keys(
            job,
            {"job_id", "job_name", "job_status", "resource_data", "totals"},
            set(),
            path,
        )
        _validate_totals(job["totals"], f"{path}.totals")
    else:
        _exact_keys(job, {"job_id", "job_name", "job_status", "resource_data"}, set(), path)
    _identifier(job["job_id"], JOB_ID_PATTERN, f"{path}.job_id")
    if not isinstance(job["job_name"], str) or not 1 <= len(job["job_name"]) <= MAX_JOB_NAME_LENGTH:
        _fail(f"{path}.job_name", f"must be a nonempty string of at most {MAX_JOB_NAME_LENGTH} characters")
    job_status = _identifier(job["job_status"], JOB_STATUS_PATTERN, f"{path}.job_status")
    terminal = job_status.startswith("FINISHED:") or job_status in LEGACY_TERMINAL_JOB_STATES
    if resource_data in {"included", "unavailable"} and not terminal:
        _fail(f"{path}.job_status", f"must be a terminal job status when resource_data is {resource_data}")
    if resource_data == "nonterminal" and terminal:
        _fail(f"{path}.job_status", "must not be a terminal job status when resource_data is nonterminal")


def _validate_study_jobs(jobs: Sequence[Mapping[str, Any]], path: str) -> None:
    if not isinstance(jobs, list) or len(jobs) > MAX_STUDY_JOBS:
        _fail(path, f"must contain 0..{MAX_STUDY_JOBS} entries")
    job_ids: list[str] = []
    for index, job in enumerate(jobs):
        _validate_study_job(job, f"{path}[{index}]")
        job_ids.append(job["job_id"])
    if len(job_ids) != len(set(job_ids)):
        _fail(path, "must contain unique job_id values")


def derive_study_totals(job_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Derive totals for selected jobs that have readable resource reports.

    Study coverage separately reports jobs without usable final reports.
    """

    job_rows = _normalize_tree(job_rows)
    _validate_study_jobs(job_rows, "jobs")
    included = [job for job in job_rows if job["resource_data"] == "included"]
    return _aggregate_totals(
        [job["totals"]["resource_time"] for job in included],
        [job["totals"]["resource_utilization"]["cpu_consumed"] for job in included],
        [job["totals"]["resource_utilization"]["retained_content"] for job in included],
        [job["totals"]["resource_utilization"]["message_traffic"] for job in included],
        complete=True,
        path="totals",
    )


def _validate_study_summary(record: Mapping[str, Any], path: str) -> None:
    _exact_keys(
        record,
        {"schema_version", "kind", "selection", "generated_at", "coverage", "jobs", "totals"},
        set(),
        path,
    )
    selection = _mapping(record["selection"], f"{path}.selection")
    _exact_keys(selection, {"study_name"}, set(), f"{path}.selection")
    _identifier(selection["study_name"], STUDY_NAME_PATTERN, f"{path}.selection.study_name")
    _timestamp(record["generated_at"], f"{path}.generated_at")
    _validate_study_jobs(record["jobs"], f"{path}.jobs")

    coverage = _mapping(record["coverage"], f"{path}.coverage")
    coverage_names = {"selected_jobs", "included_jobs", "unavailable_jobs", "nonterminal_jobs"}
    _exact_keys(coverage, coverage_names, set(), f"{path}.coverage")
    actual = {
        "selected_jobs": len(record["jobs"]),
        "included_jobs": sum(job["resource_data"] == "included" for job in record["jobs"]),
        "unavailable_jobs": sum(job["resource_data"] == "unavailable" for job in record["jobs"]),
        "nonterminal_jobs": sum(job["resource_data"] == "nonterminal" for job in record["jobs"]),
    }
    for name in sorted(coverage_names):
        reported = _integer(coverage[name], f"{path}.coverage.{name}", maximum=MAX_STUDY_JOBS)
        if reported != actual[name]:
            _fail(f"{path}.coverage.{name}", f"must equal {actual[name]} from jobs")

    _validate_totals(record["totals"], f"{path}.totals")
    expected = derive_study_totals(record["jobs"])
    if record["totals"] != expected:
        _fail(f"{path}.totals", "must exactly equal the included-job sums")


def normalize_record(record: Mapping[str, Any], *, serialized_size: int | None = None) -> dict[str, Any]:
    """Validate current-format contents and return an independent normalized record."""

    record = _mapping(record, "$")
    kind = _enum(record.get("kind"), RECORD_KINDS, "$.kind")
    if record.get("schema_version") != SCHEMA_VERSION:
        _fail("$.schema_version", f"must equal {SCHEMA_VERSION}")
    if serialized_size is not None:
        if not isinstance(serialized_size, int) or isinstance(serialized_size, bool) or serialized_size < 0:
            _fail("$", "serialized_size must be a non-negative integer")
        if serialized_size > MAX_RECORD_BYTES[kind]:
            _fail("$", f"record exceeds the {MAX_RECORD_BYTES[kind]}-byte limit")
    if _json_depth(record) > MAX_JSON_DEPTH:
        _fail("$", f"record exceeds maximum JSON depth {MAX_JSON_DEPTH}")
    _privacy_walk(record)
    record = _normalize_tree(record)
    if kind == KIND_PARTICIPANT_SUMMARY:
        _validate_participant_summary(record, "$")
    elif kind == KIND_RESOURCE_SUMMARY:
        _validate_resource_summary(record, "$")
    else:
        _validate_study_summary(record, "$")
    return record


def validate_record(record: Mapping[str, Any], *, serialized_size: int | None = None) -> None:
    """Validate input contents without imposing writer formatting or mutation."""

    normalize_record(record, serialized_size=serialized_size)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"$: duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ContractError(f"$: non-finite JSON number {value!r} is forbidden")


def load_and_validate(data: bytes, expected_kind: str | None = None) -> dict[str, Any]:
    """Decode UTF-8 JSON with duplicate-key rejection, then validate it."""

    if not isinstance(data, bytes):
        _fail("$", "input must be bytes")
    if expected_kind is not None and expected_kind not in RECORD_KINDS:
        _fail("expected_kind", "is not a supported resource-statistics record kind")
    absolute_limit = MAX_RECORD_BYTES[expected_kind] if expected_kind else max(MAX_RECORD_BYTES.values())
    if len(data) > absolute_limit:
        _fail("$", f"record exceeds the absolute {absolute_limit}-byte input limit")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError("$: input must be valid UTF-8") from exc
    try:
        record = json.loads(text, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_nonfinite)
    except ContractError:
        raise
    except RecursionError as exc:
        raise ContractError(f"$: JSON nesting exceeds maximum depth {MAX_JSON_DEPTH}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise ContractError(f"$: invalid JSON: {exc}") from exc
    if not isinstance(record, dict):
        _fail("$", "record must be a JSON object")
    record = normalize_record(record, serialized_size=len(data))
    if expected_kind is not None and record["kind"] != expected_kind:
        _fail("$.kind", f"must equal {expected_kind}")
    return record


def validate_bundle(
    resource_summary: Mapping[str, Any],
    participant_records: Mapping[str, Mapping[str, Any]],
    file_bytes: Mapping[str, bytes],
) -> None:
    """Validate a finalized archive and its summary-derived exact inventory."""

    resource_summary = normalize_record(resource_summary)
    if resource_summary["kind"] != KIND_RESOURCE_SUMMARY:
        _fail("resource_summary.kind", f"must equal {KIND_RESOURCE_SUMMARY}")
    if not isinstance(participant_records, Mapping):
        _fail("participant_records", "must be a participant_name-to-record mapping")
    if not isinstance(file_bytes, Mapping):
        _fail("file_bytes", "must be a relative-path-to-bytes mapping")

    accepted = {entry["participant_name"]: entry for entry in resource_summary["participants"] if "status" not in entry}
    if set(participant_records) != set(accepted):
        _fail("participant_records", "keys must exactly equal the accepted participant names")

    expected_paths = {"resource_summary.json"} | {
        f"participants/{participant_name}.json" for participant_name in accepted
    }
    if set(file_bytes) != expected_paths:
        _fail("file_bytes", "paths must exactly cover the summary and accepted participant records")

    decoded_summary: Mapping[str, Any] | None = None
    decoded_participants: dict[str, Mapping[str, Any]] = {}
    for relative_path in sorted(expected_paths):
        data = file_bytes[relative_path]
        if not isinstance(data, bytes):
            _fail(f"file_bytes[{relative_path!r}]", "must be exact bytes")
        decoded = load_and_validate(data)
        if relative_path == "resource_summary.json":
            decoded_summary = decoded
        else:
            participant_name = relative_path.removeprefix("participants/").removesuffix(".json")
            decoded_participants[participant_name] = decoded

    if decoded_summary != resource_summary:
        _fail("resource_summary", "does not equal the exact decoded archived record")

    for participant_name, record in participant_records.items():
        _identifier(participant_name, PARTICIPANT_NAME_PATTERN, f"participant_records[{participant_name!r}]")
        record = normalize_record(record)
        if record["kind"] != KIND_PARTICIPANT_SUMMARY:
            _fail(f"participant_records[{participant_name!r}].kind", f"must equal {KIND_PARTICIPANT_SUMMARY}")
        if record["participant_name"] != participant_name:
            _fail(f"participant_records[{participant_name!r}].participant_name", "must equal its mapping key")
        if record["job_id"] != resource_summary["job_id"]:
            _fail(f"participant_records[{participant_name!r}].job_id", "must equal summary job_id")
        if decoded_participants.get(participant_name) != record:
            _fail(f"participant_records[{participant_name!r}]", "does not equal the decoded archived record")

        entry = accepted[participant_name]
        if record["schema_version"] != resource_summary["schema_version"]:
            _fail(f"participant_records[{participant_name!r}].schema_version", "must equal summary version")
        copied = participant_measurements(record)
        for field in copied.keys() | {"duration_seconds"}:
            if entry.get(field) != copied.get(field):
                _fail(
                    f"resource_summary.participants[{participant_name!r}].{field}",
                    "must exactly copy the accepted terminal participant report",
                )
