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

"""Shared scenario constants, errors, and validation helpers."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

SCHEMA_VERSION = "1"
COMPARISON_MODE_ABLATION = "mode_ablation"
COMPARISON_AGENT = "agent_comparison"
COMPARISON_MODEL = "model_comparison"
COMPARISON_ONE = "one"
COMPARISON_TYPES = {COMPARISON_MODE_ABLATION, COMPARISON_AGENT, COMPARISON_MODEL, COMPARISON_ONE}
JOB_SCALES = {"small", "medium", "large"}
DEFAULT_PATH_BUDGET = 240
SLUG_VISIBLE_LENGTH = 48

DEFAULT_RESOURCE_POLICIES: dict[str, dict[str, int]] = {
    "small": {
        "agent_timeout_seconds": 30 * 60,
        "container_timeout_seconds": 40 * 60,
        "result_size_budget_bytes": 1 * 1024 * 1024 * 1024,
        "memory_limit_bytes": 8 * 1024 * 1024 * 1024,
        "cpu_limit_millis": 4000,
        "pids_limit": 512,
    },
    "medium": {
        "agent_timeout_seconds": 90 * 60,
        "container_timeout_seconds": 120 * 60,
        "result_size_budget_bytes": 5 * 1024 * 1024 * 1024,
        "memory_limit_bytes": 16 * 1024 * 1024 * 1024,
        "cpu_limit_millis": 8000,
        "pids_limit": 1024,
    },
    "large": {
        "agent_timeout_seconds": 240 * 60,
        "container_timeout_seconds": 300 * 60,
        "result_size_budget_bytes": 20 * 1024 * 1024 * 1024,
        "memory_limit_bytes": 32 * 1024 * 1024 * 1024,
        "cpu_limit_millis": 16000,
        "pids_limit": 2048,
    },
}

DEFAULT_QUALITY_GATE = {
    "agent_process_passed": True,
    "final_container_exit_code": 0,
    "source_input_modified": False,
    "required_validation_metric_status": ["present", "not_required"],
    "critical_quality_checks_failed": False,
    "eval_contract_status": ["passed", "not_scored"],
    "expected_skill_status": ["passed", "not_scored"],
    "required_behavior_status": ["passed", "not_scored"],
}
QUALITY_GATE_KEYS = set(DEFAULT_QUALITY_GATE)
REQUIRED_VALIDATION_METRIC_STATUSES = {"present", "not_required", "missing"}
COMPLIANCE_STATUSES = {"passed", "failed", "not_scored"}
DEFAULT_WINNER_POLICY = "median_agent_elapsed_seconds_then_tokens_with_quality_gate"
UNAVAILABLE_STRUCTURE_QUALITY_SIGNAL = {
    "status": "unavailable",
    "reason": "structure quality was not captured for this run",
}
SUMMARY_RUN_FIELDS = (
    "run_id",
    "sequence",
    "scenario_name",
    "comparison_type",
    "comparison_group_id",
    "agent",
    "agent_model",
    "workflow",
    "job_name",
    "job_slug",
    "job_path",
    "job_scale",
    "mode",
    "skills_enabled",
    "prompt_hash",
    "record_dir",
)


class ScenarioValidationError(ValueError):
    """Raised when a scenario cannot produce a valid run plan."""


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def slug_base(value: str) -> tuple[str, bool]:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return normalized[:SLUG_VISIBLE_LENGTH].rstrip("_"), len(normalized) > SLUG_VISIBLE_LENGTH


def slugify(value: str, *, force_hash: bool = False) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    hash_input = str(value) or "empty"
    suffix = stable_hash(hash_input)
    if not normalized:
        return f"item_{suffix}"
    visible = normalized[:SLUG_VISIBLE_LENGTH].rstrip("_")
    if force_hash or len(normalized) > SLUG_VISIBLE_LENGTH:
        return f"{visible}_{suffix}"
    return visible


def unique_slug_map(values: Iterable[str]) -> dict[str, str]:
    ordered = list(dict.fromkeys(str(value) for value in values))
    bases: dict[str, list[str]] = {}
    truncated: dict[str, bool] = {}
    for value in ordered:
        base, was_truncated = slug_base(value)
        base = base or "item"
        bases.setdefault(base, []).append(value)
        truncated[value] = was_truncated
    result = {}
    for value in ordered:
        base, _was_truncated = slug_base(value)
        base = base or "item"
        result[value] = slugify(value, force_hash=truncated[value] or len(bases[base]) > 1)
    return result


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScenarioValidationError(f"{label} must be a mapping")
    return value


def require_non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScenarioValidationError(f"{label} must be a non-empty string")
    return value.strip()


def as_list(value: Any, label: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    raise ScenarioValidationError(f"{label} must be a list")


def model_list(value: Any, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        raise ScenarioValidationError(f"{label} must be a string or list of strings")
    models = tuple(require_non_empty_string(item, label) for item in items)
    if len(set(models)) != len(models):
        raise ScenarioValidationError(f"{label} contains duplicate model names")
    return models


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    scenario_path = Path(path)
    try:
        raw = yaml.safe_load(scenario_path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ScenarioValidationError(f"Could not read scenario file {scenario_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ScenarioValidationError(f"Scenario file {scenario_path} must contain a YAML object")
    return raw
