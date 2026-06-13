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

"""Benchmark-record synthesis and normalization helpers."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .common import bool_from_text, flatten_numbers, load_json, write_json
from .metric_artifacts import validation_metric_from_workspace_delta_manifest
from .quality_signals import critical_quality_checks_failed, metric_signal, required_validation_metric_status
from .record_identity import record_case, record_skill
from .timing import LifecycleEpochs, finalize_timing

ALLOWED_BEHAVIOR_STATUSES = {"pass", "fail", "missing", "not_applicable", "non_scoring_note"}
MAX_JSON_RECORD_FILES = 10000
MAX_JSON_RECORD_FILE_BYTES = 5 * 1024 * 1024
MAX_EVENTS_TEXT_BYTES = 20 * 1024 * 1024
MAX_EVALS_FILE_BYTES = 512 * 1024
MAX_SKILL_NAME_EVENT_IDENTITY_CANDIDATES = 50
UNAVAILABLE_STRUCTURE_QUALITY_SIGNAL = {
    "status": "unavailable",
    "reason": "structure quality was not captured for this run",
}


@dataclass(frozen=True)
class AgentRecordSynthesisInputs:
    agent_record_path: Path
    records_dir: Path
    events_path: Path
    usage_path: Path
    activity_path: Path
    last_message_path: Path
    input_dir: Path
    mode: str
    elapsed_seconds: int
    agent_exit: int
    skills_enabled: bool
    skill_run_mode: str
    agent: str
    agent_model: str
    run_start_time_ns: int
    workspace_delta_manifest_path: Path
    input_delta_manifest_path: Path | None = None
    prompt_path: Path | None = None


def apply_record_runtime_fields(
    record: dict[str, Any],
    *,
    usage: dict[str, Any],
    mode: str,
    elapsed_seconds: int,
    agent_exit: int,
    skills_enabled: bool,
    skill_run_mode: str,
    agent: str,
    agent_model: str,
    agent_record_present: bool | None = None,
    agent_record_valid: bool | None = None,
) -> dict[str, Any]:
    """Mutate ``record`` in place and return its ``process_metrics`` mapping."""

    record["schema_version"] = "1"
    record["run_mode"] = record.get("run_mode") or skill_run_mode
    record["agent"] = record.get("agent") or agent
    record["mode"] = mode
    record["source"] = "agent_benchmark_harness"
    record["agent_model"] = record.get("agent_model") or agent_model
    record["skills_enabled"] = skills_enabled
    record["agent_process_passed"] = agent_exit == 0
    record["agent_process_exit_code"] = agent_exit
    record["agent_elapsed_seconds"] = elapsed_seconds
    record["elapsed_seconds"] = elapsed_seconds
    record["timestamp"] = record.get("timestamp") or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if agent_record_present is not None:
        record["agent_record_present"] = agent_record_present
    if agent_record_valid is not None:
        record["agent_record_valid"] = agent_record_valid

    metrics = record.get("process_metrics")
    if not isinstance(metrics, dict):
        metrics = {}
        record["process_metrics"] = metrics
    metrics.update(
        {
            "elapsed_seconds": elapsed_seconds,
            "agent_elapsed_seconds": elapsed_seconds,
            "token_count": usage.get("total_tokens"),
            "cache_tokens": usage.get("cache_tokens"),
            "cost": usage.get("cost"),
            "agent_exit_code": agent_exit,
            "agent_process_passed": 1 if agent_exit == 0 else 0,
            "token_parser": usage.get("token_parser"),
        }
    )
    parser_warnings = usage.get("parser_warnings") or usage.get("token_parser_warnings")
    if parser_warnings:
        metrics["token_parser_warning_count"] = len(parser_warnings)
    if agent_record_present is not None:
        metrics["agent_record_present"] = 1 if agent_record_present else 0
    if agent_record_valid is not None:
        metrics["agent_record_valid"] = 1 if agent_record_valid else 0
    return metrics


def same_path(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except Exception:
        return False


def record_is_candidate(record: Any) -> bool:
    return isinstance(record, dict) and (record_skill(record) or record_case(record))


def load_text(path: Path, *, max_bytes: int | None = None) -> str:
    try:
        if max_bytes is not None:
            if max_bytes <= 0:
                return ""
            with path.open("rb") as stream:
                return stream.read(max_bytes).decode("utf-8", errors="replace")
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def iter_json_records(root: Path, agent_record_path: Path | None = None) -> Iterable[tuple[Path, dict[str, Any]]]:
    scanned = 0
    record_paths = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = [name for name in dirnames if not (current / name).is_symlink()]
        record_paths.extend(current / filename for filename in filenames if filename.endswith(".json"))
    for path in sorted(record_paths):
        if not path.is_file() or path.is_symlink():
            continue
        if agent_record_path is not None and same_path(path, agent_record_path):
            continue
        if path.name.endswith("_agent_record.json") or path.name.endswith("_record.json"):
            continue
        try:
            if path.stat().st_size > MAX_JSON_RECORD_FILE_BYTES:
                continue
        except OSError:
            continue
        scanned += 1
        if scanned > MAX_JSON_RECORD_FILES:
            break
        data = load_json(path)
        if isinstance(data, dict):
            yield path, data
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield path, item


def path_is_current_run(path: Path, run_start_time_ns: int) -> bool:
    try:
        return path.stat().st_mtime_ns >= max(0, run_start_time_ns)
    except OSError:
        return False


GUIDANCE_FILE_STEMS = {
    "readme",
    "task",
    "tasks",
    "instruction",
    "instructions",
    "guidance",
    "metric",
    "metrics",
    "source",
    "notes",
}
GUIDANCE_SUFFIXES = {".md", ".rst", ".txt"}
MAX_GUIDANCE_FILES = 12
MAX_GUIDANCE_FILE_BYTES = 512 * 1024


def guidance_priority(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    stem = path.stem.lower()
    if name == "prompt.txt":
        return (0, name)
    if name == "readme.md":
        return (1, name)
    if stem == "readme":
        return (2, name)
    if stem in GUIDANCE_FILE_STEMS:
        return (3, name)
    return (4, name)


def is_guidance_file(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in GUIDANCE_SUFFIXES:
        return False
    stem = path.stem.lower()
    return stem in GUIDANCE_FILE_STEMS or path.name.lower().startswith("readme")


def guidance_identity(path: Path) -> tuple:
    try:
        return ("resolved", str(path.resolve()))
    except OSError:
        pass
    if path.is_symlink():
        try:
            target = path.readlink()
            if not target.is_absolute():
                target = path.parent / target
            return ("symlink_target", str(target))
        except OSError:
            pass
    try:
        stat_result = path.stat()
        return ("inode", stat_result.st_dev, stat_result.st_ino)
    except OSError:
        return ("path", str(path))


def discover_job_guidance(input_root: Path, prompt_path: Path | None = None) -> tuple[list[dict[str, str]], str]:
    if not input_root.is_dir():
        return [], ""
    candidates = []
    if prompt_path is not None and prompt_path.is_file():
        candidates.append((prompt_path, "prompt"))
    for path in input_root.iterdir():
        if path.is_symlink():
            continue
        if is_guidance_file(path):
            candidates.append((path, "job_documentation"))
            if len(candidates) >= MAX_GUIDANCE_FILES:
                break
    docs_dir = input_root / "docs"
    if len(candidates) < MAX_GUIDANCE_FILES and docs_dir.is_dir() and not docs_dir.is_symlink():
        for path in docs_dir.iterdir():
            if path.is_symlink():
                continue
            if is_guidance_file(path):
                candidates.append((path, "job_documentation"))
                if len(candidates) >= MAX_GUIDANCE_FILES:
                    break
    if not candidates:
        return [], ""
    candidates.sort(key=lambda item: guidance_priority(item[0]))
    sources = []
    chunks = []
    seen = set()
    for path, source_type in candidates[:MAX_GUIDANCE_FILES]:
        identity = guidance_identity(path)
        if identity in seen:
            continue
        seen.add(identity)
        try:
            if path.stat().st_size > MAX_GUIDANCE_FILE_BYTES:
                continue
        except OSError:
            continue
        text = load_text(path)
        if not text:
            continue
        sources.append({"path": str(path), "source_type": source_type, "text": text})
        chunks.append(f"\n\n# {source_type}: {path}\n{text}")
    return sources, "".join(chunks)


def discover_readme(input_root: Path) -> tuple[Path | None, str]:
    sources, guidance_text = discover_job_guidance(input_root)
    if not sources:
        return None, ""
    return Path(sources[0]["path"]), guidance_text


def agent_home_from_env() -> Path:
    return Path(os.environ.get("BENCHMARK_AGENT_HOME", "/workspace/agent-home"))


def available_skill_names() -> set[str]:
    names = set()
    skills_root = agent_home_from_env() / "skills"
    if skills_root.is_dir():
        for path in skills_root.iterdir():
            if path.is_symlink():
                continue
            if path.is_dir() and not path.name.startswith("."):
                names.add(path.name)
    return names


def eval_case_ids_for_skill(skill_name: str) -> list[str]:
    evals_path = agent_home_from_env() / "skills" / skill_name / "evals" / "evals.json"
    try:
        if not evals_path.is_file() or evals_path.stat().st_size > MAX_EVALS_FILE_BYTES:
            return []
    except OSError:
        return []
    data = load_json(evals_path)
    if not isinstance(data, dict):
        return []
    case_ids = []
    for item in data.get("evals") or []:
        if isinstance(item, dict) and item.get("id"):
            case_ids.append(str(item["id"]))
    return case_ids


IDENTITY_PLACEHOLDERS = {
    "CASE",
    "CASE_ID",
    "EVAL_ID",
    "SKILL",
    "SKILL_NAME",
    "<case>",
    "<case_id>",
    "<skill>",
    "<skill_name>",
}


def valid_identity_token(value: str) -> bool:
    normalized = value.strip()
    return (
        bool(normalized) and normalized not in IDENTITY_PLACEHOLDERS and normalized.upper() not in IDENTITY_PLACEHOLDERS
    )


def identity_occurrence_count(text: str, candidate: str) -> int:
    pattern = re.compile(rf"(?<![A-Za-z0-9_.-]){re.escape(candidate)}(?![A-Za-z0-9_.-])")
    return sum(1 for _match in pattern.finditer(text))


def infer_from_events(events_text: str) -> dict[str, Any]:
    scores: dict[str, int] = {}
    source: dict[str, str] = {}
    parser_warnings: list[str] = []

    def add(name: str, points: int, reason: str) -> None:
        if not valid_identity_token(name):
            return
        scores[name] = scores.get(name, 0) + points
        source.setdefault(name, reason)

    for match in re.finditer(r"/[^\s\"']+/skills/([^/\s\"']+)", events_text):
        add(match.group(1), 50, "agent_skill_path")

    for match in re.finditer(r"(?:^|\s)--skill(?:=|\s+)([A-Za-z0-9_.-]+)", events_text):
        add(match.group(1), 100, "agent_skill_arg")

    skill_names = sorted(
        (name for name in available_skill_names() if valid_identity_token(name)),
        key=lambda name: (-len(name), name),
    )
    if len(skill_names) > MAX_SKILL_NAME_EVENT_IDENTITY_CANDIDATES:
        parser_warnings.append(
            "installed skill name event inference was capped at "
            f"{MAX_SKILL_NAME_EVENT_IDENTITY_CANDIDATES} candidates"
        )
        skill_names = skill_names[:MAX_SKILL_NAME_EVENT_IDENTITY_CANDIDATES]
    if skill_names:
        skill_pattern = re.compile("|".join(re.escape(name) for name in skill_names))
        for name, occurrences in Counter(match.group(0) for match in skill_pattern.finditer(events_text)).items():
            add(name, occurrences, "installed_skill_name_seen_in_events")

    skill = ""
    if scores:
        skill = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[0][0]

    case_id = ""
    case_source = ""
    case_match = re.search(r"(?:^|\s)--case(?:=|\s+)([A-Za-z0-9_.-]+)", events_text)
    if case_match and valid_identity_token(case_match.group(1)):
        case_id = case_match.group(1)
        case_source = "agent_skill_arg"
    elif skill:
        case_scores = {}
        candidates = sorted(
            (candidate for candidate in eval_case_ids_for_skill(skill) if valid_identity_token(candidate)),
            key=len,
            reverse=True,
        )
        for candidate in candidates:
            count = identity_occurrence_count(events_text, candidate)
            if count:
                case_scores[candidate] = count
        if case_scores:
            case_id = sorted(case_scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[0][0]
            case_source = "installed_skill_case_seen_in_events"
    return {
        "skill": skill,
        "case_id": case_id,
        "skill_source": source.get(skill, "") if skill else "",
        "case_source": case_source,
        "skill_scores": dict(sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:5]),
        "parser_warnings": parser_warnings,
        "used_as_fallback": False,
    }


def record_score(path: Path, record: dict[str, Any]) -> tuple[int, int, str]:
    score = 0
    if record_skill(record):
        score += 2
    if record_case(record):
        score += 2
    if isinstance(record.get("mandatory_behavior"), dict) and record["mandatory_behavior"]:
        score += 1
    if isinstance(record.get("prohibited_behavior"), dict) and record["prohibited_behavior"]:
        score += 1
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = 0
    return score, mtime, str(path)


def choose_source_record(
    candidates: list[tuple[Path, dict[str, Any]]],
    *,
    expected_skill: str = "",
    expected_case: str = "",
) -> tuple[Path | None, dict[str, Any] | None, dict[str, Any]]:
    audit: dict[str, Any] = {
        "candidate_count": len(candidates),
        "expected_skill": expected_skill or None,
        "expected_case": expected_case or None,
        "selection_reason": "",
        "selected_path": None,
        "selected_skill": None,
        "selected_case": None,
    }
    explicit_candidates = [
        (path, record) for path, record in candidates if record_skill(record) and record_case(record)
    ]
    audit["explicit_identity_candidate_count"] = len(explicit_candidates)
    if not explicit_candidates:
        audit["selection_reason"] = "no_explicit_identity_candidates"
        return None, None, audit

    filtered = explicit_candidates
    if expected_skill:
        filtered = [(path, record) for path, record in filtered if str(record_skill(record)) == str(expected_skill)]
    if expected_case:
        filtered = [(path, record) for path, record in filtered if str(record_case(record)) == str(expected_case)]
    audit["identity_matched_candidate_count"] = len(filtered)
    if (expected_skill or expected_case) and not filtered:
        audit["selection_reason"] = "no_candidate_matched_expected_identity"
        return None, None, audit

    identities = sorted({(str(record_skill(record)), str(record_case(record))) for _path, record in filtered})
    audit["candidate_identities"] = [{"skill": skill, "case": case} for skill, case in identities]
    if not expected_skill and not expected_case and len(identities) != 1:
        audit["selection_reason"] = "ambiguous_identity_without_expected_filter"
        return None, None, audit

    scored = []
    for path, record in filtered:
        score, mtime, path_text = record_score(path, record)
        scored.append((score, mtime, path_text, path, record))
    if not scored:
        audit["selection_reason"] = "no_scored_candidates"
        return None, None, audit
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    selected_path = scored[0][3]
    selected_record = scored[0][4]
    audit["selection_reason"] = "selected_explicit_identity_record"
    audit["selected_path"] = str(selected_path)
    audit["selected_skill"] = str(record_skill(selected_record))
    audit["selected_case"] = str(record_case(selected_record))
    audit["selected_score"] = scored[0][0]
    return selected_path, selected_record, audit


def synthesize_agent_record(inputs: AgentRecordSynthesisInputs) -> None:
    agent_record_path = inputs.agent_record_path
    records_dir = inputs.records_dir
    events_path = inputs.events_path
    usage_path = inputs.usage_path
    activity_path = inputs.activity_path
    last_message_path = inputs.last_message_path
    input_dir = inputs.input_dir
    mode = inputs.mode
    elapsed_seconds = inputs.elapsed_seconds
    agent_exit = inputs.agent_exit
    skills_enabled = inputs.skills_enabled
    skill_run_mode = inputs.skill_run_mode
    agent = inputs.agent
    agent_model = inputs.agent_model
    run_start_time_ns = inputs.run_start_time_ns
    workspace_delta_manifest_path = inputs.workspace_delta_manifest_path
    input_delta_manifest_path = inputs.input_delta_manifest_path
    prompt_path = inputs.prompt_path
    records_dir.mkdir(parents=True, exist_ok=True)
    existing_agent_record = load_json(agent_record_path)
    usage = load_json(usage_path, {}) or {}
    activity = load_json(activity_path, {}) or {}
    workspace_delta = load_json(workspace_delta_manifest_path, {}) or {}
    input_delta: dict[str, Any] | None = None
    input_delta_not_captured_reason = "input_delta_manifest was not provided"
    if input_delta_manifest_path is not None:
        loaded_input_delta = load_json(input_delta_manifest_path)
        if isinstance(loaded_input_delta, dict) and loaded_input_delta.get("delta_scope") == "input_snapshot":
            input_delta = loaded_input_delta
        elif isinstance(loaded_input_delta, dict) and loaded_input_delta:
            input_delta_not_captured_reason = (
                f"input_delta_manifest had unexpected delta_scope={loaded_input_delta.get('delta_scope')!r}"
            )
        else:
            input_delta_not_captured_reason = "input_delta_manifest was missing, empty, or unreadable"
    events_text = load_text(events_path, max_bytes=MAX_EVENTS_TEXT_BYTES)
    last_message = load_text(last_message_path)
    guidance_sources, guidance_text = discover_job_guidance(input_dir, prompt_path)
    event_identity = infer_from_events(events_text)
    job_guidance_metric_signal = metric_signal(guidance_sources, guidance_text, last_message)
    artifact_validation_metric = validation_metric_from_workspace_delta_manifest(
        workspace_delta,
        workspace_delta_manifest_path,
        job_guidance_metric_signal.get("expected_primary_metric"),
    )
    existing_skill = str(record_skill(existing_agent_record) or "") if isinstance(existing_agent_record, dict) else ""
    existing_case = str(record_case(existing_agent_record) or "") if isinstance(existing_agent_record, dict) else ""
    if not valid_identity_token(existing_skill):
        existing_skill = ""
    if not valid_identity_token(existing_case):
        existing_case = ""

    candidates = [
        (path, record)
        for path, record in iter_json_records(records_dir, agent_record_path)
        if path_is_current_run(path, run_start_time_ns)
        and record_is_candidate(record)
        and record_skill(record)
        and record_case(record)
    ]
    source_path, source_record, source_audit = choose_source_record(
        candidates,
        expected_skill=existing_skill,
        expected_case=existing_case,
    )

    base: dict[str, Any] = {}
    record_source = "harness_synthesized"
    if isinstance(source_record, dict):
        base = copy.deepcopy(source_record)
        record_source = "agent_runtime_evidence_record"
    elif isinstance(existing_agent_record, dict):
        base = copy.deepcopy(existing_agent_record)
        record_source = "existing_mode_agent_record"

    skill = record_skill(base)
    case_id = record_case(base)
    event_skill = event_identity.get("skill") if isinstance(event_identity, dict) else ""
    event_case = event_identity.get("case_id") if isinstance(event_identity, dict) else ""
    if not skill and event_skill:
        skill = event_skill
        event_identity["used_as_fallback"] = True
    if not case_id and event_case:
        case_id = event_case
        event_identity["used_as_fallback"] = True
    if event_identity.get("used_as_fallback"):
        print(
            "warning: benchmark record identity inferred from agent event text "
            f"(skill={skill or 'unknown'}, case_id={case_id or 'unknown'})",
            file=sys.stderr,
        )

    record = base if isinstance(base, dict) else {}
    metrics = apply_record_runtime_fields(
        record,
        usage=usage,
        mode=mode,
        elapsed_seconds=elapsed_seconds,
        agent_exit=agent_exit,
        skills_enabled=skills_enabled,
        skill_run_mode=skill_run_mode,
        agent=agent,
        agent_model=agent_model,
    )
    record["agent_record_generated_by_harness"] = True
    record["agent_record_source"] = record_source
    if source_path is not None:
        record["agent_record_source_path"] = str(source_path)
    record["agent_record_source_audit"] = source_audit
    record["event_identity_inference"] = event_identity

    if skill:
        record["skill"] = skill
        record["skill_name"] = skill
    if case_id:
        record["case_id"] = case_id

    discovery = record.get("skill_discovery")
    if not isinstance(discovery, dict):
        discovery = {}
    if skill and not discovery.get("selected_skill"):
        discovery["selected_skill"] = skill
    if case_id and not discovery.get("selected_case_id"):
        discovery["selected_case_id"] = case_id
    if discovery:
        discovery.setdefault(
            "source",
            "harness_explicit_record" if not event_identity.get("used_as_fallback") else "harness_event_log_fallback",
        )
        record["skill_discovery"] = discovery

    for category in ("mandatory_behavior", "prohibited_behavior", "optional_behavior"):
        if record.get(category) is None:
            record[category] = {}

    metrics["event_count"] = activity.get("event_count")
    metrics["command_count"] = activity.get("command_count")
    metrics["unique_command_count"] = activity.get("unique_command_count")
    if isinstance(workspace_delta, dict):
        record["workspace_delta"] = workspace_delta
        for key in (
            "changed_file_count",
            "deleted_file_count",
            "workspace_added_file_count",
            "workspace_modified_file_count",
            "workspace_deleted_baseline_file_count",
            "workspace_change_count",
            "runtime_artifact_count",
            "copied_file_count",
            "copied_bytes",
        ):
            if isinstance(workspace_delta.get(key), (int, float)) and not isinstance(workspace_delta.get(key), bool):
                metrics[f"workspace_delta_{key}"] = workspace_delta[key]

    if isinstance(input_delta, dict):
        record["source_input_delta"] = input_delta
        input_delta_aliases = {
            "workspace_added_file_count": "added_file_count",
            "workspace_modified_file_count": "modified_file_count",
            "workspace_deleted_baseline_file_count": "deleted_baseline_file_count",
            "workspace_change_count": "change_count",
        }
        for key in (
            "changed_file_count",
            "deleted_file_count",
            "workspace_added_file_count",
            "workspace_modified_file_count",
            "workspace_deleted_baseline_file_count",
            "workspace_change_count",
            "copied_file_count",
            "copied_bytes",
        ):
            if isinstance(input_delta.get(key), (int, float)) and not isinstance(input_delta.get(key), bool):
                metrics[f"source_input_delta_{key}"] = input_delta[key]
                if key in input_delta_aliases:
                    metrics[f"source_input_delta_{input_delta_aliases[key]}"] = input_delta[key]
        source_input_violation = bool(input_delta.get("changed_file_count") or input_delta.get("deleted_file_count"))
        metrics["source_input_immutable_violation"] = 1 if source_input_violation else 0
        record["source_input_immutable_policy"] = {
            "status": "fail" if source_input_violation else "pass",
            "reason": (
                "The immutable input snapshot changed during the agent run."
                if source_input_violation
                else "The immutable input snapshot was unchanged; conversion output is captured separately from the writable workspace."
            ),
            "scope": str(input_delta.get("workspace_root") or ""),
            "changed_files": input_delta.get("changed_files") or [],
            "deleted_files": input_delta.get("deleted_files") or [],
        }
        if source_input_violation:
            record["source_input_immutable_violation"] = {
                "status": "fail",
                "reason": "Agent or runtime changed files inside the immutable input snapshot.",
                "changed_files": input_delta.get("changed_files") or [],
                "deleted_files": input_delta.get("deleted_files") or [],
            }
    else:
        record["source_input_immutable_policy"] = {
            "status": "not_captured",
            "reason": input_delta_not_captured_reason,
            "scope": "",
            "changed_files": [],
            "deleted_files": [],
        }
        metrics["source_input_immutable_violation"] = 0

    quality_signals = record.get("quality_signals")
    if not isinstance(quality_signals, dict):
        quality_signals = {}
    quality_signals["job_guidance_primary_validation_metric"] = job_guidance_metric_signal
    if artifact_validation_metric:
        quality_signals["artifact_validation_metric"] = {
            "status": "pass",
            "source_type": "metrics_artifact",
            "reported_validation_metric": artifact_validation_metric,
            "evidence": (
                "A captured runtime metrics artifact contains a numeric "
                f"{artifact_validation_metric.get('name')} value."
            ),
        }
    record["quality_signals"] = quality_signals
    record["required_validation_metric_status"] = (
        "present" if artifact_validation_metric else required_validation_metric_status(job_guidance_metric_signal)
    )
    record["critical_quality_checks_failed"] = bool(
        record.get("critical_quality_checks_failed") or critical_quality_checks_failed(record)
    )
    if job_guidance_metric_signal.get("expected_primary_metric"):
        record["validation_metric_policy"] = {
            "source": job_guidance_metric_signal.get("source"),
            "sources": job_guidance_metric_signal.get("sources"),
            "expected_primary_metric": job_guidance_metric_signal.get("expected_primary_metric"),
            "scoring_note": "Measured as a benchmark quality signal from the final message and job documentation.",
        }
        validation_metric = job_guidance_metric_signal.get("reported_validation_metric")
        record["reported_validation_metric"] = validation_metric
        if artifact_validation_metric:
            record["validation_metric"] = artifact_validation_metric
        metrics["validation_metric_policy_available"] = 1
        metrics["validation_metric_value_available"] = (
            1 if artifact_validation_metric or job_guidance_metric_signal.get("metric_value_available") else 0
        )
        metrics["validation_metric_artifact_available"] = 1 if artifact_validation_metric else 0
        metrics["validation_metric_aligned_with_job_guidance"] = (
            1 if artifact_validation_metric or job_guidance_metric_signal.get("aligned_with_job_guidance") else 0
        )
        metrics["validation_metric_aligned_with_readme"] = (
            1 if artifact_validation_metric or job_guidance_metric_signal.get("aligned_with_readme") else 0
        )
        metrics["validation_metric_mismatch"] = 1 if job_guidance_metric_signal.get("mismatch") else 0
    else:
        validation_metric = job_guidance_metric_signal.get("reported_validation_metric")
        if isinstance(validation_metric, dict) and validation_metric.get("name"):
            record["reported_validation_metric"] = validation_metric
        if artifact_validation_metric:
            record["validation_metric"] = artifact_validation_metric
        metrics["validation_metric_policy_available"] = 0
        metrics["validation_metric_artifact_available"] = 1 if artifact_validation_metric else 0

    record["process_metrics"] = metrics
    record["agent_usage"] = usage

    notes = record.get("notes")
    note = "Mode-specific benchmark record was synthesized by the benchmark harness, not requested through prompt text."
    if isinstance(notes, list):
        if note not in notes:
            notes.append(note)
    elif isinstance(notes, str) and notes:
        record["notes"] = [notes, note] if notes != note else [notes]
    else:
        record["notes"] = [note]

    write_json(agent_record_path, record)


def as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_behavior_map(record: dict[str, Any], category: str) -> dict[str, dict[str, str]]:
    raw = record.get(category)
    if not isinstance(raw, dict):
        raw = {}
    normalized = {}
    for behavior_id, entry in raw.items():
        if not isinstance(entry, dict):
            entry = {}
        status = entry.get("status")
        if status not in ALLOWED_BEHAVIOR_STATUSES:
            status = "missing"
        normalized[str(behavior_id)] = {
            "status": status,
            "evidence": str(entry.get("evidence") or "No evidence supplied by agent."),
        }
    record[category] = normalized
    return normalized


def status_counts(behavior_map: dict[str, dict[str, str]]) -> dict[str, int]:
    counts = {status: 0 for status in sorted(ALLOWED_BEHAVIOR_STATUSES)}
    for entry in behavior_map.values():
        status = entry.get("status")
        if status in counts:
            counts[status] += 1
    return counts


def pass_rate(behavior_map: dict[str, dict[str, str]]) -> float | None:
    total = len(behavior_map)
    if total == 0:
        return None
    return round(sum(1 for entry in behavior_map.values() if entry.get("status") == "pass") / total, 3)


def merge_record(
    agent_record_path: Path,
    final_record_path: Path,
    usage_path: Path,
    mode: str,
    elapsed_seconds: int,
    agent_exit: int,
    skills_enabled: bool,
    skill_run_mode: str,
    agent: str,
    agent_model: str,
) -> None:
    record: dict[str, Any] = {}
    agent_record_present = agent_record_path.exists()
    agent_record_valid = False
    if agent_record_present:
        try:
            data = json.loads(agent_record_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                record = data
                agent_record_valid = True
        except Exception as exc:
            record = {"notes": f"agent record was not valid JSON: {exc}"}

    if not isinstance(record, dict):
        record = {"notes": "agent record was not a JSON object"}

    usage = load_json(usage_path, {}) or {}
    metrics = apply_record_runtime_fields(
        record,
        usage=usage,
        mode=mode,
        elapsed_seconds=elapsed_seconds,
        agent_exit=agent_exit,
        skills_enabled=skills_enabled,
        skill_run_mode=skill_run_mode,
        agent=agent,
        agent_model=agent_model,
        agent_record_present=agent_record_present,
        agent_record_valid=agent_record_valid,
    )
    if "user_correction_count" not in metrics and "correction_count" in metrics:
        metrics["user_correction_count"] = metrics.get("correction_count")
    if "first_pass_accepted" not in metrics:
        corrections = as_float(metrics.get("user_correction_count"))
        if corrections is not None and agent_exit == 0:
            metrics["first_pass_accepted"] = 1 if corrections == 0 else 0

    mandatory_behavior = normalize_behavior_map(record, "mandatory_behavior")
    prohibited_behavior = normalize_behavior_map(record, "prohibited_behavior")
    optional_behavior = normalize_behavior_map(record, "optional_behavior")
    mandatory_count = len(mandatory_behavior)
    prohibited_count = len(prohibited_behavior)
    required_count = mandatory_count + prohibited_count
    required_pass_count = sum(1 for entry in mandatory_behavior.values() if entry.get("status") == "pass") + sum(
        1 for entry in prohibited_behavior.values() if entry.get("status") == "pass"
    )
    instruction_compliance = {
        "mandatory_behavior": {
            "total": mandatory_count,
            "pass_rate": pass_rate(mandatory_behavior),
            "status_counts": status_counts(mandatory_behavior),
        },
        "prohibited_behavior": {
            "total": prohibited_count,
            "avoidance_rate": pass_rate(prohibited_behavior),
            "status_counts": status_counts(prohibited_behavior),
        },
        "optional_behavior": {
            "total": len(optional_behavior),
            "coverage_rate": pass_rate(optional_behavior),
            "status_counts": status_counts(optional_behavior),
        },
        "required_behavior": {
            "total": required_count,
            "pass_rate": round(required_pass_count / required_count, 3) if required_count else None,
            "pass_count": required_pass_count,
        },
    }
    reported_instruction_compliance = record.get("instruction_compliance")
    if (
        mandatory_behavior
        or prohibited_behavior
        or optional_behavior
        or not isinstance(reported_instruction_compliance, dict)
    ):
        record["instruction_compliance"] = instruction_compliance
    else:
        instruction_compliance = reported_instruction_compliance
    required_behavior = instruction_compliance.get("required_behavior") or {}
    mandatory_behavior_summary = instruction_compliance.get("mandatory_behavior") or {}
    prohibited_behavior_summary = instruction_compliance.get("prohibited_behavior") or {}
    metrics["instruction_required_pass_rate"] = required_behavior.get("pass_rate")
    metrics["instruction_mandatory_pass_rate"] = mandatory_behavior_summary.get("pass_rate")
    metrics["instruction_prohibited_avoidance_rate"] = prohibited_behavior_summary.get("avoidance_rate")
    required_pass_rate = required_behavior.get("pass_rate")
    if required_pass_rate is not None:
        metrics["instruction_required_passed"] = 1 if required_pass_rate >= 1.0 else 0
    record["agent_usage"] = usage
    write_json(final_record_path, record)


def write_run_summary(final_record_path: Path, summary_path: Path, *, print_summary: bool = True) -> None:
    record = load_json(final_record_path, {}) or {}
    metrics = record.get("process_metrics") or {}
    prompt_metadata = load_json(summary_path.parent / "prompt_metadata.json", {}) or {}
    if not isinstance(prompt_metadata, dict):
        prompt_metadata = {}
    runtime_metadata = load_json(summary_path.parent / "runtime_image.json", {}) or {}
    if not isinstance(runtime_metadata, dict):
        runtime_metadata = {}
    skill_discovery = record.get("skill_discovery") if isinstance(record.get("skill_discovery"), dict) else {}
    quality_signals = record.get("quality_signals") if isinstance(record.get("quality_signals"), dict) else {}
    validation_signal = quality_signals.get("job_guidance_primary_validation_metric")
    validation_metric = record.get("validation_metric") or record.get("reported_validation_metric")
    if validation_metric is None and isinstance(validation_signal, dict):
        validation_metric = validation_signal.get("reported_validation_metric")
    agent_elapsed = metrics.get("agent_elapsed_seconds")
    if agent_elapsed is None:
        agent_elapsed = record.get("agent_elapsed_seconds")
    if agent_elapsed is None:
        agent_elapsed = metrics.get("elapsed_seconds")
    summary = {
        "mode": record.get("mode"),
        "run_mode": record.get("run_mode"),
        "skill": record.get("skill"),
        "skill_name": record.get("skill_name"),
        "observed_skill_name": skill_discovery.get("selected_skill") or record.get("skill") or record.get("skill_name"),
        "skill_name_source": skill_discovery.get("source") or record.get("agent_record_source"),
        "case_id": record.get("case_id"),
        "agent_elapsed_seconds": agent_elapsed,
        "elapsed_seconds": agent_elapsed,
        "phase_seconds": metrics.get("phase_seconds"),
        "prompt_hash": prompt_metadata.get("prompt_sha256"),
        "prompt_source": prompt_metadata.get("template_path"),
        "token_count": metrics.get("token_count"),
        "command_count": metrics.get("command_count"),
        "cache_tokens": metrics.get("cache_tokens"),
        "cost": metrics.get("cost"),
        "runtime_image": runtime_metadata.get("runtime_image"),
        "wheel_variant": runtime_metadata.get("sdk_image_kind"),
        "conversion_quality": metrics.get("conversion_quality"),
        "correction_count": metrics.get("correction_count"),
        "command_failures": metrics.get("command_failures"),
        "agent_process_passed": record.get("agent_process_passed"),
        "agent_process_exit_code": record.get("agent_process_exit_code"),
        "agent_exit_code": metrics.get("agent_exit_code"),
        "agent_report_exit_codes": record.get("agent_report_exit_codes") or {},
        "agent_report_exit_code": record.get("agent_report_exit_code"),
        "agent_report_failed": record.get("agent_report_failed"),
        "final_container_exit_code": record.get("final_container_exit_code"),
        "report_inclusive_exit_code": record.get("report_inclusive_exit_code"),
        "harness_failure": record.get("harness_failure") or metrics.get("harness_failure"),
        "harness_error": record.get("harness_error") or {},
        "harness_errors": record.get("harness_errors") or [],
        "failure_root_cause": record.get("failure_root_cause") or record.get("failure_category"),
        "validation_metric": validation_metric,
        "validation_metric_status": record.get("required_validation_metric_status"),
        "required_validation_metric_status": record.get("required_validation_metric_status"),
        "structure_quality_signal": record.get("structure_quality_signal") or UNAVAILABLE_STRUCTURE_QUALITY_SIGNAL,
        "skills_enabled": record.get("skills_enabled"),
        "agent": record.get("agent"),
        "agent_model": record.get("agent_model"),
        "agent_record_present": record.get("agent_record_present"),
        "agent_record_valid": record.get("agent_record_valid"),
        "process_metrics": metrics,
        "instruction_compliance": record.get("instruction_compliance") or {},
        "mandatory_behavior": record.get("mandatory_behavior") or {},
        "prohibited_behavior": record.get("prohibited_behavior") or {},
        "optional_behavior": record.get("optional_behavior") or {},
        "skill_discovery": record.get("skill_discovery") or {},
        "agent_usage": record.get("agent_usage") or {},
        "workspace_delta": record.get("workspace_delta") or {},
        "source_input_delta": record.get("source_input_delta") or {},
        "source_input_immutable_policy": record.get("source_input_immutable_policy") or {},
    }
    summary["all_metrics"] = flatten_numbers(summary)
    write_json(summary_path, summary)
    if print_summary:
        elapsed = metrics.get("elapsed_seconds")
        elapsed_text = (
            f"{elapsed:.1f}s" if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) else "NA"
        )
        print(
            "Run summary written: "
            f"{summary_path}; mode={summary.get('mode') or 'unknown'}; "
            f"elapsed={elapsed_text}; final_exit={summary.get('final_container_exit_code')}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    synth = subparsers.add_parser("synthesize")
    synth.add_argument("agent_record", type=Path)
    synth.add_argument("records_dir", type=Path)
    synth.add_argument("events_path", type=Path)
    synth.add_argument("usage_path", type=Path)
    synth.add_argument("activity_path", type=Path)
    synth.add_argument("last_message_path", type=Path)
    synth.add_argument("input_dir", type=Path)
    synth.add_argument("mode")
    synth.add_argument("elapsed_seconds", type=int)
    synth.add_argument("agent_exit", type=int)
    synth.add_argument("skills_enabled")
    synth.add_argument("skill_run_mode")
    synth.add_argument("agent_model")
    synth.add_argument("run_start_time_ns", type=int)
    synth.add_argument("workspace_delta_manifest", type=Path)
    synth.add_argument("input_delta_manifest", nargs="?", type=Path)
    synth.add_argument("--agent", default=os.environ.get("BENCHMARK_AGENT", "unknown"))
    synth.add_argument("--prompt", type=Path, default=None)

    merge = subparsers.add_parser("merge")
    merge.add_argument("agent_record", type=Path)
    merge.add_argument("final_record", type=Path)
    merge.add_argument("usage_path", type=Path)
    merge.add_argument("mode")
    merge.add_argument("elapsed_seconds", type=int)
    merge.add_argument("agent_exit", type=int)
    merge.add_argument("skills_enabled")
    merge.add_argument("skill_run_mode")
    merge.add_argument("agent_model")
    merge.add_argument("--agent", default=os.environ.get("BENCHMARK_AGENT", "unknown"))

    summary = subparsers.add_parser("summary")
    summary.add_argument("final_record", type=Path)
    summary.add_argument("summary_path", type=Path)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("summary_path", type=Path)
    finalize.add_argument("record_path", type=Path)
    finalize.add_argument("timing_path", type=Path)
    finalize.add_argument("activity_path", type=Path)
    finalize.add_argument("epochs", nargs=14, type=int)

    args = parser.parse_args()
    if args.command == "synthesize":
        synthesize_agent_record(
            AgentRecordSynthesisInputs(
                agent_record_path=args.agent_record,
                records_dir=args.records_dir,
                events_path=args.events_path,
                usage_path=args.usage_path,
                activity_path=args.activity_path,
                last_message_path=args.last_message_path,
                input_dir=args.input_dir,
                mode=args.mode,
                elapsed_seconds=args.elapsed_seconds,
                agent_exit=args.agent_exit,
                skills_enabled=bool_from_text(args.skills_enabled),
                skill_run_mode=args.skill_run_mode,
                agent=args.agent,
                agent_model=args.agent_model,
                run_start_time_ns=args.run_start_time_ns,
                workspace_delta_manifest_path=args.workspace_delta_manifest,
                input_delta_manifest_path=args.input_delta_manifest,
                prompt_path=args.prompt,
            )
        )
    elif args.command == "merge":
        merge_record(
            args.agent_record,
            args.final_record,
            args.usage_path,
            args.mode,
            args.elapsed_seconds,
            args.agent_exit,
            bool_from_text(args.skills_enabled),
            args.skill_run_mode,
            args.agent,
            args.agent_model,
        )
    elif args.command == "summary":
        write_run_summary(args.final_record, args.summary_path)
    elif args.command == "finalize":
        finalize_timing(
            args.summary_path,
            args.record_path,
            args.timing_path,
            args.activity_path,
            LifecycleEpochs.from_sequence(args.epochs),
        )


if __name__ == "__main__":
    main()
