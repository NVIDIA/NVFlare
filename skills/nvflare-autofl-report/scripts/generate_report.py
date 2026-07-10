#!/usr/bin/env python3
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

"""Generate a reproducible final report for a stopped Auto-FL campaign."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import shlex
import sys
import tempfile
import textwrap
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - NVFlare installs PyYAML
    yaml = None


SUMMARY_SCHEMA_VERSION = "nvflare.autofl.report.v1"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
ATTEMPT_STATUSES = {"candidate", "keep", "discard", "crash"}
FINALIZED_SCORE_STATUSES = {"baseline", "keep", "discard"}
RETAINED_STATUSES = {"baseline", "keep"}
LITERATURE_STATUSES = {"literature", "checkpoint", "event"}
PENDING_MANIFEST_STATUSES = {"prepared", "ready_for_external_execution"}
TRAINING_BUDGET_ARGS = {
    "aggregation_epochs",
    "alpha",
    "batch_size",
    "eval_batch_size",
    "local_train_steps",
    "max_model_params",
    "min_clients",
    "model_arch",
    "n_clients",
    "num_clients",
    "num_rounds",
    "seed",
}
FIXED_BUDGET_TO_CLI = {
    "min_clients": "min_clients",
    "num_clients": "n_clients",
    "num_rounds": "num_rounds",
}
SOURCE_RE = re.compile(r"\[src:\s*([^\]]+)\]", re.IGNORECASE)
ARXIV_RE = re.compile(r"\barxiv\s*:\s*(\d{4}\.\d{4,5})", re.IGNORECASE)


@dataclass(frozen=True)
class RunRecord:
    index: int
    status: str
    name: str
    score: Optional[float]
    runtime_seconds: float
    changed_files: str
    diff_summary: str
    run_command: str
    artifacts: str
    failure_reason: str
    candidate_manifest: str
    base_candidate: str
    patch_sha256: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", nargs="?", default=".", help="job directory containing Auto-FL artifacts")
    parser.add_argument("--results", default="results.tsv")
    parser.add_argument("--state", default=".nvflare/autofl/campaign_state.json")
    parser.add_argument("--autofl-yaml", default="autofl.yaml")
    parser.add_argument("--progress", default="progress.png")
    parser.add_argument("--output", default="autofl_final_report.md")
    parser.add_argument("--summary-json", default="autofl_report_summary.json")
    parser.add_argument(
        "--plotter",
        help="override path to nvflare-autofl plot_progress.py; relative paths resolve from campaign_dir",
    )
    parser.add_argument("--mode", choices=["max", "min"], help="override objective direction")
    parser.add_argument("--metric", help="override optimization metric label")
    parser.add_argument("--confirm-interrupted", action="store_true", help="confirm an abruptly interrupted campaign")
    parser.add_argument("--agent-model")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--agent-cost")
    parser.add_argument("--agent-context", help="optional JSON or text file with agent/tooling context")
    parser.add_argument("--max-milestones", type=int, default=12)
    parser.add_argument("--max-non-improvements", type=int, default=10)
    return parser.parse_args(argv)


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_results(path: Path) -> List[RunRecord]:
    if not path.is_file():
        raise ValueError(f"Auto-FL ledger not found: {path}")
    records = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or not {"status", "score"}.issubset(reader.fieldnames):
            raise ValueError(f"Auto-FL ledger is missing required status/score columns: {path}")
        for index, row in enumerate(reader):
            records.append(
                RunRecord(
                    index=index,
                    status=(row.get("status") or "").strip().lower(),
                    name=(row.get("name") or row.get("candidate") or row.get("commit") or f"row_{index}").strip(),
                    score=finite_float(row.get("score")),
                    runtime_seconds=max(0.0, finite_float(row.get("runtime_seconds")) or 0.0),
                    changed_files=(row.get("changed_files") or "").strip(),
                    diff_summary=(row.get("diff_summary") or row.get("description") or "").strip(),
                    run_command=(row.get("run_command") or "").strip(),
                    artifacts=(row.get("artifacts") or "").strip(),
                    failure_reason=(row.get("failure_reason") or "").strip(),
                    candidate_manifest=(row.get("candidate_manifest") or "").strip(),
                    base_candidate=(row.get("base_candidate") or "").strip(),
                    patch_sha256=(row.get("patch_sha256") or "").strip(),
                )
            )
    if not records:
        raise ValueError(f"Auto-FL ledger has no rows: {path}")
    return records


def load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    if yaml is None:
        raise ValueError("PyYAML is required to read autofl.yaml")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read Auto-FL config from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return value


def better(score: Optional[float], incumbent: Optional[float], mode: str) -> bool:
    if score is None:
        return False
    if incumbent is None:
        return True
    return score > incumbent if mode == "max" else score < incumbent


def normalize_contract_sections(config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    normalized = dict(config)
    warnings = []
    for section in ("objective", "environment", "budget"):
        if section not in normalized:
            continue
        value = normalized[section]
        if not isinstance(value, dict):
            kind = "null" if value is None else type(value).__name__
            warnings.append(
                f"autofl.yaml section '{section}' is {kind}, not a mapping; it was treated as an empty section."
            )
            normalized[section] = {}
    return normalized, warnings


def metric_contract(
    config: Dict[str, Any], state: Dict[str, Any], args: argparse.Namespace
) -> Tuple[str, str, str, str]:
    objective = config.get("objective") if isinstance(config.get("objective"), dict) else {}
    metric = (
        args.metric or objective.get("optimization_metric") or objective.get("metric") or state.get("metric") or "score"
    )
    requested = objective.get("requested_metric") or objective.get("metric") or metric
    measurement_source = objective.get("metric_source") or "NVFlare metric artifacts"
    contract_source = objective.get("source") or "not declared"
    return str(metric), str(requested), str(measurement_source), str(contract_source)


def infer_mode(config: Dict[str, Any], state: Dict[str, Any], requested: Optional[str]) -> str:
    if requested:
        return requested
    objective = config.get("objective") if isinstance(config.get("objective"), dict) else {}
    value = objective.get("mode") or objective.get("direction") or state.get("mode") or "max"
    return str(value).lower() if str(value).lower() in {"max", "min"} else "max"


def verify_stopped(state_path: Path, confirm_interrupted: bool) -> Tuple[Dict[str, Any], str, List[str]]:
    warnings = []
    if not state_path.is_file():
        if not confirm_interrupted:
            raise ValueError(
                f"campaign state not found at {state_path}; use --confirm-interrupted only after confirming execution stopped"
            )
        warnings.append("Campaign state was unavailable; the user explicitly confirmed interruption.")
        return {}, "user_confirmed_interruption", warnings

    state = load_json(state_path)
    if state.get("final_response_allowed") is True:
        return state, str(state.get("reason") or "campaign_stopped"), warnings
    if not confirm_interrupted:
        raise ValueError(
            "campaign state still has final_response_allowed=false; stop the campaign or pass --confirm-interrupted "
            "after independently confirming its processes are no longer running"
        )
    warnings.append("Campaign state remained active; the user explicitly confirmed that execution was interrupted.")
    return state, "user_confirmed_interruption", warnings


def is_baseline(record: RunRecord) -> bool:
    name = record.name.strip().lower()
    command = record.run_command.lower()
    return (
        record.status == "baseline"
        or name == "baseline"
        or name.startswith("baseline_")
        or "--name baseline" in command
    )


def is_finalized_scored_record(record: RunRecord) -> bool:
    if record.score is None or record.status in {"candidate", "crash"}:
        return False
    return is_baseline(record) or record.status in FINALIZED_SCORE_STATUSES


def candidate_manifest_evidence(
    campaign_root: Path, records: Sequence[RunRecord]
) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    candidates_root = campaign_root / ".nvflare" / "autofl" / "candidates"
    paths = set(candidates_root.glob("*/candidate_manifest.json")) if candidates_root.is_dir() else set()
    paths.update(
        resolve_path(campaign_root, record.candidate_manifest) for record in records if record.candidate_manifest
    )
    pending = []
    unreadable = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        try:
            manifest = load_json(path)
        except ValueError as exc:
            unreadable.append((path, str(exc)))
            continue
        if str(manifest.get("status") or "").strip().lower() in PENDING_MANIFEST_STATUSES:
            pending.append(path)
    return pending, unreadable


def verify_no_pending_candidates(campaign_root: Path, state: Dict[str, Any], records: Sequence[RunRecord]) -> None:
    evidence = []
    pending_count = finite_float(state.get("pending_candidates"))
    if pending_count is not None and pending_count > 0:
        evidence.append(f"campaign state reports pending_candidates={state.get('pending_candidates')}")
    if state.get("pending_candidate_manifest"):
        evidence.append(f"campaign state names {state['pending_candidate_manifest']}")

    ledger_rows = [record for record in records if record.status == "candidate"]
    if ledger_rows:
        rows = ", ".join(f"row {record.index + 1} ({record.name})" for record in ledger_rows[:5])
        if len(ledger_rows) > 5:
            rows += f", and {len(ledger_rows) - 5} more"
        evidence.append(f"results.tsv contains pending candidate rows: {rows}")

    manifests, unreadable_manifests = candidate_manifest_evidence(campaign_root, records)
    if manifests:
        paths = ", ".join(str(path.resolve()) for path in manifests[:3])
        if len(manifests) > 3:
            paths += f", and {len(manifests) - 3} more"
        evidence.append(f"candidate manifests remain prepared for execution: {paths}")
    if unreadable_manifests:
        details = "; ".join(f"{path.resolve()} ({error})" for path, error in unreadable_manifests[:3])
        if len(unreadable_manifests) > 3:
            details += f"; and {len(unreadable_manifests) - 3} more"
        evidence.append(f"candidate manifests could not be read and may contain unfinished work: {details}")

    if evidence:
        raise ValueError(
            "cannot finalize while candidate evidence is unfinished; finalize or abandon pending candidates first. "
            "--confirm-interrupted bypasses only stale campaign stop state, never unfinished candidate evidence. "
            + " ".join(evidence)
        )


def scored_records(records: Iterable[RunRecord]) -> List[RunRecord]:
    return [record for record in records if is_finalized_scored_record(record)]


def select_baseline(records: Sequence[RunRecord]) -> Optional[RunRecord]:
    return next(
        (record for record in records if record.status == "baseline" and is_finalized_scored_record(record)), None
    )


def select_best(records: Sequence[RunRecord], mode: str, retained_only: bool = False) -> Optional[RunRecord]:
    candidates = [
        record
        for record in records
        if is_finalized_scored_record(record)
        and (record.status in RETAINED_STATUSES if retained_only else record.status in FINALIZED_SCORE_STATUSES)
    ]
    if not candidates:
        return None
    return (max if mode == "max" else min)(candidates, key=lambda record: record.score)


def running_best_milestones(records: Sequence[RunRecord], mode: str, limit: int) -> List[Dict[str, Any]]:
    milestones = []
    incumbent = None
    for record in records:
        if not is_finalized_scored_record(record) or not better(record.score, incumbent, mode):
            continue
        previous = incumbent
        incumbent = record.score
        milestones.append(
            {
                "row": record.index + 1,
                "name": record.name,
                "status": record.status,
                "score": record.score,
                "delta_from_previous_best": None if previous is None else record.score - previous,
                "hypothesis": record.diff_summary,
                "changed_files": split_files(record.changed_files),
            }
        )
    if limit > 0 and len(milestones) > limit:
        if limit == 1:
            return [milestones[-1]]
        indices = [round(index * (len(milestones) - 1) / (limit - 1)) for index in range(limit)]
        return [milestones[index] for index in dict.fromkeys(indices)]
    return milestones


def split_files(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip() and item.strip().lower() != "none"]


def candidate_lineage(best: Optional[RunRecord], records: Sequence[RunRecord]) -> Dict[str, Any]:
    if best is None:
        return {"candidates": [], "changed_files": [], "complete": False}
    by_name = {record.name: record for record in records if record.name}
    chain = []
    changed_files = []
    seen = set()
    current = best
    complete = True
    while current:
        if current.name in seen:
            complete = False
            break
        seen.add(current.name)
        chain.append(current.name)
        changed_files.extend(split_files(current.changed_files))
        if not current.base_candidate:
            complete = current.status == "baseline" or current.name == "baseline"
            break
        base_candidate = current.base_candidate
        current = by_name.get(base_candidate)
        if current is None:
            chain.append(base_candidate)
            complete = False
            break
    return {
        "candidates": list(reversed(chain)),
        "changed_files": sorted(set(changed_files)),
        "complete": complete,
    }


def parse_sources(text: str) -> List[str]:
    sources = []
    for match in SOURCE_RE.findall(text or ""):
        for item in re.split(r"\s*;\s*", match):
            if item and item not in sources:
                sources.append(item)
    for identifier in ARXIV_RE.findall(text or ""):
        source = f"arXiv:{identifier}"
        if not any(identifier in existing for existing in sources):
            sources.append(source)
    return sources


def manifest_summary(record: Optional[RunRecord], campaign_root: Path) -> Dict[str, Any]:
    if record is None or not record.candidate_manifest:
        return {}
    path = resolve_path(campaign_root, record.candidate_manifest)
    if not path.is_file():
        return {"path": str(path), "available": False}
    try:
        manifest = load_json(path)
    except ValueError as exc:
        return {"path": str(path), "available": False, "error": str(exc)}
    return {
        "path": str(path.resolve()),
        "available": True,
        "schema_version": manifest.get("schema_version"),
        "candidate_id": manifest.get("candidate_id"),
        "base_candidate": manifest.get("base_candidate"),
        "hypothesis": manifest.get("hypothesis"),
        "run_args": manifest.get("run_args") or [],
        "changed_files": manifest.get("changed_files") or [],
        "created_files": manifest.get("created_files") or [],
        "source_sha256": manifest.get("candidate_source_sha256") or manifest.get("base_source_sha256"),
        "budget_sha256": manifest.get("fixed_budget_sha256") or manifest.get("budget_sha256"),
        "patch_sha256": manifest.get("patch_sha256"),
        "status": manifest.get("status"),
        "artifacts": manifest.get("artifacts") or {},
        "result": manifest.get("result") or {},
    }


def literature_outcomes(records: Sequence[RunRecord], mode: str) -> List[Dict[str, Any]]:
    event_indices = [index for index, record in enumerate(records) if record.status in LITERATURE_STATUSES]
    outcomes = []
    for event_number, start in enumerate(event_indices):
        event = records[start]
        end = event_indices[event_number + 1] if event_number + 1 < len(event_indices) else len(records)
        before = select_best(records[:start], mode)
        attempts = [
            record
            for record in records[start + 1 : end]
            if record.status in ATTEMPT_STATUSES and not is_baseline(record)
        ]
        scored = [record for record in attempts if is_finalized_scored_record(record)]
        segment_best = select_best(scored, mode)
        if segment_best is None:
            outcome = "failed" if attempts else "not_evaluated"
            delta = None
        elif before is None or better(segment_best.score, before.score, mode):
            outcome = "helped"
            delta = None if before is None else segment_best.score - before.score
        elif segment_best.score == before.score:
            outcome = "matched"
            delta = 0.0
        else:
            outcome = "not_confirmed"
            delta = segment_best.score - before.score
        outcomes.append(
            {
                "event": event.name,
                "row": event.index + 1,
                "hypothesis": event.diff_summary,
                "sources": parse_sources(event.diff_summary),
                "outcome": outcome,
                "incumbent_score": None if before is None else before.score,
                "best_candidate": None if segment_best is None else segment_best.name,
                "best_score": None if segment_best is None else segment_best.score,
                "delta_from_incumbent": delta,
                "candidate_attempts": [record.name for record in attempts],
                "candidate_results": [
                    {"name": record.name, "status": record.status, "score": record.score} for record in attempts
                ],
                "failures": [record.name for record in attempts if record.status == "crash"],
            }
        )
    return outcomes


def command_options(command: str) -> Dict[str, Any]:
    if not command:
        return {}
    try:
        tokens = shlex.split(command)
    except ValueError:
        return {}
    options: Dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        key_value = token[2:].split("=", 1)
        key = key_value[0].replace("-", "_")
        if len(key_value) == 2:
            options[key] = key_value[1]
        elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            options[key] = tokens[index + 1]
            index += 1
        else:
            options[key] = True
        index += 1
    return options


def command_changes(baseline: Optional[RunRecord], best: Optional[RunRecord]) -> Dict[str, Dict[str, Any]]:
    if baseline is None or best is None:
        return {}
    base = command_options(baseline.run_command)
    candidate = command_options(best.run_command)
    changes = {}
    for key in sorted(set(base) | set(candidate)):
        if key == "name" or base.get(key) == candidate.get(key):
            continue
        changes[key] = {"baseline": base.get(key), "best": candidate.get(key)}
    return changes


def values_equal(left: Any, right: Any) -> bool:
    if str(left) == str(right):
        return True
    left_number = finite_float(left)
    right_number = finite_float(right)
    return left_number is not None and right_number is not None and math.isclose(left_number, right_number)


def comparability_warnings(
    config: Dict[str, Any],
    records: Sequence[RunRecord],
    baseline: Optional[RunRecord],
    best: Optional[RunRecord],
    metric: str,
    metric_source: str,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    warnings = []
    changes = command_changes(baseline, best)
    changed_budget = sorted(set(changes).intersection(TRAINING_BUDGET_ARGS))
    if changed_budget:
        warnings.append(
            "The best run changed executed training/comparison arguments relative to baseline: "
            + ", ".join(changed_budget)
            + ". Treat the gain as non-equal-compute unless this was explicitly approved."
        )

    budget = config.get("budget") if isinstance(config.get("budget"), dict) else {}
    fixed = budget.get("fixed_training_budget", {})
    baseline_options = command_options(baseline.run_command) if baseline else {}
    fixed_mismatches = []
    if isinstance(fixed, dict):
        for config_key, expected in fixed.items():
            cli_key = FIXED_BUDGET_TO_CLI.get(config_key, config_key)
            actual = baseline_options.get(cli_key)
            if actual is not None and not values_equal(actual, expected):
                fixed_mismatches.append(f"{config_key}: autofl.yaml={expected}, executed={actual}")
    if fixed_mismatches:
        warnings.append(
            "The imported fixed budget differs from the executed baseline command ("
            + "; ".join(fixed_mismatches)
            + ")."
        )

    candidate_count = len(
        [record for record in records if record.status in ATTEMPT_STATUSES and not is_baseline(record)]
    )
    metric_text = f"{metric} {metric_source}".lower()
    if candidate_count > 1 and "test" in metric_text:
        warnings.append(
            "Multiple candidates were selected against a test-like metric. Re-evaluate the chosen candidate once on an "
            "untouched holdout before making generalization claims."
        )
    if not config:
        warnings.append(
            "autofl.yaml was unavailable, so the declared objective and fixed-budget contract could not be verified."
        )
    return warnings, changes


def default_plotter_path() -> Path:
    return Path(__file__).resolve().parents[2] / "nvflare-autofl" / "scripts" / "plot_progress.py"


def refresh_plot(results: Path, output: Path, mode: str, metric: str, plotter_path: Path) -> Optional[str]:
    if not plotter_path.is_file():
        return f"Auto-FL progress plotter not found at {plotter_path}; existing plot was preserved."
    spec = importlib.util.spec_from_file_location("nvflare_autofl_report_plotter", plotter_path)
    if spec is None or spec.loader is None:
        return f"Could not load Auto-FL progress plotter from {plotter_path}; existing plot was preserved."
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        module.plot_progress(module.load_results(results), output, mode, metric)
    except Exception as exc:  # plotting should not destroy an otherwise useful stopped-campaign report
        return f"Could not refresh progress plot ({type(exc).__name__}: {exc}); existing plot was preserved."
    return None


def is_png(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE
    except OSError:
        return False


def compact_lineage(lineage: Sequence[str], edge_items: int = 4) -> str:
    if len(lineage) <= edge_items * 2 + 1:
        return " -> ".join(lineage)
    hidden = len(lineage) - edge_items * 2
    return " -> ".join([*lineage[:edge_items], f"... ({hidden} intermediate)", *lineage[-edge_items:]])


def read_agent_context(path: Optional[Path], args: argparse.Namespace) -> Dict[str, Any]:
    context = {}
    if path:
        if not path.is_file():
            raise ValueError(f"agent context file not found: {path}")
        text = path.read_text(encoding="utf-8").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            context["notes"] = text
        else:
            context.update(parsed if isinstance(parsed, dict) else {"notes": parsed})
    if args.agent_model:
        context["model"] = args.agent_model
    if args.reasoning_effort:
        context["reasoning_effort"] = args.reasoning_effort
    if args.agent_cost:
        context["cost"] = args.agent_cost
    return context


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as f:
            f.write(text)
            temp_path = Path(f.name)
        os.replace(temp_path, path)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def md_cell(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def format_score(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def format_delta(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:+.6f}"


def format_runtime(seconds: float) -> str:
    if seconds <= 0:
        return ""
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def format_command(command: str) -> str:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return command or "not recorded"
    if not tokens:
        return "not recorded"
    chunks = []
    current = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--") and current:
            chunks.append(current)
            current = []
        current.append(token)
        if token.startswith("--") and index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            index += 1
            current.append(tokens[index])
        index += 1
    if current:
        chunks.append(current)
    return " \\\n  ".join(shlex.join(chunk) for chunk in chunks)


def wrap_markdown_bullet(prefix: str, text: str, width: int = 120) -> List[str]:
    return textwrap.wrap(
        f"{prefix}{' '.join(text.split())}",
        width=width,
        subsequent_indent="  ",
        break_long_words=False,
        break_on_hyphens=False,
    )


def report_markdown(summary: Dict[str, Any], records: Sequence[RunRecord], max_non_improvements: int) -> str:
    baseline = summary["baseline"]
    best = summary["best"]
    counts = summary["status_counts"]
    runtime_label = format_runtime(summary["runtime_seconds"]) or "n/a"
    lines = [
        "# Auto-FL Final Report",
        "",
        "## Executive Summary",
        "",
        f"The stopped campaign optimized `{summary['objective']['optimization_metric']}` in "
        f"`{summary['environment']}` mode. It evaluated `{summary['candidate_attempts']}` candidate attempts "
        f"across `{len(records)}` ledger rows in `{runtime_label}`.",
        "",
    ]
    if best:
        baseline_score = baseline["score"] if baseline else None
        delta = None if baseline_score is None else best["score"] - baseline_score
        lines.append(
            f"Best retained candidate: `{best['name']}` at "
            f"`{summary['objective']['optimization_metric']}={format_score(best['score'])}` "
            f"(baseline `{format_score(baseline_score)}`, delta `{format_delta(delta)}`)."
        )
    else:
        observed = summary["best_observed"]
        if observed:
            lines.append(
                "No scored result was retained. Best unretained observed result: "
                f"`{observed['name']}` at "
                f"`{summary['objective']['optimization_metric']}={format_score(observed['score'])}` "
                f"with status `{observed['status']}`."
            )
        else:
            lines.append("No finalized scored result was available.")
    progress_lines = (
        [f"![Auto-FL progress]({summary['artifacts']['progress_plot']})"]
        if summary["artifacts"]["progress_plot_available"]
        else ["Progress plot unavailable; see the validation and comparability warnings below."]
    )
    lines.extend(
        [
            "",
            f"Termination: `{summary['termination']['reason']}`. "
            f"Final campaign state allowed: `{summary['termination']['state_allowed_final_response']}`.",
            "",
            "## Campaign Contract",
            "",
            f"- Requested metric: `{summary['objective']['requested_metric']}`",
            f"- Optimization metric: `{summary['objective']['optimization_metric']}`",
            f"- Metric source: `{summary['objective']['metric_source']}`",
            f"- Metric contract source: `{summary['objective']['metric_contract_source']}`",
            f"- Direction: `{summary['objective']['mode']}`",
            f"- Candidate cap: `{summary['candidate_cap']}`",
            f"- Declared fixed budget: `{json.dumps(summary['declared_fixed_budget'], sort_keys=True)}`",
            "",
            "## Progress",
            "",
            *progress_lines,
            "",
            "## Optimization Trajectory",
            "",
            "| Row | Candidate | Status | Score | Delta from previous best | Hypothesis |",
            "| ---: | --- | --- | ---: | ---: | --- |",
        ]
    )
    for item in summary["milestones"]:
        lines.append(
            f"| {item['row']} | `{md_cell(item['name'], 50)}` | {item['status']} | {format_score(item['score'])} | "
            f"{format_delta(item['delta_from_previous_best'])} | {md_cell(item['hypothesis'])} |"
        )
    if not summary["milestones"]:
        lines.append("|  |  |  |  |  | No scored milestones. |")

    lines.extend(["", "## Best Candidate Provenance", ""])
    if best:
        lines.extend(
            [
                f"- Candidate: `{best['name']}`",
                f"- Status: `{best['status']}`",
                f"- Base lineage: `{compact_lineage(summary['best_lineage']['candidates']) or 'unavailable'}`",
                f"- Cumulative changed files: `{', '.join(summary['best_lineage']['changed_files']) or 'none recorded'}`",
                f"- Candidate manifest: `{best['candidate_manifest'] or 'not recorded'}`",
                f"- Manifest available: `{summary['best_manifest'].get('available', False)}`",
                f"- Manifest budget SHA-256: `{summary['best_manifest'].get('budget_sha256') or 'not recorded'}`",
                f"- Patch SHA-256: `{best['patch_sha256'] or 'not recorded'}`",
                f"- Artifacts: `{best['artifacts'] or 'not recorded'}`",
                "",
                "Executed baseline command:",
                "",
                "```text",
                format_command(baseline["run_command"]) if baseline else "not recorded",
                "```",
                "",
                "Executed best-candidate command:",
                "",
                "```text",
                format_command(best["run_command"]),
                "```",
            ]
        )
    else:
        lines.append("No retained scored candidate provenance was available.")

    lines.extend(["", "## Literature Review Outcomes", ""])
    if summary["literature_reviews"]:
        lines.extend(
            [
                "| Checkpoint | Sources | Outcome | Candidate evidence | Delta vs incumbent |",
                "| --- | --- | --- | --- | ---: |",
            ]
        )
        for item in summary["literature_reviews"]:
            evidence_items = []
            for candidate in item["candidate_results"]:
                if candidate["status"] == "crash":
                    result = "crash"
                else:
                    result = format_score(candidate["score"])
                evidence_items.append(f"{candidate['name']}={result}")
            evidence = "; ".join(evidence_items) or "no candidate recorded"
            lines.append(
                f"| `{md_cell(item['event'], 45)}` | {md_cell('; '.join(item['sources']) or 'not recorded', 100)} | "
                f"{item['outcome']} | `{md_cell(evidence, 130)}` | {format_delta(item['delta_from_incumbent'])} |"
            )
        lines.extend(["", "Checkpoint hypotheses and decisions:", ""])
        for item in summary["literature_reviews"]:
            candidates = ", ".join(item["candidate_attempts"]) or "none recorded"
            lines.extend(
                wrap_markdown_bullet(
                    f"- **{item['event']} ({item['outcome']}):** ",
                    f"{item['hypothesis']} Measured candidates: `{md_cell(candidates, 300)}`.",
                )
            )
        lines.extend(
            [
                "",
                "Source labels above are campaign-recorded identifiers and were not independently verified by the report helper.",
            ]
        )
    else:
        lines.append("No literature checkpoints were recorded in the ledger.")

    lines.extend(
        [
            "",
            "## Runtime And Reliability",
            "",
            f"- Total recorded runtime: `{runtime_label}`",
            f"- Status counts: `{json.dumps(counts, sort_keys=True)}`",
            f"- Scored comparable runs: `{summary['scored_runs']}`",
            f"- Crashes: `{counts.get('crash', 0)}`",
            "",
            "## Null, Worse, Or Unstable Ideas",
            "",
        ]
    )
    non_improvements = [record for record in records if record.status in {"discard", "crash"}]
    for record in non_improvements[: max(0, max_non_improvements)]:
        evidence = record.failure_reason or f"score={format_score(record.score)}"
        lines.append(f"- `{record.name}`: {md_cell(record.diff_summary, 240)} Evidence: `{md_cell(evidence, 120)}`.")
    if not non_improvements:
        lines.append("No discarded or crashed candidates were recorded.")

    lines.extend(["", "## Validation And Comparability Notes", ""])
    if summary["warnings"]:
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    else:
        lines.append("- No deterministic comparability warning was detected from the available artifacts.")
    if summary["best_command_changes"]:
        lines.extend(["", "Executed argument changes from baseline to best:", ""])
        for key, value in summary["best_command_changes"].items():
            lines.append(f"- `{key}`: `{value['baseline']}` -> `{value['best']}`")

    lines.extend(["", "## Agent And Tooling Context", ""])
    if summary["agent_context"]:
        lines.append("```json")
        lines.append(json.dumps(summary["agent_context"], indent=2, sort_keys=True))
        lines.append("```")
    else:
        lines.append("Agent model, reasoning effort, and cost were not supplied to the report generator.")

    lines.extend(
        [
            "",
            "## Reproduction Recommendations",
            "",
            "1. Re-run the baseline and selected candidate from the exact commands above in the same NVFLARE environment.",
            "2. Confirm the candidate on additional seeds or sites before treating a single-run improvement as robust.",
            "3. When a test-like metric guided selection, perform one final evaluation on an untouched holdout.",
            "4. Preserve the candidate manifest, patch hash, ledger, campaign config, and downloaded NVFLARE artifacts.",
            "",
            "## Report Artifacts",
            "",
            f"- Auto-FL config: `{summary['artifacts']['autofl_yaml']}`",
            f"- Results ledger: `{summary['artifacts']['results']}`",
            f"- Campaign state: `{summary['artifacts']['campaign_state']}`",
            f"- Progress plot: `{summary['artifacts']['progress_plot']}`",
            f"- Progress plot available: `{summary['artifacts']['progress_plot_available']}`",
            f"- Machine-readable summary: `{summary['artifacts']['summary_json']}`",
            f"- Markdown report: `{summary['artifacts']['report']}`",
            "",
            "## Technical Appendix",
            "",
            f"Generated at `{summary['generated_at']}` with schema `{summary['schema_version']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def record_payload(record: Optional[RunRecord]) -> Optional[Dict[str, Any]]:
    return asdict(record) if record else None


def generate(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.campaign_dir).expanduser().resolve()
    results_path = resolve_path(root, args.results)
    state_path = resolve_path(root, args.state)
    config_path = resolve_path(root, args.autofl_yaml)
    progress_path = resolve_path(root, args.progress)
    report_path = resolve_path(root, args.output)
    summary_path = resolve_path(root, args.summary_json)
    agent_context_path = resolve_path(root, args.agent_context) if args.agent_context else None

    records = load_results(results_path)
    state, termination_reason, warnings = verify_stopped(state_path, args.confirm_interrupted)
    verify_no_pending_candidates(root, state, records)
    config = load_config(config_path)
    config, config_warnings = normalize_contract_sections(config)
    warnings.extend(config_warnings)
    mode = infer_mode(config, state, args.mode)
    metric, requested_metric, metric_source, metric_contract_source = metric_contract(config, state, args)
    baseline = select_baseline(records)
    best = select_best(records, mode, retained_only=True)
    observed_best = select_best(records, mode)
    if best is None and observed_best is not None:
        warnings.append(
            f"Best observed row {observed_best.name} was not retained; no scored baseline or kept candidate is available."
        )
    elif best and observed_best and best.name != observed_best.name:
        warnings.append(
            f"Best observed row {observed_best.name} was not retained; the report identifies retained candidate {best.name}."
        )

    plot_warning = refresh_plot(
        results_path,
        progress_path,
        mode,
        metric,
        resolve_path(root, args.plotter).resolve() if args.plotter else default_plotter_path(),
    )
    if plot_warning:
        warnings.append(plot_warning)
    progress_plot_available = is_png(progress_path)
    if not progress_plot_available:
        warnings.append(
            f"Progress plot at {progress_path} is missing or is not a valid PNG; the report was generated without "
            "embedding it."
        )

    comparison_warnings, changes = comparability_warnings(config, records, baseline, best, metric, metric_source)
    warnings.extend(comparison_warnings)
    status_counts = dict(sorted(Counter(record.status or "unknown" for record in records).items()))
    scored = scored_records(records)
    candidate_attempts = len(
        [record for record in records if record.status in ATTEMPT_STATUSES and not is_baseline(record)]
    )
    environment = config.get("environment") if isinstance(config.get("environment"), dict) else {}
    budget = config.get("budget") if isinstance(config.get("budget"), dict) else {}
    fixed_budget = budget.get("fixed_training_budget", {})
    cap = state.get("candidate_cap")
    cap_label = "uncapped" if cap in {None, "", 0} else cap
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "termination": {
            "reason": termination_reason,
            "state_allowed_final_response": state.get("final_response_allowed") is True,
            "user_confirmed_interruption": bool(args.confirm_interrupted),
        },
        "objective": {
            "requested_metric": requested_metric,
            "optimization_metric": metric,
            "metric_source": metric_source,
            "metric_contract_source": metric_contract_source,
            "mode": mode,
        },
        "environment": str(environment.get("requested") or state.get("environment") or "not declared"),
        "candidate_cap": cap_label,
        "candidate_cap_source": state.get("candidate_cap_source") or "not recorded",
        "declared_fixed_budget": fixed_budget if isinstance(fixed_budget, dict) else {},
        "candidate_attempts": candidate_attempts,
        "scored_runs": len(scored),
        "runtime_seconds": sum(record.runtime_seconds for record in records),
        "status_counts": status_counts,
        "baseline": record_payload(baseline),
        "best": record_payload(best),
        "best_observed": record_payload(observed_best),
        "best_lineage": candidate_lineage(best, records),
        "best_manifest": manifest_summary(best, root),
        "best_command_changes": changes,
        "milestones": running_best_milestones(records, mode, args.max_milestones),
        "literature_reviews": literature_outcomes(records, mode),
        "warnings": warnings,
        "agent_context": read_agent_context(agent_context_path, args),
        "artifacts": {
            "autofl_yaml": str(config_path.resolve()),
            "results": str(results_path.resolve()),
            "campaign_state": str(state_path.resolve()),
            "progress_plot": str(progress_path.resolve()),
            "progress_plot_available": progress_plot_available,
            "summary_json": str(summary_path.resolve()),
            "report": str(report_path.resolve()),
        },
    }
    atomic_write_text(report_path, report_markdown(summary, records, args.max_non_improvements))
    atomic_write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        summary = generate(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "ok", "artifacts": summary["artifacts"], "best": summary["best"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
