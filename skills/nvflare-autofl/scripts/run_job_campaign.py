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

"""Small deterministic Auto-FL campaign runner for an existing NVFlare job.py.

This script is intentionally modest: it runs a baseline plus a fixed number of
same-budget CLI-argument candidates, records state and artifacts, and never
depends on the research/auto-fl-research harness. It is a skill-side scaffold
for validating the product UX while the supported Auto-FL APIs mature.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - launcher installs PyYAML through NVFlare deps
    yaml = None


RESULT_FIELDS = [
    "status",
    "name",
    "score",
    "runtime_seconds",
    "changed_files",
    "diff_summary",
    "run_command",
    "artifacts",
    "failure_reason",
]

INFRASTRUCTURE_RETRY = "infrastructure_retry"
SIMULATOR_STALL_EXIT_CODE = 125
SIMULATOR_STALL_PATTERNS = (
    "Failed to create connection to the child process in SimulatorClientRunner",
    "SimulatorClientRunner - ERROR - run_client_thread error",
)
SIMULATOR_STALL_LOG_LIMIT = 65536
SIMULATOR_AGGREGATION_RE = re.compile(r"Aggregated\s+(\d+)/(\d+)\s+results")
SIMULATOR_PROGRESS_PATTERNS = (
    "Round ",
    "Aggregated ",
    "Beginning model validation",
    "Saved validation result",
    "Finished FedAvg",
    "Finished ScatterAndGather",
    "cross_site_val",
    "round=",
    "epoch",
    "returning result",
    "sent result",
    "task completed",
)
SIMULATOR_PROGRESS_LOG_LIMIT = 131072
DEFAULT_SIMULATOR_NO_PROGRESS_TIMEOUT = 240
DEFAULT_HARD_CRASH_THRESHOLD = 6
DEFAULT_PLATEAU_MIN_DELTA = 0.0005
DEFAULT_PLATEAU_THRESHOLD = 32

FIXED_BUDGET_TO_CLI = {
    "num_clients": "n_clients",
    "num_rounds": "num_rounds",
}

PROFILE_BUDGET_TO_CLI = {
    "n_clients": "n_clients",
    "num_rounds": "num_rounds",
    "aggregation_epochs": "aggregation_epochs",
    "local_train_steps": "local_train_steps",
    "batch_size": "batch_size",
    "eval_batch_size": "eval_batch_size",
    "alpha": "alpha",
    "seed": "seed",
    "model_arch": "model_arch",
    "max_model_params": "max_model_params",
    "aggregator": "aggregator",
    "final_eval_clients": "final_eval_clients",
}


@dataclass
class RunRecord:
    status: str
    name: str
    score: Optional[float]
    runtime_seconds: float
    changed_files: str
    diff_summary: str
    run_command: str
    artifacts: str
    failure_reason: str = ""


@dataclass
class JobRun:
    name: str
    args: List[str]
    description: str
    status: str = "candidate"
    score: Optional[float] = None
    runtime_seconds: float = 0.0
    artifacts: str = ""
    failure_reason: str = ""
    command: List[str] = field(default_factory=list)


def env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value


def env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", help="NVFlare job.py to optimize")
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--mode", choices=["max", "min"], default="max")
    parser.add_argument("--env", dest="target_env", choices=["sim"], default="sim")
    parser.add_argument(
        "--max-candidates",
        type=int,
        help="optional candidate cap; omit to continue until interrupted or blocked",
    )
    parser.add_argument("--autofl-yaml", default="autofl.yaml")
    parser.add_argument("--results", default="results.tsv")
    parser.add_argument("--state", default=".nvflare/autofl/campaign_state.json")
    parser.add_argument("--progress", default="progress.png")
    parser.add_argument("--report", default="autofl_report.md")
    parser.add_argument("--output-root", default="autofl_runs")
    parser.add_argument(
        "--plateau-threshold",
        type=int,
        default=env_int("AUTOFL_PLATEAU_THRESHOLD", DEFAULT_PLATEAU_THRESHOLD),
        help=(
            "scored comparable candidate attempts after the last material improvement or literature event "
            "before campaign state requests run_literature_loop"
        ),
    )
    parser.add_argument(
        "--plateau-min-delta",
        type=float,
        default=env_float("AUTOFL_PLATEAU_MIN_DELTA", DEFAULT_PLATEAU_MIN_DELTA),
        help="minimum metric delta required to reset the plateau clock",
    )
    parser.add_argument(
        "--hard-crash-threshold",
        type=int,
        default=env_int("AUTOFL_HARD_CRASH_THRESHOLD", DEFAULT_HARD_CRASH_THRESHOLD),
        help="stop after this many consecutive real candidate crashes; set 0 to disable",
    )
    parser.add_argument(
        "--stop-file",
        action="append",
        help="manual stop-file path; defaults to STOP_AUTOFL and .nvflare/autofl/STOP under the job directory",
    )
    parser.add_argument("--base-args", default="", help="extra job args applied to baseline and all candidates")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--simulator-no-progress-timeout",
        type=int,
        default=env_int("AUTOFL_SIMULATOR_NO_PROGRESS_TIMEOUT_SECONDS", DEFAULT_SIMULATOR_NO_PROGRESS_TIMEOUT),
        help=(
            "candidate-level simulator no-progress timeout in seconds; set 0 to disable. "
            "This is separate from the full run timeout and only applies after progress markers appear."
        ),
    )
    parser.add_argument("--python", default=os.environ.get("PYTHON") or sys.executable)
    parser.add_argument("--prefer-synthetic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--synthetic-train-size", type=int, default=2048)
    parser.add_argument("--synthetic-test-size", type=int, default=256)
    return parser.parse_args(argv)


def terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            process.terminate()
    else:
        process.terminate()

    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass

    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except Exception:
            process.kill()
    else:
        process.kill()


def recent_text(path: Path, limit: int = SIMULATOR_STALL_LOG_LIMIT) -> str:
    try:
        with path.open("rb") as f:
            try:
                f.seek(-limit, os.SEEK_END)
            except OSError:
                f.seek(0)
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def detect_nvflare_simulator_stall(sim_root: Path) -> Optional[str]:
    if not sim_root.exists():
        return None

    log_paths = [
        sim_root / "server" / "log_fl.txt",
        sim_root / "server" / "log.txt",
        sim_root / "server" / "error_log.txt",
    ]
    for path in log_paths:
        text = recent_text(path)
        if not text:
            continue
        for pattern in SIMULATOR_STALL_PATTERNS:
            if pattern in text:
                return f"{pattern} in {path}"
    return None


def simulator_stall_message(simulator_stall_roots: Sequence[Path]) -> Optional[str]:
    for root in simulator_stall_roots:
        message = detect_nvflare_simulator_stall(root)
        if message:
            return message
    return None


def simulator_progress_signature(sim_root: Path) -> str:
    if not sim_root.exists():
        return ""

    log_paths = [
        sim_root / "server" / "log_fl.txt",
        sim_root / "server" / "log.txt",
        *sorted(sim_root.glob("site-*/log_fl.txt")),
        *sorted(sim_root.glob("site-*/log.txt")),
    ]
    markers: List[str] = []
    for path in log_paths:
        text = recent_text(path, limit=SIMULATOR_PROGRESS_LOG_LIMIT)
        if not text:
            continue
        for line in text.splitlines():
            line_lower = line.lower()
            if any(pattern.lower() in line_lower for pattern in SIMULATOR_PROGRESS_PATTERNS):
                markers.append(f"{path.relative_to(sim_root)}:{line}")
    return "\n".join(markers[-200:])


def simulator_progress_signature_for_roots(simulator_stall_roots: Sequence[Path]) -> str:
    markers = []
    for root in simulator_stall_roots:
        signature = simulator_progress_signature(root)
        if signature:
            markers.append(f"{root}:\n{signature}")
    return "\n".join(markers)


def simulator_partial_aggregation_signature(sim_root: Path) -> str:
    signatures = []
    for relative_log_path in ("server/log_fl.txt", "server/simulate_job/log_fl.txt"):
        path = sim_root / relative_log_path
        text = recent_text(path, SIMULATOR_PROGRESS_LOG_LIMIT)
        if not text:
            continue
        for line in reversed(text.splitlines()):
            match = SIMULATOR_AGGREGATION_RE.search(line)
            if not match:
                continue
            received = int(match.group(1))
            expected = int(match.group(2))
            if received < expected:
                signatures.append(f"{path.relative_to(sim_root)}:{line}")
            break
    return "\n".join(signatures)


def simulator_partial_aggregation_signature_for_roots(simulator_stall_roots: Sequence[Path]) -> str:
    markers = []
    for root in simulator_stall_roots:
        signature = simulator_partial_aggregation_signature(root)
        if signature:
            markers.append(f"{root}:\n{signature}")
    return "\n".join(markers)


def run(
    argv: Sequence[str],
    cwd: Path,
    timeout: int,
    log_path: Path,
    simulator_stall_roots: Sequence[Path] = (),
    stall_check_interval: float = 5.0,
    simulator_no_progress_timeout: int = DEFAULT_SIMULATOR_NO_PROGRESS_TIMEOUT,
) -> Tuple[int, str, float]:
    started = time.monotonic()
    next_stall_check = started
    last_progress_check = started
    last_progress_seen = started
    last_progress_signature = ""
    last_partial_aggregation_seen = started
    last_partial_aggregation_signature = ""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    chunks: List[str] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=os.name != "nt",
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        timed_out = False
        stall_message = ""
        while True:
            now = time.monotonic()
            if timeout and now - started > timeout:
                timed_out = True
                terminate_process(process)
            if simulator_stall_roots and now >= next_stall_check:
                stall_message = simulator_stall_message(simulator_stall_roots) or ""
                next_stall_check = now + stall_check_interval
                if stall_message:
                    terminate_process(process)
                elif simulator_no_progress_timeout:
                    partial_aggregation_signature = simulator_partial_aggregation_signature_for_roots(
                        simulator_stall_roots
                    )
                    if (
                        partial_aggregation_signature
                        and partial_aggregation_signature != last_partial_aggregation_signature
                    ):
                        last_partial_aggregation_signature = partial_aggregation_signature
                        last_partial_aggregation_seen = now
                    elif (
                        last_partial_aggregation_signature
                        and now - last_partial_aggregation_seen > simulator_no_progress_timeout
                    ):
                        stall_message = (
                            "partial simulator aggregation made no server-side progress for "
                            f"{int(now - last_partial_aggregation_seen)}s: {last_partial_aggregation_signature}"
                        )
                        terminate_process(process)
                    progress_signature = simulator_progress_signature_for_roots(simulator_stall_roots)
                    if stall_message:
                        pass
                    elif progress_signature and progress_signature != last_progress_signature:
                        last_progress_signature = progress_signature
                        last_progress_seen = now
                    elif (
                        last_progress_signature
                        and now - last_progress_seen > simulator_no_progress_timeout
                        and now - last_progress_check >= stall_check_interval
                    ):
                        stall_message = (
                            f"no simulator progress markers changed for {int(now - last_progress_seen)}s "
                            f"across {', '.join(str(root) for root in simulator_stall_roots)}"
                        )
                        terminate_process(process)
                    last_progress_check = now
            events = selector.select(timeout=0.2)
            for key, _ in events:
                chunk = key.fileobj.readline()
                if chunk:
                    chunks.append(chunk)
                    log_file.write(chunk)
                    log_file.flush()
            if process.poll() is not None:
                remainder = process.stdout.read()
                if remainder:
                    chunks.append(remainder)
                    log_file.write(remainder)
                    log_file.flush()
                break
        selector.close()
        if timed_out:
            timeout_msg = f"\nTIMEOUT after {timeout}s\n"
            chunks.append(timeout_msg)
            log_file.write(timeout_msg)
            log_file.flush()
            return 124, "".join(chunks), time.monotonic() - started
        if stall_message:
            stall_text = f"\nSIMULATOR_STALL: {stall_message}\n"
            chunks.append(stall_text)
            log_file.write(stall_text)
            log_file.flush()
            return SIMULATOR_STALL_EXIT_CODE, "".join(chunks), time.monotonic() - started
    return process.returncode or 0, "".join(chunks), time.monotonic() - started


def run_allow_timeout(
    argv: Sequence[str],
    cwd: Path,
    timeout: int,
    log_path: Path,
    simulator_stall_roots: Sequence[Path] = (),
    simulator_no_progress_timeout: int = DEFAULT_SIMULATOR_NO_PROGRESS_TIMEOUT,
) -> Tuple[int, str, float]:
    try:
        return run(
            argv,
            cwd,
            timeout,
            log_path,
            simulator_stall_roots=simulator_stall_roots,
            simulator_no_progress_timeout=simulator_no_progress_timeout,
        )
    except subprocess.TimeoutExpired as e:
        output = e.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output + f"\nTIMEOUT after {timeout}s\n", encoding="utf-8")
        return 124, output, float(timeout)


def read_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read autofl.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


_CAMPAIGN_GUARD = None


def load_campaign_guard():
    global _CAMPAIGN_GUARD
    if _CAMPAIGN_GUARD is not None:
        return _CAMPAIGN_GUARD

    guard_path = Path(__file__).resolve().with_name("campaign_guard.py")
    spec = importlib.util.spec_from_file_location("nvflare_autofl_skill_campaign_guard", guard_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load campaign guard from {guard_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _CAMPAIGN_GUARD = module
    return module


def resolve_output_path(cwd: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return cwd / path


def resolve_stop_files(cwd: Path, values: Optional[Sequence[str]]) -> List[str]:
    guard = load_campaign_guard()
    stop_files = values if values is not None else list(guard.DEFAULT_STOP_FILES)
    return [str(resolve_output_path(cwd, value)) for value in stop_files]


def extract_result_dir(output: str) -> Optional[Path]:
    patterns = [
        r"Result can be found in\s*:\s*(?P<path>\S+)",
        r"Results:\s*(?P<path>\S+)",
        r"result_dir=(?P<path>\S+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            return Path(match.group("path")).expanduser()
    return None


def find_metric_value(payload: Any, metric: str) -> Optional[float]:
    if isinstance(payload, dict):
        for key in ("final_aggregated_metrics", "best_metrics", "metrics"):
            value = metric_from_list(payload.get(key), metric)
            if value is not None:
                return value
        if metric in payload and isinstance(payload[metric], (int, float)):
            return float(payload[metric])
        for value in payload.values():
            score = find_metric_value(value, metric)
            if score is not None:
                return score
    elif isinstance(payload, list):
        value = metric_from_list(payload, metric)
        if value is not None:
            return value
        for item in payload:
            score = find_metric_value(item, metric)
            if score is not None:
                return score
    return None


def metric_from_list(items: Any, metric: str) -> Optional[float]:
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("name") == metric and isinstance(item.get("value"), (int, float)):
            return float(item["value"])
    return None


def extract_score(artifact_root: Path, metric: str) -> Optional[float]:
    metric_files = list(artifact_root.glob("**/metrics_summary.json")) + list(
        artifact_root.glob("**/cross_val_results.json")
    )
    for path in metric_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        score = find_metric_value(payload, metric)
        if score is not None:
            return score

    number_patterns = [
        rf"{re.escape(metric)}[^0-9+\-.]+([+-]?[0-9]+(?:\.[0-9]+)?)",
        r"Accuracy of the network[^0-9+\-.]+([+-]?[0-9]+(?:\.[0-9]+)?)",
    ]
    for path in artifact_root.glob("**/*.log"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for pattern in number_patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if matches:
                return float(matches[-1])
    return None


def is_sandbox_socket_failure(output: str) -> bool:
    text = output.lower()
    return (
        "permissionerror" in text
        and ("operation not permitted" in text or "[errno 1]" in text)
        and ("socket" in text or "sock" in text)
    )


def is_nvflare_simulator_stall(output: str) -> bool:
    return "SIMULATOR_STALL:" in output


def collect_artifacts(result_dir: Optional[Path], output_root: Path, name: str, log_path: Path) -> Path:
    dest = output_root / name / "simulation"
    run_log = output_root / name / "run.log"
    if dest.exists():
        shutil.rmtree(dest)
    if result_dir and result_dir.exists():
        shutil.copytree(result_dir, dest)
        shutil.rmtree(result_dir, ignore_errors=True)
    else:
        dest.mkdir(parents=True, exist_ok=True)
    if log_path.resolve() != run_log.resolve():
        shutil.copy2(log_path, run_log)
    return dest


def job_help(python: str, job: Path, cwd: Path) -> str:
    process = subprocess.run(
        [python, str(job), "--help"],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return process.stdout


def supports_flag(help_text: str, flag: str) -> bool:
    return flag in help_text


def mutable_arg_specs(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    specs = schema.get("mutable_args")
    return specs if isinstance(specs, dict) else {}


def candidate_arg_values(candidate_args: Sequence[str]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    idx = 0
    while idx < len(candidate_args):
        raw = candidate_args[idx]
        if not raw.startswith("--"):
            idx += 1
            continue
        name = raw[2:].replace("-", "_")
        if idx + 1 >= len(candidate_args) or candidate_args[idx + 1].startswith("--"):
            values[name] = True
            idx += 1
        else:
            values[name] = candidate_args[idx + 1]
            idx += 2
    return values


def coerce_schema_value(value: Any, spec: Dict[str, Any]) -> Any:
    value_type = spec.get("type")
    if value_type == "int":
        return int(value)
    if value_type == "float":
        return float(value)
    if value_type == "bool":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    return str(value)


def candidate_args_allowed(candidate_args: Sequence[str], schema: Dict[str, Any]) -> Tuple[bool, str]:
    specs = mutable_arg_specs(schema)
    if not specs:
        return True, ""

    for name, value in candidate_arg_values(candidate_args).items():
        spec = specs.get(name)
        if not isinstance(spec, dict):
            continue
        try:
            coerced = coerce_schema_value(value, spec)
        except (TypeError, ValueError) as e:
            return False, f"{name}={value!r} cannot be parsed as {spec.get('type')}: {e}"

        choices = spec.get("choices")
        if choices is not None and coerced not in choices:
            return False, f"{name}={coerced!r} is not in allowed choices {choices!r}"
        minimum = spec.get("min")
        if minimum is not None and coerced < minimum:
            return False, f"{name}={coerced!r} is below schema min {minimum!r}"
        maximum = spec.get("max")
        if maximum is not None and coerced > maximum:
            return False, f"{name}={coerced!r} is above schema max {maximum!r}"

    return True, ""


def load_mutation_schema(cwd: Path) -> Dict[str, Any]:
    path = cwd / "mutation_schema.yaml"
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to read mutation_schema.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def h100_comparison_budget(schema: Dict[str, Any]) -> Dict[str, Any]:
    return (
        schema.get("comparison_budget_args", {}).get("h100_default_candidate_budget", {})
        if isinstance(schema.get("comparison_budget_args"), dict)
        else {}
    )


def fixed_within_campaign(schema: Dict[str, Any]) -> set:
    values = []
    comparison = schema.get("comparison_budget_args")
    if isinstance(comparison, dict):
        values = comparison.get("fixed_within_campaign") or []
    return set(values) if isinstance(values, list) else set()


def build_profile_args(schema: Dict[str, Any], help_text: str) -> List[str]:
    budget = h100_comparison_budget(schema)
    args: List[str] = []
    for field, cli_name in PROFILE_BUDGET_TO_CLI.items():
        value = budget.get(field)
        if value is not None and supports_flag(help_text, f"--{cli_name}"):
            args.extend([f"--{cli_name}", str(value)])
    if budget.get("cross_site_eval") and supports_flag(help_text, "--cross_site_eval"):
        args.append("--cross_site_eval")
    return args


def build_fixed_args(config: Dict[str, Any], help_text: str, schema: Dict[str, Any]) -> List[str]:
    fixed = config.get("budget", {}).get("fixed_training_budget", {}) or {}
    profile_budget = h100_comparison_budget(schema)
    profile_cli_names = {
        cli_name for field, cli_name in PROFILE_BUDGET_TO_CLI.items() if profile_budget.get(field) is not None
    }
    args: List[str] = []
    for field, cli_name in FIXED_BUDGET_TO_CLI.items():
        if cli_name in profile_cli_names:
            continue
        value = fixed.get(field)
        if value is not None and supports_flag(help_text, f"--{cli_name}"):
            args.extend([f"--{cli_name}", str(value)])
    return args


def build_base_args(args: argparse.Namespace, help_text: str, schema: Dict[str, Any]) -> List[str]:
    base = shlex.split(args.base_args)
    profile_args = build_profile_args(schema, help_text)
    if profile_args:
        base.extend(profile_args)
    if args.prefer_synthetic and supports_flag(help_text, "--synthetic_data"):
        if "--synthetic_data" not in base:
            base.append("--synthetic_data")
        if supports_flag(help_text, "--train_size") and "--train_size" not in base:
            base.extend(["--train_size", str(args.synthetic_train_size)])
        if supports_flag(help_text, "--test_size") and "--test_size" not in base:
            base.extend(["--test_size", str(args.synthetic_test_size)])
    return base


def suggested_arg_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    suggested = config.get("search_space", {}).get("suggested", {}) or {}
    defaults = {}
    for name, spec in suggested.items():
        if isinstance(spec, dict) and "default" in spec:
            defaults[name] = spec["default"]
    return defaults


def candidate_plan(
    config: Dict[str, Any],
    help_text: str,
    max_candidates: Optional[int],
    schema: Optional[Dict[str, Any]] = None,
) -> Iterable[JobRun]:
    defaults = suggested_arg_defaults(config)
    candidates: List[JobRun] = []
    seen_args = set()
    fixed_fields = fixed_within_campaign(schema or {})

    def can_mutate(name: str) -> bool:
        return name not in fixed_fields

    def make_candidate(name: str, candidate_args: List[str], description: str) -> Optional[JobRun]:
        allowed, _ = candidate_args_allowed(candidate_args, schema or {})
        if not allowed:
            return None
        key = tuple(candidate_args)
        if key in seen_args:
            return None
        seen_args.add(key)
        return JobRun(name=name, args=candidate_args, description=description)

    def add(name: str, candidate_args: List[str], description: str) -> None:
        if max_candidates is not None and len(candidates) >= max_candidates:
            return
        candidate = make_candidate(name, candidate_args, description)
        if candidate is not None:
            candidates.append(candidate)

    if can_mutate("aggregator") and supports_flag(help_text, "--aggregator"):
        for value in ["default", "fedavg", "fedavgm", "fedadam", "scaffold"]:
            add(f"aggregator_{value}", ["--aggregator", value], f"aggregator={value}")
        if can_mutate("fedproxloss_mu") and supports_flag(help_text, "--fedproxloss_mu"):
            add(
                "fedprox_mu_1e-5",
                ["--aggregator", "weighted", "--fedproxloss_mu", "1e-5"],
                "aggregator=weighted fedproxloss_mu=1e-5",
            )
            add(
                "fedprox_mu_1e-4",
                ["--aggregator", "weighted", "--fedproxloss_mu", "1e-4"],
                "aggregator=weighted fedproxloss_mu=1e-4",
            )

    if can_mutate("aggregation_epochs") and supports_flag(help_text, "--aggregation_epochs"):
        default = int(defaults.get("aggregation_epochs") or 4)
        for value in [1, 2, 3, 4, 6, 8]:
            if value != default:
                add(f"aggregation_epochs_{value}", ["--aggregation_epochs", str(value)], f"aggregation_epochs={value}")

    if can_mutate("local_train_steps") and supports_flag(help_text, "--local_train_steps"):
        for value in [50, 100, 200, 400]:
            add(f"local_train_steps_{value}", ["--local_train_steps", str(value)], f"local_train_steps={value}")

    if can_mutate("lr") and supports_flag(help_text, "--lr"):
        default = float(defaults.get("lr") or 0.05)
        for value in [default / 4, default / 2, default * 2, default * 4]:
            value_text = f"{value:.6g}"
            add(f"lr_{value_text.replace('.', 'p').replace('-', 'm')}", ["--lr", value_text], f"lr={value_text}")

    if can_mutate("momentum") and supports_flag(help_text, "--momentum"):
        for value in [0.0, 0.5, 0.8, 0.95]:
            add(
                f"momentum_{str(value).replace('.', 'p')}",
                ["--momentum", str(value)],
                f"momentum={value}",
            )

    if can_mutate("weight_decay") and supports_flag(help_text, "--weight_decay"):
        for value in ["1e-5", "1e-4", "5e-4"]:
            add(f"weight_decay_{value.replace('-', 'm')}", ["--weight_decay", value], f"weight_decay={value}")

    if can_mutate("batch_size") and supports_flag(help_text, "--batch_size"):
        default = int(defaults.get("batch_size") or 16)
        values = [max(1, default // 2), default * 2, default * 4, max(1, default // 4), 24, 40, 64, 96, 128, 256]
        for value in values:
            if value != default:
                add(f"batch_size_{value}", ["--batch_size", str(value)], f"batch_size={value}")

    if can_mutate("epochs") and supports_flag(help_text, "--epochs"):
        for value in [1, 2, 3, 4, 5]:
            add(f"epochs_{value}", ["--epochs", str(value)], f"epochs={value}")

    if can_mutate("num_workers") and supports_flag(help_text, "--num_workers"):
        for value in [0, 1, 2, 4]:
            add(f"num_workers_{value}", ["--num_workers", str(value)], f"num_workers={value}")

    if supports_flag(help_text, "--client_memory_gc_rounds"):
        add("client_memory_gc_1", ["--client_memory_gc_rounds", "1"], "client_memory_gc_rounds=1")

    if not candidates:
        add("rerun", [], "repeat baseline command to test campaign plumbing")

    if max_candidates is not None:
        return candidates[:max_candidates]

    def uncapped() -> Iterable[JobRun]:
        for template in candidates:
            yield JobRun(name=template.name, args=list(template.args), description=template.description)

        idx = 1
        batch_default = int(defaults.get("batch_size") or 16)
        while True:
            generated = False
            if can_mutate("batch_size") and supports_flag(help_text, "--batch_size"):
                # Walk a broad conservative range before repeats so uncapped
                # campaigns keep doing comparable, reviewable work for hours.
                value = 1 + ((batch_default + idx * 7) % 512)
                candidate_args = ["--batch_size", str(value)]
                candidate = make_candidate(
                    f"batch_size_auto_{value}",
                    candidate_args,
                    f"batch_size={value}",
                )
                if value != batch_default and candidate is not None:
                    yield candidate
                    generated = True

            if not generated and can_mutate("aggregation_epochs") and supports_flag(help_text, "--aggregation_epochs"):
                value = 1 + ((idx - 1) % 8)
                candidate_args = ["--aggregation_epochs", str(value)]
                candidate = make_candidate(
                    f"aggregation_epochs_auto_{value}",
                    candidate_args,
                    f"aggregation_epochs={value}",
                )
                if candidate is not None:
                    yield candidate
                    generated = True

            if not generated and can_mutate("lr") and supports_flag(help_text, "--lr"):
                value = 10 ** (-4 + ((idx - 1) % 25) / 10)
                value_text = f"{value:.6g}"
                candidate_args = ["--lr", value_text]
                candidate = make_candidate(
                    f"lr_auto_{value_text.replace('.', 'p').replace('-', 'm')}",
                    candidate_args,
                    f"lr={value_text}",
                )
                if candidate is not None:
                    yield candidate
                    generated = True

            if not generated and can_mutate("epochs") and supports_flag(help_text, "--epochs"):
                value = 1 + ((idx - 1) % 20)
                candidate_args = ["--epochs", str(value)]
                candidate = make_candidate(f"epochs_auto_{value}", candidate_args, f"epochs={value}")
                if candidate is not None:
                    yield candidate
                    generated = True

            if not generated and can_mutate("num_workers") and supports_flag(help_text, "--num_workers"):
                value = (idx - 1) % 9
                candidate_args = ["--num_workers", str(value)]
                candidate = make_candidate(f"num_workers_auto_{value}", candidate_args, f"num_workers={value}")
                if candidate is not None:
                    yield candidate
                    generated = True

            if not generated:
                template = candidates[(idx - 1) % len(candidates)]
                cycle = (idx - 1) // len(candidates) + 2
                yield JobRun(
                    name=f"{template.name}_repeat_{cycle:04d}",
                    args=list(template.args),
                    description=f"{template.description}; repeat_cycle={cycle}",
                )
            idx += 1

    return uncapped()


def remove_known_result_dir(config: Dict[str, Any]) -> None:
    recipe_args = config.get("job", {}).get("recipe_args", {}) or {}
    name = recipe_args.get("name", {}).get("value") if isinstance(recipe_args.get("name"), dict) else None
    if isinstance(name, str) and name:
        shutil.rmtree(Path("/tmp/nvflare/simulation") / name, ignore_errors=True)


def run_job(
    run_def: JobRun,
    *,
    python: str,
    job: Path,
    cwd: Path,
    help_text: str,
    fixed_args: List[str],
    base_args: List[str],
    output_root: Path,
    timeout: int,
    simulator_no_progress_timeout: int,
    metric: str,
    config: Dict[str, Any],
) -> RunRecord:
    log_path = output_root / run_def.name / "run.log"
    run_name = f"autofl_{run_def.name}"
    simulator_root = Path("/tmp/nvflare/simulation") / run_name
    name_args = ["--name", run_name] if supports_flag(help_text, "--name") else []
    command = [python, str(job), *fixed_args, *base_args, *name_args, *run_def.args]
    run_def.command = command
    remove_known_result_dir(config)
    if name_args:
        shutil.rmtree(simulator_root, ignore_errors=True)
    rc, stdout, runtime = run_allow_timeout(
        command,
        cwd,
        timeout,
        log_path,
        simulator_stall_roots=[simulator_root],
        simulator_no_progress_timeout=simulator_no_progress_timeout,
    )
    run_def.runtime_seconds = runtime
    result_dir = extract_result_dir(stdout) or (simulator_root if simulator_root.exists() else None)
    artifact_dir = collect_artifacts(result_dir, output_root, run_def.name, log_path)
    run_def.artifacts = str(artifact_dir)

    if rc != 0:
        if is_sandbox_socket_failure(stdout):
            run_def.status = INFRASTRUCTURE_RETRY
            run_def.failure_reason = "sandbox/socket permission failure; rerun runner with escalated execution"
        elif is_nvflare_simulator_stall(stdout):
            run_def.status = "crash"
            run_def.failure_reason = (
                "nvflare simulator watchdog detected a child connection/no-progress stall; "
                "candidate killed and campaign continued"
            )
        else:
            run_def.status = "crash"
            run_def.failure_reason = f"exit_code={rc}"
    else:
        score = extract_score(artifact_dir, metric)
        if score is None:
            run_def.status = "crash"
            run_def.failure_reason = f"metric '{metric}' not found"
        else:
            run_def.score = score

    return RunRecord(
        status=run_def.status,
        name=run_def.name,
        score=run_def.score,
        runtime_seconds=run_def.runtime_seconds,
        changed_files="none",
        diff_summary=run_def.description,
        run_command=shlex.join(command),
        artifacts=run_def.artifacts,
        failure_reason=run_def.failure_reason,
    )


def write_results(path: Path, records: List[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "status": record.status,
                    "name": record.name,
                    "score": "" if record.score is None else f"{record.score:.6f}",
                    "runtime_seconds": f"{record.runtime_seconds:.3f}",
                    "changed_files": record.changed_files,
                    "diff_summary": record.diff_summary,
                    "run_command": record.run_command,
                    "artifacts": record.artifacts,
                    "failure_reason": record.failure_reason,
                }
            )


def better(new_score: Optional[float], old_score: Optional[float], mode: str) -> bool:
    if new_score is None:
        return False
    if old_score is None:
        return True
    return new_score > old_score if mode == "max" else new_score < old_score


def finalize_candidates(records: List[RunRecord], mode: str) -> None:
    baseline = next((r for r in records if r.status == "baseline"), None)
    best_score = baseline.score if baseline else None
    for record in records:
        if record.status == "keep" and better(record.score, best_score, mode):
            best_score = record.score
    best_idx: Optional[int] = None
    for idx, record in enumerate(records):
        if record.status != "candidate":
            continue
        if better(record.score, best_score, mode):
            best_score = record.score
            best_idx = idx
    for idx, record in enumerate(records):
        if record.status != "candidate":
            continue
        record.status = "keep" if idx == best_idx else "discard"


def write_state(
    path: Path,
    results_path: Path,
    records: List[RunRecord],
    max_candidates: Optional[int],
    *,
    mode: str = "max",
    stop_files: Optional[List[str]] = None,
    plateau_threshold: int = DEFAULT_PLATEAU_THRESHOLD,
    plateau_min_delta: float = DEFAULT_PLATEAU_MIN_DELTA,
    hard_crash_threshold: int = DEFAULT_HARD_CRASH_THRESHOLD,
    manual_stop: bool = False,
) -> Dict[str, Any]:
    guard = load_campaign_guard()
    attempts = len([r for r in records if r.status in {"keep", "discard", "crash"}])
    if manual_stop:
        state = guard.guard_state(
            results_path,
            max_candidates=max_candidates,
            stop_files=stop_files,
            plateau_threshold=plateau_threshold,
            min_delta=plateau_min_delta,
            hard_crash_threshold=hard_crash_threshold,
            mode=mode,
        )
        state.update(
            {
                "candidate_attempts": attempts,
                "decision": "stop",
                "reason": "manual_interrupt",
                "next_action": "final_report",
                "final_response_allowed": True,
                "agent_instruction": "Final report is allowed because the campaign was manually interrupted.",
            }
        )
        write_json(path, state)
        return state

    if any(r.status == INFRASTRUCTURE_RETRY for r in records):
        state = guard.guard_state(
            results_path,
            max_candidates=max_candidates,
            stop_files=stop_files,
            plateau_threshold=plateau_threshold,
            min_delta=plateau_min_delta,
            hard_crash_threshold=hard_crash_threshold,
            mode=mode,
        )
        state.update(
            {
                "candidate_attempts": attempts,
                "decision": "retry_infrastructure",
                "reason": "infrastructure_retry",
                "next_action": "rerun_with_escalated_execution",
                "final_response_allowed": False,
                "agent_instruction": (
                    "Do not produce a final answer. Rerun the same command with escalated execution or repaired "
                    "runtime permissions; infrastructure retries do not count against the candidate budget."
                ),
            }
        )
        write_json(path, state)
        return state

    state = guard.guard_state(
        results_path,
        max_candidates=max_candidates,
        stop_files=stop_files,
        plateau_threshold=plateau_threshold,
        min_delta=plateau_min_delta,
        hard_crash_threshold=hard_crash_threshold,
        mode=mode,
    )
    write_json(path, state)
    return state


def write_progress(path: Path, records: List[RunRecord], mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        path.write_text("Pillow is not installed; progress image unavailable.\n", encoding="utf-8")
        return

    width, height = 1000, 560
    margin = 70
    scores = [r.score for r in records if r.score is not None]
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((margin, 24), f"Auto-FL Progress: {len(records)} rows, {len(scores)} scored", fill="black", font=font)
    draw.line((margin, height - margin, width - margin, height - margin), fill=(80, 80, 80), width=2)
    draw.line((margin, margin, margin, height - margin), fill=(80, 80, 80), width=2)

    if scores:
        lo, hi = min(scores), max(scores)
        if lo == hi:
            lo -= 1.0
            hi += 1.0
        span = hi - lo
        running_best: Optional[float] = None
        last_point: Optional[Tuple[float, float]] = None
        denom = max(1, len(records) - 1)
        for idx, record in enumerate(records):
            if record.score is None:
                continue
            x = margin + (width - 2 * margin) * idx / denom
            y = height - margin - (height - 2 * margin) * (record.score - lo) / span
            color = (40, 160, 90) if record.status in {"baseline", "keep"} else (150, 150, 150)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="black")
            draw.text((x + 6, y - 14), f"{record.name}: {record.score:.3f}", fill=color, font=font)
            if better(record.score, running_best, mode):
                running_best = record.score
            if running_best == record.score:
                if last_point:
                    draw.line((last_point[0], last_point[1], x, y), fill=(40, 160, 90), width=2)
                last_point = (x, y)
    image.save(path)


def write_report(path: Path, config: Dict[str, Any], records: List[RunRecord], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best = None
    for record in records:
        if record.status in {"baseline", "keep"} and better(record.score, best.score if best else None, args.mode):
            best = record
    candidate_budget = (
        str(args.max_candidates) if args.max_candidates is not None else "uncapped; runs until manual interruption"
    )
    lines = [
        "# Auto-FL Report",
        "",
        f"Objective: optimize `{args.metric}` in `{args.target_env}`.",
        f"Candidate budget: `{candidate_budget}`.",
        f"Config: `{args.autofl_yaml}`.",
        f"Fixed budget: `{json.dumps(config.get('budget', {}).get('fixed_training_budget', {}), sort_keys=True)}`.",
        "",
        "## Leaderboard",
        "",
        "| Status | Name | Score | Artifacts | Notes |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for record in records:
        score = "" if record.score is None else f"{record.score:.6f}"
        lines.append(f"| {record.status} | {record.name} | {score} | `{record.artifacts}` | {record.diff_summary} |")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Ledger: `{args.results}`",
            f"- Progress plot: `{args.progress}`",
            f"- Campaign state: `{args.state}`",
            "",
            "## Outcome",
            "",
        ]
    )
    if best:
        lines.append(f"Best retained run: `{best.name}` with `{args.metric}={best.score:.6f}`.")
    else:
        lines.append("No scored run was retained.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def candidate_attempts(records: List[RunRecord]) -> int:
    return len([r for r in records if r.status in {"keep", "discard", "crash"}])


def campaign_summary(
    autofl_yaml: Path,
    results: Path,
    state: Path,
    progress: Path,
    report: Path,
    records: List[RunRecord],
    state_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "autofl_yaml": str(autofl_yaml.resolve()),
        "results": str(results.resolve()),
        "state": str(state.resolve()),
        "progress": str(progress.resolve()),
        "report": str(report.resolve()),
        "candidate_attempts": candidate_attempts(records),
    }
    if state_payload:
        for key in ["decision", "reason", "next_action", "final_response_allowed", "agent_instruction"]:
            if key in state_payload:
                payload[key] = state_payload[key]
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.max_candidates is not None and args.max_candidates < 1:
        print("--max-candidates must be positive when provided", file=sys.stderr)
        return 2
    if args.plateau_threshold < 1:
        print("--plateau-threshold must be positive", file=sys.stderr)
        return 2
    if args.plateau_min_delta < 0:
        print("--plateau-min-delta must be non-negative", file=sys.stderr)
        return 2
    if args.hard_crash_threshold < 0:
        print("--hard-crash-threshold must be non-negative", file=sys.stderr)
        return 2
    job = Path(args.job).resolve()
    cwd = job.parent
    autofl_yaml = resolve_output_path(cwd, args.autofl_yaml)
    results = resolve_output_path(cwd, args.results)
    state = resolve_output_path(cwd, args.state)
    progress = resolve_output_path(cwd, args.progress)
    report = resolve_output_path(cwd, args.report)
    output_root = resolve_output_path(cwd, args.output_root)
    stop_files = resolve_stop_files(cwd, args.stop_file)
    output_root.mkdir(parents=True, exist_ok=True)
    schema = load_mutation_schema(cwd)
    profile_budget = h100_comparison_budget(schema)
    profile_timeout = profile_budget.get("run_timeout_seconds")
    timeout = max(args.timeout, int(profile_timeout)) if profile_timeout is not None else args.timeout
    profile_no_progress_timeout = profile_budget.get("simulator_no_progress_timeout_seconds")
    simulator_no_progress_timeout = (
        int(profile_no_progress_timeout)
        if profile_no_progress_timeout is not None
        else args.simulator_no_progress_timeout
    )

    import_cmd = [
        args.python,
        "-m",
        "nvflare.app_common.autofl.job_importer",
        str(job),
        "--metric",
        args.metric,
        "--env",
        args.target_env,
        "--output",
        str(autofl_yaml),
    ]
    if args.max_candidates is not None:
        import_cmd.extend(["--max-candidates", str(args.max_candidates)])
    rc, output, _ = run(import_cmd, cwd, timeout, output_root / "import.log")
    if rc != 0:
        print(output, file=sys.stderr)
        return rc

    config = read_yaml(autofl_yaml)
    help_text = job_help(args.python, job, cwd)
    fixed_args = build_fixed_args(config, help_text, schema)
    base_args = build_base_args(args, help_text, schema)

    records: List[RunRecord] = []
    baseline = JobRun(name="baseline", args=[], description="baseline", status="baseline")
    baseline_record = run_job(
        baseline,
        python=args.python,
        job=job,
        cwd=cwd,
        help_text=help_text,
        fixed_args=fixed_args,
        base_args=base_args,
        output_root=output_root,
        timeout=timeout,
        simulator_no_progress_timeout=simulator_no_progress_timeout,
        metric=args.metric,
        config=config,
    )
    records.append(baseline_record)
    write_results(results, records)
    state_payload = write_state(
        state,
        results,
        records,
        args.max_candidates,
        mode=args.mode,
        stop_files=stop_files,
        plateau_threshold=args.plateau_threshold,
        plateau_min_delta=args.plateau_min_delta,
        hard_crash_threshold=args.hard_crash_threshold,
    )
    write_progress(progress, records, args.mode)
    write_report(report, config, records, args)
    if baseline_record.status == INFRASTRUCTURE_RETRY:
        print(
            json.dumps(
                campaign_summary(autofl_yaml, results, state, progress, report, records, state_payload),
                indent=2,
                sort_keys=True,
            )
        )
        return 75
    if baseline_record.score is None:
        write_report(report, config, records, args)
        print(f"Baseline run did not produce metric '{args.metric}'. See {baseline_record.artifacts}", file=sys.stderr)
        return 1

    try:
        for candidate in candidate_plan(config, help_text, args.max_candidates, schema):
            record = run_job(
                candidate,
                python=args.python,
                job=job,
                cwd=cwd,
                help_text=help_text,
                fixed_args=fixed_args,
                base_args=base_args,
                output_root=output_root,
                timeout=timeout,
                simulator_no_progress_timeout=simulator_no_progress_timeout,
                metric=args.metric,
                config=config,
            )
            records.append(record)
            finalize_candidates(records, args.mode)
            write_results(results, records)
            state_payload = write_state(
                state,
                results,
                records,
                args.max_candidates,
                mode=args.mode,
                stop_files=stop_files,
                plateau_threshold=args.plateau_threshold,
                plateau_min_delta=args.plateau_min_delta,
                hard_crash_threshold=args.hard_crash_threshold,
            )
            write_progress(progress, records, args.mode)
            write_report(report, config, records, args)
            if record.status == INFRASTRUCTURE_RETRY:
                print(
                    json.dumps(
                        campaign_summary(autofl_yaml, results, state, progress, report, records, state_payload),
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 75
            if state_payload.get("next_action") == "run_literature_loop":
                print(
                    json.dumps(
                        campaign_summary(autofl_yaml, results, state, progress, report, records, state_payload),
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
            if state_payload.get("final_response_allowed"):
                break
    except KeyboardInterrupt:
        write_results(results, records)
        state_payload = write_state(
            state,
            results,
            records,
            args.max_candidates,
            mode=args.mode,
            stop_files=stop_files,
            plateau_threshold=args.plateau_threshold,
            plateau_min_delta=args.plateau_min_delta,
            hard_crash_threshold=args.hard_crash_threshold,
            manual_stop=True,
        )
        write_progress(progress, records, args.mode)
        write_report(report, config, records, args)
        print(
            json.dumps(
                campaign_summary(autofl_yaml, results, state, progress, report, records, state_payload),
                indent=2,
                sort_keys=True,
            )
        )
        return 130

    write_report(report, config, records, args)
    state_payload = write_state(
        state,
        results,
        records,
        args.max_candidates,
        mode=args.mode,
        stop_files=stop_files,
        plateau_threshold=args.plateau_threshold,
        plateau_min_delta=args.plateau_min_delta,
        hard_crash_threshold=args.hard_crash_threshold,
    )
    print(
        json.dumps(
            campaign_summary(autofl_yaml, results, state, progress, report, records, state_payload),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
