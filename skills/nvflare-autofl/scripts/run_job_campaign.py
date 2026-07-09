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

"""Manage agent-authored Auto-FL candidates for an existing NVFlare job.py.

The coding agent owns hypotheses and source edits. This helper snapshots the
current best source, validates and evaluates candidate manifests, restores
discarded candidates, and records reproducible campaign state and artifacts.
"""

from __future__ import annotations

import argparse
import ast
import codecs
import csv
import difflib
import hashlib
import importlib.util
import io
import json
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback keeps per-root locks
    fcntl = None

try:
    import yaml
except ImportError:  # pragma: no cover - NVFlare installs PyYAML
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
    "candidate_manifest",
    "base_candidate",
    "patch_sha256",
    "metric_name",
    "metric_source",
    "metric_artifact",
    "candidate_kind",
    "algorithm_family",
    "literature_event_id",
]

CANDIDATE_MANIFEST_SCHEMA_VERSION = "nvflare.autofl.candidate.v1"
CANDIDATE_MANIFEST_STATUSES = {"prepared", "ready_for_external_execution", "keep", "discard", "crash", "abandoned"}
CAMPAIGN_METADATA_SCHEMA_VERSION = "nvflare.autofl.campaign.v1"
CAMPAIGN_METADATA_PATH = ".nvflare/autofl/campaign.json"
CANDIDATE_ROOT = ".nvflare/autofl/candidates"
BEST_SNAPSHOT_ROOT = ".nvflare/autofl/snapshots/best"
ALLOWED_CREATE_PATTERNS = ["**/*.py"]
RESERVED_CANDIDATE_PATH_PARTS = {".git", ".nvflare", "__pycache__", "autofl_runs"}

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
DEFAULT_JOB_HELP_TIMEOUT = 30
SIMULATOR_WORKSPACE_ROOT_ENV_VAR = "NVFLARE_SIMULATOR_WORKSPACE_ROOT"

FIXED_BUDGET_TO_CLI = {
    "num_clients": "n_clients",
    "min_clients": "min_clients",
    "num_rounds": "num_rounds",
}

COMPARISON_BUDGET_TO_CLI = {
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
    candidate_manifest: str = ""
    base_candidate: str = ""
    patch_sha256: str = ""
    metric_name: str = ""
    metric_source: str = ""
    metric_artifact: str = ""
    candidate_kind: str = ""
    algorithm_family: str = ""
    literature_event_id: str = ""


@dataclass(frozen=True)
class MetricEvidence:
    score: float
    metric_name: str
    source: str
    artifact: str


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
    guard = load_campaign_guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=["initialize", "prepare", "evaluate", "abandon", "suggest", "record", "status"],
        help="skill-internal campaign lifecycle action",
    )
    parser.add_argument("job", help="NVFlare job.py to optimize")
    parser.add_argument("--metric", help="optimization metric; omit to use job.py key_metric")
    parser.add_argument("--mode", choices=["max", "min"], default="max")
    parser.add_argument("--env", dest="target_env", choices=["sim", "poc", "prod"], default="sim")
    cap_group = parser.add_mutually_exclusive_group()
    cap_group.add_argument(
        "--max-candidates",
        type=int,
        help="optional candidate cap; omit to continue until interrupted or blocked",
    )
    cap_group.add_argument("--uncapped", action="store_true", help="remove a previously configured candidate cap")
    parser.add_argument("--autofl-yaml", default="autofl.yaml")
    parser.add_argument("--results", default="results.tsv")
    parser.add_argument("--state", default=".nvflare/autofl/campaign_state.json")
    parser.add_argument("--progress", default="progress.png")
    parser.add_argument("--report", default="autofl_report.md")
    parser.add_argument("--output-root", default="autofl_runs")
    parser.add_argument(
        "--plateau-threshold",
        type=int,
        default=env_int("AUTOFL_PLATEAU_THRESHOLD", guard.DEFAULT_PLATEAU_THRESHOLD),
        help=(
            "scored comparable candidate attempts after the last material improvement or literature event "
            "before campaign state requests run_literature_loop"
        ),
    )
    parser.add_argument(
        "--plateau-min-delta",
        type=float,
        default=env_float("AUTOFL_PLATEAU_MIN_DELTA", guard.DEFAULT_MIN_DELTA),
        help="minimum metric delta required to reset the plateau clock",
    )
    parser.add_argument(
        "--hard-crash-threshold",
        type=int,
        default=env_int("AUTOFL_HARD_CRASH_THRESHOLD", guard.DEFAULT_HARD_CRASH_THRESHOLD),
        help="stop after this many consecutive real candidate crashes; set 0 to disable",
    )
    parser.add_argument(
        "--exploration-batch-size",
        type=int,
        default=env_int("AUTOFL_EXPLORATION_BATCH_SIZE", guard.DEFAULT_EXPLORATION_BATCH_SIZE),
        help=(
            "scored source-backed candidates required per literature event before normal candidate "
            "flow resumes; set 0 to disable"
        ),
    )
    parser.add_argument(
        "--family-repeat-limit",
        type=int,
        default=env_int("AUTOFL_FAMILY_REPEAT_LIMIT", guard.DEFAULT_FAMILY_REPEAT_LIMIT),
        help=(
            "consecutive same-family argument-only scored attempts before campaign state requires "
            "diversification; set 0 to disable"
        ),
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
    parser.add_argument("--name", help="candidate name for prepare")
    parser.add_argument("--hypothesis", help="candidate hypothesis for prepare")
    parser.add_argument("--family", help="algorithm family slug for prepare or record --literature")
    parser.add_argument("--literature-event", help="literature event id this candidate develops (prepare)")
    parser.add_argument("--manifest", help="candidate_manifest.json path")
    parser.add_argument("--run-args", default="", help="candidate-only job.py arguments")
    parser.add_argument("--score", type=float, help="externally measured metric for record")
    parser.add_argument("--artifacts", dest="external_artifacts", help="external POC/production artifacts")
    parser.add_argument("--job-id", help="standard NVFlare job ID for an external result")
    parser.add_argument("--failure-reason", default="", help="external execution failure")
    parser.add_argument("--baseline", action="store_true", help="record an externally executed baseline")
    parser.add_argument("--literature", action="store_true", help="record a literature-review checkpoint")
    parser.add_argument("--limit", type=int, default=10, help="maximum fallback suggestions")
    args = parser.parse_args(argv)
    tokens = list(argv) if argv is not None else sys.argv[1:]
    explicit_settings = set()
    for action in parser._actions:
        for option in action.option_strings:
            if any(token == option or token.startswith(f"{option}=") for token in tokens):
                explicit_settings.add(action.dest)
                break
    args._explicit_settings = explicit_settings
    if args.uncapped:
        args.max_candidates = None
        args._explicit_settings.add("max_candidates")
    return args


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
    env: Optional[Dict[str, str]] = None,
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
            text=False,
            bufsize=0,
            start_new_session=os.name != "nt",
            env=env,
        )
        assert process.stdout is not None
        output_queue: queue.Queue[Optional[bytes]] = queue.Queue()

        def read_output() -> None:
            try:
                while True:
                    chunk = process.stdout.read(65536)
                    if not chunk:
                        break
                    output_queue.put(chunk)
            finally:
                output_queue.put(None)

        reader = threading.Thread(target=read_output, name="autofl-process-output", daemon=True)
        reader.start()
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        output_finished = False
        timed_out = False
        stall_message = ""
        while not output_finished or process.poll() is None:
            now = time.monotonic()
            if not timed_out and timeout and now - started > timeout:
                timed_out = True
                terminate_process(process)
            if not timed_out and not stall_message and simulator_stall_roots and now >= next_stall_check:
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
            try:
                raw_chunk = output_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if raw_chunk is None:
                output_finished = True
                continue
            chunk = decoder.decode(raw_chunk)
            if chunk:
                chunks.append(chunk)
                log_file.write(chunk)
                log_file.flush()
        reader.join(timeout=1)
        remainder = decoder.decode(b"", final=True)
        if remainder:
            chunks.append(remainder)
            log_file.write(remainder)
            log_file.flush()
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


def read_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML files")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"invalid YAML in {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"invalid YAML in {path}: expected a mapping")
    return data


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to write autofl.yaml")
    atomic_write_bytes(path, yaml.safe_dump(data, sort_keys=False).encode("utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    atomic_write_bytes(path, (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def capture_file_versions(paths: Iterable[Path]) -> Dict[Path, Optional[bytes]]:
    return {path: path.read_bytes() if path.is_file() else None for path in paths}


def restore_file_versions(versions: Dict[Path, Optional[bytes]]) -> None:
    for path, data in versions.items():
        if data is None:
            path.unlink(missing_ok=True)
        else:
            atomic_write_bytes(path, data)


def read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"failed to read JSON from {path}: {e}") from e
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def utc_now() -> str:
    return load_campaign_guard().utc_now()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(data: Any) -> str:
    return sha256_bytes(json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def safe_relative_path(workspace: Path, value: str) -> str:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (workspace / path).resolve()
    try:
        relative = resolved.relative_to(workspace.resolve())
    except ValueError as e:
        raise ValueError(f"path escapes the Auto-FL job workspace: {value}") from e
    if not relative.parts or any(part in RESERVED_CANDIDATE_PATH_PARTS for part in relative.parts):
        raise ValueError(f"path is reserved for Auto-FL or repository metadata: {value}")
    return relative.as_posix()


def allowed_edit_paths(config: Dict[str, Any], workspace: Path) -> List[str]:
    values = config.get("trust_contract", {}).get("allowed_edit_paths", []) or []
    if not isinstance(values, list):
        raise ValueError("autofl.yaml trust_contract.allowed_edit_paths must be a list")
    return list(dict.fromkeys(safe_relative_path(workspace, str(value)) for value in values))


def allowed_create_patterns(config: Dict[str, Any]) -> List[str]:
    values = config.get("trust_contract", {}).get("allowed_create_patterns", []) or []
    if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
        raise ValueError("autofl.yaml trust_contract.allowed_create_patterns must be a list of patterns")
    return list(dict.fromkeys(values))


def is_allowed_new_source(path: str, patterns: Sequence[str]) -> bool:
    relative = Path(path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or any(part in RESERVED_CANDIDATE_PATH_PARTS for part in relative.parts)
    ):
        return False
    posix_path = PurePosixPath(relative.as_posix())
    return any(
        posix_path.match(pattern) or (pattern.startswith("**/") and posix_path.match(pattern[3:]))
        for pattern in patterns
    )


def file_map(root: Path) -> Dict[str, str]:
    if not root.exists():
        return {}
    files: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"candidate source contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            files[relative] = sha256_bytes(path.read_bytes())
    return files


def source_hash(files: Dict[str, str]) -> str:
    return sha256_json(files)


def copy_relative_file(source_root: Path, destination_root: Path, relative: str) -> None:
    relative = safe_relative_path(destination_root, relative)
    source = source_root / relative
    destination = destination_root / relative
    try:
        source.resolve().relative_to(source_root.resolve())
    except ValueError as e:
        raise ValueError(f"source path escapes its managed root: {relative}") from e
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"managed source path is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def stage_best_snapshot(workspace: Path, config: Dict[str, Any], snapshot_root: Path) -> Tuple[Path, Dict[str, str]]:
    snapshot_root.parent.mkdir(parents=True, exist_ok=True)
    staged_root = snapshot_root.parent / f".{snapshot_root.name}.staged-{uuid.uuid4().hex}"
    source_root = staged_root / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    try:
        for relative in allowed_edit_paths(config, workspace):
            source = workspace / relative
            if source.is_symlink():
                raise ValueError(f"allowed edit path is a symlink: {source}")
            if source.is_file():
                copy_relative_file(workspace, source_root, relative)
        files = file_map(source_root)
        write_json(staged_root / "snapshot.json", {"source_sha256": source_hash(files), "files": files})
        return staged_root, files
    except BaseException:
        shutil.rmtree(staged_root, ignore_errors=True)
        raise


def activate_best_snapshot(snapshot_root: Path, staged_root: Path) -> Optional[Path]:
    previous_root = None
    if snapshot_root.exists():
        previous_root = snapshot_root.parent / f".{snapshot_root.name}.previous-{uuid.uuid4().hex}"
        os.replace(snapshot_root, previous_root)
    try:
        os.replace(staged_root, snapshot_root)
    except BaseException:
        if previous_root is not None:
            os.replace(previous_root, snapshot_root)
        raise
    return previous_root


def rollback_best_snapshot(snapshot_root: Path, previous_root: Optional[Path]) -> None:
    if previous_root is None:
        shutil.rmtree(snapshot_root, ignore_errors=True)
        return
    failed_root = snapshot_root.parent / f".{snapshot_root.name}.failed-{uuid.uuid4().hex}"
    if snapshot_root.exists():
        os.replace(snapshot_root, failed_root)
    try:
        os.replace(previous_root, snapshot_root)
    except BaseException:
        if failed_root.exists():
            os.replace(failed_root, snapshot_root)
        raise
    shutil.rmtree(failed_root, ignore_errors=True)


def create_best_snapshot(workspace: Path, config: Dict[str, Any], snapshot_root: Path) -> Dict[str, str]:
    staged_root, files = stage_best_snapshot(workspace, config, snapshot_root)
    try:
        previous_root = activate_best_snapshot(snapshot_root, staged_root)
    finally:
        if staged_root.exists():
            shutil.rmtree(staged_root, ignore_errors=True)
    if previous_root is not None:
        shutil.rmtree(previous_root, ignore_errors=True)
    return files


def load_best_snapshot(snapshot_root: Path) -> Tuple[Path, Dict[str, str]]:
    metadata = read_json(snapshot_root / "snapshot.json")
    files = metadata.get("files")
    if not isinstance(files, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in files.items()):
        raise ValueError("best snapshot metadata has an invalid files mapping")
    source_root = snapshot_root / "source"
    if source_hash(files) != metadata.get("source_sha256") or file_map(source_root) != files:
        raise ValueError("best snapshot failed its integrity check")
    return source_root, files


def workspace_matches_snapshot(workspace: Path, source_root: Path, files: Dict[str, str]) -> bool:
    for relative, digest in files.items():
        path = workspace / relative
        if path.is_symlink() or not path.is_file() or sha256_bytes(path.read_bytes()) != digest:
            return False
    return True


def validate_candidate_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", value or ""):
        raise ValueError("candidate name must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
    return value


def candidate_manifest_path(workspace: Path, candidate_id: str) -> Path:
    return workspace / CANDIDATE_ROOT / candidate_id / "candidate_manifest.json"


def validate_candidate_manifest_identity(path: Path, manifest: Dict[str, Any]) -> str:
    if manifest.get("schema_version") != CANDIDATE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported candidate manifest schema in {path}")
    candidate_id = validate_candidate_id(str(manifest.get("candidate_id") or ""))
    expected = candidate_manifest_path(Path(str(manifest.get("workspace_root") or "")), candidate_id).resolve()
    if path.resolve() != expected:
        raise ValueError("candidate manifest path does not match its workspace and candidate ID")
    if manifest.get("status") not in CANDIDATE_MANIFEST_STATUSES:
        raise ValueError(f"candidate {candidate_id} has an invalid manifest status")
    return candidate_id


def load_candidate_manifest(path: Path) -> Dict[str, Any]:
    manifest = read_json(path)
    candidate_id = validate_candidate_manifest_identity(path, manifest)
    if manifest.get("status") not in {"prepared", "ready_for_external_execution"}:
        raise ValueError(f"candidate {candidate_id} is not pending evaluation")
    return manifest


def campaign_metadata_path(workspace: Path) -> Path:
    return workspace / CAMPAIGN_METADATA_PATH


def load_campaign_metadata(workspace: Path, job: Path) -> Dict[str, Any]:
    metadata = read_json(campaign_metadata_path(workspace))
    if metadata.get("schema_version") != CAMPAIGN_METADATA_SCHEMA_VERSION:
        raise ValueError("unsupported Auto-FL campaign metadata schema")
    if Path(str(metadata.get("job") or "")).resolve() != job.resolve():
        raise ValueError("campaign metadata belongs to a different job.py")
    return metadata


def fixed_budget_hash(config: Dict[str, Any]) -> str:
    return sha256_json(config.get("budget", {}).get("fixed_training_budget", {}) or {})


def candidate_changes(
    workspace: Path,
    config: Dict[str, Any],
    best_source: Path,
    best_files: Dict[str, str],
    draft_source: Path,
    *,
    allow_materialized: bool = False,
) -> Tuple[List[str], List[str]]:
    draft_files = file_map(draft_source)
    deleted = sorted(set(best_files) - set(draft_files))
    if deleted:
        raise ValueError(f"candidate deletes managed source files: {', '.join(deleted)}")

    allowed = set(allowed_edit_paths(config, workspace))
    create_patterns = allowed_create_patterns(config)
    changed = sorted(path for path, digest in draft_files.items() if best_files.get(path) != digest)
    created = []
    for relative in changed:
        if relative in best_files:
            if relative not in allowed:
                raise ValueError(f"candidate modifies a path outside trust_contract.allowed_edit_paths: {relative}")
            continue
        if ((workspace / relative).exists() and not allow_materialized) or not is_allowed_new_source(
            relative, create_patterns
        ):
            raise ValueError(f"candidate creates an unauthorized source path: {relative}")
        created.append(relative)
    return changed, created


def text_for_diff(path: Path) -> List[str]:
    data = path.read_bytes()
    if b"\0" in data:
        raise ValueError(f"candidate diff does not support binary file: {path}")
    return data.decode("utf-8", errors="replace").splitlines(keepends=True)


def render_candidate_patch(best_source: Path, draft_source: Path, changed: Sequence[str]) -> str:
    chunks: List[str] = []
    for relative in changed:
        before = text_for_diff(best_source / relative) if (best_source / relative).is_file() else []
        after = text_for_diff(draft_source / relative)
        chunks.extend(
            difflib.unified_diff(
                before,
                after,
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    return "".join(chunks)


def apply_candidate_source(workspace: Path, draft_source: Path, changed: Sequence[str]) -> None:
    for relative in changed:
        copy_relative_file(draft_source, workspace, relative)


def restore_best_source(
    workspace: Path,
    best_source: Path,
    best_files: Dict[str, str],
    changed: Sequence[str],
    created: Sequence[str],
) -> None:
    restore_paths = {safe_relative_path(workspace, value) for value in list(changed) + list(created)}
    for relative in sorted(restore_paths):
        destination = workspace / relative
        if relative in best_files:
            copy_relative_file(best_source, workspace, relative)
        elif destination.exists() and not destination.is_dir():
            destination.unlink()


def validated_manifest_paths(manifest: Dict[str, Any], workspace: Path, field: str) -> List[str]:
    values = manifest.get(field) or []
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"candidate manifest {field} must be a list of paths")
    paths = []
    for value in values:
        normalized = safe_relative_path(workspace, value)
        if value != normalized:
            raise ValueError(f"candidate manifest {field} contains a non-canonical path: {value}")
        paths.append(normalized)
    if len(paths) != len(set(paths)):
        raise ValueError(f"candidate manifest {field} contains duplicate paths")
    return sorted(paths)


def validate_manifest_change_set(
    manifest: Dict[str, Any], workspace: Path, changed: Sequence[str], created: Sequence[str]
) -> None:
    recorded_changed = validated_manifest_paths(manifest, workspace, "changed_files")
    recorded_created = validated_manifest_paths(manifest, workspace, "created_files")
    if recorded_changed != sorted(changed) or recorded_created != sorted(created):
        raise ValueError("candidate manifest source lists do not match the deterministic candidate diff")


def load_sibling_module(filename: str, module_name: str):
    """Load a sibling skill script as a module, cached in sys.modules."""
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    module_path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load skill module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_campaign_guard():
    return load_sibling_module("campaign_guard.py", "nvflare_autofl_skill_campaign_guard")


def load_job_importer():
    return load_sibling_module("job_importer.py", "nvflare_autofl_skill_job_importer")


def resolve_output_path(cwd: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return cwd / path


def resolve_stop_files(cwd: Path, values: Optional[Sequence[str]]) -> List[str]:
    guard = load_campaign_guard()
    stop_files = values if values is not None else list(guard.DEFAULT_STOP_FILES)
    return [str(resolve_output_path(cwd, value)) for value in stop_files]


def extract_result_dir(output: str, cwd: Optional[Path] = None) -> Optional[Path]:
    patterns = [
        r"Result can be found in\s*:\s*(?P<path>\S+)",
        r"Results:\s*(?P<path>\S+)",
        r"result_dir=(?P<path>\S+)",
        r"result location\s*[:=]\s*(?P<path>\S+)",
        r"simulation logs can be found at\s+(?P<path>\S+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            path = Path(match.group("path")).expanduser()
            if not path.is_absolute() and cwd is not None:
                path = cwd / path
            return path.resolve()
    return None


def objective_contract(config: Dict[str, Any], requested_metric: Optional[str]) -> Dict[str, Any]:
    objective = config.get("objective", {})
    if not isinstance(objective, dict):
        objective = {}
    metric = str(objective.get("metric") or requested_metric or "accuracy")
    requested = str(objective.get("requested_metric") or metric)
    optimization = str(objective.get("optimization_metric") or metric)
    extraction_order = objective.get("metric_extraction_order")
    if not isinstance(extraction_order, list) or not extraction_order:
        extraction_order = [optimization]
    extraction_order = [str(item) for item in extraction_order if item]
    if optimization not in extraction_order:
        extraction_order.insert(0, optimization)
    return {
        **objective,
        "metric": metric,
        "requested_metric": requested,
        "optimization_metric": optimization,
        "metric_extraction_order": extraction_order,
    }


def apply_metric_contract(
    config: Dict[str, Any], requested_metric: Optional[str], schema: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    objective = objective_contract(config, requested_metric)
    schema_objective = (schema or {}).get("objective", {})
    if isinstance(schema_objective, dict):
        schema_requested = schema_objective.get("requested_metric") or schema_objective.get("metric")
        if not schema_requested or schema_requested == objective["requested_metric"]:
            for key in ("optimization_metric", "metric_extraction_order", "metric_source"):
                if key in schema_objective:
                    objective[key] = schema_objective[key]
    config["objective"] = objective_contract({"objective": objective}, requested_metric)
    return config


def metric_extraction_order(config: Dict[str, Any], requested_metric: Optional[str]) -> List[str]:
    return list(objective_contract(config, requested_metric)["metric_extraction_order"])


def optimization_metric(config: Dict[str, Any], requested_metric: Optional[str]) -> str:
    return str(objective_contract(config, requested_metric)["optimization_metric"])


def metric_source(config: Dict[str, Any]) -> str:
    source = config.get("objective", {}).get("metric_source", "")
    return str(source) if source else "NVFlare metric artifacts"


def normalize_metric_order(metrics: Sequence[str] | str) -> List[str]:
    if isinstance(metrics, str):
        return [metrics]
    return [str(metric) for metric in metrics if metric]


def parse_finite_metric_value(value: Any) -> Optional[float]:
    return load_campaign_guard().parse_score(value)


def is_numeric_metric_value(value: Any) -> bool:
    return isinstance(value, (int, float)) and parse_finite_metric_value(value) is not None


def find_metric_value(payload: Any, metric_order: Sequence[str] | str) -> Optional[float]:
    metric_keys = normalize_metric_order(metric_order)
    if isinstance(payload, dict):
        for key in ("final_aggregated_metrics", "best_metrics", "metrics"):
            for metric_key in metric_keys:
                value = metric_from_list(payload.get(key), metric_key)
                if value is not None:
                    return value
        for metric_key in metric_keys:
            if metric_key in payload and is_numeric_metric_value(payload[metric_key]):
                return parse_finite_metric_value(payload[metric_key])
        for value in payload.values():
            score = find_metric_value(value, metric_keys)
            if score is not None:
                return score
    elif isinstance(payload, list):
        for metric_key in metric_keys:
            value = metric_from_list(payload, metric_key)
            if value is not None:
                return value
        for item in payload:
            score = find_metric_value(item, metric_keys)
            if score is not None:
                return score
    return None


def metric_from_list(items: Any, metric: str) -> Optional[float]:
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("name") == metric and is_numeric_metric_value(item.get("value")):
            return parse_finite_metric_value(item["value"])
    return None


def text_metric_matches(text: str, metric: str) -> List[Tuple[int, float]]:
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    context = r"(?:(?:final|test|validation|cross[-_ ]site)\s+)*"
    line_prefix = r"^\s*(?:\[[^\]]+\]\s*)*(?:\{|\[)?\s*"
    direct_pattern = re.compile(
        line_prefix + rf"{context}[\"']?{re.escape(metric)}[\"']?" rf"\s*[:=]\s*({number})(?:\s|[,}}\]]|$)",
        flags=re.IGNORECASE,
    )
    evaluation_pattern = re.compile(
        rf"\b{re.escape(metric)}\s+of\s+the\s+received\s+model\b.*?:\s*({number})(?:\s*%|\s|$)",
        flags=re.IGNORECASE,
    )
    framework_pattern = re.compile(
        rf"(?<![\w]){re.escape(metric)}(?![\w])\s*[:=]\s*({number})(?:\s|[,}}\]]|$)",
        flags=re.IGNORECASE,
    )
    mapping_pattern = re.compile(
        rf"(?:\{{|,)\s*[\"']{re.escape(metric)}[\"']\s*:\s*({number})(?:\s|[,}}]|$)",
        flags=re.IGNORECASE,
    )
    direct_matches = []
    evaluation_matches = []
    mapping_matches = []
    framework_matches = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        log_prefix = re.match(r"^\d{4}-\d{2}-\d{2}[^\n]*? - [A-Z]+ - (.*)$", line)
        message = log_prefix.group(1) if log_prefix else line
        direct_match = direct_pattern.search(message)
        if direct_match:
            score = parse_finite_metric_value(direct_match.group(1))
            if score is not None:
                direct_matches.append((line_number, score))
        evaluation_match = evaluation_pattern.search(message)
        if evaluation_match:
            score = parse_finite_metric_value(evaluation_match.group(1))
            if score is not None:
                evaluation_matches.append((line_number, score))
        mapping_match = mapping_pattern.search(message)
        if mapping_match:
            score = parse_finite_metric_value(mapping_match.group(1))
            if score is not None:
                mapping_matches.append((line_number, score))
        if re.search(r"\d+/\d+.*\d+(?:\.\d+)?(?:ms|s)/step\b", message):
            matches = list(framework_pattern.finditer(message))
            if matches:
                score = parse_finite_metric_value(matches[-1].group(1))
                if score is not None:
                    framework_matches.append((line_number, score))
    return evaluation_matches or direct_matches or mapping_matches or framework_matches


def extract_metric_evidence(artifact_root: Path, metrics: Sequence[str] | str) -> Optional[MetricEvidence]:
    metric_order = normalize_metric_order(metrics)
    metric_files = sorted(artifact_root.glob("**/metrics_summary.json")) + sorted(
        artifact_root.glob("**/cross_val_results.json")
    )
    payloads = []
    for path in metric_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payloads.append((path, payload))
    for metric in metric_order:
        for path, payload in payloads:
            score = find_metric_value(payload, [metric])
            if score is not None:
                return MetricEvidence(
                    score=score,
                    metric_name=metric,
                    source=f"structured:{path.name}",
                    artifact=str(path.resolve()),
                )

    text_artifacts = []
    for name in ("run.log", "log.txt", "log_fl.txt", "error_log.txt"):
        text_artifacts.extend(sorted(artifact_root.glob(f"**/{name}")))
    for metric in metric_order:
        for path in text_artifacts:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            matches = text_metric_matches(text, metric)
            if matches:
                line_number, score = matches[-1]
                return MetricEvidence(
                    score=score,
                    metric_name=metric,
                    source=f"text:{path.name}:line={line_number}",
                    artifact=str(path.resolve()),
                )
    return None


def extract_score(artifact_root: Path, metrics: Sequence[str] | str) -> Optional[float]:
    evidence = extract_metric_evidence(artifact_root, metrics)
    return evidence.score if evidence else None


def metric_search_description(artifact_root: Path, metrics: Sequence[str] | str) -> str:
    names = ["metrics_summary.json", "cross_val_results.json", "run.log", "log.txt", "log_fl.txt", "error_log.txt"]
    searched = [str(path) for name in names for path in sorted(artifact_root.glob(f"**/{name}"))]
    return f"metrics={normalize_metric_order(metrics)!r}; searched={searched or [str(artifact_root)]!r}"


def is_sandbox_socket_failure(output: str) -> bool:
    text = output.lower()
    return (
        "permissionerror" in text
        and ("operation not permitted" in text or "[errno 1]" in text)
        and ("socket" in text or "sock" in text or "get_open_ports" in text or ".bind(" in text)
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
    else:
        dest.mkdir(parents=True, exist_ok=True)
    if log_path.resolve() != run_log.resolve():
        shutil.copy2(log_path, run_log)
    return dest


def job_help(python: str, job: Path, cwd: Path, timeout: int = DEFAULT_JOB_HELP_TIMEOUT) -> str:
    del python, cwd, timeout
    try:
        tree = ast.parse(job.read_text(encoding="utf-8"), filename=str(job))
    except (OSError, SyntaxError) as e:
        raise RuntimeError(f"cannot inspect job CLI flags in {job}: {e}") from e
    flags = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                if argument.value.startswith("--"):
                    flags.add(argument.value)
    return " ".join(sorted(flags))


def supported_long_flags(help_text: str) -> set[str]:
    return set(re.findall(r"(?<![\w-])(--[A-Za-z][A-Za-z0-9_-]*)", help_text))


def supports_flag(help_text: str, flag: str) -> bool:
    return flag in supported_long_flags(help_text)


def mutable_arg_specs(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    specs = schema.get("mutable_args")
    return specs if isinstance(specs, dict) else {}


def candidate_arg_values(candidate_args: Sequence[str]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    idx = 0
    while idx < len(candidate_args):
        raw = candidate_args[idx]
        if raw.startswith("--") and not re.match(r"^--[A-Za-z][A-Za-z0-9_-]*(?:=.*)?$", raw):
            raise ValueError(f"invalid canonical long option: {raw!r}")
        if (
            raw.startswith("-")
            and not raw.startswith("--")
            and not re.match(r"^-(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$", raw)
        ):
            raise ValueError(f"candidate run arguments require canonical long options, not {raw!r}")
        if not raw.startswith("--"):
            idx += 1
            continue
        option = raw[2:]
        if "=" in option:
            name, value = option.split("=", 1)
            values[name.replace("-", "_")] = value
            idx += 1
            continue
        name = option.replace("-", "_")
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


def candidate_preserves_fixed_args(
    candidate_args: Sequence[str], config: Dict[str, Any], schema: Dict[str, Any]
) -> Tuple[bool, str]:
    values = candidate_arg_values(candidate_args)
    fixed_names = set(fixed_within_campaign(schema))
    fixed_budget = config.get("budget", {}).get("fixed_training_budget", {}) or {}
    fixed_names.update(FIXED_BUDGET_TO_CLI[field] for field in fixed_budget if field in FIXED_BUDGET_TO_CLI)
    normalized_fixed_names = {name.replace("-", "_") for name in fixed_names}
    changed = sorted(normalized_fixed_names.intersection(values))
    if changed:
        return False, f"candidate run arguments change fixed-budget fields: {', '.join(changed)}"
    for raw in candidate_args:
        if not raw.startswith("--"):
            continue
        supplied = raw[2:].split("=", 1)[0].replace("-", "_")
        abbreviated = sorted(name for name in normalized_fixed_names if name.startswith(supplied) and name != supplied)
        if abbreviated:
            return False, (
                f"candidate run argument {raw!r} is a strict prefix of fixed-budget option(s): "
                f"{', '.join('--' + name for name in abbreviated)}"
            )
    return True, ""


def load_mutation_schema(cwd: Path) -> Dict[str, Any]:
    path = cwd / "mutation_schema.yaml"
    if not path.exists():
        return {}
    return read_yaml(path)


def apply_mutation_schema_contract(config: Dict[str, Any], schema: Dict[str, Any], workspace: Path) -> Dict[str, Any]:
    preferred_targets = schema.get("preferred_targets")
    if preferred_targets is None:
        return config
    if not isinstance(preferred_targets, list) or not all(isinstance(value, str) for value in preferred_targets):
        raise ValueError("mutation_schema.yaml preferred_targets must be a list of paths")

    trust_contract = config.setdefault("trust_contract", {})
    trust_paths = trust_contract.setdefault("allowed_edit_paths", [])
    if not isinstance(trust_paths, list):
        raise ValueError("autofl.yaml trust_contract.allowed_edit_paths must be a list")

    resolved_targets = []
    unresolved_targets = []
    for target in preferred_targets:
        try:
            relative = safe_relative_path(workspace, target)
        except ValueError as e:
            unresolved_targets.append(_schema_target_issue(target, str(e)))
            continue
        path = workspace / relative
        if path.is_symlink():
            unresolved_targets.append(_schema_target_issue(target, "preferred target is a symlink"))
            continue
        if not path.is_file():
            unresolved_targets.append(
                _schema_target_issue(target, "preferred target is not an existing file under the job workspace")
            )
            continue
        resolved_targets.append(relative)
        if relative not in trust_paths:
            trust_paths.append(relative)

    trust_contract["preferred_targets"] = list(resolved_targets)
    if unresolved_targets:
        unresolved = config.setdefault("unresolved", [])
        trust_unresolved = trust_contract.setdefault("unresolved", [])
        if not isinstance(unresolved, list) or not isinstance(trust_unresolved, list):
            raise ValueError("autofl.yaml unresolved fields must be lists")
        unresolved.extend(unresolved_targets)
        trust_unresolved.extend(dict(item) for item in unresolved_targets)
    return config


def _schema_target_issue(target: str, reason: str) -> Dict[str, str]:
    return {"field": "mutation_schema.preferred_targets", "reason": f"{target}: {reason}"}


def comparison_budget(schema: Dict[str, Any]) -> Dict[str, Any]:
    comparison = schema.get("comparison_budget_args")
    if not isinstance(comparison, dict):
        return {}
    budget = comparison.get("default_candidate_budget")
    return budget if isinstance(budget, dict) else {}


def fixed_within_campaign(schema: Dict[str, Any]) -> set:
    values = []
    comparison = schema.get("comparison_budget_args")
    if isinstance(comparison, dict):
        values = comparison.get("fixed_within_campaign") or []
    return set(values) if isinstance(values, list) else set()


def build_comparison_budget_args(schema: Dict[str, Any], help_text: str) -> List[str]:
    budget = comparison_budget(schema)
    args: List[str] = []
    for budget_field, cli_name in COMPARISON_BUDGET_TO_CLI.items():
        value = budget.get(budget_field)
        if value is not None and supports_flag(help_text, f"--{cli_name}"):
            args.extend([f"--{cli_name}", str(value)])
    if budget.get("cross_site_eval") and supports_flag(help_text, "--cross_site_eval"):
        args.append("--cross_site_eval")
    return args


def build_fixed_args(config: Dict[str, Any], help_text: str, schema: Dict[str, Any]) -> List[str]:
    fixed = config.get("budget", {}).get("fixed_training_budget", {}) or {}
    budget = comparison_budget(schema)
    budget_cli_names = {
        cli_name for budget_field, cli_name in COMPARISON_BUDGET_TO_CLI.items() if budget.get(budget_field) is not None
    }
    args: List[str] = []
    for budget_field, cli_name in FIXED_BUDGET_TO_CLI.items():
        if cli_name in budget_cli_names:
            continue
        value = fixed.get(budget_field)
        if value is not None and supports_flag(help_text, f"--{cli_name}"):
            args.extend([f"--{cli_name}", str(value)])
    return args


def build_base_args(args: argparse.Namespace, help_text: str, schema: Dict[str, Any]) -> List[str]:
    base = shlex.split(args.base_args)
    budget_args = build_comparison_budget_args(schema, help_text)
    if budget_args:
        base.extend(budget_args)
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


def imported_job_names(config: Dict[str, Any]) -> List[str]:
    names = []
    job_config = config.get("job", {})
    if not isinstance(job_config, dict):
        return names
    for key in ("recipe_args", "fed_job_args"):
        arguments = job_config.get(key)
        if not isinstance(arguments, dict):
            continue
        name = arguments.get("name")
        value = name.get("value") if isinstance(name, dict) else name
        if isinstance(name, dict) and name.get("confidence") != "high":
            continue
        if isinstance(value, str) and value and value not in names:
            names.append(value)
    discovered_name = config.get("artifacts", {}).get("simulator_result_name")
    if isinstance(discovered_name, str) and discovered_name and discovered_name not in names:
        names.append(discovered_name)
    return names


def simulator_workspace_root(config: Dict[str, Any], cwd: Path) -> Path:
    discovered = config.get("environment", {}).get("discovered", {})
    if isinstance(discovered, dict):
        arguments = discovered.get("args", {})
        workspace_arg = arguments.get("workspace_root") if isinstance(arguments, dict) else None
        if isinstance(workspace_arg, dict):
            value = workspace_arg.get("value")
            if workspace_arg.get("confidence") == "high" and isinstance(value, str) and value:
                root = Path(value).expanduser()
                return (root if root.is_absolute() else cwd / root).resolve()
    return (Path(tempfile.gettempdir()) / "nvflare" / "simulation").resolve()


def simulator_run_name(candidate_name: str, cwd: Path) -> str:
    campaign_namespace = sha256_bytes(str(cwd.resolve()).encode("utf-8"))[:12]
    return f"autofl_{campaign_namespace}_{candidate_name}"


def expected_simulator_roots(
    config: Dict[str, Any],
    injected_name: Optional[str],
    cwd: Path,
    simulator_base: Optional[Path] = None,
) -> List[Path]:
    names = ([injected_name] if injected_name else []) + imported_job_names(config)
    roots = []
    simulator_base = (simulator_base or simulator_workspace_root(config, cwd)).resolve()
    for name in names:
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"unsafe simulator job name: {name!r}")
        root = (simulator_base / name).resolve()
        if root.parent != simulator_base:
            raise ValueError(f"unsafe simulator result root: {root}")
        if root not in roots:
            roots.append(root)
    return roots


@contextmanager
def locked_simulator_roots(roots: Sequence[Path]) -> Iterator[None]:
    lock_paths = []
    try:
        for root in sorted(set(roots), key=str):
            root.parent.mkdir(parents=True, exist_ok=True)
            lock_path = root.parent / f".{root.name}.autofl.lock"
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError as e:
                raise RuntimeError(
                    f"simulator result root is already in use: {root}; "
                    f"wait for the other campaign or remove the stale lock {lock_path}"
                ) from e
            lock_paths.append(lock_path)
            with os.fdopen(descriptor, "w", encoding="utf-8") as lock_file:
                lock_file.write(json.dumps({"pid": os.getpid(), "workspace": str(Path.cwd().resolve())}) + "\n")
        yield
    finally:
        for lock_path in reversed(lock_paths):
            lock_path.unlink(missing_ok=True)


@contextmanager
def locked_simulator_workspace(simulator_base: Path, *, exclusive: bool) -> Iterator[None]:
    """Coordinate unknown result roots with named jobs in the same simulator workspace."""

    if fcntl is None:
        yield
        return
    simulator_base.mkdir(parents=True, exist_ok=True)
    lock_path = simulator_base / ".autofl-workspace.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
        try:
            fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
        except BlockingIOError as e:
            mode = "unnamed" if exclusive else "named"
            raise RuntimeError(
                f"simulator workspace is already in use by a conflicting {mode} campaign: {simulator_base}"
            ) from e
        yield
    finally:
        os.close(descriptor)


def validated_printed_simulator_root(result_dir: Optional[Path], simulator_base: Path) -> Optional[Path]:
    if result_dir is None or not result_dir.exists():
        return None
    resolved = result_dir.resolve()
    if resolved.parent != simulator_base.resolve():
        return None
    return resolved


def simulator_root_snapshot(simulator_base: Path) -> Dict[Path, int]:
    snapshot = {}
    if not simulator_base.exists():
        return snapshot
    for path in simulator_base.iterdir():
        if path.name.startswith(".") or path.is_symlink() or not path.is_dir():
            continue
        snapshot[path.resolve()] = path.stat().st_mtime_ns
    return snapshot


def changed_simulator_roots(simulator_base: Path, before: Dict[Path, int]) -> List[Path]:
    after = simulator_root_snapshot(simulator_base)
    return sorted(path for path, modified in after.items() if before.get(path) != modified)


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
    metrics: Sequence[str],
    config: Dict[str, Any],
) -> RunRecord:
    baseline_run = run_def.status == "baseline"
    log_path = output_root / run_def.name / "run.log"
    run_name = simulator_run_name(run_def.name, cwd)
    name_args = ["--name", run_name] if supports_flag(help_text, "--name") else []
    command = [python, str(job), *fixed_args, *base_args, *name_args, *run_def.args]
    run_def.command = command
    with tempfile.TemporaryDirectory(prefix="nvflare-autofl-sim-") as trial_workspace:
        simulator_base = Path(trial_workspace).resolve()
        simulator_roots = expected_simulator_roots(
            config, run_name if name_args else None, cwd, simulator_base=simulator_base
        )
        run_env = os.environ.copy()
        run_env[SIMULATOR_WORKSPACE_ROOT_ENV_VAR] = str(simulator_base)
        with locked_simulator_workspace(simulator_base, exclusive=not simulator_roots):
            with locked_simulator_roots(simulator_roots):
                unnamed_root_snapshot = simulator_root_snapshot(simulator_base) if not simulator_roots else {}
                rc, stdout, runtime = run(
                    command,
                    cwd,
                    timeout,
                    log_path,
                    env=run_env,
                    simulator_stall_roots=simulator_roots,
                    simulator_no_progress_timeout=simulator_no_progress_timeout,
                )
                run_def.runtime_seconds = runtime
                printed_result_dir = extract_result_dir(stdout, cwd)
                existing_roots = [root.resolve() for root in simulator_roots if root.exists()]
                result_dir = printed_result_dir if printed_result_dir in existing_roots else None
                injected_root = (simulator_base / run_name).resolve() if name_args else None
                if result_dir is None and injected_root in existing_roots:
                    result_dir = injected_root
                if result_dir is None and len(existing_roots) == 1:
                    result_dir = existing_roots[0]
                if result_dir is None and not simulator_roots:
                    result_dir = validated_printed_simulator_root(printed_result_dir, simulator_base)
                    if result_dir is None:
                        changed_roots = changed_simulator_roots(simulator_base, unnamed_root_snapshot)
                        if len(changed_roots) == 1:
                            result_dir = changed_roots[0]
                    if result_dir is not None:
                        config.setdefault("artifacts", {})["simulator_result_name"] = result_dir.name
                artifact_dir = collect_artifacts(result_dir, output_root, run_def.name, log_path)
    run_def.artifacts = str(artifact_dir)

    evidence = None
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
    elif result_dir is None:
        run_def.status = "crash"
        run_def.failure_reason = (
            "job exited successfully but no deterministic NVFlare result directory was resolved; "
            "expose a literal job name, support --name, or print the direct simulator result directory"
        )
    else:
        artifact_root = artifact_dir.parent
        evidence = extract_metric_evidence(artifact_root, metrics)
        if evidence is None:
            run_def.status = "crash"
            run_def.failure_reason = f"matching metric not found; {metric_search_description(artifact_root, metrics)}"
        else:
            run_def.score = evidence.score

    record_status = "baseline" if baseline_run and run_def.status == "crash" else run_def.status
    return RunRecord(
        status=record_status,
        name=run_def.name,
        score=run_def.score,
        runtime_seconds=run_def.runtime_seconds,
        changed_files="none",
        diff_summary=run_def.description,
        run_command=shlex.join(command),
        artifacts=run_def.artifacts,
        failure_reason=run_def.failure_reason,
        metric_name=evidence.metric_name if evidence else "",
        metric_source=evidence.source if evidence else "",
        metric_artifact=evidence.artifact if evidence else "",
    )


def write_results(path: Path, records: List[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS, delimiter="\t")
    writer.writeheader()
    for record in records:
        score = parse_finite_metric_value(record.score) if record.score is not None else None
        if record.score is not None and score is None:
            raise ValueError(f"record {record.name!r} has a non-finite score")
        writer.writerow(
            {
                "status": record.status,
                "name": record.name,
                "score": "" if score is None else f"{score:.6f}",
                "runtime_seconds": f"{record.runtime_seconds:.3f}",
                "changed_files": record.changed_files,
                "diff_summary": record.diff_summary,
                "run_command": record.run_command,
                "artifacts": record.artifacts,
                "failure_reason": record.failure_reason,
                "candidate_manifest": record.candidate_manifest,
                "base_candidate": record.base_candidate,
                "patch_sha256": record.patch_sha256,
                "metric_name": record.metric_name,
                "metric_source": record.metric_source,
                "metric_artifact": record.metric_artifact,
                "candidate_kind": record.candidate_kind,
                "algorithm_family": record.algorithm_family,
                "literature_event_id": record.literature_event_id,
            }
        )
    atomic_write_bytes(path, stream.getvalue().encode("utf-8"))


def load_results(path: Path) -> List[RunRecord]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            score_text = row.get("score", "")
            records.append(
                RunRecord(
                    status=row.get("status", ""),
                    name=row.get("name", ""),
                    score=parse_finite_metric_value(score_text) if score_text else None,
                    runtime_seconds=float(row.get("runtime_seconds") or 0.0),
                    changed_files=row.get("changed_files", ""),
                    diff_summary=row.get("diff_summary", ""),
                    run_command=row.get("run_command", ""),
                    artifacts=row.get("artifacts", ""),
                    failure_reason=row.get("failure_reason", ""),
                    candidate_manifest=row.get("candidate_manifest", ""),
                    base_candidate=row.get("base_candidate", ""),
                    patch_sha256=row.get("patch_sha256", ""),
                    metric_name=row.get("metric_name", ""),
                    metric_source=row.get("metric_source", ""),
                    metric_artifact=row.get("metric_artifact", ""),
                    candidate_kind=row.get("candidate_kind", "") or "",
                    algorithm_family=row.get("algorithm_family", "") or "",
                    literature_event_id=row.get("literature_event_id", "") or "",
                )
            )
    return records


def better(new_score: Optional[float], old_score: Optional[float], mode: str) -> bool:
    return load_campaign_guard().better(new_score, old_score, mode)


def write_state(
    path: Path,
    results_path: Path,
    records: List[RunRecord],
    max_candidates: Optional[int],
    *,
    mode: str = "max",
    stop_files: Optional[List[str]] = None,
    plateau_threshold: Optional[int] = None,
    plateau_min_delta: Optional[float] = None,
    hard_crash_threshold: Optional[int] = None,
    exploration_batch_size: Optional[int] = None,
    family_repeat_limit: Optional[int] = None,
    pending_manifest_count: int = 0,
    persist: bool = True,
) -> Dict[str, Any]:
    guard = load_campaign_guard()
    plateau_threshold = guard.DEFAULT_PLATEAU_THRESHOLD if plateau_threshold is None else plateau_threshold
    plateau_min_delta = guard.DEFAULT_MIN_DELTA if plateau_min_delta is None else plateau_min_delta
    hard_crash_threshold = guard.DEFAULT_HARD_CRASH_THRESHOLD if hard_crash_threshold is None else hard_crash_threshold
    exploration_batch_size = (
        guard.DEFAULT_EXPLORATION_BATCH_SIZE if exploration_batch_size is None else exploration_batch_size
    )
    family_repeat_limit = guard.DEFAULT_FAMILY_REPEAT_LIMIT if family_repeat_limit is None else family_repeat_limit
    state = guard.guard_state(
        results_path,
        max_candidates=max_candidates,
        stop_files=stop_files,
        plateau_threshold=plateau_threshold,
        min_delta=plateau_min_delta,
        hard_crash_threshold=hard_crash_threshold,
        mode=mode,
        pending_manifest_count=pending_manifest_count,
        exploration_batch_size=exploration_batch_size,
        family_repeat_limit=family_repeat_limit,
    )
    if records and records[-1].status == INFRASTRUCTURE_RETRY:
        attempts = len(
            [
                record
                for record in records
                if record.status in {"candidate", "keep", "discard", "crash"} and not is_baseline_record(record)
            ]
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
    if persist:
        write_json(path, state)
    return state


def write_state_if_changed(path: Path, state: Dict[str, Any]) -> bool:
    previous = read_json(path) if path.exists() else {}
    previous_comparable = dict(previous)
    state_comparable = dict(state)
    previous_comparable.pop("updated_at", None)
    state_comparable.pop("updated_at", None)
    if previous_comparable == state_comparable:
        if previous.get("updated_at"):
            state["updated_at"] = previous["updated_at"]
        return False
    write_json(path, state)
    return True


def load_progress_plotter():
    return load_sibling_module("plot_progress.py", "nvflare_autofl_plot_progress")


def write_progress_fallback(path: Path, records: List[RunRecord], mode: str, metric_label: str) -> None:
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
    draw.text(
        (margin, 24),
        f"Auto-FL Progress ({metric_label}): {len(records)} rows, {len(scores)} scored",
        fill="black",
        font=font,
    )
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


def write_progress(path: Path, records: List[RunRecord], mode: str, metric_label: str) -> None:
    plotter = load_progress_plotter()
    try:
        plotter.plot_progress(records, path, mode, metric_label)
    except (plotter.NoScoredResultsError, plotter.PlotDependencyError):
        write_progress_fallback(path, records, mode, metric_label)


def write_report(path: Path, config: Dict[str, Any], records: List[RunRecord], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best = None
    for record in records:
        if record.status in {"baseline", "keep"} and better(record.score, best.score if best else None, args.mode):
            best = record
    candidate_budget = (
        str(args.max_candidates) if args.max_candidates is not None else "uncapped; runs until manual interruption"
    )
    objective = objective_contract(config, args.metric)
    lines = [
        "# Auto-FL Report",
        "",
        f"Objective: optimize `{objective['optimization_metric']}` in `{args.target_env}`.",
        f"Requested metric: `{objective['requested_metric']}`.",
        f"Metric source: `{metric_source(config)}`.",
        f"Metric extraction order: `{', '.join(objective['metric_extraction_order'])}`.",
        f"Candidate budget: `{candidate_budget}`.",
        f"Config: `{args.autofl_yaml}`.",
        f"Fixed budget: `{json.dumps(config.get('budget', {}).get('fixed_training_budget', {}), sort_keys=True)}`.",
        "",
        "## Leaderboard",
        "",
        "| Status | Name | Score | Changed files | Manifest | Artifacts | Notes |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for record in records:
        score = "" if record.score is None else f"{record.score:.6f}"
        lines.append(
            f"| {record.status} | {record.name} | {score} | `{record.changed_files}` | "
            f"`{record.candidate_manifest}` | `{record.artifacts}` | {record.diff_summary} |"
        )
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
        lines.append(f"Best retained run: `{best.name}` with `{objective['optimization_metric']}={best.score:.6f}`.")
    else:
        lines.append("No scored run was retained.")
    atomic_write_text(path, "\n".join(lines) + "\n")


def candidate_attempts(records: List[RunRecord]) -> int:
    return len(
        [
            record
            for record in records
            if record.status in {"candidate", "keep", "discard", "crash"} and not is_baseline_record(record)
        ]
    )


def is_baseline_record(record: RunRecord) -> bool:
    return record.status == "baseline"


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
        for key in [
            "decision",
            "reason",
            "next_action",
            "final_response_allowed",
            "candidate_cap",
            "candidate_cap_source",
            "agent_instruction",
            "required_exploration",
        ]:
            if key in state_payload:
                payload[key] = state_payload[key]
    return payload


def campaign_paths(args: argparse.Namespace, job: Path) -> Dict[str, Path]:
    workspace = job.parent
    return {
        "workspace": workspace,
        "autofl_yaml": resolve_output_path(workspace, args.autofl_yaml),
        "results": resolve_output_path(workspace, args.results),
        "state": resolve_output_path(workspace, args.state),
        "progress": resolve_output_path(workspace, args.progress),
        "report": resolve_output_path(workspace, args.report),
        "output_root": resolve_output_path(workspace, args.output_root),
        "snapshot_root": workspace / BEST_SNAPSHOT_ROOT,
    }


CAMPAIGN_SETTING_NAMES = (
    "metric",
    "mode",
    "target_env",
    "max_candidates",
    "autofl_yaml",
    "results",
    "state",
    "progress",
    "report",
    "output_root",
    "plateau_threshold",
    "plateau_min_delta",
    "hard_crash_threshold",
    "exploration_batch_size",
    "family_repeat_limit",
    "stop_file",
    "base_args",
    "timeout",
    "simulator_no_progress_timeout",
    "python",
    "prefer_synthetic",
    "synthetic_train_size",
    "synthetic_test_size",
)

MUTABLE_CAMPAIGN_SETTING_NAMES = {
    "max_candidates",
    "plateau_threshold",
    "plateau_min_delta",
    "hard_crash_threshold",
    "exploration_batch_size",
    "family_repeat_limit",
    "stop_file",
    "timeout",
    "simulator_no_progress_timeout",
}


def campaign_settings(args: argparse.Namespace) -> Dict[str, Any]:
    return {name: getattr(args, name) for name in CAMPAIGN_SETTING_NAMES}


def restore_campaign_settings(args: argparse.Namespace, metadata: Dict[str, Any]) -> None:
    settings = metadata.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("campaign metadata is missing settings")
    explicit = getattr(args, "_explicit_settings", set())
    changed = False
    for name in CAMPAIGN_SETTING_NAMES:
        if name not in settings:
            continue
        if name in explicit:
            requested = getattr(args, name)
            if name in MUTABLE_CAMPAIGN_SETTING_NAMES:
                if settings.get(name) != requested:
                    settings[name] = requested
                    changed = True
                continue
            if requested != settings[name]:
                raise ValueError(
                    f"campaign setting {name} is immutable after initialization: "
                    f"configured={settings[name]!r}, requested={requested!r}"
                )
        setattr(args, name, settings[name])
    if changed:
        metadata["updated_at"] = utc_now()
        workspace_value = metadata.get("workspace_root")
        if not isinstance(workspace_value, str) or not workspace_value:
            raise ValueError("campaign metadata is missing workspace_root")
        workspace = Path(workspace_value)
        write_json(campaign_metadata_path(workspace), metadata)


def campaign_timeout(args: argparse.Namespace, schema: Dict[str, Any]) -> Tuple[int, int]:
    budget = comparison_budget(schema)
    budget_timeout = parse_timeout_setting(budget.get("run_timeout_seconds"), "run_timeout_seconds")
    timeout = max(args.timeout, budget_timeout) if budget_timeout is not None else args.timeout
    budget_no_progress_timeout = parse_timeout_setting(
        budget.get("simulator_no_progress_timeout_seconds"), "simulator_no_progress_timeout_seconds"
    )
    no_progress_timeout = (
        budget_no_progress_timeout if budget_no_progress_timeout is not None else args.simulator_no_progress_timeout
    )
    return timeout, no_progress_timeout


def parse_timeout_setting(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be a non-negative integer") from e
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def import_job_config(
    args: argparse.Namespace,
    job: Path,
    output: Path,
    log_path: Path,
    timeout: int,
) -> Dict[str, Any]:
    del timeout
    importer = load_job_importer()
    config = importer.import_job_to_autofl_config(
        str(job),
        workspace_root=str(job.parent),
        metric=args.metric,
        mode=args.mode,
        target_env=args.target_env,
        max_candidates=args.max_candidates,
        job_args=shlex.split(args.base_args),
    )
    atomic_write_bytes(output, importer.dump_autofl_yaml(config).encode("utf-8"))
    atomic_write_bytes(
        log_path,
        (
            json.dumps(
                {
                    "importer": str(Path(importer.__file__).resolve()),
                    "job": str(job.resolve()),
                    "output": str(output.resolve()),
                    "source_sha256": config.get("import", {}).get("source_sha256"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )
    return config


def campaign_admission_errors(config: Dict[str, Any]) -> List[str]:
    errors = []
    support = config.get("import", {}).get("support", {})
    if not isinstance(support, dict) or support.get("status") != "supported":
        reason = support.get("reason") if isinstance(support, dict) else None
        errors.append(f"job surface is not deterministically supported{f': {reason}' if reason else ''}")
    fixed_budget = config.get("budget", {}).get("fixed_training_budget")
    if not isinstance(fixed_budget, dict) or not fixed_budget:
        errors.append("fixed comparison budget is unresolved")
    unresolved = config.get("unresolved", [])
    if isinstance(unresolved, list):
        critical_fields = []
        for item in unresolved:
            field = item.get("field", "") if isinstance(item, dict) else ""
            if field.startswith("budget.fixed_training_budget"):
                critical_fields.append(field)
        if critical_fields:
            errors.append(f"safety-critical fields are unresolved: {', '.join(sorted(set(critical_fields)))}")
    return errors


def best_retained_record(records: Sequence[RunRecord], mode: str) -> Optional[RunRecord]:
    best = None
    for record in records:
        if record.status in {"baseline", "keep"} and better(record.score, best.score if best else None, mode):
            best = record
    return best


def refresh_campaign_artifacts(
    args: argparse.Namespace,
    paths: Dict[str, Path],
    config: Dict[str, Any],
    records: List[RunRecord],
    metadata: Dict[str, Any],
    *,
    pending_manifest: Optional[Path] = None,
    next_action: Optional[str] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    pending_manifests = pending_candidate_manifests(paths["workspace"])
    if pending_manifest is not None and pending_manifest not in pending_manifests:
        pending_manifests.append(pending_manifest)
    write_results(paths["results"], records)
    state = write_state(
        paths["state"],
        paths["results"],
        records,
        args.max_candidates,
        mode=args.mode,
        stop_files=resolve_stop_files(paths["workspace"], args.stop_file),
        plateau_threshold=args.plateau_threshold,
        plateau_min_delta=args.plateau_min_delta,
        hard_crash_threshold=args.hard_crash_threshold,
        exploration_batch_size=args.exploration_batch_size,
        family_repeat_limit=args.family_repeat_limit,
        pending_manifest_count=len(pending_manifests),
    )
    if state.get("next_action") == "abandon_candidate":
        next_action = "abandon_candidate"
        reason = state.get("reason")
    elif pending_manifests and next_action is None:
        pending_status = read_json(pending_manifests[0]).get("status")
        next_action = "submit_candidate" if pending_status == "ready_for_external_execution" else "edit_candidate"
        reason = "pending_candidates"
    if not state.get("final_response_allowed") and next_action:
        state["next_action"] = next_action
        state["reason"] = reason or next_action
        state["agent_instruction"] = f"Do not produce a final answer. Execute `{next_action}` for this campaign."
    state.update(
        {
            "best_candidate": metadata.get("best_candidate"),
            "best_source_sha256": metadata.get("best_source_sha256"),
            "pending_candidate_manifest": str(pending_manifests[0].resolve()) if pending_manifests else None,
        }
    )
    write_json(paths["state"], state)
    write_progress(paths["progress"], records, args.mode, optimization_metric(config, args.metric))
    write_report(paths["report"], config, records, args)
    return state


def print_campaign_result(
    paths: Dict[str, Path], records: List[RunRecord], state: Dict[str, Any], **extra: Any
) -> None:
    payload = campaign_summary(
        paths["autofl_yaml"],
        paths["results"],
        paths["state"],
        paths["progress"],
        paths["report"],
        records,
        state,
    )
    payload.update(extra)
    print(json.dumps(payload, indent=2, sort_keys=True))


def execute_sim_baseline(
    args: argparse.Namespace,
    job: Path,
    paths: Dict[str, Path],
    config: Dict[str, Any],
    schema: Dict[str, Any],
    name: str = "baseline",
) -> RunRecord:
    timeout, no_progress_timeout = campaign_timeout(args, schema)
    help_text = job_help(args.python, job, job.parent)
    return run_job(
        JobRun(name=name, args=[], description="baseline", status="baseline"),
        python=args.python,
        job=job,
        cwd=job.parent,
        help_text=help_text,
        fixed_args=build_fixed_args(config, help_text, schema),
        base_args=build_base_args(args, help_text, schema),
        output_root=paths["output_root"],
        timeout=timeout,
        simulator_no_progress_timeout=no_progress_timeout,
        metrics=metric_extraction_order(config, args.metric),
        config=config,
    )


def initialize_campaign(args: argparse.Namespace, job: Path) -> int:
    workspace = job.parent
    metadata_path = campaign_metadata_path(workspace)
    if metadata_path.exists():
        metadata = load_campaign_metadata(workspace, job)
        restore_campaign_settings(args, metadata)
        paths = campaign_paths(args, job)
        records = load_results(paths["results"])
        if args.target_env == "sim" and not any(
            record.status == "baseline" and record.score is not None for record in records
        ):
            config = read_yaml(paths["autofl_yaml"])
            retry_number = sum(is_baseline_record(record) for record in records) + 1
            baseline = execute_sim_baseline(
                args,
                job,
                paths,
                config,
                load_mutation_schema(workspace),
                name=f"baseline_retry_{retry_number}",
            )
            write_yaml(paths["autofl_yaml"], config)
            records.append(baseline)
            metadata["best_score"] = baseline.score
            if baseline.score is not None:
                metadata["best_candidate"] = baseline.name
            metadata["updated_at"] = utc_now()
            write_json(metadata_path, metadata)
            next_action = "propose_candidate" if baseline.score is not None else "repair_baseline"
            state = refresh_campaign_artifacts(
                args, paths, config, records, metadata, next_action=next_action, reason="baseline_retried"
            )
            print_campaign_result(paths, records, state, initialized=False, baseline_retried=True)
            if baseline.status == INFRASTRUCTURE_RETRY:
                return 75
            return 0 if baseline.score is not None else 1
        print_campaign_result(paths, records, read_json(paths["state"]), initialized=False)
        return 0

    paths = campaign_paths(args, job)
    paths["output_root"].mkdir(parents=True, exist_ok=True)
    schema = load_mutation_schema(workspace)
    timeout, _ = campaign_timeout(args, schema)
    config = import_job_config(args, job, paths["autofl_yaml"], paths["output_root"] / "import.log", timeout)
    config = apply_metric_contract(config, args.metric, schema)
    args.metric = str(config.get("objective", {}).get("requested_metric") or config["objective"]["metric"])
    config = apply_mutation_schema_contract(config, schema, workspace)
    config.setdefault("trust_contract", {})["allowed_create_patterns"] = list(ALLOWED_CREATE_PATTERNS)
    write_yaml(paths["autofl_yaml"], config)
    admission_errors = campaign_admission_errors(config)
    if admission_errors:
        raise ValueError(
            f"autofl.yaml was generated at {paths['autofl_yaml']}, but baseline execution is unsafe: "
            f"{'; '.join(admission_errors)}. Resolve these fields and initialize again."
        )
    snapshot_files = create_best_snapshot(workspace, config, paths["snapshot_root"])
    metadata = {
        "schema_version": CAMPAIGN_METADATA_SCHEMA_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "job": str(job.resolve()),
        "workspace_root": str(workspace.resolve()),
        "settings": campaign_settings(args),
        "best_candidate": "baseline",
        "best_score": None,
        "best_source_sha256": source_hash(snapshot_files),
        "fixed_budget_sha256": fixed_budget_hash(config),
    }
    write_json(metadata_path, metadata)

    records: List[RunRecord] = []
    if args.target_env == "sim":
        baseline = execute_sim_baseline(args, job, paths, config, schema)
        write_yaml(paths["autofl_yaml"], config)
        records.append(baseline)
        metadata["best_score"] = baseline.score
        metadata["updated_at"] = utc_now()
        write_json(metadata_path, metadata)
        next_action = "propose_candidate" if baseline.score is not None else "repair_baseline"
    else:
        next_action = "submit_baseline"

    state = refresh_campaign_artifacts(
        args, paths, config, records, metadata, next_action=next_action, reason="campaign_initialized"
    )
    print_campaign_result(paths, records, state, initialized=True)
    if records and records[0].status == INFRASTRUCTURE_RETRY:
        return 75
    if args.target_env == "sim" and (not records or records[0].score is None):
        return 1
    return 0


def pending_candidate_manifests(workspace: Path) -> List[Path]:
    root = workspace / CANDIDATE_ROOT
    pending = []
    if not root.exists():
        return pending
    for path in sorted(root.glob("*/candidate_manifest.json")):
        manifest = read_json(path)
        validate_candidate_manifest_identity(path, manifest)
        status = manifest.get("status")
        if status in {"prepared", "ready_for_external_execution"}:
            pending.append(path)
    return pending


def prepare_candidate(args: argparse.Namespace, job: Path) -> int:
    workspace = job.parent
    metadata = load_campaign_metadata(workspace, job)
    restore_campaign_settings(args, metadata)
    paths = campaign_paths(args, job)
    records = load_results(paths["results"])
    if not any(record.status == "baseline" and record.score is not None for record in records):
        raise ValueError("a scored baseline is required before preparing candidates")
    state = read_json(paths["state"])
    if state.get("final_response_allowed"):
        raise ValueError(f"campaign is already final: {state.get('reason')}")
    pending = pending_candidate_manifests(workspace)
    if pending:
        raise ValueError(f"campaign already has a pending candidate: {pending[0]}")
    candidate_id = validate_candidate_id(args.name or "")
    if not args.hypothesis:
        raise ValueError("--hypothesis is required for prepare")
    algorithm_family = (args.family or "").strip().lower()
    literature_event = (args.literature_event or "").strip()
    if literature_event:
        known_events = {record.literature_event_id for record in records if record.status == "literature"}
        if literature_event not in known_events:
            raise ValueError(f"unknown literature event id: {literature_event}")
    manifest_path = candidate_manifest_path(workspace, candidate_id)
    candidate_dir = manifest_path.parent
    if candidate_dir.exists():
        raise ValueError(f"candidate already exists: {candidate_id}")

    best_source, best_files = load_best_snapshot(paths["snapshot_root"])
    if source_hash(best_files) != metadata.get("best_source_sha256"):
        raise ValueError("campaign best-source hash is stale")
    if not workspace_matches_snapshot(workspace, best_source, best_files):
        raise ValueError("job workspace differs from the recorded best candidate; reconcile edits before preparing")

    draft_source = candidate_dir / "source"
    shutil.copytree(best_source, draft_source)
    manifest = {
        "schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "name": candidate_id,
        "hypothesis": args.hypothesis,
        "algorithm_family": algorithm_family,
        "literature_event_id": literature_event,
        "status": "prepared",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "workspace_root": str(workspace.resolve()),
        "base_candidate": metadata.get("best_candidate"),
        "base_source_sha256": source_hash(best_files),
        "fixed_budget_sha256": metadata.get("fixed_budget_sha256"),
        "objective": {"metric": args.metric, "mode": args.mode},
        "environment": args.target_env,
        "run_args": shlex.split(args.run_args),
        "changed_files": [],
        "patch_sha256": "",
        "candidate_source_sha256": source_hash(best_files),
        "provenance": {
            "job": str(job.resolve()),
            "autofl_yaml": str(paths["autofl_yaml"].resolve()),
            "import_source_sha256": read_yaml(paths["autofl_yaml"]).get("import", {}).get("source_sha256"),
        },
        "artifacts": {},
        "result": {},
    }
    write_json(manifest_path, manifest)
    config = read_yaml(paths["autofl_yaml"])
    state = refresh_campaign_artifacts(
        args,
        paths,
        config,
        records,
        metadata,
        pending_manifest=manifest_path,
        next_action="edit_candidate",
        reason="candidate_prepared",
    )
    print_campaign_result(
        paths,
        records,
        state,
        candidate_manifest=str(manifest_path.resolve()),
        candidate_source=str(draft_source.resolve()),
    )
    return 0


def validate_candidate_for_evaluation(
    args: argparse.Namespace,
    job: Path,
    metadata: Dict[str, Any],
    paths: Dict[str, Path],
    manifest_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Path, Dict[str, str], List[str], List[str], str]:
    manifest = load_candidate_manifest(manifest_path)
    if Path(str(manifest.get("workspace_root") or "")).resolve() != job.parent.resolve():
        raise ValueError("candidate manifest belongs to a different job workspace")
    config = read_yaml(paths["autofl_yaml"])
    best_source, best_files = load_best_snapshot(paths["snapshot_root"])
    best_hash = source_hash(best_files)
    if manifest.get("base_source_sha256") != best_hash or metadata.get("best_source_sha256") != best_hash:
        raise ValueError("candidate was prepared from a stale best candidate")
    if manifest.get("fixed_budget_sha256") != metadata.get("fixed_budget_sha256"):
        raise ValueError("candidate fixed-budget provenance is stale")
    if not workspace_matches_snapshot(job.parent, best_source, best_files):
        raise ValueError("job workspace differs from the recorded best candidate")
    draft_source = manifest_path.parent / "source"
    changed, created = candidate_changes(job.parent, config, best_source, best_files, draft_source)
    recorded_changed = validated_manifest_paths(manifest, job.parent, "changed_files")
    recorded_created = validated_manifest_paths(manifest, job.parent, "created_files")
    if (recorded_changed or recorded_created) and (recorded_changed != changed or recorded_created != created):
        raise ValueError("candidate manifest source lists do not match the deterministic candidate diff")
    run_args = manifest.get("run_args")
    if not isinstance(run_args, list) or not all(isinstance(item, str) for item in run_args):
        raise ValueError("candidate manifest run_args must be a list of strings")
    allowed, reason = candidate_args_allowed(run_args, load_mutation_schema(job.parent))
    if not allowed:
        raise ValueError(reason)
    allowed, reason = candidate_preserves_fixed_args(run_args, config, load_mutation_schema(job.parent))
    if not allowed:
        raise ValueError(reason)
    if not changed and not run_args:
        raise ValueError("candidate has no source changes or run arguments")
    if (str(manifest.get("literature_event_id") or "")).strip() and not changed and not created:
        raise ValueError("literature-linked candidates must include source edits")
    patch = render_candidate_patch(best_source, draft_source, changed)
    return manifest, config, best_source, best_files, changed, created, patch


def update_config_for_kept_sources(config: Dict[str, Any], created: Sequence[str]) -> None:
    if not created:
        return
    trust_paths = config.setdefault("trust_contract", {}).setdefault("allowed_edit_paths", [])
    for relative in created:
        if relative not in trust_paths:
            trust_paths.append(relative)


def candidate_campaign_config(
    candidate_config: Dict[str, Any], current_config: Dict[str, Any], args: argparse.Namespace, schema: Dict[str, Any]
) -> Dict[str, Any]:
    candidate_config = apply_metric_contract(candidate_config, args.metric, schema)
    candidate_config["objective"] = dict(current_config.get("objective", {}))
    current_paths = current_config.get("trust_contract", {}).get("allowed_edit_paths", []) or []
    trust_paths = candidate_config.setdefault("trust_contract", {}).setdefault("allowed_edit_paths", [])
    for path in current_paths:
        if path not in trust_paths:
            trust_paths.append(path)
    candidate_config["trust_contract"]["allowed_create_patterns"] = allowed_create_patterns(current_config)
    return candidate_config


def finalize_candidate_result(
    args: argparse.Namespace,
    job: Path,
    metadata: Dict[str, Any],
    paths: Dict[str, Path],
    config: Dict[str, Any],
    manifest_path: Path,
    manifest: Dict[str, Any],
    best_source: Path,
    best_files: Dict[str, str],
    changed: List[str],
    created: List[str],
    patch: str,
    record: RunRecord,
) -> Tuple[List[RunRecord], Dict[str, Any]]:
    rollback_files: Dict[Path, Optional[bytes]] = {}
    staged_snapshot = None
    previous_snapshot = None
    try:
        records = load_results(paths["results"])
        previous_best = best_retained_record(records, args.mode)
        if record.status == "candidate":
            record.status = (
                "keep" if better(record.score, previous_best.score if previous_best else None, args.mode) else "discard"
            )
        patch_path = manifest_path.parent / "candidate.patch"
        rollback_files = capture_file_versions(
            [
                paths["autofl_yaml"],
                manifest_path,
                campaign_metadata_path(job.parent),
                paths["results"],
                paths["state"],
                paths["progress"],
                paths["report"],
                patch_path,
            ]
        )
        atomic_write_text(patch_path, patch)
        patch_sha256 = sha256_bytes(patch.encode("utf-8"))
        record.changed_files = ",".join(changed) if changed else "none"
        record.diff_summary = str(manifest.get("hypothesis") or "candidate")
        record.candidate_manifest = str(manifest_path.resolve())
        record.base_candidate = str(manifest.get("base_candidate") or "")
        record.patch_sha256 = patch_sha256
        record.candidate_kind = "source_edit" if changed or created else "argument_only"
        record.algorithm_family = str(manifest.get("algorithm_family") or "")
        record.literature_event_id = str(manifest.get("literature_event_id") or "")

        if record.status == "keep":
            update_config_for_kept_sources(config, created)
            staged_snapshot, snapshot_files = stage_best_snapshot(job.parent, config, paths["snapshot_root"])
            write_yaml(paths["autofl_yaml"], config)
            previous_snapshot = activate_best_snapshot(paths["snapshot_root"], staged_snapshot)
            staged_snapshot = None
            metadata.update(
                {
                    "best_candidate": record.name,
                    "best_score": record.score,
                    "best_source_sha256": source_hash(snapshot_files),
                    "updated_at": utc_now(),
                }
            )
        else:
            restore_best_source(job.parent, best_source, best_files, changed, created)

        manifest.update(
            {
                "status": record.status,
                "updated_at": utc_now(),
                "changed_files": changed,
                "created_files": created,
                "candidate_kind": record.candidate_kind,
                "patch_sha256": patch_sha256,
                "artifacts": {"patch": str(patch_path.resolve()), "run": record.artifacts},
                "result": {
                    "score": record.score,
                    "metric_name": record.metric_name,
                    "metric_source": record.metric_source,
                    "metric_artifact": record.metric_artifact,
                    "runtime_seconds": record.runtime_seconds,
                    "run_command": record.run_command,
                    "failure_reason": record.failure_reason,
                },
            }
        )
        write_json(manifest_path, manifest)
        write_json(campaign_metadata_path(job.parent), metadata)
        records.append(record)
        state = refresh_campaign_artifacts(args, paths, config, records, metadata)
    except BaseException as error:
        try:
            if previous_snapshot is not None:
                rollback_best_snapshot(paths["snapshot_root"], previous_snapshot)
                previous_snapshot = None
            restore_best_source(job.parent, paths["snapshot_root"] / "source", best_files, changed, created)
            if rollback_files:
                restore_file_versions(rollback_files)
        except BaseException as rollback_error:
            raise RuntimeError(
                f"candidate finalization failed ({error}); automatic workspace rollback also failed ({rollback_error})"
            ) from rollback_error
        raise
    finally:
        if staged_snapshot is not None:
            shutil.rmtree(staged_snapshot, ignore_errors=True)

    if previous_snapshot is not None:
        shutil.rmtree(previous_snapshot, ignore_errors=True)
    return records, state


def evaluate_candidate(args: argparse.Namespace, job: Path) -> int:
    workspace = job.parent
    metadata = load_campaign_metadata(workspace, job)
    restore_campaign_settings(args, metadata)
    paths = campaign_paths(args, job)
    manifest_path = Path(args.manifest).resolve() if args.manifest else None
    if manifest_path is None:
        pending = pending_candidate_manifests(workspace)
        if len(pending) != 1:
            raise ValueError("--manifest is required when there is not exactly one pending candidate")
        manifest_path = pending[0]
    manifest, config, best_source, best_files, changed, created, patch = validate_candidate_for_evaluation(
        args, job, metadata, paths, manifest_path
    )
    patch_path = manifest_path.parent / "candidate.patch"
    atomic_write_text(patch_path, patch)
    manifest.update(
        {
            "updated_at": utc_now(),
            "changed_files": changed,
            "created_files": created,
            "patch_sha256": sha256_bytes(patch.encode("utf-8")),
            "candidate_source_sha256": source_hash(file_map(manifest_path.parent / "source")),
        }
    )
    write_json(manifest_path, manifest)
    schema = load_mutation_schema(workspace)
    timeout, no_progress_timeout = campaign_timeout(args, schema)
    candidate_config_path = manifest_path.parent / "candidate_autofl.yaml"
    try:
        apply_candidate_source(workspace, manifest_path.parent / "source", changed)
        candidate_config = import_job_config(
            args,
            job,
            candidate_config_path,
            manifest_path.parent / "import.log",
            timeout,
        )
        if fixed_budget_hash(candidate_config) != metadata.get("fixed_budget_sha256"):
            raise ValueError("candidate changes budget.fixed_training_budget")
        candidate_config = candidate_campaign_config(candidate_config, config, args, schema)
    except BaseException:
        restore_best_source(workspace, best_source, best_files, changed, created)
        raise

    if args.target_env != "sim":
        manifest["status"] = "ready_for_external_execution"
        manifest["updated_at"] = utc_now()
        write_json(manifest_path, manifest)
        records = load_results(paths["results"])
        state = refresh_campaign_artifacts(
            args,
            paths,
            config,
            records,
            metadata,
            pending_manifest=manifest_path,
            next_action="submit_candidate",
            reason="candidate_validated",
        )
        print_campaign_result(
            paths,
            records,
            state,
            candidate_manifest=str(manifest_path.resolve()),
            job=str(job.resolve()),
        )
        return 0

    try:
        help_text = job_help(args.python, job, workspace)
        run_record = run_job(
            JobRun(
                name=str(manifest["candidate_id"]),
                args=list(manifest.get("run_args") or []),
                description=str(manifest.get("hypothesis") or "candidate"),
            ),
            python=args.python,
            job=job,
            cwd=workspace,
            help_text=help_text,
            fixed_args=build_fixed_args(config, help_text, schema),
            base_args=build_base_args(args, help_text, schema),
            output_root=paths["output_root"],
            timeout=timeout,
            simulator_no_progress_timeout=no_progress_timeout,
            metrics=metric_extraction_order(config, args.metric),
            config=config,
        )
    except BaseException:
        restore_best_source(workspace, best_source, best_files, changed, created)
        raise
    if run_record.status == INFRASTRUCTURE_RETRY:
        restore_best_source(workspace, best_source, best_files, changed, created)
        manifest["status"] = "prepared"
        manifest["result"] = {"failure_reason": run_record.failure_reason}
        manifest["updated_at"] = utc_now()
        write_json(manifest_path, manifest)
        records = load_results(paths["results"])
        state = refresh_campaign_artifacts(
            args,
            paths,
            config,
            records,
            metadata,
            pending_manifest=manifest_path,
            next_action="rerun_with_escalated_execution",
            reason="infrastructure_retry",
        )
        print_campaign_result(paths, records, state, candidate_manifest=str(manifest_path.resolve()))
        return 75

    records, state = finalize_candidate_result(
        args,
        job,
        metadata,
        paths,
        candidate_config,
        manifest_path,
        manifest,
        best_source,
        best_files,
        changed,
        created,
        patch,
        run_record,
    )
    print_campaign_result(paths, records, state, candidate_manifest=str(manifest_path.resolve()))
    return 0


def abandon_candidate(args: argparse.Namespace, job: Path) -> int:
    workspace = job.parent
    metadata = load_campaign_metadata(workspace, job)
    restore_campaign_settings(args, metadata)
    paths = campaign_paths(args, job)
    manifest_path = Path(args.manifest).resolve() if args.manifest else None
    if manifest_path is None:
        pending = pending_candidate_manifests(workspace)
        if len(pending) != 1:
            raise ValueError("--manifest is required when there is not exactly one pending candidate")
        manifest_path = pending[0]
    manifest = load_candidate_manifest(manifest_path)
    best_source, best_files = load_best_snapshot(paths["snapshot_root"])
    config = read_yaml(paths["autofl_yaml"])
    draft_source = manifest_path.parent / "source"
    changed, created = candidate_changes(
        workspace,
        config,
        best_source,
        best_files,
        draft_source,
        allow_materialized=manifest.get("status") == "ready_for_external_execution",
    )
    recorded_changed = validated_manifest_paths(manifest, workspace, "changed_files")
    recorded_created = validated_manifest_paths(manifest, workspace, "created_files")
    if (recorded_changed or recorded_created) and (recorded_changed != changed or recorded_created != created):
        raise ValueError("candidate manifest source lists do not match the deterministic candidate diff")
    if manifest.get("status") == "ready_for_external_execution":
        if not workspace_matches_snapshot(workspace, draft_source, file_map(draft_source)):
            raise ValueError("materialized candidate source changed after validation")
        restore_best_source(workspace, best_source, best_files, changed, created)
    elif not workspace_matches_snapshot(workspace, best_source, best_files):
        raise ValueError("job workspace differs from the recorded best candidate")
    manifest["status"] = "abandoned"
    manifest["updated_at"] = utc_now()
    write_json(manifest_path, manifest)
    records = load_results(paths["results"])
    state = refresh_campaign_artifacts(
        args,
        paths,
        config,
        records,
        metadata,
        next_action="propose_candidate",
        reason="candidate_abandoned",
    )
    print_campaign_result(paths, records, state, candidate_manifest=str(manifest_path.resolve()))
    return 0


def suggest_candidates(args: argparse.Namespace, job: Path) -> int:
    metadata = load_campaign_metadata(job.parent, job)
    restore_campaign_settings(args, metadata)
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    config = read_yaml(campaign_paths(args, job)["autofl_yaml"])
    help_text = job_help(args.python, job, job.parent)
    suggestions = [
        {"name": candidate.name, "run_args": candidate.args, "hypothesis": candidate.description}
        for candidate in candidate_plan(
            config,
            help_text,
            args.limit,
            load_mutation_schema(job.parent),
        )
    ]
    print(json.dumps({"suggestions": suggestions}, indent=2, sort_keys=True))
    return 0


def record_external_result(args: argparse.Namespace, job: Path) -> int:
    workspace = job.parent
    metadata = load_campaign_metadata(workspace, job)
    restore_campaign_settings(args, metadata)
    paths = campaign_paths(args, job)
    config = read_yaml(paths["autofl_yaml"])
    artifact_path = Path(args.external_artifacts).resolve() if args.external_artifacts else None
    score = parse_finite_metric_value(args.score) if args.score is not None else None
    if args.score is not None and score is None:
        raise ValueError("--score must be a finite number")
    evidence = None
    if score is None and artifact_path:
        evidence = extract_metric_evidence(artifact_path, metric_extraction_order(config, args.metric))
        score = evidence.score if evidence else None
    elif score is not None:
        evidence = MetricEvidence(
            score=score,
            metric_name=optimization_metric(config, args.metric),
            source="provided",
            artifact=str(artifact_path or ""),
        )
    if args.literature:
        records = load_results(paths["results"])
        event_number = sum(record.status == "literature" for record in records) + 1
        event_id = f"lit-{event_number:04d}"
        name = f"literature_review_{event_number}"
        records.append(
            RunRecord(
                status="literature",
                name=name,
                score=None,
                runtime_seconds=0.0,
                changed_files="none",
                diff_summary=args.hypothesis or "source-backed literature review",
                run_command="agent literature review",
                artifacts=str(artifact_path or ""),
                algorithm_family=(args.family or "").strip().lower(),
                literature_event_id=event_id,
            )
        )
        # The guard now derives next_action=develop_literature_batch and keeps
        # required_exploration active until the linked exploration batch completes.
        state = refresh_campaign_artifacts(args, paths, config, records, metadata)
        print_campaign_result(paths, records, state, literature_event=name, literature_event_id=event_id)
        return 0
    if args.baseline:
        records = load_results(paths["results"])
        if any(
            record.status == "baseline"
            or (record.status in {"candidate", "keep", "discard", "crash"} and not is_baseline_record(record))
            for record in records
        ):
            raise ValueError("external baseline can only be recorded before a baseline or campaign candidate")
        if score is None:
            raise ValueError("a score or extractable --artifacts path is required for the baseline")
        record = RunRecord(
            status="baseline",
            name="baseline",
            score=score,
            runtime_seconds=0.0,
            changed_files="none",
            diff_summary="external baseline",
            run_command=f"nvflare job id={args.job_id or 'unreported'}",
            artifacts=str(artifact_path or ""),
            metric_name=evidence.metric_name if evidence else "",
            metric_source=evidence.source if evidence else "",
            metric_artifact=evidence.artifact if evidence else "",
        )
        records.append(record)
        metadata["best_score"] = score
        metadata["updated_at"] = utc_now()
        write_json(campaign_metadata_path(workspace), metadata)
        state = refresh_campaign_artifacts(
            args,
            paths,
            config,
            records,
            metadata,
            next_action="propose_candidate",
            reason="baseline_recorded",
        )
        print_campaign_result(paths, records, state, job_id=args.job_id)
        return 0

    manifest_path = Path(args.manifest).resolve() if args.manifest else None
    if manifest_path is None:
        pending = pending_candidate_manifests(workspace)
        if len(pending) != 1:
            raise ValueError("--manifest is required when there is not exactly one pending candidate")
        manifest_path = pending[0]
    manifest = load_candidate_manifest(manifest_path)
    if manifest.get("status") != "ready_for_external_execution":
        raise ValueError("external candidate must be validated before its result is recorded")
    draft_source = manifest_path.parent / "source"
    draft_files = file_map(draft_source)
    if source_hash(draft_files) != manifest.get("candidate_source_sha256") or not workspace_matches_snapshot(
        workspace, draft_source, draft_files
    ):
        raise ValueError("materialized candidate source changed after validation")
    best_source, best_files = load_best_snapshot(paths["snapshot_root"])
    changed, created = candidate_changes(
        workspace, config, best_source, best_files, draft_source, allow_materialized=True
    )
    validate_manifest_change_set(manifest, workspace, changed, created)
    patch_path = manifest_path.parent / "candidate.patch"
    patch = patch_path.read_text(encoding="utf-8") if patch_path.exists() else ""
    candidate_config_path = manifest_path.parent / "candidate_autofl.yaml"
    schema = load_mutation_schema(workspace)
    timeout, _ = campaign_timeout(args, schema)
    candidate_config = import_job_config(
        args,
        job,
        candidate_config_path,
        manifest_path.parent / "record_import.log",
        timeout,
    )
    if fixed_budget_hash(candidate_config) != metadata.get("fixed_budget_sha256"):
        raise ValueError("candidate changes budget.fixed_training_budget")
    candidate_config = candidate_campaign_config(candidate_config, config, args, schema)
    status = "crash" if args.failure_reason or score is None else "candidate"
    record = RunRecord(
        status=status,
        name=str(manifest["candidate_id"]),
        score=score,
        runtime_seconds=0.0,
        changed_files="none",
        diff_summary=str(manifest.get("hypothesis") or "candidate"),
        run_command=f"nvflare job id={args.job_id or 'unreported'}",
        artifacts=str(artifact_path or ""),
        failure_reason=args.failure_reason or ("metric not found" if score is None else ""),
        metric_name=evidence.metric_name if evidence else "",
        metric_source=evidence.source if evidence else "",
        metric_artifact=evidence.artifact if evidence else "",
    )
    records, state = finalize_candidate_result(
        args,
        job,
        metadata,
        paths,
        candidate_config,
        manifest_path,
        manifest,
        best_source,
        best_files,
        changed,
        created,
        patch,
        record,
    )
    updated_manifest = read_json(manifest_path)
    updated_manifest.setdefault("artifacts", {})["job_id"] = args.job_id
    write_json(manifest_path, updated_manifest)
    print_campaign_result(paths, records, state, candidate_manifest=str(manifest_path.resolve()), job_id=args.job_id)
    return 0


def show_campaign_status(args: argparse.Namespace, job: Path) -> int:
    metadata = load_campaign_metadata(job.parent, job)
    restore_campaign_settings(args, metadata)
    paths = campaign_paths(args, job)
    records = load_results(paths["results"])
    pending = pending_candidate_manifests(job.parent)
    state = write_state(
        paths["state"],
        paths["results"],
        records,
        args.max_candidates,
        mode=args.mode,
        stop_files=resolve_stop_files(job.parent, args.stop_file),
        plateau_threshold=args.plateau_threshold,
        plateau_min_delta=args.plateau_min_delta,
        hard_crash_threshold=args.hard_crash_threshold,
        exploration_batch_size=args.exploration_batch_size,
        family_repeat_limit=args.family_repeat_limit,
        pending_manifest_count=len(pending),
        persist=False,
    )
    state.update(
        {
            "best_candidate": metadata.get("best_candidate"),
            "best_source_sha256": metadata.get("best_source_sha256"),
            "pending_candidate_manifest": str(pending[0].resolve()) if pending else None,
        }
    )
    write_state_if_changed(paths["state"], state)
    print_campaign_result(paths, records, state, campaign=metadata)
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.max_candidates is not None and args.max_candidates < 1:
        raise ValueError("--max-candidates must be positive when provided")
    if args.plateau_threshold < 1:
        raise ValueError("--plateau-threshold must be positive")
    if args.plateau_min_delta < 0:
        raise ValueError("--plateau-min-delta must be non-negative")
    if args.hard_crash_threshold < 0:
        raise ValueError("--hard-crash-threshold must be non-negative")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        validate_args(args)
        job = Path(args.job).resolve()
        if not job.is_file():
            raise ValueError(f"job.py does not exist: {job}")
        actions = {
            "initialize": initialize_campaign,
            "prepare": prepare_candidate,
            "evaluate": evaluate_candidate,
            "abandon": abandon_candidate,
            "suggest": suggest_candidates,
            "record": record_external_result,
            "status": show_campaign_status,
        }
        return actions[args.action](args, job)
    except (OSError, RuntimeError, ValueError) as e:
        print(f"Auto-FL {args.action} failed: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
