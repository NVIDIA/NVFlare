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

"""Server-local mock-data preflight for generated NVFlare training packages."""

from __future__ import annotations

import copy
import os
import re
import shutil
import signal
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from fedready.data.contracts import training as training_contracts
from fedready.flare.training_job import (
    DEFAULT_TRAINING_TASK_NAME,
    FedAvgTrainingConfig,
    _client_runtime_training_config,
    _read_json,
    _simulator_error_snippets,
    export_fedavg_training_job,
    summarize_prepared_dataset,
    summarize_simulator_status,
)
from fedready.training_code import (
    LOCAL_TRAINING_SIMULATION_SCHEMA_VERSION,
    MOCK_TRAINING_DATASET_SCHEMA_VERSION,
    validate_training_code_spec,
)
from fedready.utils.io import atomic_write_json

PREFLIGHT_CLIENT_ID = "MOCK_CLIENT"
PREFLIGHT_TERMINAL_WATCHDOG_PATTERNS = (
    "ModuleNotFoundError",
    "ImportError:",
    "RuntimeError:",
    "ValueError:",
    "AttributeError:",
    "TypeError:",
    "Error(s) in loading state_dict",
    ": error:",
    " - ERROR - ",
)
PREFLIGHT_TERMINAL_EXCERPT_PATTERNS = (
    "Exception in thread",
    "Traceback (most recent call last):",
    *PREFLIGHT_TERMINAL_WATCHDOG_PATTERNS,
)


class TrainingCodePreflightError(RuntimeError):
    """Raised when generated training code fails the local NVFlare preflight."""

    def __init__(self, status: dict[str, Any]) -> None:
        super().__init__("server-local NVFlare SimEnv preflight failed")
        self.status = status


def build_training_mock_dataset_contract(
    *,
    code_workspace: str | Path,
    training_plan: dict[str, Any],
    training_code_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the fake prepared-client dataset used for local NVFlare preflight."""

    task_contract = training_contracts.contract_from_plan(training_plan)
    local_training = training_plan.get("local_training", {}) if isinstance(training_plan, dict) else {}
    target_size = local_training.get("target_size") if isinstance(local_training, dict) else None
    if not (
        isinstance(target_size, list)
        and len(target_size) == 2
        and all(isinstance(value, int) and value > 0 for value in target_size)
    ):
        target_size = [32, 32]
    simulation_size = [min(int(target_size[0]), 64), min(int(target_size[1]), 64)]
    dataset_root = Path(code_workspace).resolve() / "_fedready_mock_training_dataset"
    fake_data = copy.deepcopy(dict(task_contract.mock_preflight))
    fake_data["target_size"] = simulation_size
    label_values = _label_values_from_training_plan(training_plan)
    target_spec = dict(fake_data.get("target") or {})
    if target_spec.get("field") == "label" and label_values:
        target_spec["stored_values"] = label_values
        target_spec["num_classes"] = len(label_values)
    if target_spec:
        fake_data["target"] = target_spec
    else:
        fake_data.pop("target", None)
    image_spec = dict(fake_data.get("image") or {})
    image_spec["shape"] = [simulation_size[1], simulation_size[0], 3]
    fake_data["image"] = image_spec
    contract_payload = task_contract.as_data_contract()
    plan_data_contract = training_plan.get("data_contract") if isinstance(training_plan, dict) else None
    generated_contract = (
        plan_data_contract.get("generated_data_contract") if isinstance(plan_data_contract, Mapping) else None
    )
    if isinstance(generated_contract, Mapping):
        contract_payload["generated_data_contract"] = copy.deepcopy(dict(generated_contract))
        approved_template = fake_data.get("record_template")
        approved_template_fields = set(approved_template) if isinstance(approved_template, Mapping) else set()
        if set(task_contract.sample_fields) - approved_template_fields:
            supplied_template = (
                training_code_spec.get("mock_record_template") if isinstance(training_code_spec, Mapping) else None
            )
            if isinstance(training_code_spec, Mapping) and not (
                isinstance(supplied_template, Mapping) and supplied_template
            ):
                raise ValueError("generated training code spec missing mock_record_template")
            if isinstance(supplied_template, Mapping):
                missing_fields = sorted(set(task_contract.sample_fields) - set(supplied_template))
                if missing_fields:
                    raise ValueError(
                        "generated training mock_record_template missing required sample fields: "
                        + ", ".join(missing_fields)
                    )
                fake_data["record_template"] = copy.deepcopy(dict(supplied_template))
    return {
        "schema_version": MOCK_TRAINING_DATASET_SCHEMA_VERSION,
        "purpose": (
            "FedReady server-local NVFlare SimEnv preflight only. FedReady creates this dataset "
            "under code_workspace, runs a one-client recipe job against the generated package, "
            "and does not deploy anything to real clients."
        ),
        "dataset_root": str(dataset_root),
        "client_id": PREFLIGHT_CLIENT_ID,
        "site_folder": PREFLIGHT_CLIENT_ID,
        **contract_payload,
        "fake_data": fake_data,
        "simulation_requirements": [
            "FedReady runs a one-client local NVFlare SimEnv preflight with this dataset after basic package validation.",
            f"Generated code must load {task_contract.samples_file} through the same CLI arguments used for real clients.",
            "The preflight exercises data loading, transforms, model construction, loss, validation metric, and FLModel receive/send path.",
            "Do not read real client data for this check; failed preflight logs will be returned as revision feedback.",
        ],
        "preflight_report": {
            "field": "local_simulation",
            "schema_version": LOCAL_TRAINING_SIMULATION_SCHEMA_VERSION,
            "status": "FedReady attaches passed or failed after running local SimEnv",
            "client_deployment": False,
        },
    }


def _training_plan_with_mock_simulation_contract(
    *,
    training_plan: dict[str, Any],
    code_workspace: str | Path,
) -> dict[str, Any]:
    plan = copy.deepcopy(training_plan)
    plan["local_simulation_contract"] = build_training_mock_dataset_contract(
        code_workspace=code_workspace,
        training_plan=training_plan,
    )
    return plan


def run_training_code_preflight(
    *,
    training_plan: dict[str, Any],
    training_code_spec: dict[str, Any],
    code_workspace: str | Path,
    run_dir: str | Path,
    attempt: int,
    stage: str,
) -> dict[str, Any]:
    """Export and run a one-client local NVFlare SimEnv preflight on fake data."""

    run_path = Path(run_dir)
    stage_slug = _preflight_stage_slug(stage)
    contract = build_training_mock_dataset_contract(
        code_workspace=code_workspace,
        training_plan=training_plan,
        training_code_spec=training_code_spec,
    )
    dataset_root = Path(str(contract["dataset_root"]))
    _write_mock_preflight_dataset(dataset_root=dataset_root, contract=contract)

    preflight_config = _preflight_training_config(
        training_plan=training_plan,
        dataset_root=dataset_root,
        contract=contract,
    )
    preflight_plan = _preflight_training_plan(
        training_plan=training_plan,
        config=preflight_config,
        dataset_root=dataset_root,
    )
    preflight_spec = dict(training_code_spec)
    preflight_spec.pop("local_simulation", None)
    preflight_client_config = _preflight_client_training_config(
        training_plan=preflight_plan,
        training_code_spec=preflight_spec,
        config=preflight_config,
    )
    preflight_job_name = f"fedready_preflight_{stage_slug}_attempt_{int(attempt)}"
    job_root = run_path / "training_phase" / "server_preflight_jobs" / stage_slug / f"attempt_{int(attempt)}"
    candidate_job_path = job_root / preflight_job_name
    if candidate_job_path.exists():
        shutil.rmtree(candidate_job_path)
    preflight_export = export_fedavg_training_job(
        job_root=job_root,
        job_name=preflight_job_name,
        training_plan=preflight_plan,
        training_code_spec=preflight_spec,
        client_configs={PREFLIGHT_CLIENT_ID: preflight_client_config},
        min_clients=1,
        require_local_simulation=False,
    )
    validated_preflight_spec = validate_training_code_spec(
        preflight_spec,
        code_workspace=code_workspace,
        require_local_simulation=False,
    )
    workspace_root = run_path / "training_phase" / "server_preflight_workspace" / stage_slug / f"attempt_{int(attempt)}"
    expected_workspace = workspace_root / preflight_job_name
    if expected_workspace.exists():
        shutil.rmtree(expected_workspace)

    runner_result = _run_preflight_recipe_with_terminal_watchdog(
        job_name=str(preflight_export["job_name"]),
        training_plan=preflight_plan,
        training_code_spec=validated_preflight_spec,
        client_configs={PREFLIGHT_CLIENT_ID: preflight_client_config},
        workspace_root=workspace_root,
        attempt=attempt,
        stage=stage_slug,
    )
    recipe_run = runner_result.get("recipe_run") if isinstance(runner_result.get("recipe_run"), dict) else None
    terminal_status = runner_result.get("terminal_status", {})
    workspace_path = Path(recipe_run["workspace"]) if recipe_run is not None else expected_workspace
    simulator_error: Exception | None = None
    if runner_result.get("error_message"):
        simulator_error = RuntimeError(str(runner_result["error_message"]))

    simulator_status = summarize_simulator_status(workspace_path)
    if recipe_run is not None:
        simulator_status["recipe_run"] = recipe_run
    cleanup_warning_accepted = bool(
        simulator_error is not None
        and _completed_preflight_before_aio_cleanup_error(
            simulator_status=simulator_status,
            terminal_status=terminal_status,
        )
    )
    if cleanup_warning_accepted:
        simulator_error = None
        warning_codes = simulator_status.setdefault("warning_codes", [])
        if "aio_loop_cleanup_after_success" not in warning_codes:
            warning_codes.append("aio_loop_cleanup_after_success")
    if simulator_error is not None:
        simulator_status["simulator_exception_type"] = type(simulator_error).__name__
        simulator_status["simulator_exception"] = str(simulator_error)[-4000:]
    if terminal_status:
        if cleanup_warning_accepted:
            terminal_status = {
                **terminal_status,
                "accepted_as_cleanup_warning": True,
            }
        simulator_status["terminal_status"] = terminal_status
        terminal_excerpt = terminal_status.get("error_excerpt")
        terminal_log_path = terminal_status.get("terminal_log")
        if terminal_excerpt:
            simulator_status["terminal_error_snippets"] = [
                {
                    "path": str(terminal_log_path or "preflight terminal output"),
                    "excerpt": str(terminal_excerpt),
                }
            ]
    log_error_snippets = _simulator_error_snippets(workspace_path)
    if log_error_snippets:
        simulator_status["log_error_snippets"] = log_error_snippets

    succeeded = simulator_error is None and bool(simulator_status.get("succeeded"))
    metric_artifacts = simulator_status.get("metric_artifacts")
    if not isinstance(metric_artifacts, list):
        metric_artifacts = []
    nonempty_metric_artifacts = simulator_status.get("nonempty_metric_artifacts")
    if not isinstance(nonempty_metric_artifacts, list):
        nonempty_metric_artifacts = []
    metric_artifact_available = bool(simulator_status.get("metric_artifact_available")) and bool(
        nonempty_metric_artifacts
    )
    local_simulation = {
        "schema_version": LOCAL_TRAINING_SIMULATION_SCHEMA_VERSION,
        "status": "passed" if succeeded else "failed",
        "client_deployment": False,
        "validator": "fedready.local_nvflare_simenv_preflight",
        "stage": stage_slug,
        "attempt": int(attempt),
        "mock_client_id": PREFLIGHT_CLIENT_ID,
        "workspace": str(workspace_path),
        "job_path": str(preflight_export["job_path"]),
        "dataset_contract": contract,
        "metric_artifact_available": metric_artifact_available,
        "metric_artifacts": metric_artifacts,
        "nonempty_metric_artifacts": nonempty_metric_artifacts,
        "summary": (
            "one-client local NVFlare SimEnv preflight passed on fake prepared data"
            if succeeded
            else "one-client local NVFlare SimEnv preflight failed on fake prepared data"
        ),
        "simulator_status": {
            "finished_fedavg": simulator_status.get("finished_fedavg"),
            "server_error_count": simulator_status.get("server_error_count"),
            "empty_result_clients": simulator_status.get("empty_result_clients"),
            "non_tensor_param_warning_count": simulator_status.get("non_tensor_param_warning_count"),
            "persisted_global_model": simulator_status.get("persisted_global_model"),
            "metric_artifacts": metric_artifacts,
            "nonempty_metric_artifacts": nonempty_metric_artifacts,
            "metric_artifact_available": metric_artifact_available,
            "terminal_status": simulator_status.get("terminal_status"),
        },
    }
    return {
        "schema_version": "fedready.training_code_preflight.v1",
        "succeeded": succeeded,
        "stage": stage_slug,
        "attempt": int(attempt),
        "mock_client_id": PREFLIGHT_CLIENT_ID,
        "dataset_root": str(dataset_root),
        "job_name": preflight_job_name,
        "job_path": str(preflight_export["job_path"]),
        "workspace_root": str(workspace_root),
        "workspace": str(workspace_path),
        "recipe_run": recipe_run,
        "simulator_status": simulator_status,
        "terminal_status": terminal_status,
        "local_simulation": local_simulation,
    }


def _completed_preflight_before_aio_cleanup_error(
    *, simulator_status: Mapping[str, Any], terminal_status: Mapping[str, Any]
) -> bool:
    """Accept only NVFlare's known post-success AIO-loop shutdown diagnostic."""

    matched_line = str(terminal_status.get("matched_line") or "")
    return bool(
        "could not stop AIO loop" in matched_line
        and simulator_status.get("succeeded") is True
        and simulator_status.get("finished_fedavg") is True
        and simulator_status.get("persisted_global_model") is True
        and simulator_status.get("metric_artifact_available") is True
        and simulator_status.get("nonempty_metric_artifacts")
        and not simulator_status.get("empty_result_clients")
        and not simulator_status.get("client_communication_error_clients")
        and not simulator_status.get("non_tensor_param_warnings")
    )


def _run_preflight_recipe_with_terminal_watchdog(
    *,
    job_name: str,
    training_plan: dict[str, Any],
    training_code_spec: dict[str, Any],
    client_configs: dict[str, dict[str, Any]],
    workspace_root: str | Path,
    attempt: int,
    stage: str,
) -> dict[str, Any]:
    workspace_root_path = Path(workspace_root)
    workspace_root_path.mkdir(parents=True, exist_ok=True)
    runner_dir = workspace_root_path / "_fedready_preflight_runner"
    runner_dir.mkdir(parents=True, exist_ok=True)
    input_path = runner_dir / "input.json"
    result_path = runner_dir / "recipe_run.json"
    terminal_log_path = runner_dir / "terminal.log"
    payload = {
        "job_name": job_name,
        "training_plan": training_plan,
        "training_code_spec": training_code_spec,
        "client_configs": client_configs,
        "min_clients": 1,
        "workspace_root": str(workspace_root_path),
        "clients": [PREFLIGHT_CLIENT_ID],
        "threads": 1,
        "log_config": "concise",
        "result_path": str(result_path),
    }
    atomic_write_json(input_path, payload)

    research_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(research_root) if not existing_pythonpath else f"{research_root}{os.pathsep}{existing_pythonpath}"
    )
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        sys.executable,
        "-m",
        "fedready.flare.training_preflight_runner",
        str(input_path),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(research_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    matched_line: str | None = None
    matched_pattern: str | None = None
    with terminal_log_path.open("w", encoding="utf-8", errors="replace") as terminal_log:
        assert proc.stdout is not None
        for line in proc.stdout:
            terminal_log.write(line)
            terminal_log.flush()
            if matched_line is None:
                matched_pattern = _preflight_terminal_error_pattern(line)
                if matched_pattern is not None:
                    matched_line = line.rstrip()
                    _terminate_process_group(proc, marker=str(workspace_root_path))
        returncode = proc.wait()

    recipe_run = _read_json(result_path) if result_path.exists() else None
    error_excerpt = (
        _preflight_terminal_excerpt(terminal_log_path) if matched_line is not None or returncode != 0 else ""
    )
    terminal_status = {
        "schema_version": "fedready.preflight_terminal_status.v1",
        "stage": stage,
        "attempt": int(attempt),
        "terminal_log": str(terminal_log_path),
        "returncode": returncode,
        "killed_by_watchdog": matched_line is not None,
        "matched_pattern": matched_pattern,
        "matched_line": matched_line,
        "error_excerpt": error_excerpt,
    }
    error_message = None
    if matched_line is not None:
        error_message = f"Preflight terminal watchdog detected {matched_pattern}: {matched_line}"
    elif returncode != 0:
        error_message = f"Preflight runner failed with return code {returncode}"
    return {
        "recipe_run": recipe_run,
        "terminal_status": terminal_status,
        "error_message": error_message,
    }


def _preflight_terminal_error_pattern(line: str) -> str | None:
    for pattern in PREFLIGHT_TERMINAL_WATCHDOG_PATTERNS:
        if pattern in line:
            return pattern
    return None


def _terminate_process_group(proc: subprocess.Popen[str], *, marker: str | None = None) -> None:
    target_groups = _process_groups_for_preflight(proc.pid, marker=marker)
    if not target_groups and proc.poll() is not None:
        return
    try:
        if hasattr(os, "killpg"):
            for pgid in target_groups:
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    continue
        else:  # pragma: no cover - non-posix fallback
            proc.terminate()
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if hasattr(os, "killpg"):
            for pgid in _process_groups_for_preflight(proc.pid, marker=marker):
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    continue
        else:  # pragma: no cover - non-posix fallback
            proc.kill()
        proc.wait(timeout=10)
    except ProcessLookupError:
        return


def _process_groups_for_preflight(pid: int, *, marker: str | None = None) -> list[int]:
    if not hasattr(os, "getpgid"):
        return []
    own_pgid = os.getpgid(0)
    target_pids = [pid]
    if marker:
        target_pids.extend(_process_ids_matching_cmdline(marker))
    groups: set[int] = set()
    for target_pid in target_pids:
        if target_pid == os.getpid():
            continue
        try:
            pgid = os.getpgid(target_pid)
        except ProcessLookupError:
            continue
        if pgid != own_pgid:
            groups.add(pgid)
    return sorted(groups)


def _process_ids_matching_cmdline(marker: str) -> list[int]:
    if not marker:
        return []
    proc_root = Path("/proc")
    if not proc_root.exists():
        return []
    matches: list[int] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        try:
            raw_cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        cmdline = raw_cmdline.replace(b"\0", b" ").decode("utf-8", errors="replace")
        if marker in cmdline:
            matches.append(pid)
    return matches


def _preflight_terminal_excerpt(path: str | Path, *, limit: int = 5000) -> str:
    terminal_path = Path(path)
    try:
        text = terminal_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    markers = list(PREFLIGHT_TERMINAL_EXCERPT_PATTERNS) + ["ERROR", "Exception"]
    positions = [text.find(marker) for marker in markers if text.find(marker) >= 0]
    if not positions:
        return text[-limit:]
    start = max(0, min(positions) - 1200)
    end = min(len(text), min(positions) + limit)
    return text[start:end]


def _preflight_training_config(
    *,
    training_plan: dict[str, Any],
    dataset_root: str | Path,
    contract: dict[str, Any],
) -> FedAvgTrainingConfig:
    local_training = training_plan.get("local_training", {}) if isinstance(training_plan, dict) else {}
    if not isinstance(local_training, dict):
        local_training = {}
    fake_data = contract.get("fake_data", {}) if isinstance(contract, dict) else {}
    target_size = fake_data.get("target_size") if isinstance(fake_data, dict) else None
    if not (
        isinstance(target_size, list)
        and len(target_size) == 2
        and all(isinstance(value, int) and value > 0 for value in target_size)
    ):
        target_size = [32, 32]
    try:
        learning_rate = float(local_training.get("learning_rate", 1e-3))
    except (TypeError, ValueError):
        learning_rate = 1e-3
    algorithm = training_plan.get("algorithm", {}) if isinstance(training_plan, dict) else {}
    task_name = (
        algorithm.get("task_name", DEFAULT_TRAINING_TASK_NAME)
        if isinstance(algorithm, dict)
        else DEFAULT_TRAINING_TASK_NAME
    )
    return FedAvgTrainingConfig(
        num_rounds=1,
        local_epochs=1,
        batch_size=1,
        learning_rate=learning_rate,
        target_size=(int(target_size[0]), int(target_size[1])),
        num_workers=0,
        device="cpu",
        dataset_root=str(Path(dataset_root).resolve()),
        task_name=str(task_name or DEFAULT_TRAINING_TASK_NAME),
    )


def _preflight_training_plan(
    *,
    training_plan: dict[str, Any],
    config: FedAvgTrainingConfig,
    dataset_root: str | Path,
) -> dict[str, Any]:
    plan = copy.deepcopy(training_plan)
    plan["ready_clients"] = [{"client_id": PREFLIGHT_CLIENT_ID}]
    plan["ready_client_count"] = 1
    algorithm = dict(plan.get("algorithm") or {}) if isinstance(plan.get("algorithm"), dict) else {}
    algorithm["num_rounds"] = 1
    algorithm["task_name"] = config.task_name
    plan["algorithm"] = algorithm
    local_training = dict(plan.get("local_training") or {}) if isinstance(plan.get("local_training"), dict) else {}
    local_training.update(
        {
            "local_epochs": config.local_epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "target_size": list(config.target_size),
            "num_workers": config.num_workers,
            "device": config.device,
        }
    )
    plan["local_training"] = local_training
    data_contract = dict(plan.get("data_contract") or {}) if isinstance(plan.get("data_contract"), dict) else {}
    data_contract["dataset_root"] = str(Path(dataset_root).resolve())
    data_contract["site_folder"] = PREFLIGHT_CLIENT_ID
    data_contract["server_local_preflight"] = True
    plan["data_contract"] = data_contract
    privacy = dict(plan.get("privacy") or {}) if isinstance(plan.get("privacy"), dict) else {}
    privacy["server_local_preflight"] = True
    plan["privacy"] = privacy
    return plan


def _preflight_client_training_config(
    *,
    training_plan: dict[str, Any],
    training_code_spec: dict[str, Any],
    config: FedAvgTrainingConfig,
) -> dict[str, Any]:
    dataset_dir = Path(config.dataset_root) / PREFLIGHT_CLIENT_ID
    summary = summarize_prepared_dataset(dataset_dir, client_id=PREFLIGHT_CLIENT_ID)
    task_contract = training_contracts.contract_from_plan(training_plan)
    qc_required = bool(task_contract.qc_contract.get("visual_qc_required"))
    qc = summary.get("visual_qc", {}) if isinstance(summary.get("visual_qc"), dict) else {}
    verification = summary.get("verification", {}) if isinstance(summary.get("verification"), dict) else {}
    return _client_runtime_training_config(
        client_entry={
            "client_id": PREFLIGHT_CLIENT_ID,
            "train_sample_count": int(summary.get("train_sample_count") or 0),
            "validation_sample_count": int(summary.get("validation_sample_count") or 0),
            "test_sample_count": int(summary.get("test_sample_count") or 0),
            "verification_passed": verification.get("passed") is True,
            "visual_qc_required": qc_required,
            "visual_qc_passed": qc.get("passed") is True if qc_required else None,
            "record_type": task_contract.record_type,
            "samples_file": task_contract.samples_file,
            "sample_manifest_format": task_contract.sample_manifest_format,
            "source": "server_local_preflight_mock_dataset",
        },
        training_plan=training_plan,
        training_code_spec=training_code_spec,
        config=config,
    )


def _label_values_from_training_plan(training_plan: dict[str, Any]) -> list[int]:
    data_contract = training_plan.get("data_contract") if isinstance(training_plan, dict) else None
    if not isinstance(data_contract, dict):
        return []
    observed = data_contract.get("observed_label_values")
    values: list[int] = []
    if isinstance(observed, list):
        for value in observed:
            if isinstance(value, int):
                values.append(value)
            elif isinstance(value, str) and value.isdigit():
                values.append(int(value))
    if values:
        return sorted(set(values))
    labels = data_contract.get("labels")
    if isinstance(labels, dict):
        for key in labels:
            if isinstance(key, int):
                values.append(key)
            elif isinstance(key, str) and key.isdigit():
                values.append(int(key))
    return sorted(set(values))


def _label_values_from_mock_contract(contract: dict[str, Any]) -> list[int]:
    fake_data = contract.get("fake_data") if isinstance(contract, dict) else None
    target = fake_data.get("target") if isinstance(fake_data, dict) else None
    values = target.get("stored_values") if isinstance(target, dict) else None
    result: list[int] = []
    if isinstance(values, list):
        for value in values:
            if isinstance(value, int):
                result.append(value)
            elif isinstance(value, str) and value.isdigit():
                result.append(int(value))
    return sorted(set(result))


def _write_mock_preflight_dataset(*, dataset_root: str | Path, contract: dict[str, Any]) -> None:
    root = Path(dataset_root)
    if root.exists():
        shutil.rmtree(root)
    task_contract = (
        training_contracts.contract_from_generated_data_contract(contract.get("generated_data_contract"))
        or training_contracts.contract_from_generated_data_contract(contract)
        or training_contracts.contract_from_record_type(str(contract.get("record_type") or ""))
    )
    fake_data = contract.get("fake_data", {}) if isinstance(contract, dict) else {}
    preflight_template = fake_data.get("record_template") if isinstance(fake_data, Mapping) else None
    if isinstance(preflight_template, Mapping):
        mock_preflight = dict(task_contract.mock_preflight)
        mock_preflight["record_template"] = copy.deepcopy(dict(preflight_template))
        task_contract = replace(task_contract, mock_preflight=mock_preflight)
    target_size = fake_data.get("target_size") if isinstance(fake_data, dict) else [32, 32]
    if not (
        isinstance(target_size, list)
        and len(target_size) == 2
        and all(isinstance(value, int) and value > 0 for value in target_size)
    ):
        target_size = [32, 32]
    width, height = int(target_size[0]), int(target_size[1])
    label_values = _label_values_from_mock_contract(contract)
    training_contracts.write_mock_preflight_dataset(
        dataset_root=root,
        contract=task_contract,
        client_id=PREFLIGHT_CLIENT_ID,
        target_size=(width, height),
        label_values=label_values,
    )
    site_dir = root / PREFLIGHT_CLIENT_ID
    samples_path = site_dir / task_contract.samples_file
    split_counts, sample_count = training_contracts.read_split_counts(
        samples_path,
        sample_manifest_format=task_contract.sample_manifest_format,
    )
    visual_qc, visual_qc_decision = training_contracts.mock_visual_qc_payload(
        contract=task_contract,
        sample_count=sample_count,
        client_id=PREFLIGHT_CLIENT_ID,
    )
    atomic_write_json(
        site_dir / "manifest.json",
        {
            "schema_version": "fedready.prepared_dataset_manifest.v1",
            "client_id": PREFLIGHT_CLIENT_ID,
            "source": "server_local_mock_preflight",
            "record_type": task_contract.record_type,
            "sample_manifest": task_contract.samples_file,
            "sample_manifest_format": task_contract.sample_manifest_format,
            "counts": {"total": sample_count, "by_split": split_counts},
            "visual_qc": visual_qc,
            "visual_qc_decision": visual_qc_decision,
            "verification": {"passed": True, "extracted_count": sample_count},
            "local_adapter": {
                "status": "implemented",
                "source_label_type": "mock_training_contract",
                "record_count": sample_count,
            },
        },
    )
    atomic_write_json(
        site_dir / "training_transforms.json",
        {
            "schema_version": "fedready.training_transforms.v1",
            "target_size": [width, height],
            "normalization": "mock_preflight_rgb_uint8_to_float",
            "label_values": training_contracts.mock_label_values(task_contract, label_values=label_values),
        },
    )


def _preflight_stage_slug(stage: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", str(stage)).strip("_").lower()
    return slug or "training_code"
