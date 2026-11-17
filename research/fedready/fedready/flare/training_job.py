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

"""FedAvg recipe construction, client runtime arguments, and simulator results."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fedready.data.contracts import training as training_contracts
from fedready.job_data import export_fed_job_recipe
from fedready.training_code import TRAINING_PACKAGE_NAME, _safe_code_spec, validate_training_code_spec
from fedready.utils.io import atomic_write_json

from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor, ExecutionMode
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.file_model_locator import PTFileModelLocator
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.client.config import ExchangeFormat
from nvflare.job_config.api import FedJob

TRAINING_JOB_SCHEMA_VERSION = "fedready.nvflare_fedavg_training_job.v1"
CLIENT_TRAINING_CONFIG_SCHEMA_VERSION = "fedready.client_training_config.v1"
CLIENT_TRAINING_LAUNCHER_SCRIPT = "fedready_training_client_runtime.py"
DEFAULT_TRAINING_TASK_NAME = "train"
DEFAULT_DATASET_ROOT = "data/dataset_fl"
DEFAULT_METRICS_JSONL = "fedready_training_metrics.jsonl"


@dataclass(frozen=True)
class FedAvgTrainingConfig:
    """Training knobs for the current FedAvg exporter."""

    num_rounds: int = 1
    local_epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 1e-3
    target_size: tuple[int, int] = (128, 128)
    num_workers: int = 0
    device: str = "auto"
    dataset_root: str = DEFAULT_DATASET_ROOT
    task_name: str = DEFAULT_TRAINING_TASK_NAME


def prepare_client_training_configs(
    *,
    training_plan: dict[str, Any],
    training_code_spec: dict[str, Any],
    config: FedAvgTrainingConfig,
) -> dict[str, dict[str, Any]]:
    """Prepare client runtime configs without reading client-local datasets centrally.

    The exported config is derived from server-visible extraction-summary facts.
    Actual manifest.json and contract-declared sample manifest checks run inside each NVFlare client app
    via ``fedready_training_client_runtime.py`` immediately before the generated
    training entry script is invoked.
    """

    _require_relative_dataset_root_for_client_runtime(config.dataset_root)
    training_code_spec = validate_training_code_spec(training_code_spec)
    configs: dict[str, dict[str, Any]] = {}
    for client in training_plan.get("ready_clients", []):
        if not isinstance(client, dict) or not isinstance(client.get("client_id"), str):
            continue
        cfg = _client_runtime_training_config(
            client_entry=client,
            training_plan=training_plan,
            training_code_spec=training_code_spec,
            config=config,
        )
        configs[str(client["client_id"])] = cfg
    ready = {client_id: cfg for client_id, cfg in configs.items() if cfg.get("status") == "ready"}
    if not ready:
        raise ValueError("no client training configs are ready")
    return ready


def summarize_prepared_dataset(dataset_dir: str | Path, *, client_id: str) -> dict[str, Any]:
    """Return a safe local summary of an extracted training dataset."""

    root = Path(dataset_dir)
    manifest = _read_json(root / "manifest.json")
    sample_manifest = str(manifest.get("sample_manifest") or "") if isinstance(manifest, dict) else ""
    manifest_format = str(manifest.get("sample_manifest_format") or "") if isinstance(manifest, dict) else ""
    samples_path = root / sample_manifest if sample_manifest else root / "__missing_sample_manifest__"
    split_counts, sample_count = ({}, 0)
    if sample_manifest:
        split_counts, sample_count = _read_prepared_sample_counts(
            samples_path,
            sample_manifest_format=manifest_format,
        )
    visual_qc_artifacts = manifest.get("visual_qc", {}) if isinstance(manifest, dict) else {}
    visual_qc_decision = manifest.get("visual_qc_decision", {}) if isinstance(manifest, dict) else {}
    if not isinstance(visual_qc_decision, dict):
        visual_qc_decision = {}
    if not visual_qc_decision and isinstance(visual_qc_artifacts, dict) and "passed" in visual_qc_artifacts:
        visual_qc_decision = visual_qc_artifacts
    verification = manifest.get("verification", {}) if isinstance(manifest, dict) else {}
    local_adapter = manifest.get("local_adapter") if isinstance(manifest, dict) else None
    qc_available = False
    qc_sample_count = None
    if isinstance(visual_qc_artifacts, dict):
        artifacts = visual_qc_artifacts.get("artifacts")
        qc_available = bool(visual_qc_artifacts.get("available") or artifacts)
        qc_sample_count = visual_qc_artifacts.get("sample_count")
    return {
        "schema_version": "fedready.prepared_dataset_training_summary.v1",
        "client_id": client_id,
        "manifest_available": bool(manifest),
        "samples_file": sample_manifest,
        "sample_manifest_format": manifest_format,
        "samples_file_available": samples_path.exists(),
        "samples_jsonl_available": samples_path.exists(),
        "sample_count": sample_count,
        "split_counts": split_counts,
        "train_sample_count": split_counts.get("train", 0),
        "validation_sample_count": split_counts.get("validation", 0),
        "test_sample_count": split_counts.get("test", 0),
        "visual_qc": {
            "status": visual_qc_decision.get("status"),
            "passed": visual_qc_decision.get("passed"),
            "selected_transform": visual_qc_decision.get("selected_transform"),
            "available": qc_available,
            "sample_count": qc_sample_count,
        },
        "verification": {
            "passed": (verification.get("passed") if isinstance(verification, dict) else None),
            "extracted_count": (verification.get("extracted_count") if isinstance(verification, dict) else None),
        },
        "local_adapter": _safe_adapter_summary(local_adapter),
        "privacy": {
            "safe_to_share": True,
            "redacted": [
                "local_dataset_path",
                "filenames",
                "sample_ids",
                "raw_images",
                "raw_masks",
            ],
        },
    }


def _read_prepared_sample_counts(path: Path, *, sample_manifest_format: str = "") -> tuple[dict[str, int], int]:
    try:
        return training_contracts.read_split_counts(path, sample_manifest_format=sample_manifest_format)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}, 0


def export_fedavg_training_job(
    *,
    job_root: str | Path,
    job_name: str,
    training_plan: dict[str, Any],
    training_code_spec: dict[str, Any],
    client_configs: dict[str, dict[str, Any]],
    min_clients: int | None = None,
    require_local_simulation: bool = True,
) -> dict[str, Any]:
    """Export a regular NVFlare FedAvg job from generated training code."""

    code_spec = validate_training_code_spec(
        training_code_spec,
        require_local_simulation=require_local_simulation,
    )
    ready_configs = {client_id: cfg for client_id, cfg in client_configs.items() if cfg.get("status") == "ready"}
    client_ids = sorted(ready_configs)
    if not client_ids:
        raise ValueError("no ready client training configs")
    job = _build_fedavg_job_object(
        job_name=job_name,
        training_plan=training_plan,
        training_code_spec=code_spec,
        client_configs=ready_configs,
        min_clients=min_clients,
    )
    min_count = len(client_ids) if min_clients is None else min_clients

    job_path = export_fed_job_recipe(job, job_root)
    atomic_write_json(job_path / "fedready_training_plan.json", training_plan)
    atomic_write_json(job_path / "fedready_training_code_spec.json", _safe_code_spec(code_spec))
    atomic_write_json(job_path / "fedready_client_training_configs.json", ready_configs)
    return {
        "schema_version": TRAINING_JOB_SCHEMA_VERSION,
        "job_name": job.name,
        "job_path": str(job_path),
        "client_ids": client_ids,
        "min_clients": min_count,
        "training_plan": training_plan,
        "training_code_spec": _safe_code_spec(code_spec),
        "client_training_configs": ready_configs,
        "base_example": training_plan.get("algorithm", {}).get("base_example"),
        "recipe_api": "nvflare.recipe",
    }


def summarize_simulator_status(workspace: str | Path) -> dict[str, Any]:
    """Summarize the NVFlare simulator run from server logs and model artifacts."""

    root = Path(workspace)
    server_dir = root / "server"
    log_json = server_dir / "log.json"
    log_txt = server_dir / "log.txt"
    finished_fedavg = False
    error_count = 0
    aggregated_result_messages: list[str] = []
    empty_result_clients: set[str] = set()
    communication_error_clients: set[str] = set()
    non_tensor_param_warnings: list[str] = []
    training_error_count = 0
    cleanup_error_count = 0
    aggregation_counts: list[tuple[int, int]] = []

    def record_message(message: str, *, is_error: bool = False, server: bool = False) -> None:
        nonlocal training_error_count, cleanup_error_count
        if is_error:
            if server and finished_fedavg and message.rsplit(" - ERROR - ", 1)[-1].strip() == "could not stop AIO loop":
                cleanup_error_count += 1
            else:
                training_error_count += 1
        # FedAvg emits intermediate counts for every client; retain the final
        # count of each round rather than treating progress as missing results.
        if server and re.search(r"Round \d+ started\.", message):
            aggregation_counts.append((0, 0))
        match = re.search(r"Aggregated (\d+)/(\d+) results", message)
        if server and match:
            if not aggregation_counts:
                aggregation_counts.append((0, 0))
            aggregation_counts[-1] = (int(match.group(1)), int(match.group(2)))
        match = re.search(r"Empty result from client ([^,\s]+), skipping", message)
        if match:
            empty_result_clients.add(match.group(1))
        match = re.search(r"Communication error with client ([^:,\s]+)", message)
        if match:
            communication_error_clients.add(match.group(1))
        if "vars excluded as they were non-tensor type" in message:
            non_tensor_param_warnings.append(message)

    if log_json.exists():
        with log_json.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message = str(record.get("message") or "")
                record_message(message, is_error=record.get("levelname") in {"ERROR", "CRITICAL"}, server=True)
                if record.get("levelname") == "ERROR":
                    error_count += 1
                if "Finished FedAvg" in message:
                    finished_fedavg = True
                if "Aggregated " in message and " results" in message:
                    aggregated_result_messages.append(message)
    elif log_txt.exists():
        with log_txt.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                record_message(line, is_error=" - ERROR - " in line or " - CRITICAL - " in line, server=True)
                if " - ERROR - " in line:
                    error_count += 1
                if "Finished FedAvg" in line:
                    finished_fedavg = True
                if "Aggregated " in line and " results" in line:
                    aggregated_result_messages.append(line.strip())

    for path in sorted(root.rglob("log.json")):
        if path == log_json:
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record_message(
                    str(record.get("message") or ""), is_error=record.get("levelname") in {"ERROR", "CRITICAL"}
                )
    for path in sorted(root.rglob("log.txt")):
        if path == log_txt:
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                record_message(line, is_error=" - ERROR - " in line or " - CRITICAL - " in line)

    model_paths = sorted(str(path) for path in server_dir.rglob("FL_global_model.pt"))
    metric_paths = sorted(path for path in root.rglob(DEFAULT_METRICS_JSONL))
    metric_artifacts = [str(path) for path in metric_paths]
    nonempty_metric_artifacts = [str(path) for path in metric_paths if path.is_file() and path.stat().st_size > 0]
    valid_metric_artifacts = [str(path) for path in metric_paths if _metric_artifact_has_valid_records(path)]
    has_empty_results = bool(empty_result_clients)
    has_non_tensor_params = bool(non_tensor_param_warnings)
    has_metric_artifact = bool(valid_metric_artifacts)
    all_results_aggregated = bool(aggregation_counts) and all(
        received == expected and expected > 0 for received, expected in aggregation_counts
    )
    completed_with_warnings = bool(error_count or has_empty_results)
    warning_codes: list[str] = []
    if error_count:
        warning_codes.append("server_errors_observed")
    if has_empty_results:
        warning_codes.append("client_results_missing")
    return {
        "schema_version": "fedready.nvflare_simulator_status.v1",
        "workspace": str(root),
        "finished_fedavg": finished_fedavg,
        "server_error_count": error_count,
        "training_error_count": training_error_count,
        "aio_cleanup_error_count": cleanup_error_count,
        "all_results_aggregated": all_results_aggregated,
        "empty_result_clients": sorted(empty_result_clients),
        "client_communication_error_clients": sorted(communication_error_clients),
        "non_tensor_param_warning_count": len(non_tensor_param_warnings),
        "non_tensor_param_warnings": non_tensor_param_warnings[-3:],
        "global_model_paths": model_paths,
        "metric_artifacts": metric_artifacts,
        "nonempty_metric_artifacts": nonempty_metric_artifacts,
        "valid_metric_artifacts": valid_metric_artifacts,
        "metric_artifact_available": has_metric_artifact,
        "persisted_global_model": bool(model_paths),
        "aggregated_result_messages": aggregated_result_messages[-3:],
        "completed_with_warnings": completed_with_warnings,
        "warning_codes": warning_codes,
        "succeeded": (
            finished_fedavg
            and all_results_aggregated
            and not has_empty_results
            and not communication_error_clients
            and not training_error_count
            and not has_non_tensor_params
            and bool(model_paths)
            and has_metric_artifact
        ),
    }


def _metric_artifact_has_valid_records(path: Path) -> bool:
    """Accept non-empty JSONL containing finite aggregate metric values."""

    try:
        records = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            return False
        for line in records:
            record = json.loads(line)
            if not isinstance(record, dict):
                return False
            round_value = record.get("round")
            if round_value is not None and (
                isinstance(round_value, bool)
                or not isinstance(round_value, (int, float))
                or not math.isfinite(float(round_value))
            ):
                return False
            metrics = [
                float(value)
                for name, value in record.items()
                if name not in {"round", "client_id"}
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ]
            if not metrics or not all(math.isfinite(value) for value in metrics):
                return False
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return False
    return True


def _runtime_attempt_job_name(base_job_name: str, attempt: int) -> str:
    if attempt <= 1:
        return base_job_name
    return f"{base_job_name}_runtime_retry{attempt}"


def _refresh_client_training_configs_for_code_spec(
    configs: dict[str, Any],
    *,
    training_plan: dict[str, Any],
    training_code_spec: dict[str, Any],
    config: FedAvgTrainingConfig,
) -> dict[str, dict[str, Any]]:
    refreshed: dict[str, dict[str, Any]] = {}
    for client_id, cfg in configs.items():
        if not isinstance(cfg, dict):
            continue
        updated = dict(cfg)
        if updated.get("status") == "ready":
            updated["task_script_path"] = CLIENT_TRAINING_LAUNCHER_SCRIPT
            updated["training_entry_script"] = training_code_spec["entry_script"]
            updated["task_script_args"] = _format_client_runtime_task_args(
                client_id=str(client_id),
                training_plan=training_plan,
                training_code_spec=training_code_spec,
                config=config,
                train_sample_count=int(updated.get("train_sample_count") or 0),
                validation_sample_count=int(updated.get("validation_sample_count") or 0),
                test_sample_count=int(updated.get("test_sample_count") or 0),
            )
        refreshed[str(client_id)] = updated
    return refreshed


def _simulator_error_snippets(workspace: str | Path, *, max_files: int = 8) -> list[dict[str, str]]:
    root = Path(workspace)
    snippets: list[dict[str, str]] = []
    log_paths = sorted(root.glob("*/log.txt")) + sorted(root.glob("*/log.json"))
    seen: set[Path] = set()
    for path in log_paths:
        if path in seen or len(snippets) >= max_files:
            continue
        seen.add(path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        excerpt = _interesting_log_excerpt(text)
        if excerpt:
            snippets.append({"path": str(path), "excerpt": excerpt})
    return snippets


def _interesting_log_excerpt(text: str, *, limit: int = 5000) -> str:
    markers = ["Traceback", "AttributeError", "RuntimeError", " - ERROR - ", " ERROR "]
    positions = [text.find(marker) for marker in markers if text.find(marker) >= 0]
    if not positions:
        return ""
    start = max(0, min(positions) - 1200)
    end = min(len(text), min(positions) + limit)
    return text[start:end]


def _build_fedavg_job_object(
    *,
    job_name: str,
    training_plan: dict[str, Any],
    training_code_spec: dict[str, Any],
    client_configs: dict[str, dict[str, Any]],
    min_clients: int | None,
) -> FedJob:
    ready_configs = {client_id: cfg for client_id, cfg in client_configs.items() if cfg.get("status") == "ready"}
    client_ids = sorted(ready_configs)
    if not client_ids:
        raise ValueError("no ready client training configs")
    min_count = len(client_ids) if min_clients is None else min_clients
    if not isinstance(min_count, int) or isinstance(min_count, bool) or not 1 <= min_count <= len(client_ids):
        raise ValueError(f"min_clients must be between 1 and {len(client_ids)}")
    job = FedJob(name=job_name, min_clients=min_count, mandatory_clients=client_ids)
    job.to_server(
        FedAvg(
            num_clients=min_count,
            num_rounds=int(training_plan.get("algorithm", {}).get("num_rounds", 1)),
            persistor_id="persistor",
            task_name=training_plan.get("algorithm", {}).get("task_name", DEFAULT_TRAINING_TASK_NAME),
        ),
        id="controller",
    )
    job.to_server(
        PTFileModelPersistor(
            model={
                "path": training_code_spec["model_class_path"],
                "args": training_code_spec.get("model_args", {}),
            }
        ),
        id="persistor",
    )
    job.to_server(PTFileModelLocator(pt_persistor_id="persistor"), id="locator")
    package_dir = Path(training_code_spec["package_dir"]).resolve()
    _add_training_package(job, package_dir=package_dir, target="server")
    task_name = training_plan.get("algorithm", {}).get("task_name", DEFAULT_TRAINING_TASK_NAME)
    for client_id in client_ids:
        _add_training_package(job, package_dir=package_dir, target=client_id)
        _add_client_runtime_launcher(job, target=client_id)
        job.to(
            ClientAPIExecutor(
                execution_mode=ExecutionMode.IN_PROCESS,
                task_script_path=CLIENT_TRAINING_LAUNCHER_SCRIPT,
                task_script_args=str(ready_configs[client_id]["task_script_args"]),
                params_exchange_format=ExchangeFormat.PYTORCH,
                server_expected_format=ExchangeFormat.NUMPY,
                train_with_evaluation=True,
            ),
            client_id,
            id="fedready_training_executor",
            tasks=[task_name],
        )
    return job


def _require_relative_dataset_root_for_client_runtime(dataset_root: str) -> None:
    if Path(dataset_root).expanduser().is_absolute():
        raise ValueError(
            "training dataset_root must be relative for client-side training-site preparation; "
            "configure absolute client-local roots inside the client runtime environment"
        )


def _client_runtime_training_config(
    *,
    client_entry: dict[str, Any],
    training_plan: dict[str, Any],
    training_code_spec: dict[str, Any],
    config: FedAvgTrainingConfig,
) -> dict[str, Any]:
    client_id = str(client_entry["client_id"])
    task_contract = training_contracts.contract_from_plan(training_plan)
    train_count = int(client_entry.get("train_sample_count") or 0)
    validation_count = int(client_entry.get("validation_sample_count") or 0)
    test_count = int(client_entry.get("test_sample_count") or 0)
    visual_qc_required = bool(task_contract.qc_contract.get("visual_qc_required"))
    visual_qc_ready = (not visual_qc_required) or client_entry.get("visual_qc_passed") is True
    ready = bool(train_count > 0 and client_entry.get("verification_passed") is True and visual_qc_ready)
    return {
        "schema_version": CLIENT_TRAINING_CONFIG_SCHEMA_VERSION,
        "client_id": client_id,
        "status": "ready" if ready else "not_ready",
        "task_script_path": CLIENT_TRAINING_LAUNCHER_SCRIPT,
        "training_entry_script": training_code_spec["entry_script"],
        "task_script_args": _format_client_runtime_task_args(
            client_id=client_id,
            training_plan=training_plan,
            training_code_spec=training_code_spec,
            config=config,
            train_sample_count=train_count,
            validation_sample_count=validation_count,
            test_sample_count=test_count,
        ),
        "dataset_root": config.dataset_root,
        "dataset_client_folder": client_id,
        "record_type": task_contract.record_type,
        "samples_file": task_contract.samples_file,
        "sample_manifest_format": task_contract.sample_manifest_format,
        "train_sample_count": train_count,
        "validation_sample_count": validation_count,
        "test_sample_count": test_count,
        "visual_qc_required": visual_qc_required,
        "visual_qc_available": bool(client_entry.get("visual_qc_passed")),
        "visual_qc_passed": (client_entry.get("visual_qc_passed") if visual_qc_required else None),
        "verification_passed": bool(client_entry.get("verification_passed")),
        "client_side_prepare": {
            "script": CLIENT_TRAINING_LAUNCHER_SCRIPT,
            "checks": list(task_contract.runtime_checks),
            "runs_inside": "nvflare_client_app",
        },
        "reason": (
            "client runtime will validate prepared extracted dataset before training"
            if ready
            else "server-visible extraction summary is missing a contract-required readiness signal"
        ),
        "privacy": {
            "safe_to_share": True,
            "redacted": [
                "local_paths",
                "filenames",
                "sample_ids",
                "raw_images",
                "raw_masks",
            ],
            "local_binding_scope": "client_executor_runtime",
        },
    }


def _format_client_runtime_task_args(
    *,
    client_id: str,
    training_plan: dict[str, Any],
    training_code_spec: dict[str, Any],
    config: FedAvgTrainingConfig,
    train_sample_count: int,
    validation_sample_count: int,
    test_sample_count: int,
) -> str:
    task_contract = training_contracts.contract_from_plan(training_plan)
    training_args = _format_task_args(
        client_id=client_id,
        config=config,
    )
    launcher_args = [
        "--fedready-entry-script",
        _task_arg_token("entry_script", f"{TRAINING_PACKAGE_NAME}/{training_code_spec['entry_script']}"),
        "--fedready-expected-train",
        str(int(train_sample_count)),
        "--fedready-expected-validation",
        str(int(validation_sample_count)),
        "--fedready-expected-test",
        str(int(test_sample_count)),
        "--fedready-package-dir",
        TRAINING_PACKAGE_NAME,
        "--fedready-record-type",
        _task_arg_token("record_type", task_contract.record_type),
        "--fedready-samples-file",
        _task_arg_token("samples_file", task_contract.samples_file),
        "--fedready-sample-manifest-format",
        _task_arg_token("sample_manifest_format", task_contract.sample_manifest_format),
        "--fedready-required-fields",
        json.dumps(list(task_contract.sample_fields), separators=(",", ":")),
        "--fedready-visual-qc-required",
        "true" if task_contract.qc_contract.get("visual_qc_required") else "false",
    ]
    return " ".join(launcher_args + [training_args])


def _add_client_runtime_launcher(job: FedJob, *, target: str) -> None:
    runtime_dir = Path(__file__).resolve().parent
    job.add_file_to(
        str(runtime_dir / CLIENT_TRAINING_LAUNCHER_SCRIPT),
        target,
        app_folder_type="custom",
    )


def _task_arg_token(name: str, value: Any) -> str:
    """Return one TaskScriptRunner argument token.

    TaskScriptRunner splits task arguments on whitespace without shell quote
    handling, so shell quoting would leak literal quote characters into argv.
    Values are passed through unchanged and must not contain whitespace.
    """

    text = str(value)
    if not text or any(char.isspace() for char in text):
        raise ValueError(f"task argument {name} must be non-empty and contain no whitespace: {text!r}")
    return text


def _format_task_args(
    *,
    client_id: str,
    config: FedAvgTrainingConfig,
) -> str:
    values = {
        # TaskScriptRunner uses whitespace splitting, so encode the path as one
        # token; the client runtime launcher decodes it before local access.
        "dataset_root": quote(config.dataset_root, safe="/"),
        "client_id": _task_arg_token("client_id", client_id),
        "local_epochs": config.local_epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "target_width": config.target_size[0],
        "target_height": config.target_size[1],
        "num_workers": config.num_workers,
        "device": _task_arg_token("device", config.device),
        "metrics_jsonl": DEFAULT_METRICS_JSONL,
    }
    return (
        "--dataset-root {dataset_root} --client-id {client_id} "
        "--local-epochs {local_epochs} --batch-size {batch_size} "
        "--learning-rate {learning_rate} --target-size {target_width} {target_height} "
        "--num-workers {num_workers} --device {device} --metrics-jsonl {metrics_jsonl}"
    ).format(**values)


def _add_training_package(job: FedJob, *, package_dir: Path, target: str) -> None:
    job.add_file_to(str(package_dir), target, dest_dir=TRAINING_PACKAGE_NAME, app_folder_type="custom")


def _safe_adapter_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "status": value.get("status"),
        "source_label_type": value.get("source_label_type"),
        "record_count": value.get("record_count"),
        "safe_to_share": True,
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload
