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

"""NVFlare FedAvg training job and direct entry point for FedReady.

The server Codex agent writes task-specific code into the run workspace and
returns an ``fedready.training_code_spec.v1``. This module validates that spec,
packages a regular NVFlare FedAvg job, and defers prepared-data checks to a
client-local runtime launcher. There is no alternate-agent or fixed-reference
fallback in the contributed workflow.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import py_compile
import re
import shlex
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from fedready.agents import ServerAgent
from fedready.agents.bridge import build_agent_backend
from fedready.agents.training_reference import (
    NVFLARE_EXAMPLE_SELECTION_ID,
    NVFLARE_EXAMPLE_SELECTION_PATTERN,
    NVFLARE_EXAMPLE_SELECTION_PROMPT,
)
from fedready.data.contracts import training as training_contracts
from fedready.data.contracts.training import TrainingContract
from fedready.data.qc import visual_qc_decision_passed
from fedready.job_data import export_fed_job_recipe, resolve_recipe_workspace_root, run_fed_job_recipe
from fedready.prompts import render_server_prompt, render_server_prompt_object
from fedready.utils.io import atomic_write_json, safe_path_slug
from fedready.utils.logging import payload_digest, timestamp_utc
from fedready.utils.training_metrics import export_tensorboard_metrics

from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor, ExecutionMode
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.file_model_locator import PTFileModelLocator
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.client.config import ExchangeFormat
from nvflare.job_config.api import FedJob

TRAINING_JOB_SCHEMA_VERSION = "fedready.nvflare_fedavg_training_job.v1"
TRAINING_PLAN_SCHEMA_VERSION = "fedready.training_plan.v1"
TRAINING_CODE_SPEC_SCHEMA_VERSION = "fedready.training_code_spec.v1"
LOCAL_TRAINING_SIMULATION_SCHEMA_VERSION = "fedready.local_training_simulation.v1"
MOCK_TRAINING_DATASET_SCHEMA_VERSION = "fedready.mock_training_dataset.v1"
CLIENT_TRAINING_CONFIG_SCHEMA_VERSION = "fedready.client_training_config.v1"
TRAINING_PACKAGE_INTEGRITY_SCHEMA_VERSION = "fedready.training_package_integrity.v1"
CLIENT_TRAINING_LAUNCHER_SCRIPT = "fedready_training_client_runtime.py"
DEFAULT_TRAINING_TASK_NAME = "train"
DEFAULT_DATASET_ROOT = "data/dataset_fl"
DEFAULT_METRICS_JSONL = "fedready_training_metrics.jsonl"
DEFAULT_TRAINING_CODE_AGENT_MAX_ATTEMPTS = 8
DEFAULT_TRAINING_RUNTIME_AGENT_MAX_ATTEMPTS = 3
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


@dataclass(frozen=True)
class FedAvgTrainingConfig:
    """Training knobs for the current FedAvg exporter."""

    num_rounds: int = 1
    local_epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 1e-3
    target_size: tuple[int, int] = (128, 128)
    num_workers: int = 0
    use_site_intensity_stats: bool = False
    device: str = "auto"
    dataset_root: str = DEFAULT_DATASET_ROOT
    task_name: str = DEFAULT_TRAINING_TASK_NAME


class TrainingCodePreflightError(RuntimeError):
    """Raised when generated training code fails the local NVFlare preflight."""

    def __init__(self, status: dict[str, Any]) -> None:
        super().__init__("server-local NVFlare SimEnv preflight failed")
        self.status = status


def load_extraction_summary(path: str | Path) -> dict[str, Any]:
    """Load a server extraction summary JSON."""

    summary_path = Path(path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("output"), dict):
        payload = payload["output"]
    if not isinstance(payload, dict):
        raise ValueError("extraction summary must be a JSON object")
    if payload.get("schema_version") != "fedready.extraction_round_summary.v1":
        raise ValueError("expected fedready.extraction_round_summary.v1")
    results = payload.get("extraction_results")
    if not isinstance(results, dict) or not results:
        raise ValueError("extraction summary must include non-empty extraction_results for training")
    _attach_generated_contract_from_sibling_strategy(payload, summary_path=summary_path)
    return payload


def _attach_generated_contract_from_sibling_strategy(payload: dict[str, Any], *, summary_path: Path) -> None:
    if isinstance(payload.get("generated_data_contract"), dict):
        return
    strategy_path = summary_path.with_name("extraction_strategy.json")
    try:
        strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return
    if not isinstance(strategy, dict):
        return
    contract = strategy.get("generated_data_contract")
    if isinstance(contract, dict):
        payload["generated_data_contract"] = contract
    materializer = strategy.get("generated_data_materializer")
    if isinstance(materializer, dict):
        payload["generated_data_materializer"] = {
            "schema_version": materializer.get("schema_version"),
            "status": materializer.get("status"),
            "record_type": materializer.get("record_type"),
            "sample_manifest": materializer.get("sample_manifest"),
            "sample_manifest_format": materializer.get("sample_manifest_format"),
            "source_digest": materializer.get("source_digest"),
            "safe_to_share": False,
            "source_files_redacted_for_training_summary": True,
        }


def _training_contract_from_task(*, task: str, extraction_summary: dict[str, Any] | None = None) -> TrainingContract:
    return training_contracts.contract_from_task(task, extraction_summary=extraction_summary)


def _training_contract_from_plan(training_plan: dict[str, Any]) -> TrainingContract:
    return training_contracts.contract_from_plan(training_plan)


def _training_data_contract_payload(
    *,
    task: str,
    task_contract: TrainingContract,
    dataset_root: str,
    extraction_summary: dict[str, Any],
) -> dict[str, Any]:
    payload = task_contract.as_data_contract()
    payload.update(
        {
            "dataset_root": dataset_root,
            "site_folder": "<client_id>",
            "label_rule_source": "extraction_summary",
            "resize_policy": "training_time_resize_or_padding",
        }
    )
    label_summary = _safe_training_label_summary_from_extraction_summary(extraction_summary)
    label_harmonization = training_contracts.label_harmonization_for_training(
        task=task,
        contract=task_contract,
        label_summary=label_summary,
    )
    if label_harmonization:
        payload["label_harmonization"] = label_harmonization
        payload["labels"] = label_harmonization["shared_label_space"]
        payload["observed_label_values"] = [0, 1]
        payload["num_classes"] = 2
        raw_values = label_summary.get("observed_label_values", [])
        if raw_values:
            payload["raw_observed_label_values"] = raw_values
        class_counts = training_contracts.harmonized_binary_class_counts(
            label_summary.get("class_counts") if isinstance(label_summary.get("class_counts"), dict) else {}
        )
        if class_counts:
            payload["class_counts"] = class_counts
    else:
        labels = label_summary.get("labels") if isinstance(label_summary.get("labels"), dict) else {}
        if labels:
            payload["labels"] = labels
            payload["observed_label_values"] = label_summary.get("observed_label_values", [])
            payload["num_classes"] = len(labels)
        class_counts = label_summary.get("class_counts") if isinstance(label_summary.get("class_counts"), dict) else {}
        if class_counts:
            payload["class_counts"] = class_counts
    canonical = _canonical_labels_from_extraction_summary(extraction_summary)
    if canonical:
        payload["canonical_labels"] = canonical
    generated_contract = extraction_summary.get("generated_data_contract")
    if isinstance(generated_contract, dict):
        payload["generated_data_contract"] = generated_contract
    generated_materializer = extraction_summary.get("generated_data_materializer")
    if isinstance(generated_materializer, dict):
        payload["generated_data_materializer"] = generated_materializer
    return payload


def _safe_training_labels_from_extraction_summary(extraction_summary: dict[str, Any]) -> dict[str, Any]:
    label_summary = _safe_training_label_summary_from_extraction_summary(extraction_summary)
    labels = label_summary.get("labels")
    return labels if isinstance(labels, dict) else {}


def _safe_training_label_summary_from_extraction_summary(extraction_summary: dict[str, Any]) -> dict[str, Any]:
    labels: dict[str, str] = {}
    class_counts: dict[str, int] = {}
    canonical = _canonical_labels_from_extraction_summary(extraction_summary)
    for key, value in canonical.items():
        labels[str(value) if isinstance(value, int) else str(key)] = str(key)
    results = extraction_summary.get("extraction_results")
    if isinstance(results, dict):
        for result in results.values():
            if not isinstance(result, dict):
                continue
            extraction = result.get("extraction")
            storage = extraction.get("classification_storage") if isinstance(extraction, dict) else None
            storage_labels = storage.get("labels") if isinstance(storage, dict) else None
            if isinstance(storage_labels, dict):
                for key, value in storage_labels.items():
                    labels.setdefault(str(key), str(value))
            storage_counts = storage.get("class_counts") if isinstance(storage, dict) else None
            if isinstance(storage_counts, dict):
                for key, value in storage_counts.items():
                    try:
                        class_counts[str(key)] = class_counts.get(str(key), 0) + int(value)
                    except (TypeError, ValueError):
                        continue
    ordered_keys = sorted(labels, key=_label_sort_key)
    return {
        "labels": {key: labels[key] for key in ordered_keys},
        "observed_label_values": [_coerce_label_value(key) for key in ordered_keys],
        "class_counts": {key: class_counts[key] for key in sorted(class_counts, key=_label_sort_key)},
    }


def _label_sort_key(value: str) -> tuple[int, Any]:
    coerced = _coerce_label_value(value)
    return (0, coerced) if isinstance(coerced, int) else (1, str(value))


def _coerce_label_value(value: Any) -> int | str:
    text = str(value)
    return int(text) if text.isdigit() else text


def _canonical_labels_from_extraction_summary(extraction_summary: dict[str, Any]) -> dict[str, Any]:
    strategy = extraction_summary.get("extraction_strategy")
    if isinstance(strategy, dict):
        label_rule = strategy.get("label_rule")
        if isinstance(label_rule, dict) and isinstance(label_rule.get("canonical_labels"), dict):
            return dict(label_rule["canonical_labels"])
    results = extraction_summary.get("extraction_results")
    if isinstance(results, dict):
        for result in results.values():
            if not isinstance(result, dict):
                continue
            for key in ("label_rule_applied", "label_rule"):
                label_rule = result.get(key)
                if isinstance(label_rule, dict) and isinstance(label_rule.get("canonical_labels"), dict):
                    return dict(label_rule["canonical_labels"])
    return {}


def build_training_mock_dataset_contract(
    *,
    code_workspace: str | Path,
    training_plan: dict[str, Any],
    training_code_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the fake prepared-client dataset used for local NVFlare preflight."""

    task_contract = _training_contract_from_plan(training_plan)
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


def _ready_client_training_entries(extraction_summary: dict[str, Any], *, task: str) -> list[dict[str, Any]]:
    results = extraction_summary.get("extraction_results", {})
    if not isinstance(results, dict):
        return []
    task_contract = _training_contract_from_task(task=task, extraction_summary=extraction_summary)
    qc_required = bool(task_contract.qc_contract.get("visual_qc_required"))
    entries: list[dict[str, Any]] = []
    for client_id in ready_training_clients(extraction_summary, task=task):
        result = results.get(client_id)
        if not isinstance(result, dict):
            continue
        counts = result.get("counts", {}) if isinstance(result.get("counts"), dict) else {}
        by_split = counts.get("by_split", {}) if isinstance(counts.get("by_split"), dict) else {}
        visual_qc = result.get("visual_qc", {}) if isinstance(result.get("visual_qc"), dict) else {}
        entries.append(
            {
                "client_id": client_id,
                "train_sample_count": int(by_split.get("train") or 0),
                "validation_sample_count": int(by_split.get("validation") or 0),
                "test_sample_count": int(by_split.get("test") or 0),
                "verification_passed": True,
                "visual_qc_required": qc_required,
                "visual_qc_passed": (
                    visual_qc_decision_passed(
                        visual_qc,
                        label_orientation=(
                            result.get("label_orientation")
                            if isinstance(result.get("label_orientation"), dict)
                            else None
                        ),
                    )
                    if qc_required
                    else None
                ),
                "record_type": task_contract.record_type,
                "samples_file": task_contract.samples_file,
                "sample_manifest_format": task_contract.sample_manifest_format,
                "source": "extraction_summary",
            }
        )
    return entries


def _training_implementation_policy(
    *,
    config: FedAvgTrainingConfig,
    client_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    policy = render_server_prompt_object("training_implementation_policy")
    if not isinstance(policy, dict):
        raise TypeError("training_implementation_policy prompt entry must be an object")
    policy = copy.deepcopy(policy)
    policy["run_profile"] = _training_run_profile(config=config, client_entries=client_entries)
    return policy


def _training_run_profile(
    *,
    config: FedAvgTrainingConfig,
    client_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    client_count = len(client_entries)
    train_records = sum(max(0, int(entry.get("train_sample_count") or 0)) for entry in client_entries)
    validation_records = sum(max(0, int(entry.get("validation_sample_count") or 0)) for entry in client_entries)
    test_records = sum(max(0, int(entry.get("test_sample_count") or 0)) for entry in client_entries)
    total_records = train_records + validation_records + test_records
    per_client_records = total_records / client_count if client_count else 0.0
    is_smoke = int(config.num_rounds) <= 3 or (client_count > 0 and per_client_records <= 64)
    profile = "smoke_test" if is_smoke else "actual_training"
    return {
        "schema_version": "fedready.training_run_profile.v1",
        "profile": profile,
        "selection_rule": (
            "smoke_test when num_rounds <= 3 or aggregate prepared records average <= 64 per ready client; "
            "otherwise actual_training"
        ),
        "num_rounds": int(config.num_rounds),
        "ready_client_count": client_count,
        "aggregate_record_count": total_records,
        "aggregate_train_record_count": train_records,
        "aggregate_validation_record_count": validation_records,
        "aggregate_test_record_count": test_records,
        "average_records_per_ready_client": per_client_records,
        "model_guidance": (
            "prefer lightweight representative architecture"
            if profile == "smoke_test"
            else "prefer mainstream task-appropriate architecture over a tiny custom network"
        ),
    }


def build_training_plan(
    *,
    task: str,
    extraction_summary: dict[str, Any],
    config: FedAvgTrainingConfig,
) -> dict[str, Any]:
    """Build a FedAvg server training plan from safe extraction outcomes."""

    task_contract = _training_contract_from_task(task=task, extraction_summary=extraction_summary)
    client_entries = _ready_client_training_entries(extraction_summary, task=task)
    clients = [entry["client_id"] for entry in client_entries]
    if not clients:
        raise ValueError("no clients are ready for training")
    training_implementation_policy = _training_implementation_policy(
        config=config,
        client_entries=client_entries,
    )
    return {
        "schema_version": TRAINING_PLAN_SCHEMA_VERSION,
        "task": task,
        "phase": "federated_learning",
        "algorithm": {
            "name": "FedAvg",
            "implementation": "nvflare.app_common.workflows.fedavg.FedAvg",
            "base_example": NVFLARE_EXAMPLE_SELECTION_ID,
            "base_example_pattern": NVFLARE_EXAMPLE_SELECTION_PATTERN,
            "base_example_selection_prompt": NVFLARE_EXAMPLE_SELECTION_PROMPT,
            "num_rounds": config.num_rounds,
            "task_name": config.task_name,
        },
        "ready_clients": client_entries,
        "ready_client_count": len(clients),
        "fl_update_contract": {
            "params_type": "nvflare.app_common.abstract.fl_model.ParamsType.DIFF",
            "outgoing_params": "model_delta_relative_to_incoming_FLModel_params",
            "required_meta": ["NUM_STEPS_CURRENT_ROUND"],
            "server_aggregator": "nvflare.app_common.workflows.fedavg.FedAvg",
        },
        "training_framework": {
            "selection": "agent_selected_from_task_and_data_contract",
            "base_guidance": NVFLARE_EXAMPLE_SELECTION_PATTERN,
            "required_client_components": ["nvflare.client"],
        },
        "training_implementation_policy": training_implementation_policy,
        "local_training": {
            "local_epochs": config.local_epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "target_size": list(config.target_size),
            "num_workers": config.num_workers,
            "use_site_intensity_stats": config.use_site_intensity_stats,
            "device": config.device,
        },
        "data_contract": _training_data_contract_payload(
            task=task,
            task_contract=task_contract,
            dataset_root=config.dataset_root,
            extraction_summary=extraction_summary,
        ),
        "metric_contract": copy.deepcopy(dict(task_contract.metric_contract)),
        "privacy": {
            "safe_to_share": True,
            "redacted": [
                "local_paths",
                "filenames",
                "raw_sample_ids",
                "raw_images",
                "raw_masks",
                "adapter_manifests",
                "adapter_scripts",
            ],
        },
        "source_extraction_summary_digest": payload_digest(extraction_summary),
    }


def ready_training_clients(extraction_summary: dict[str, Any], *, task: str | None = None) -> list[str]:
    """Return clients that satisfy the task contract's training admission gates."""

    extracted = extraction_summary.get("extracted_clients")
    qc = extraction_summary.get("visual_qc", {})
    qc_passed = qc.get("passed_clients") if isinstance(qc, dict) else None
    task_contract = _training_contract_from_task(task=task or "", extraction_summary=extraction_summary)
    visual_qc_required = bool(task_contract.qc_contract.get("visual_qc_required"))
    results = extraction_summary.get("extraction_results", {})
    if not isinstance(results, dict) or not results:
        return []

    clients = []
    for client_id, result in results.items():
        if not isinstance(result, dict) or result.get("data") != "extracted":
            continue
        verification = result.get("verification", {})
        if not (isinstance(verification, dict) and verification.get("passed") is True):
            continue
        if visual_qc_required:
            visual_qc = result.get("visual_qc", {})
            if not visual_qc_decision_passed(
                visual_qc if isinstance(visual_qc, dict) else None,
                label_orientation=(
                    result.get("label_orientation") if isinstance(result.get("label_orientation"), dict) else None
                ),
            ):
                continue
        if isinstance(extracted, list) and client_id not in extracted:
            continue
        if visual_qc_required and isinstance(qc_passed, list) and client_id not in qc_passed:
            continue
        counts = result.get("counts", {})
        by_split = counts.get("by_split", {}) if isinstance(counts, dict) else {}
        if not isinstance(by_split, dict) or int(by_split.get("train") or 0) <= 0:
            continue
        clients.append(str(client_id))
    return sorted(clients)


def prepare_training_code_with_server_agent(
    *,
    task: str,
    extraction_summary: dict[str, Any],
    training_plan: dict[str, Any],
    run_dir: str | Path,
    session_id: str,
    agent_backend: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
    max_attempts: int = DEFAULT_TRAINING_CODE_AGENT_MAX_ATTEMPTS,
) -> dict[str, Any]:
    """Request task-dependent training code and give the agent one local validation loop."""

    code_workspace = Path(run_dir) / "training_phase" / "server_generated_code"
    code_workspace.mkdir(parents=True, exist_ok=True)
    feedback_dir = Path(run_dir) / "training_phase" / "server"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    backend = _build_agent_backend(
        kind=agent_backend,
        run_dir=Path(run_dir),
        session_id=session_id,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    server_agent = ServerAgent(backend)
    agent_training_plan = _training_plan_with_mock_simulation_contract(
        training_plan=training_plan,
        code_workspace=code_workspace,
    )
    turn = server_agent.implement_training_code(
        task={"task": task},
        extraction_summary=extraction_summary,
        training_plan=agent_training_plan,
        code_workspace=str(code_workspace.resolve()),
    )
    attempts = max(1, int(max_attempts))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            if turn.output.get("status") != "implemented":
                raise RuntimeError(
                    "server agent did not implement training code: "
                    f"{turn.output.get('status')} {turn.output.get('reason')}"
                )
            return _validate_training_code_with_local_preflight(
                spec=turn.output,
                training_plan=agent_training_plan,
                code_workspace=code_workspace,
                run_dir=run_dir,
                attempt=attempt,
                stage="initial",
            )
        except Exception as exc:
            last_error = exc
            feedback = build_training_code_validation_feedback(
                spec=turn.output,
                error=exc,
                attempt=attempt,
                max_attempts=attempts,
                code_workspace=code_workspace,
                require_local_simulation=False,
            )
            if isinstance(exc, TrainingCodePreflightError):
                feedback["preflight_status"] = exc.status
                feedback["agent_instruction"] = render_server_prompt("training_preflight_revision_initial")
            _write_json(
                feedback_dir / f"training_code_validation_attempt_{attempt}.json",
                feedback,
            )
            if attempt >= attempts:
                break
            turn = server_agent.revise_training_code(
                task={"task": task},
                extraction_summary=extraction_summary,
                training_plan=agent_training_plan,
                code_workspace=str(code_workspace.resolve()),
                previous_code_spec=turn.output,
                validation_feedback=feedback,
            )
    raise RuntimeError(
        "server agent training code failed local validation after " f"{attempts} attempt(s): {last_error}"
    ) from last_error


def _validate_training_code_with_local_preflight(
    *,
    spec: dict[str, Any],
    training_plan: dict[str, Any],
    code_workspace: str | Path,
    run_dir: str | Path,
    attempt: int,
    stage: str,
) -> dict[str, Any]:
    """Run basic checks, then validate generated code with a local SimEnv preflight."""

    code_spec = validate_training_code_spec(
        spec,
        code_workspace=code_workspace,
        require_local_simulation=False,
    )
    preflight = run_training_code_preflight(
        training_plan=training_plan,
        training_code_spec=code_spec,
        code_workspace=code_workspace,
        run_dir=run_dir,
        attempt=attempt,
        stage=stage,
    )
    feedback_dir = Path(run_dir) / "training_phase" / "server"
    stage_slug = _preflight_stage_slug(stage)
    _write_json(
        feedback_dir / f"training_code_preflight_{stage_slug}_attempt_{attempt}.json",
        preflight,
    )
    if not preflight.get("succeeded"):
        raise TrainingCodePreflightError(preflight)

    validated = dict(code_spec)
    validated["local_simulation"] = preflight["local_simulation"]
    validated["package_integrity"] = build_training_package_integrity(Path(validated["package_dir"]))
    return validate_training_code_spec(validated, code_workspace=code_workspace)


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
    _write_json(input_path, payload)

    research_root = Path(__file__).resolve().parents[1]
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
        use_site_intensity_stats=False,
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
    task_contract = _training_contract_from_plan(training_plan)
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
    _write_json(
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
    _write_json(
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


def prepare_client_training_configs(
    *,
    training_plan: dict[str, Any],
    training_code_spec: dict[str, Any],
    project_root: str | Path,
    run_dir: str | Path,
    session_id: str,
    config: FedAvgTrainingConfig,
    agent_backend: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, dict[str, Any]]:
    """Prepare client runtime configs without reading client-local datasets centrally.

    The exported config is derived from server-visible extraction-summary facts.
    Actual manifest.json and contract-declared sample manifest checks run inside each NVFlare client app
    via ``fedready_training_client_runtime.py`` immediately before the generated
    training entry script is invoked.
    """

    _ = (
        project_root,
        run_dir,
        session_id,
        agent_backend,
        timeout_seconds,
        poll_interval_seconds,
    )
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
    _write_json(job_path / "fedready_training_plan.json", training_plan)
    _write_json(job_path / "fedready_training_code_spec.json", _safe_code_spec(code_spec))
    _write_json(job_path / "fedready_client_training_configs.json", ready_configs)
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


def prepare_and_export_fedavg_training_job(
    *,
    extraction_summary_path: str | Path,
    project_root: str | Path,
    output_dir: str | Path,
    session_id: str | None,
    job_root: str | Path,
    job_name: str,
    task: str = "<TASK_DESCRIPTION>",
    config: FedAvgTrainingConfig | None = None,
    training_code_spec_path: str | Path | None = None,
    agent_backend: str = "codex",
    agent_timeout_seconds: float = 3600.0,
    agent_poll_interval_seconds: float = 2.0,
    min_clients: int | None = None,
) -> dict[str, Any]:
    """Prepare agent-generated training code/configs and export a FedAvg job."""

    cfg = config or FedAvgTrainingConfig()
    session = session_id or _default_training_session_id(task)
    run_dir = Path(output_dir) / session
    run_dir.mkdir(parents=True, exist_ok=True)
    extraction_summary = load_extraction_summary(extraction_summary_path)
    training_plan = build_training_plan(task=task, extraction_summary=extraction_summary, config=cfg)
    _write_json(run_dir / "training_phase" / "server" / "training_plan.json", training_plan)

    if training_code_spec_path is not None:
        code_spec = validate_training_code_spec(_read_json(training_code_spec_path))
    else:
        code_spec = prepare_training_code_with_server_agent(
            task=task,
            extraction_summary=extraction_summary,
            training_plan=training_plan,
            run_dir=run_dir,
            session_id=session,
            agent_backend=agent_backend,
            timeout_seconds=agent_timeout_seconds,
            poll_interval_seconds=agent_poll_interval_seconds,
        )

    client_configs = prepare_client_training_configs(
        training_plan=training_plan,
        training_code_spec=code_spec,
        project_root=project_root,
        run_dir=run_dir,
        session_id=session,
        config=cfg,
        agent_backend=agent_backend,
        timeout_seconds=agent_timeout_seconds,
        poll_interval_seconds=agent_poll_interval_seconds,
    )
    _write_json(
        run_dir / "training_phase" / "server" / "client_training_configs.json",
        client_configs,
    )

    export = export_fedavg_training_job(
        job_root=job_root,
        job_name=job_name,
        training_plan=training_plan,
        training_code_spec=code_spec,
        client_configs=client_configs,
        min_clients=min_clients,
    )
    _write_json(run_dir / "training_phase" / "server" / "fedavg_job_export.json", export)
    return {
        **export,
        "session_id": session,
        "run_dir": str(run_dir.resolve()),
    }


def run_fedavg_training_job(
    *,
    workspace: str | Path,
    threads: int | None = None,
    log_config: str | None = "concise",
    **kwargs: Any,
) -> dict[str, Any]:
    """Prepare/export and run the generated FedAvg training job."""

    export = prepare_and_export_fedavg_training_job(**kwargs)
    session = str(export["session_id"])
    run_dir = Path(export["run_dir"])
    requested_workspace = Path(workspace)
    base_job_name = str(export["job_name"])
    recipe_workspace_root = resolve_recipe_workspace_root(
        requested_workspace,
        session_id=session,
        job_name=base_job_name,
    )
    cfg = kwargs.get("config") or FedAvgTrainingConfig()
    project_root = Path(kwargs.get("project_root", "."))
    agent_backend = str(kwargs.get("agent_backend", "codex"))
    can_revise_runtime = kwargs.get("training_code_spec_path") is None
    runtime_attempts = DEFAULT_TRAINING_RUNTIME_AGENT_MAX_ATTEMPTS if can_revise_runtime else 1
    last_status: dict[str, Any] | None = None
    last_error: Exception | None = None

    for runtime_attempt in range(1, runtime_attempts + 1):
        job_path = Path(export["job_path"])
        training_plan = export["training_plan"]
        code_spec = validate_training_code_spec(_read_json(job_path / "fedready_training_code_spec.json"))
        raw_configs = _read_json(job_path / "fedready_client_training_configs.json")
        if not isinstance(raw_configs, dict):
            raise ValueError("client training configs must be a JSON object")
        job = _build_fedavg_job_object(
            job_name=str(export["job_name"]),
            training_plan=training_plan,
            training_code_spec=code_spec,
            client_configs=raw_configs,
            min_clients=export.get("min_clients"),
        )
        client_ids = list(export["client_ids"])
        workspace_path = recipe_workspace_root / str(export["job_name"])
        simulator_error: Exception | None = None
        recipe_run: dict[str, str] | None = None
        try:
            recipe_run = run_fed_job_recipe(
                job,
                workspace_root=recipe_workspace_root,
                clients=client_ids,
                threads=threads,
                log_config=log_config,
            )
            workspace_path = Path(recipe_run["workspace"])
        except Exception as exc:  # pragma: no cover - simulator failures depend on generated code
            simulator_error = exc
        simulator_status = summarize_simulator_status(workspace_path)
        if recipe_run is not None:
            simulator_status["recipe_run"] = recipe_run
        if simulator_error is not None:
            simulator_status["simulator_exception_type"] = type(simulator_error).__name__
            simulator_status["simulator_exception"] = str(simulator_error)[-4000:]
        if simulator_error is None and simulator_status["succeeded"]:
            try:
                simulator_status["tensorboard"] = export_tensorboard_metrics(workspace_path)
            except Exception as exc:  # visualization export must not invalidate completed training
                simulator_status["tensorboard"] = {
                    "schema_version": "fedready.tensorboard_export.v1",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        _write_json(
            run_dir / "training_phase" / "server" / f"simulator_status_attempt_{runtime_attempt}.json",
            simulator_status,
        )
        if simulator_error is None and simulator_status["succeeded"]:
            return {
                **export,
                "workspace_arg": str(requested_workspace),
                "workspace_root": str(recipe_workspace_root),
                "workspace": str(workspace_path),
                "recipe_run": recipe_run,
                "client_ids": client_ids,
                "simulator_status": simulator_status,
                "runtime_attempt": runtime_attempt,
            }

        last_status = simulator_status
        last_error = simulator_error
        if runtime_attempt >= runtime_attempts or not can_revise_runtime:
            break

        runtime_feedback = build_training_runtime_feedback(
            spec=code_spec,
            simulator_status=simulator_status,
            workspace=workspace_path,
            attempt=runtime_attempt,
            max_attempts=runtime_attempts,
        )
        _write_json(
            run_dir / "training_phase" / "server" / f"training_runtime_feedback_attempt_{runtime_attempt}.json",
            runtime_feedback,
        )
        extraction_summary = load_extraction_summary(kwargs["extraction_summary_path"])
        revised_code_spec = revise_training_code_after_runtime_feedback(
            task=str(training_plan.get("task") or kwargs.get("task") or "<TASK_DESCRIPTION>"),
            extraction_summary=extraction_summary,
            training_plan=training_plan,
            previous_code_spec=code_spec,
            runtime_feedback=runtime_feedback,
            run_dir=run_dir,
            session_id=session,
            agent_backend=agent_backend,
            timeout_seconds=float(kwargs.get("agent_timeout_seconds", 3600.0)),
            poll_interval_seconds=float(kwargs.get("agent_poll_interval_seconds", 2.0)),
        )
        refreshed_configs = _refresh_client_training_configs_for_code_spec(
            raw_configs,
            training_plan=training_plan,
            training_code_spec=revised_code_spec,
            config=cfg,
            project_root=project_root,
        )
        retry_job_name = _runtime_attempt_job_name(base_job_name, runtime_attempt + 1)
        export = export_fedavg_training_job(
            job_root=kwargs["job_root"],
            job_name=retry_job_name,
            training_plan=training_plan,
            training_code_spec=revised_code_spec,
            client_configs=refreshed_configs,
            min_clients=export.get("min_clients"),
        )
        export = {
            **export,
            "session_id": session,
            "run_dir": str(run_dir.resolve()),
            "runtime_retry_of": base_job_name,
            "runtime_attempt": runtime_attempt + 1,
        }
        _write_json(
            run_dir / "training_phase" / "server" / f"fedavg_job_export_attempt_{runtime_attempt + 1}.json",
            export,
        )
        _write_json(run_dir / "training_phase" / "server" / "fedavg_job_export.json", export)

    error_suffix = f": {last_error}" if last_error is not None else ""
    raise RuntimeError(f"NVFlare simulator training did not finish successfully: {last_status}{error_suffix}")


def revise_training_code_after_runtime_feedback(
    *,
    task: str,
    extraction_summary: dict[str, Any],
    training_plan: dict[str, Any],
    previous_code_spec: dict[str, Any],
    runtime_feedback: dict[str, Any],
    run_dir: str | Path,
    session_id: str,
    agent_backend: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
    max_attempts: int = DEFAULT_TRAINING_CODE_AGENT_MAX_ATTEMPTS,
) -> dict[str, Any]:
    """Ask the server coding agent to revise code after an actual simulator failure."""

    run_path = Path(run_dir)
    code_workspace = run_path / "training_phase" / "server_generated_code"
    feedback_dir = run_path / "training_phase" / "server"
    backend = _build_agent_backend(
        kind=agent_backend,
        run_dir=run_path,
        session_id=session_id,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    server_agent = ServerAgent(backend)
    agent_training_plan = _training_plan_with_mock_simulation_contract(
        training_plan=training_plan,
        code_workspace=code_workspace,
    )
    turn = server_agent.revise_training_code(
        task={"task": task},
        extraction_summary=extraction_summary,
        training_plan=agent_training_plan,
        code_workspace=str(code_workspace.resolve()),
        previous_code_spec=previous_code_spec,
        validation_feedback=runtime_feedback,
    )
    last_error: Exception | None = None
    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        try:
            if turn.output.get("status") != "implemented":
                raise RuntimeError(
                    "server agent did not implement revised training code: "
                    f"{turn.output.get('status')} {turn.output.get('reason')}"
                )
            return _validate_training_code_with_local_preflight(
                spec=turn.output,
                training_plan=agent_training_plan,
                code_workspace=code_workspace,
                run_dir=run_dir,
                attempt=attempt,
                stage="runtime_revision",
            )
        except Exception as exc:
            last_error = exc
            feedback = build_training_code_validation_feedback(
                spec=turn.output,
                error=exc,
                attempt=attempt,
                max_attempts=attempts,
                code_workspace=code_workspace,
                require_local_simulation=False,
            )
            if isinstance(exc, TrainingCodePreflightError):
                feedback["preflight_status"] = exc.status
                feedback["agent_instruction"] = render_server_prompt("training_preflight_revision_runtime")
            feedback["source_runtime_feedback"] = runtime_feedback
            _write_json(
                feedback_dir / f"runtime_training_code_validation_attempt_{attempt}.json",
                feedback,
            )
            if attempt >= attempts:
                break
            turn = server_agent.revise_training_code(
                task={"task": task},
                extraction_summary=extraction_summary,
                training_plan=agent_training_plan,
                code_workspace=str(code_workspace.resolve()),
                previous_code_spec=turn.output,
                validation_feedback=feedback,
            )
    raise RuntimeError(
        "server agent runtime training-code revision failed local validation after "
        f"{attempts} attempt(s): {last_error}"
    ) from last_error


def build_training_runtime_feedback(
    *,
    spec: dict[str, Any],
    simulator_status: dict[str, Any],
    workspace: str | Path,
    attempt: int,
    max_attempts: int,
) -> dict[str, Any]:
    """Build concise simulator runtime feedback for the server coding agent."""

    snippets = _simulator_error_snippets(workspace)
    return {
        "schema_version": "fedready.training_runtime_feedback.v1",
        "status": "failed",
        "stage": "nvflare_simulator_runtime",
        "attempt": attempt,
        "max_attempts": max_attempts,
        "error_message": "NVFlare simulator did not finish successfully with the generated training package.",
        "simulator_status": simulator_status,
        "previous_output_summary": {
            "schema_version": spec.get("schema_version"),
            "status": spec.get("status"),
            "package_dir": spec.get("package_dir"),
            "entry_script": spec.get("entry_script"),
            "model_class_path": spec.get("model_class_path"),
            "selected_reference_example_path": _selected_reference_example_path(spec),
            "reference_api_evidence_count": (
                len(_reference_api_evidence(spec)) if isinstance(_reference_api_evidence(spec), list) else 0
            ),
        },
        "log_error_snippets": snippets,
        "required_fix": render_server_prompt("training_runtime_required_fix"),
    }


def build_training_code_validation_feedback(
    *,
    spec: dict[str, Any],
    error: Exception,
    attempt: int,
    max_attempts: int,
    code_workspace: str | Path,
    require_local_simulation: bool = False,
) -> dict[str, Any]:
    """Build local validation feedback for an agent revision pass."""

    errors = _training_code_spec_validation_errors(
        spec,
        code_workspace=code_workspace,
        require_local_simulation=require_local_simulation,
    )
    if not errors:
        errors = [{"kind": type(error).__name__, "message": str(error)}]
    code_workspace_path = Path(code_workspace).resolve()
    expected_package_dir = code_workspace_path / "fedready_task_training"
    return {
        "schema_version": "fedready.training_code_validation_feedback.v1",
        "status": "failed",
        "attempt": attempt,
        "max_attempts": max_attempts,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "errors": errors,
        "code_workspace": str(code_workspace_path),
        "expected_package_dir": str(expected_package_dir),
        "expected_schema_version": TRAINING_CODE_SPEC_SCHEMA_VERSION,
        "expected_status": "implemented",
        "previous_output_summary": {
            "schema_version": (spec.get("schema_version") if isinstance(spec, dict) else None),
            "status": spec.get("status") if isinstance(spec, dict) else None,
            "package_dir": spec.get("package_dir") if isinstance(spec, dict) else None,
            "entry_script": (spec.get("entry_script") if isinstance(spec, dict) else None),
            "model_class_path": (spec.get("model_class_path") if isinstance(spec, dict) else None),
            "selected_reference_example_path": (
                _selected_reference_example_path(spec) if isinstance(spec, dict) else None
            ),
            "reference_api_evidence_count": (
                len(_reference_api_evidence(spec))
                if isinstance(spec, dict) and isinstance(_reference_api_evidence(spec), list)
                else 0
            ),
        },
        "required_fields_to_preserve": [
            "schema_version",
            "status",
            "package_dir",
            "entry_script",
            "model_class_path",
            "selected_reference_example_path",
            "reference_api_evidence",
            "task_script_args_template",
            "mock_record_template",
        ],
        "agent_instruction": render_server_prompt("training_validation_agent_instruction"),
    }


def validate_training_code_spec(
    spec: dict[str, Any],
    *,
    code_workspace: str | Path | None = None,
    require_local_simulation: bool = True,
) -> dict[str, Any]:
    errors = _training_code_spec_validation_errors(
        spec,
        code_workspace=code_workspace,
        require_local_simulation=require_local_simulation,
    )
    if errors:
        message = "; ".join(str(error.get("message")) for error in errors)
        if any(str(error.get("kind", "")).endswith("not_found") for error in errors):
            raise FileNotFoundError(message)
        raise ValueError(message)
    validated = dict(spec)
    package_dir = _resolve_training_package_dir(str(spec["package_dir"]), code_workspace=code_workspace)
    validated["package_dir"] = str(package_dir.resolve())
    if require_local_simulation:
        validated["package_integrity"] = build_training_package_integrity(package_dir)
    return validated


def _training_code_spec_validation_errors(
    spec: dict[str, Any],
    *,
    code_workspace: str | Path | None = None,
    require_local_simulation: bool = True,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not isinstance(spec, dict):
        return [
            {
                "kind": "invalid_spec",
                "message": "training code spec must be a JSON object",
            }
        ]
    if spec.get("deterministic_ground_truth") is True:
        errors.append(
            {
                "kind": "deterministic_training_spec",
                "message": "deterministic or reference training-code specs are unsupported",
            }
        )
    if spec.get("schema_version") != TRAINING_CODE_SPEC_SCHEMA_VERSION:
        errors.append(
            {
                "kind": "schema_mismatch",
                "message": f"expected {TRAINING_CODE_SPEC_SCHEMA_VERSION}",
                "actual": spec.get("schema_version"),
            }
        )
    if spec.get("status") != "implemented":
        errors.append(
            {
                "kind": "status_mismatch",
                "message": "training code spec status must be implemented",
                "actual": spec.get("status"),
            }
        )
    for key in ("package_dir", "entry_script", "model_class_path"):
        value = spec.get(key)
        if not isinstance(value, str) or not value:
            error = {
                "kind": "missing_required_key",
                "message": f"training code spec {key} must be a non-empty string",
                "key": key,
                "actual_type": type(value).__name__,
            }
            if isinstance(value, dict):
                error["actual_keys"] = sorted(str(item) for item in value)
            errors.append(error)
    if errors:
        return errors

    package_dir = _resolve_training_package_dir(str(spec["package_dir"]), code_workspace=code_workspace)
    if not package_dir.exists() or not package_dir.is_dir():
        return [
            {
                "kind": "package_dir_not_found",
                "message": f"training code package_dir not found: {package_dir}",
                "package_dir": str(spec["package_dir"]),
                "resolved_package_dir": str(package_dir),
            }
        ]

    entry_path = _training_entry_path(package_dir=package_dir, entry_script=spec["entry_script"])
    if not entry_path.exists():
        error: dict[str, Any] = {
            "kind": "entry_script_not_found",
            "message": f"training entry script not found under package_dir: {spec['entry_script']}",
            "package_dir": str(package_dir),
            "expected_entry_path": str(entry_path),
            "entry_script": spec["entry_script"],
        }
        nested_entry_path = package_dir / spec["entry_script"]
        if nested_entry_path.exists():
            parts = Path(spec["entry_script"]).parts
            if len(parts) >= 2:
                suggested_package_dir = package_dir / parts[0]
                error["likely_cause"] = (
                    "package_dir points at the parent code workspace instead of the Python package directory"
                )
                error["suggested_package_dir"] = str(suggested_package_dir.resolve())
        errors.append(error)

    model_file = _training_model_file_path(package_dir=package_dir, model_class_path=spec["model_class_path"])
    if model_file is not None and not model_file.exists():
        error = {
            "kind": "model_file_not_found",
            "message": f"training model module not found for model_class_path: {spec['model_class_path']}",
            "expected_model_file": str(model_file),
        }
        model_class_path = str(spec["model_class_path"])
        if "/" in model_class_path or model_class_path.endswith(".py"):
            error["likely_cause"] = (
                "model_class_path is a filesystem path; NVFlare requires a dotted Python import path "
                "that includes the model class"
            )
            error["required_format"] = f"{package_dir.name}.model.ModelClass"
        errors.append(error)

    model_args = spec.get("model_args", {})
    if not isinstance(model_args, dict):
        errors.append(
            {
                "kind": "model_args_type",
                "message": "training code spec model_args must be an object",
            }
        )
    mock_record_template = spec.get("mock_record_template")
    if mock_record_template is not None and not isinstance(mock_record_template, dict):
        errors.append(
            {
                "kind": "mock_record_template_type",
                "message": "training code spec mock_record_template must be an object when provided",
            }
        )

    py_files = sorted(package_dir.rglob("*.py"))
    if len(py_files) <= 50:
        for py_file in py_files:
            try:
                py_compile.compile(str(py_file), doraise=True)
            except py_compile.PyCompileError as exc:
                errors.append(
                    {
                        "kind": "python_compile_failed",
                        "message": f"generated Python file failed to compile: {py_file}",
                        "path": str(py_file),
                        "details": str(exc),
                    }
                )
    # Package integrity is recorded by FedReady as provenance after local
    # preflight, but it is not a deterministic admission gate here. Artifact
    # immutability and deployment integrity belong to the FL/job system rather
    # than the coding-agent validation loop.
    if require_local_simulation and not errors:
        errors.extend(_training_local_simulation_errors(spec))
    return errors


def build_training_package_integrity(package_dir: str | Path) -> dict[str, Any]:
    """Return a deterministic content manifest for deployable training files."""

    root = Path(package_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"training package directory is missing: {root}")
    files: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        relative_text = relative.as_posix()
        aggregate.update(relative_text.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\0")
        files.append(
            {
                "path": relative_text,
                "sha256": f"sha256:{digest}",
                "size_bytes": len(content),
            }
        )
    if not files:
        raise ValueError(f"training package contains no deployable files: {root}")
    return {
        "schema_version": TRAINING_PACKAGE_INTEGRITY_SCHEMA_VERSION,
        "sha256": f"sha256:{aggregate.hexdigest()}",
        "file_count": len(files),
        "files": files,
    }


def _training_local_simulation_errors(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Require live agent specs to include a FedReady local SimEnv preflight report."""

    simulation = spec.get("local_simulation")
    if not isinstance(simulation, dict):
        return [
            {
                "kind": "local_simulation_missing",
                "message": (
                    "live training code spec must include local_simulation from FedReady local NVFlare preflight"
                ),
            }
        ]

    errors: list[dict[str, Any]] = []
    if simulation.get("schema_version") != LOCAL_TRAINING_SIMULATION_SCHEMA_VERSION:
        errors.append(
            {
                "kind": "local_simulation_schema_mismatch",
                "message": f"local_simulation.schema_version must be {LOCAL_TRAINING_SIMULATION_SCHEMA_VERSION}",
                "actual": simulation.get("schema_version"),
            }
        )
    if simulation.get("status") != "passed":
        errors.append(
            {
                "kind": "local_simulation_not_passed",
                "message": "FedReady local NVFlare SimEnv preflight must pass before training job export",
                "actual": simulation.get("status"),
            }
        )
    if simulation.get("client_deployment") is not False:
        errors.append(
            {
                "kind": "local_simulation_client_deployment_not_false",
                "message": "local_simulation.client_deployment must be false; pre-export simulation must not deploy to clients",
                "actual": simulation.get("client_deployment"),
            }
        )
    nonempty_metric_artifacts = simulation.get("nonempty_metric_artifacts")
    has_metric_artifact = (
        simulation.get("metric_artifact_available") is True
        and isinstance(nonempty_metric_artifacts, list)
        and any(isinstance(path, str) and path for path in nonempty_metric_artifacts)
    )
    if not has_metric_artifact:
        errors.append(
            {
                "kind": "local_simulation_metric_artifact_missing",
                "message": "local_simulation must include a non-empty metric artifact from the local NVFlare preflight",
                "metric_artifact_available": simulation.get("metric_artifact_available"),
                "nonempty_metric_artifacts": nonempty_metric_artifacts,
            }
        )
    dataset_contract = simulation.get("dataset_contract")
    if isinstance(dataset_contract, dict) and dataset_contract.get("schema_version") not in (
        None,
        MOCK_TRAINING_DATASET_SCHEMA_VERSION,
    ):
        errors.append(
            {
                "kind": "local_simulation_dataset_contract_schema_mismatch",
                "message": f"local_simulation.dataset_contract.schema_version must be {MOCK_TRAINING_DATASET_SCHEMA_VERSION}",
                "actual": dataset_contract.get("schema_version"),
            }
        )
    return errors


def _selected_reference_example_path(spec: dict[str, Any]) -> str | None:
    for key in (
        "selected_reference_example_path",
        "selected_reference_example_url",
        "reference_example_path",
        "reference_example_url",
    ):
        value = spec.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    selection = spec.get("reference_example_selection")
    if isinstance(selection, dict):
        for key in (
            "selected_reference_example_path",
            "path",
            "url",
            "source",
            "example_path",
            "example_url",
        ):
            value = selection.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    value = spec.get("base_reference")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _reference_api_evidence(spec: dict[str, Any]) -> Any:
    value = spec.get("reference_api_evidence")
    if isinstance(value, list) and value:
        return value
    selection = spec.get("reference_example_selection")
    if isinstance(selection, dict):
        for key in (
            "reference_api_evidence",
            "api_evidence",
            "api_observations",
            "evidence",
        ):
            value = selection.get(key)
            if isinstance(value, list) and value:
                return value
    return spec.get("reference_api_evidence")


def _resolve_training_package_dir(package_dir: str, *, code_workspace: str | Path | None = None) -> Path:
    path = Path(package_dir)
    if path.is_absolute() or code_workspace is None:
        return path
    return Path(code_workspace) / path


def _training_entry_path(*, package_dir: Path, entry_script: str) -> Path:
    entry_path = package_dir / Path(entry_script).name
    if "/" in entry_script:
        parts = Path(entry_script).parts
        if parts[-2:] and parts[-2] == package_dir.name:
            entry_path = package_dir / parts[-1]
    return entry_path


def _training_model_file_path(*, package_dir: Path, model_class_path: str) -> Path | None:
    parts = model_class_path.split(".")
    if len(parts) < 2:
        return None
    module_parts = parts[:-1]
    if module_parts and module_parts[0] == package_dir.name:
        module_parts = module_parts[1:]
    if not module_parts:
        return None
    return package_dir.joinpath(*module_parts).with_suffix(".py")


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

    def record_message(message: str) -> None:
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
                record_message(message)
                if record.get("levelname") == "ERROR":
                    error_count += 1
                if "Finished FedAvg" in message:
                    finished_fedavg = True
                if "Aggregated " in message and " results" in message:
                    aggregated_result_messages.append(message)
    elif log_txt.exists():
        with log_txt.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                record_message(line)
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
                record_message(str(record.get("message") or ""))
    for path in sorted(root.rglob("log.txt")):
        if path == log_txt:
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                record_message(line)

    model_paths = sorted(str(path) for path in server_dir.rglob("FL_global_model.pt"))
    metric_paths = sorted(path for path in root.rglob(DEFAULT_METRICS_JSONL))
    metric_artifacts = [str(path) for path in metric_paths]
    nonempty_metric_artifacts = [str(path) for path in metric_paths if path.is_file() and path.stat().st_size > 0]
    has_empty_results = bool(empty_result_clients)
    has_non_tensor_params = bool(non_tensor_param_warnings)
    has_metric_artifact = bool(nonempty_metric_artifacts)
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
        "empty_result_clients": sorted(empty_result_clients),
        "client_communication_error_clients": sorted(communication_error_clients),
        "non_tensor_param_warning_count": len(non_tensor_param_warnings),
        "non_tensor_param_warnings": non_tensor_param_warnings[-3:],
        "global_model_paths": model_paths,
        "metric_artifacts": metric_artifacts,
        "nonempty_metric_artifacts": nonempty_metric_artifacts,
        "metric_artifact_available": has_metric_artifact,
        "persisted_global_model": bool(model_paths),
        "aggregated_result_messages": aggregated_result_messages[-3:],
        "completed_with_warnings": completed_with_warnings,
        "warning_codes": warning_codes,
        "succeeded": (finished_fedavg and not has_non_tensor_params and bool(model_paths) and has_metric_artifact),
    }


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
    project_root: Path,
) -> dict[str, dict[str, Any]]:
    _ = project_root
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
        _add_client_runtime_launcher_if_needed(job, target=client_id, client_config=ready_configs[client_id])
        job.to(
            ClientAPIExecutor(
                execution_mode=ExecutionMode.IN_PROCESS,
                task_script_path=_client_task_script_path(ready_configs[client_id], training_code_spec),
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
    task_contract = _training_contract_from_plan(training_plan)
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
    task_contract = _training_contract_from_plan(training_plan)
    training_args = _format_task_args(
        template=str(training_code_spec.get("task_script_args_template") or ""),
        client_id=client_id,
        config=config,
    )
    launcher_args = [
        "--fedready-entry-script",
        shlex.quote(str(training_code_spec["entry_script"])),
        "--fedready-expected-train",
        str(int(train_sample_count)),
        "--fedready-expected-validation",
        str(int(validation_sample_count)),
        "--fedready-expected-test",
        str(int(test_sample_count)),
        "--fedready-package-dir",
        shlex.quote(Path(str(training_code_spec["package_dir"])).name),
        "--fedready-record-type",
        shlex.quote(str(task_contract.record_type)),
        "--fedready-samples-file",
        shlex.quote(str(task_contract.samples_file)),
        "--fedready-sample-manifest-format",
        shlex.quote(str(task_contract.sample_manifest_format)),
        "--fedready-required-fields",
        json.dumps(list(task_contract.sample_fields), separators=(",", ":")),
        "--fedready-visual-qc-required",
        "true" if task_contract.qc_contract.get("visual_qc_required") else "false",
    ]
    return " ".join(launcher_args + [training_args])


def _client_task_script_path(config: dict[str, Any], training_code_spec: dict[str, Any]) -> str:
    value = config.get("task_script_path")
    return str(value) if isinstance(value, str) and value.strip() else str(training_code_spec["entry_script"])


def _add_client_runtime_launcher_if_needed(job: FedJob, *, target: str, client_config: dict[str, Any]) -> None:
    if client_config.get("task_script_path") != CLIENT_TRAINING_LAUNCHER_SCRIPT:
        return
    runtime_dir = Path(__file__).resolve().parent / "flare"
    job.add_file_to(
        str(runtime_dir / CLIENT_TRAINING_LAUNCHER_SCRIPT),
        target,
        app_folder_type="custom",
    )


def _format_task_args(
    *,
    template: str,
    client_id: str,
    config: FedAvgTrainingConfig,
    dataset_root: str | None = None,
) -> str:
    values = {
        "dataset_root": shlex.quote(dataset_root or config.dataset_root),
        "client_id": shlex.quote(client_id),
        "local_epochs": config.local_epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "target_width": config.target_size[0],
        "target_height": config.target_size[1],
        "num_workers": config.num_workers,
        "device": shlex.quote(config.device),
        "metrics_jsonl": shlex.quote(DEFAULT_METRICS_JSONL),
    }
    if not template:
        template = (
            "--dataset-root {dataset_root} --client-id {client_id} "
            "--local-epochs {local_epochs} --batch-size {batch_size} "
            "--learning-rate {learning_rate} --target-size {target_width} {target_height} "
            "--num-workers {num_workers} --device {device} --metrics-jsonl {metrics_jsonl}"
        )
    args = template.format(**values)
    if config.use_site_intensity_stats and "--use-site-intensity-stats" not in args:
        args += " --use-site-intensity-stats"
    return args


def _add_training_package(job: FedJob, *, package_dir: Path, target: str) -> None:
    if package_dir.name == "src":
        job.add_file_to(str(package_dir), target, dest_dir="src", app_folder_type="custom")
    elif (package_dir / "__init__.py").exists():
        job.add_file_to(str(package_dir.parent), target, app_folder_type="custom")
    else:
        job.add_file_to(str(package_dir), target, app_folder_type="custom")


def _safe_code_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": spec.get("schema_version"),
        "status": spec.get("status"),
        "package_dir": spec.get("package_dir"),
        "entry_script": spec.get("entry_script"),
        "model_class_path": spec.get("model_class_path"),
        "model_args": spec.get("model_args", {}),
        "task_script_args_template": spec.get("task_script_args_template"),
        "framework": spec.get("framework"),
        "base_reference": spec.get("base_reference"),
        "selected_reference_example_path": spec.get("selected_reference_example_path"),
        "reference_api_evidence": spec.get("reference_api_evidence"),
        "metric_artifacts": spec.get("metric_artifacts"),
        "local_simulation": spec.get("local_simulation"),
        "package_integrity": spec.get("package_integrity"),
        "safe_to_share": False,
        "local_paths_redacted_for_server_summary": True,
    }


def _safe_adapter_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "status": value.get("status"),
        "source_label_type": value.get("source_label_type"),
        "record_count": value.get("record_count"),
        "safe_to_share": True,
    }


def _build_agent_backend(
    *,
    kind: str,
    run_dir: Path,
    session_id: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> object:
    return build_agent_backend(
        kind=kind,
        run_dir=run_dir,
        session_id=session_id,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _default_training_session_id(task: str) -> str:
    slug = safe_path_slug(task.lower(), fallback="training")
    timestamp = timestamp_utc().replace(":", "").replace(".", "_")
    return f"{slug}_fedavg_{timestamp}"


def _dataset_root_from_extraction_summary_path(summary_path: Path) -> str:
    """Recover the fixed data-phase run-scoped dataset root from its decision path."""

    if summary_path.parent.name != "decisions" or summary_path.parent.parent.name != "server":
        raise ValueError(
            "extraction summary must use runs/<session-id>/server/decisions/" "extraction_round_summary.json"
        )
    source_session = summary_path.parent.parent.parent.name
    if not source_session:
        raise ValueError("could not resolve the data-phase session from extraction summary")
    dataset_slug = safe_path_slug(source_session, fallback="fedready_experiment")
    return str(Path("data") / "dataset_fl_runs" / dataset_slug)


def main(argv: list[str] | None = None) -> int:
    """Run generated FedAvg training from one completed extraction summary."""

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) not in {1, 2}:
        print(
            "usage: python -m fedready.job_train EXTRACTION_SUMMARY [PROJECT_ROOT]",
            file=sys.stderr,
        )
        return 2
    summary_path = Path(args[0]).resolve()
    project_root = Path(args[1] if len(args) == 2 else ".").resolve()
    summary = load_extraction_summary(summary_path)
    task = str(summary.get("task") or "").strip()
    if not task:
        raise ValueError("extraction summary must contain a non-empty task")

    result = run_fedavg_training_job(
        workspace=project_root / "workspace",
        extraction_summary_path=summary_path,
        project_root=project_root,
        output_dir=project_root / "runs",
        session_id=None,
        job_root=project_root / "jobs",
        job_name="fedready_train",
        task=task,
        config=FedAvgTrainingConfig(
            num_rounds=100,
            dataset_root=_dataset_root_from_extraction_summary_path(summary_path),
        ),
        agent_backend="codex",
    )
    print(
        json.dumps(
            {key: result[key] for key in ("session_id", "run_dir", "job_path", "workspace")},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
