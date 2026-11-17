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
fallback in the contributed workflow."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

from fedready.agents.bridge import CodexWorkerBackend
from fedready.agents.roles import ServerAgent
from fedready.agents.training_reference import (
    NVFLARE_EXAMPLE_SELECTION_ID,
    NVFLARE_EXAMPLE_SELECTION_PATTERN,
    NVFLARE_EXAMPLE_SELECTION_PROMPT,
)
from fedready.data.contracts import training as training_contracts
from fedready.data.contracts.training import TrainingContract
from fedready.data.qc import visual_qc_decision_passed
from fedready.flare.training_job import (
    FedAvgTrainingConfig,
    _build_fedavg_job_object,
    _read_json,
    _refresh_client_training_configs_for_code_spec,
    _runtime_attempt_job_name,
    _simulator_error_snippets,
    export_fedavg_training_job,
    prepare_client_training_configs,
    summarize_simulator_status,
)
from fedready.flare.training_preflight import (
    TrainingCodePreflightError,
    _preflight_stage_slug,
    _training_plan_with_mock_simulation_contract,
    run_training_code_preflight,
)
from fedready.job_data import resolve_recipe_workspace_root, run_fed_job_recipe
from fedready.prompts import render_server_prompt, render_server_prompt_object
from fedready.training_code import (
    _reference_api_evidence,
    _selected_reference_example_path,
    build_training_code_validation_feedback,
    build_training_package_integrity,
    validate_training_code_spec,
)
from fedready.utils.io import atomic_write_json, safe_path_slug
from fedready.utils.logging import payload_digest, timestamp_utc
from fedready.utils.training_metrics import export_tensorboard_metrics

TRAINING_PLAN_SCHEMA_VERSION = "fedready.training_plan.v1"
DEFAULT_TRAINING_CODE_AGENT_MAX_ATTEMPTS = 8
DEFAULT_TRAINING_RUNTIME_AGENT_MAX_ATTEMPTS = 3


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
    labels = label_summary.get("labels")
    if labels:
        payload["labels"] = labels
        payload["observed_label_values"] = label_summary["observed_label_values"]
        payload["num_classes"] = len(labels)
    if label_summary["class_counts"]:
        payload["class_counts"] = label_summary["class_counts"]
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


def _ready_client_training_entries(extraction_summary: dict[str, Any], *, task: str) -> list[dict[str, Any]]:
    results = extraction_summary.get("extraction_results", {})
    if not isinstance(results, dict):
        return []
    task_contract = training_contracts.contract_from_task(task, extraction_summary=extraction_summary)
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

    task_contract = training_contracts.contract_from_task(task, extraction_summary=extraction_summary)
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
    task_contract = training_contracts.contract_from_task(task or "", extraction_summary=extraction_summary)
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
    timeout_seconds: float,
    max_attempts: int = DEFAULT_TRAINING_CODE_AGENT_MAX_ATTEMPTS,
    previous_code_spec: dict[str, Any] | None = None,
    runtime_feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate or repair training code through a bounded validation and revision loop."""

    code_workspace = Path(run_dir) / "training_phase" / "server_generated_code"
    code_workspace.mkdir(parents=True, exist_ok=True)
    feedback_dir = Path(run_dir) / "training_phase" / "server"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    backend = CodexWorkerBackend.from_env(
        run_dir=Path(run_dir),
        session_id=session_id,
        timeout_seconds=timeout_seconds,
    )
    server_agent = ServerAgent(backend)
    agent_training_plan = _training_plan_with_mock_simulation_contract(
        training_plan=training_plan,
        code_workspace=code_workspace,
    )
    runtime_revision = runtime_feedback is not None
    stage = "runtime_revision" if runtime_revision else "initial"
    prefix = "runtime_" if runtime_revision else ""
    request_context = dict(
        task={"task": task},
        extraction_summary=extraction_summary,
        training_plan=agent_training_plan,
        code_workspace=str(code_workspace.resolve()),
    )
    if runtime_revision:
        turn = server_agent.revise_training_code(
            **request_context, previous_code_spec=previous_code_spec, validation_feedback=runtime_feedback
        )
    else:
        turn = server_agent.implement_training_code(**request_context)
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
                stage=stage,
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
                feedback["agent_instruction"] = render_server_prompt(
                    "training_preflight_revision_runtime" if runtime_revision else "training_preflight_revision_initial"
                )
            if runtime_revision:
                feedback["source_runtime_feedback"] = runtime_feedback
            atomic_write_json(
                feedback_dir / f"{prefix}training_code_validation_attempt_{attempt}.json",
                feedback,
            )
            if attempt >= attempts:
                break
            turn = server_agent.revise_training_code(
                **request_context,
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
    atomic_write_json(
        feedback_dir / f"training_code_preflight_{stage_slug}_attempt_{attempt}.json",
        preflight,
    )
    if not preflight.get("succeeded"):
        raise TrainingCodePreflightError(preflight)

    validated = dict(code_spec)
    validated["local_simulation"] = preflight["local_simulation"]
    validated["package_integrity"] = build_training_package_integrity(Path(validated["package_dir"]))
    return validate_training_code_spec(validated, code_workspace=code_workspace)


def prepare_and_export_fedavg_training_job(
    *,
    extraction_summary_path: str | Path,
    output_dir: str | Path,
    session_id: str | None,
    job_root: str | Path,
    job_name: str,
    task: str = "<TASK_DESCRIPTION>",
    config: FedAvgTrainingConfig | None = None,
    training_code_spec_path: str | Path | None = None,
    agent_timeout_seconds: float = 3600.0,
    min_clients: int | None = None,
) -> dict[str, Any]:
    """Prepare agent-generated training code/configs and export a FedAvg job."""

    cfg = config or FedAvgTrainingConfig()
    session = session_id or _default_training_session_id(task)
    run_dir = Path(output_dir) / session
    run_dir.mkdir(parents=True, exist_ok=True)
    extraction_summary = load_extraction_summary(extraction_summary_path)
    training_plan = build_training_plan(task=task, extraction_summary=extraction_summary, config=cfg)
    atomic_write_json(run_dir / "training_phase" / "server" / "training_plan.json", training_plan)

    if training_code_spec_path is not None:
        code_spec = validate_training_code_spec(_read_json(training_code_spec_path))
    else:
        code_spec = prepare_training_code_with_server_agent(
            task=task,
            extraction_summary=extraction_summary,
            training_plan=training_plan,
            run_dir=run_dir,
            session_id=session,
            timeout_seconds=agent_timeout_seconds,
        )

    client_configs = prepare_client_training_configs(
        training_plan=training_plan,
        training_code_spec=code_spec,
        config=cfg,
    )
    atomic_write_json(
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
    atomic_write_json(run_dir / "training_phase" / "server" / "fedavg_job_export.json", export)
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
        atomic_write_json(
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
        atomic_write_json(
            run_dir / "training_phase" / "server" / f"training_runtime_feedback_attempt_{runtime_attempt}.json",
            runtime_feedback,
        )
        extraction_summary = load_extraction_summary(kwargs["extraction_summary_path"])
        revised_code_spec = prepare_training_code_with_server_agent(
            task=str(training_plan.get("task") or kwargs.get("task") or "<TASK_DESCRIPTION>"),
            extraction_summary=extraction_summary,
            training_plan=training_plan,
            previous_code_spec=code_spec,
            runtime_feedback=runtime_feedback,
            run_dir=run_dir,
            session_id=session,
            timeout_seconds=float(kwargs.get("agent_timeout_seconds", 3600.0)),
        )
        refreshed_configs = _refresh_client_training_configs_for_code_spec(
            raw_configs,
            training_plan=training_plan,
            training_code_spec=revised_code_spec,
            config=cfg,
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
        atomic_write_json(
            run_dir / "training_phase" / "server" / f"fedavg_job_export_attempt_{runtime_attempt + 1}.json",
            export,
        )
        atomic_write_json(run_dir / "training_phase" / "server" / "fedavg_job_export.json", export)

    error_suffix = f": {last_error}" if last_error is not None else ""
    raise RuntimeError(f"NVFlare simulator training did not finish successfully: {last_status}{error_suffix}")


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
        output_dir=project_root / "runs",
        session_id=None,
        job_root=project_root / "jobs",
        job_name="fedready_train",
        task=task,
        config=FedAvgTrainingConfig(
            num_rounds=100,
            dataset_root=_dataset_root_from_extraction_summary_path(summary_path),
        ),
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
