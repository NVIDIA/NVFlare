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

"""Lightweight server/client agents for FedReady request negotiation.

These classes own guardrail checks, prompts, and Codex backend dispatch. Live
runtime code contains only schema templates and harness-owned summaries;
visual-review references are selected from prepared local data before execution.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from fedready.agents.local_adapter import (
    DEFAULT_LOCAL_ADAPTER_MAX_ATTEMPTS,
    task_example_context_for_task,
    task_example_image_path_for_task,
)
from fedready.agents.training_reference import NVFLARE_EXAMPLE_SELECTION_ID, NVFLARE_EXAMPLE_SELECTION_PROMPT
from fedready.data.contracts import (
    CLASSIFICATION,
    available_contract_summaries,
    generated_data_contract_validation_errors,
)
from fedready.data.qc import VISUAL_QC_TRANSFORM_SET, VISUAL_QC_TRANSFORMS, visual_qc_result_ready_for_training
from fedready.prompts import render_client_prompt, render_server_prompt, render_server_prompt_object
from PIL import Image

TRAINING_CODE_SPEC_OUTPUT_SCHEMA = "fedready.training_code_spec.v1"
DATA_MATERIALIZER_SPEC_OUTPUT_SCHEMA = "fedready.data_materializer_spec.v1"
LOCAL_VLM_MAX_TRANSPORT_ATTEMPTS = 3
LOCAL_VLM_RETRY_DELAY_SECONDS = 1.0
LOCAL_VLM_TRANSPORT_MAX_LONG_SIDE = 768
LOCAL_VLM_TRANSPORT_JPEG_QUALITY = 90
GENERIC_TRAINING_TASK_ARGS_TEMPLATE = (
    "--dataset-root {dataset_root} --client-id {client_id} "
    "--local-epochs {local_epochs} --batch-size {batch_size} "
    "--learning-rate {learning_rate} --target-size {target_width} {target_height} "
    "--num-workers {num_workers} --device {device} --metrics-jsonl {metrics_jsonl}"
)


def _required_training_package_contract(training_plan: dict[str, Any]) -> dict[str, Any]:
    """Return static server-agent training contract with runtime plan fields injected."""

    contract = render_server_prompt_object(
        "required_training_package_contract",
        nvflare_example_selection_id=NVFLARE_EXAMPLE_SELECTION_ID,
        nvflare_example_selection_prompt=NVFLARE_EXAMPLE_SELECTION_PROMPT,
    )
    if not isinstance(contract, dict):
        raise TypeError("required_training_package_contract prompt entry must be an object")

    data_contract = training_plan.get("data_contract")
    data_contract = data_contract if isinstance(data_contract, dict) else None
    contract["input_data_contract"] = data_contract
    contract["label_harmonization_contract"] = (data_contract or {}).get("label_harmonization")

    preflight_contract = contract.get("local_preflight_contract")
    if isinstance(preflight_contract, dict):
        preflight_contract["dataset_contract"] = training_plan.get("local_simulation_contract")

    metric_contract = training_plan.get("metric_contract")
    contract["output_metrics"] = (metric_contract if isinstance(metric_contract, dict) else {}).get("safe_metrics")
    training_policy = training_plan.get("training_implementation_policy")
    contract["training_implementation_policy"] = training_policy if isinstance(training_policy, dict) else None
    return contract


def _training_code_agent_context(
    *,
    extraction_summary: dict[str, Any],
    training_plan: dict[str, Any],
) -> dict[str, Any]:
    """Return only the aggregate data contract needed to implement training code."""

    split_totals = {"train": 0, "validation": 0, "test": 0}
    record_total = 0
    verified_count = 0
    visual_qc_count = 0
    source_label_types: set[str] = set()
    results = extraction_summary.get("extraction_results")
    if isinstance(results, dict):
        for result in results.values():
            if not isinstance(result, dict) or result.get("data") != "extracted":
                continue
            counts = result.get("counts")
            counts = counts if isinstance(counts, dict) else {}
            total = counts.get("total")
            if isinstance(total, int) and not isinstance(total, bool):
                record_total += max(total, 0)
            by_split = counts.get("by_split")
            if isinstance(by_split, dict):
                for split in split_totals:
                    value = by_split.get(split)
                    if isinstance(value, int) and not isinstance(value, bool):
                        split_totals[split] += max(value, 0)
            verification = result.get("verification")
            if isinstance(verification, dict) and verification.get("passed") is True:
                verified_count += 1
            visual_qc = result.get("visual_qc")
            if isinstance(visual_qc, dict) and visual_qc.get("passed") is True:
                visual_qc_count += 1
            label_type = result.get("source_label_type")
            if isinstance(label_type, str) and label_type:
                source_label_types.add(label_type)

    data_contract = training_plan.get("data_contract")
    data_contract = dict(data_contract) if isinstance(data_contract, dict) else {}
    data_contract.pop("dataset_root", None)
    data_contract["dataset_root"] = "provided locally at client runtime"
    data_contract["stored_label_normalization"] = render_server_prompt("stored_label_normalization_guidance")

    simulation_contract = training_plan.get("local_simulation_contract")
    simulation_contract = dict(simulation_contract) if isinstance(simulation_contract, dict) else {}
    simulation_contract.pop("dataset_root", None)

    compact_plan = {
        "schema_version": training_plan.get("schema_version"),
        "task": training_plan.get("task"),
        "phase": training_plan.get("phase"),
        "algorithm": training_plan.get("algorithm"),
        "ready_client_count": training_plan.get("ready_client_count"),
        "fl_update_contract": training_plan.get("fl_update_contract"),
        "training_framework": training_plan.get("training_framework"),
        "training_implementation_policy": training_plan.get("training_implementation_policy"),
        "local_training": training_plan.get("local_training"),
        "data_contract": data_contract,
        "metric_contract": training_plan.get("metric_contract"),
        "local_simulation_contract": simulation_contract,
        "privacy": training_plan.get("privacy"),
        "source_extraction_summary_digest": training_plan.get("source_extraction_summary_digest"),
    }
    return {
        "extraction_contract": {
            "schema_version": "fedready.training_extraction_contract.v1",
            "source_schema_version": extraction_summary.get("schema_version"),
            "ready_client_count": training_plan.get("ready_client_count"),
            "aggregate_record_count": record_total,
            "aggregate_split_counts": split_totals,
            "verified_client_count": verified_count,
            "visual_qc_passed_client_count": visual_qc_count,
            "source_label_types": sorted(source_label_types),
        },
        "training_plan": compact_plan,
    }


@dataclass(frozen=True)
class AgentTurn:
    """Prompt and structured output for an agent step."""

    prompt: str
    output: dict[str, Any]
    guardrail: dict[str, Any] | None = None


@dataclass(frozen=True)
class GuardrailCheck:
    """Minimal explicit allow-list request descriptor."""

    role: str
    source: str
    channel: str
    action: str
    phase: str
    input_schema: str
    output_schema: str

    def key(self) -> tuple[str, str, str, str, str, str, str]:
        return (
            self.role,
            self.source,
            self.channel,
            self.action,
            self.phase,
            self.input_schema,
            self.output_schema,
        )


@dataclass(frozen=True)
class GuardrailDecision:
    """Explicit allow-list decision suitable for audit logs."""

    allowed: bool
    rule_id: str
    reason_code: str | None
    reason: str
    redacted_fields: tuple[str, ...] = ()

    @property
    def redaction_applied(self) -> bool:
        return bool(self.redacted_fields)

    def effective_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.allowed:
            raise ValueError("cannot produce an effective payload for a denied guardrail decision")
        if not self.redacted_fields:
            return payload
        return _guardrail_apply_redactions(payload, self.redacted_fields)

    def as_policy(self) -> dict[str, Any]:
        return {
            "guardrail_checked": True,
            "decision": "redacted" if self.redaction_applied else "allowed" if self.allowed else "denied",
            "allow_list_rule": self.rule_id,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "redactions": list(self.redacted_fields),
        }

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "fedready.guardrail_decision.v1",
            "allowed": self.allowed,
            "allow_list_rule": self.rule_id,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "redaction_applied": self.redaction_applied,
            "redacted_fields": list(self.redacted_fields),
        }


GUARDRAIL_DECISION_OUTPUT_SCHEMA = "fedready.guardrail_decision.v1"
GUARDRAIL_REVIEW_MODE_ENV = "FEDREADY_GUARDRAIL_REVIEW_MODE"
GUARDRAIL_REVIEW_MODES = {"agent"}


class GuardrailError(RuntimeError):
    """Raised when a local agent request is outside the explicit allow list."""

    def __init__(self, decision: GuardrailDecision) -> None:
        super().__init__(decision.reason)
        self.decision = decision


ALLOW_LIST_RULES = {
    (
        "server_agent",
        "human_user",
        "server_console",
        "SERVER.DEFINE_PROFILE_REQUEST",
        "data_profile",
        "TaskSpec@v1",
        "FedReadyClientInquiry@v1",
    ): "server.profile_request.v1",
    (
        "server_agent",
        "client_agent_via_flare",
        "flare_task_result",
        "SERVER.AGGREGATE_CLIENT_PROFILES",
        "data_profile",
        "FedReadyClientResponse@v1",
        "FedReadyServerState@v1",
    ): "server.aggregate_client_profiles.v1",
    (
        "server_agent",
        "server_state",
        "flare_task",
        "SERVER.DISPATCH_EXTRACTION_POLICY",
        "data_harmonization",
        "FedReadyServerState@v1",
        "FedReadySiteExtractionPolicy@v1",
    ): "server.dispatch_extraction_policy.v1",
    (
        "server_agent",
        "client_agent_via_flare",
        "flare_task_result",
        "SERVER.AGGREGATE_EXTRACTION_RESULTS",
        "data_harmonization",
        "FedReadyClientExtractionResponse@v1",
        "FedReadyExtractionRoundSummary@v1",
    ): "server.aggregate_extraction_results.v1",
    (
        "server_agent",
        "server_state",
        "local_training_workspace",
        "SERVER.IMPLEMENT_TRAINING_CODE",
        "fl_training",
        "FedReadyExtractionRoundSummary@v1",
        "FedReadyTrainingCodeSpec@v1",
    ): "server.implement_training_code.v1",
    (
        "server_agent",
        "server_state",
        "local_materializer_workspace",
        "SERVER.IMPLEMENT_DATA_MATERIALIZER",
        "data_harmonization",
        "FedReadyGeneratedDataContract@v1",
        "FedReadyDataMaterializerSpec@v1",
    ): "server.implement_data_materializer.v1",
    (
        "client_agent",
        "server_agent_via_flare",
        "flare_task",
        "CLIENT.REPORT_DATA_PROFILE",
        "data_profile",
        "FedReadyClientInquiry@v1",
        "FedReadyClientResponse@v1",
    ): "client.task_inquiry.v1",
    (
        "client_agent",
        "server_agent_via_flare",
        "flare_task",
        "CLIENT.EXTRACT_DATA",
        "data_harmonization",
        "FedReadySiteExtractionPolicy@v1",
        "FedReadyClientExtractionResponse@v1",
    ): "client.extract_data.v1",
    (
        "client_agent",
        "client_local_extractor",
        "local_adapter_workspace",
        "CLIENT.IMPLEMENT_LOCAL_ADAPTER",
        "data_harmonization",
        "FedReadyLocalAdapterRequest@v1",
        "FedReadyLocalAdapterSpec@v1",
    ): "client.implement_local_adapter.v1",
    (
        "client_agent",
        "client_local_extractor",
        "local_visual_qc",
        "CLIENT.VISUAL_QC_EXTRACTION",
        "data_harmonization",
        "FedReadyExtractionQCBundle@v1",
        "FedReadyExtractionQCDecision@v1",
    ): "client.visual_qc_extraction.v1",
    (
        "client_agent",
        "server_agent_via_flare",
        "local_training_workspace",
        "CLIENT.PREPARE_TRAINING_SITE",
        "fl_training",
        "FedReadyTrainingPlan@v1",
        "FedReadyClientTrainingConfig@v1",
    ): "client.prepare_training_site.v1",
}


VISUAL_QC_ALLOWED_SELECTIONS = VISUAL_QC_TRANSFORM_SET | {"undecided"}
LOCAL_VISION_API_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_LOCAL_VISION_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
LOCAL_VLM_QC_SCHEMA = "fedready.local_vlm_visual_qc.v1"
VISUAL_QC_COUNT_FIELDS = (
    "selected_transform_counts",
    "transform_vote_counts",
    "transform_counts",
    "vote_counts",
)
VISUAL_QC_TRANSFORM_LIST_FIELDS = ("selected_transforms", "sample_selected_transforms")
VISUAL_QC_PER_SAMPLE_FIELDS = (
    "per_sample_decisions",
    "sample_decisions",
    "per_sample_results",
    "per_sample_records",
)


def check_allow_list(check: GuardrailCheck) -> GuardrailDecision:
    rule_id = ALLOW_LIST_RULES.get(check.key())
    if rule_id is None:
        return GuardrailDecision(
            allowed=False,
            rule_id="none",
            reason_code="NOT_IN_ALLOW_LIST",
            reason=(
                "The requested action is not in the agent allow list for this "
                "source, channel, phase, input schema, and output schema."
            ),
        )
    return GuardrailDecision(
        allowed=True,
        rule_id=rule_id,
        reason_code=None,
        reason="Request matched the explicit FedReady allow list.",
    )


def require_allowed(check: GuardrailCheck) -> GuardrailDecision:
    decision = check_allow_list(check)
    if not decision.allowed:
        raise GuardrailError(decision)
    return decision


def _require_configured_agent_backend(agent_backend: Any) -> Any:
    if agent_backend is None:
        raise ValueError("A live Codex agent backend is required; deterministic fallback execution is disabled.")
    return agent_backend


class GuardrailAgent:
    """Agent-mediated FLARE I/O guard with local fail-closed audit signals."""

    def __init__(
        self,
        *,
        party_role: str,
        party_id: str,
        agent_backend: Any,
        review_mode: str | None = None,
    ) -> None:
        self.party_role = party_role
        self.party_id = party_id
        self.agent_backend = _require_configured_agent_backend(agent_backend)
        self.review_mode = _guardrail_review_mode(review_mode)

    def inspect(
        self,
        *,
        direction: str,
        check: GuardrailCheck,
        payload: dict[str, Any],
        counterpart: str | None = None,
    ) -> GuardrailDecision:
        shape_decision = check_allow_list(check)
        if not shape_decision.allowed:
            return shape_decision

        audit = _guardrail_local_audit(payload=payload, direction=direction)
        allow_outgoing_redaction = direction == "outgoing" and self.party_role == "client"
        baseline_decision = shape_decision
        output_template = baseline_decision.as_payload()
        outgoing_redaction_prompt = (
            render_client_prompt("guardrail_outgoing_redaction") if allow_outgoing_redaction else ""
        )
        prompt = render_client_prompt(
            "guardrail_inspect",
            party_role=self.party_role,
            party_id=self.party_id,
            direction=direction,
            action=check.action,
            outgoing_redaction_prompt=outgoing_redaction_prompt,
        )
        output = self.agent_backend.request(
            role=f"{self.party_role}_guardrail_agent",
            site_id=None if self.party_role == "server" else self.party_id,
            action=f"GUARDRAIL.{check.action}",
            phase=check.phase,
            input_schema=check.input_schema,
            output_schema="FedReadyGuardrailDecision@v1",
            prompt=prompt,
            context={
                "direction": direction,
                "party": {"role": self.party_role, "id": self.party_id},
                "counterpart": counterpart,
                "allow_list_decision": shape_decision.as_payload(),
                "local_audit": audit,
                "payload_for_review": _guardrail_review_payload(payload),
                "output_template": output_template,
                "output_template_instruction": render_client_prompt("guardrail_output_template_instruction"),
            },
        )
        agent_decision = _normalize_guardrail_agent_output(
            output,
            baseline_decision=baseline_decision,
            rule_id=shape_decision.rule_id,
        )
        if allow_outgoing_redaction and agent_decision.redacted_fields:
            return _guardrail_redaction_decision(
                payload=payload,
                redacted_fields=agent_decision.redacted_fields,
                rule_id=shape_decision.rule_id,
            )
        return agent_decision


def _guardrail_review_mode(value: str | None = None) -> str:
    mode = str(value if value is not None else os.environ.get(GUARDRAIL_REVIEW_MODE_ENV, "agent"))
    mode = mode.strip().lower()
    if mode not in GUARDRAIL_REVIEW_MODES:
        allowed = ", ".join(sorted(GUARDRAIL_REVIEW_MODES))
        raise ValueError(f"{GUARDRAIL_REVIEW_MODE_ENV} must be one of: {allowed}")
    return mode


class ServerAgent:
    """Server-side agent that expands user tasks and summarizes responses."""

    def __init__(self, agent_backend: Any) -> None:
        self.agent_backend = _require_configured_agent_backend(agent_backend)

    def compose_client_inquiry(self, *, task: dict[str, Any], client_ids: list[str]) -> AgentTurn:
        decision = require_allowed(
            GuardrailCheck(
                role="server_agent",
                source="human_user",
                channel="server_console",
                action="SERVER.DEFINE_PROFILE_REQUEST",
                phase="data_profile",
                input_schema="TaskSpec@v1",
                output_schema="FedReadyClientInquiry@v1",
            )
        )
        prompt = render_server_prompt(
            "define_profile_request",
            task_json=json.dumps(task, sort_keys=True),
        )
        baseline_output = _live_client_inquiry_template(task=task, client_ids=client_ids)
        output = self._agent_output(
            role="server_agent",
            action="SERVER.DEFINE_PROFILE_REQUEST",
            phase="data_profile",
            input_schema="TaskSpec@v1",
            output_schema="FedReadyClientInquiry@v1",
            prompt=prompt,
            context={"task": task, "client_ids": client_ids},
            output_template=baseline_output,
        )
        output = _lock_client_inquiry_policy(output, baseline=baseline_output)
        _require_live_client_inquiry_output(output)
        return AgentTurn(prompt=prompt, output=output, guardrail=decision.as_policy())

    def compose_extraction_dispatch(
        self,
        *,
        task: dict[str, Any],
        extraction_strategy: dict[str, Any],
        target_client_ids: list[str],
        extraction_config: dict[str, Any],
    ) -> AgentTurn:
        decision = require_allowed(
            GuardrailCheck(
                role="server_agent",
                source="server_state",
                channel="flare_task",
                action="SERVER.DISPATCH_EXTRACTION_POLICY",
                phase="data_harmonization",
                input_schema="FedReadyServerState@v1",
                output_schema="FedReadySiteExtractionPolicy@v1",
            )
        )
        prompt = render_server_prompt(
            "dispatch_extraction_policy",
            task_json=json.dumps(task, sort_keys=True),
        )
        extraction_strategy = _with_prepared_split_policy_in_strategy_output(
            {"extraction_strategy": extraction_strategy}
        )["extraction_strategy"]
        generated_contract = extraction_strategy.get("generated_data_contract")
        if isinstance(generated_contract, dict):
            _require_generated_data_contract(
                generated_contract,
                action="SERVER.DISPATCH_EXTRACTION_POLICY",
            )
        generated_materializer = extraction_strategy.get("generated_data_materializer")
        generated_materializer = generated_materializer if isinstance(generated_materializer, dict) else None
        agent_extraction_strategy = _summarize_generated_materializer_payload(extraction_strategy)
        output = _live_extraction_dispatch_template(
            task=task,
            extraction_strategy=extraction_strategy,
            target_client_ids=target_client_ids,
            extraction_config=extraction_config,
        )
        agent_output_template = _summarize_generated_materializer_payload(output)
        output = self._agent_output(
            role="server_agent",
            action="SERVER.DISPATCH_EXTRACTION_POLICY",
            phase="data_harmonization",
            input_schema="FedReadyServerState@v1",
            output_schema="FedReadySiteExtractionPolicy@v1",
            prompt=prompt,
            context={
                "task": task,
                "extraction_strategy": agent_extraction_strategy,
                "target_client_ids": target_client_ids,
                "extraction_config": extraction_config,
            },
            output_template=agent_output_template,
        )
        output = _restore_generated_materializer_payload(output, materializer=generated_materializer)
        output = _lock_extraction_runtime_config(
            output,
            extraction_config=extraction_config,
        )
        output = _lock_live_local_datalist_contract(output, target_client_ids=target_client_ids)
        return AgentTurn(prompt=prompt, output=output, guardrail=decision.as_policy())

    def summarize_extraction_results(
        self,
        *,
        task: dict[str, Any],
        extraction_results: dict[str, Any],
    ) -> AgentTurn:
        decision = require_allowed(
            GuardrailCheck(
                role="server_agent",
                source="client_agent_via_flare",
                channel="flare_task_result",
                action="SERVER.AGGREGATE_EXTRACTION_RESULTS",
                phase="data_harmonization",
                input_schema="FedReadyClientExtractionResponse@v1",
                output_schema="FedReadyExtractionRoundSummary@v1",
            )
        )
        prompt = render_server_prompt(
            "aggregate_extraction_results",
            task_json=json.dumps(task, sort_keys=True),
        )
        baseline = _generic_extraction_round_summary(task=task, extraction_results=extraction_results)
        output = self._agent_output(
            role="server_agent",
            action="SERVER.AGGREGATE_EXTRACTION_RESULTS",
            phase="data_harmonization",
            input_schema="FedReadyClientExtractionResponse@v1",
            output_schema="FedReadyExtractionRoundSummary@v1",
            prompt=prompt,
            context={"task": task, "extraction_results": extraction_results},
            output_template=baseline,
        )
        output = _lock_harness_owned_fields(output, baseline=baseline)
        return AgentTurn(prompt=prompt, output=output, guardrail=decision.as_policy())

    def implement_training_code(
        self,
        *,
        task: dict[str, Any],
        extraction_summary: dict[str, Any],
        training_plan: dict[str, Any],
        code_workspace: str,
    ) -> AgentTurn:
        """Ask the server agent to implement task-dependent training code.

        Agentic backends emit task-dependent trainers into ``code_workspace``
        for the current run. No fixed reference trainer is substituted here.
        """

        decision = require_allowed(
            GuardrailCheck(
                role="server_agent",
                source="server_state",
                channel="local_training_workspace",
                action="SERVER.IMPLEMENT_TRAINING_CODE",
                phase="fl_training",
                input_schema="FedReadyExtractionRoundSummary@v1",
                output_schema="FedReadyTrainingCodeSpec@v1",
            )
        )
        prompt = render_server_prompt(
            "implement_training_code",
            task_json=json.dumps(task, sort_keys=True),
            nvflare_example_selection_prompt=NVFLARE_EXAMPLE_SELECTION_PROMPT,
        )
        output = _generic_training_code_spec_template()
        agent_context = _training_code_agent_context(
            extraction_summary=extraction_summary,
            training_plan=training_plan,
        )
        output = self._agent_output(
            role="server_agent",
            action="SERVER.IMPLEMENT_TRAINING_CODE",
            phase="fl_training",
            input_schema="FedReadyExtractionRoundSummary@v1",
            output_schema=TRAINING_CODE_SPEC_OUTPUT_SCHEMA,
            prompt=prompt,
            context={
                "task": task,
                **agent_context,
                "code_workspace": code_workspace,
                "code_workspace_allowed_for_agent_writes": True,
                "dependency_policy": render_server_prompt_object("training_dependency_policy"),
                "required_package_contract": _required_training_package_contract(training_plan),
            },
            output_template=output,
            allow_template_passthrough=False,
        )
        output = _normalize_training_code_output(output, code_workspace=code_workspace)
        _reject_reference_training_code_output(output, action="SERVER.IMPLEMENT_TRAINING_CODE")
        return AgentTurn(prompt=prompt, output=output, guardrail=decision.as_policy())

    def revise_training_code(
        self,
        *,
        task: dict[str, Any],
        extraction_summary: dict[str, Any],
        training_plan: dict[str, Any],
        code_workspace: str,
        previous_code_spec: dict[str, Any],
        validation_feedback: dict[str, Any],
    ) -> AgentTurn:
        """Ask the server agent to revise training code after local validation fails."""

        decision = require_allowed(
            GuardrailCheck(
                role="server_agent",
                source="server_state",
                channel="local_training_workspace",
                action="SERVER.IMPLEMENT_TRAINING_CODE",
                phase="fl_training",
                input_schema="FedReadyExtractionRoundSummary@v1",
                output_schema="FedReadyTrainingCodeSpec@v1",
            )
        )
        prompt = render_server_prompt("revise_training_code")
        output = (
            dict(previous_code_spec) if isinstance(previous_code_spec, dict) else _generic_training_code_spec_template()
        )
        output["schema_version"] = TRAINING_CODE_SPEC_OUTPUT_SCHEMA
        output["status"] = "implemented"
        agent_context = _training_code_agent_context(
            extraction_summary=extraction_summary,
            training_plan=training_plan,
        )
        output = self._agent_output(
            role="server_agent",
            action="SERVER.IMPLEMENT_TRAINING_CODE",
            phase="fl_training",
            input_schema="FedReadyExtractionRoundSummary@v1",
            output_schema=TRAINING_CODE_SPEC_OUTPUT_SCHEMA,
            prompt=prompt,
            context={
                "task": task,
                **agent_context,
                "code_workspace": code_workspace,
                "code_workspace_allowed_for_agent_writes": True,
                "previous_training_code_spec": previous_code_spec,
                "validation_feedback": validation_feedback,
                "dependency_policy": render_server_prompt_object("training_dependency_policy"),
                "revision_instruction": render_server_prompt("training_code_revision_instruction"),
            },
            output_template=output,
            allow_template_passthrough=True,
        )
        output = _normalize_training_code_output(output, code_workspace=code_workspace)
        _reject_reference_training_code_output(output, action="SERVER.IMPLEMENT_TRAINING_CODE")
        return AgentTurn(prompt=prompt, output=output, guardrail=decision.as_policy())

    def implement_data_materializer(
        self,
        *,
        task: dict[str, Any],
        extraction_strategy: dict[str, Any],
        code_workspace: str,
    ) -> AgentTurn:
        """Ask the server agent to implement a generated-contract materializer.

        This is used only when the server planning agent created a task-family
        contract that is not one of the built-in executable contracts. The
        materializer receives client-local adapter manifests at runtime, but it
        is generated from aggregate contract metadata and must not contain
        dataset-specific client rules.
        """

        decision = require_allowed(
            GuardrailCheck(
                role="server_agent",
                source="server_state",
                channel="local_materializer_workspace",
                action="SERVER.IMPLEMENT_DATA_MATERIALIZER",
                phase="data_harmonization",
                input_schema="FedReadyGeneratedDataContract@v1",
                output_schema="FedReadyDataMaterializerSpec@v1",
            )
        )
        generated_contract = _generated_contract_with_prepared_split_policy(
            extraction_strategy.get("generated_data_contract")
        )
        generated_contract = generated_contract if isinstance(generated_contract, dict) else {}
        prompt = render_server_prompt(
            "implement_data_materializer",
            task_json=json.dumps(task, sort_keys=True),
        )
        output = _generic_data_materializer_spec_template(generated_contract=generated_contract)
        output = self._agent_output(
            role="server_agent",
            action="SERVER.IMPLEMENT_DATA_MATERIALIZER",
            phase="data_harmonization",
            input_schema="FedReadyGeneratedDataContract@v1",
            output_schema=DATA_MATERIALIZER_SPEC_OUTPUT_SCHEMA,
            prompt=prompt,
            context={
                "task": task,
                "generated_data_contract": generated_contract,
                "code_workspace": code_workspace,
                "code_workspace_allowed_for_agent_writes": True,
                "required_materializer_contract": render_server_prompt_object("required_data_materializer_contract"),
                "built_in_data_contract_examples": available_contract_summaries(),
                "task_examples": task_example_context_for_task(_task_text_for_contract(task)),
            },
            output_template=output,
            allow_template_passthrough=False,
        )
        output = _normalize_data_materializer_output(output, code_workspace=code_workspace)
        _require_data_materializer_agent_output(output)
        return AgentTurn(prompt=prompt, output=output, guardrail=decision.as_policy())

    def summarize_responses(
        self,
        *,
        task: dict[str, Any],
        inquiry: dict[str, Any],
        site_wise_info: dict[str, Any],
    ) -> AgentTurn:
        decision = require_allowed(
            GuardrailCheck(
                role="server_agent",
                source="client_agent_via_flare",
                channel="flare_task_result",
                action="SERVER.AGGREGATE_CLIENT_PROFILES",
                phase="data_profile",
                input_schema="FedReadyClientResponse@v1",
                output_schema="FedReadyServerState@v1",
            )
        )
        base_prompt = render_server_prompt(
            "aggregate_client_profiles_base",
            task_json=json.dumps(task, sort_keys=True),
        )
        revision_prompt = render_server_prompt("aggregate_client_profiles_revision")
        prompt = base_prompt
        expected_applicable_clients = sorted(
            client_id
            for client_id, response in site_wise_info.items()
            if isinstance(response, dict) and response.get("data") == "applicable"
        )
        output_template = _live_server_site_summary_template(
            task=task,
            applicable_client_ids=expected_applicable_clients,
        )
        attempt_previous_output: dict[str, Any] | None = None
        attempt_feedback: dict[str, Any] | None = None
        api_output: dict[str, Any] | None = None
        for attempt_index in range(2):
            prompt = base_prompt + (revision_prompt if attempt_feedback is not None else "")
            request_context = {
                "task": task,
                "inquiry": inquiry,
                "site_profile_summary": _compact_site_profile_summary(site_wise_info),
                "built_in_data_contract_examples": available_contract_summaries(),
                "task_examples": task_example_context_for_task(_task_text_for_contract(task)),
                "generated_contract_policy": render_server_prompt_object("generated_contract_policy"),
                "site_wise_info_storage": "Full site_wise_info remains in server-local state and is logged separately; do not echo it.",
                "output_template": output_template,
                "output_template_instruction": render_server_prompt(
                    "aggregate_client_profiles_output_template_instruction"
                ),
            }
            if attempt_previous_output is not None:
                request_context["previous_agent_output"] = attempt_previous_output
            if attempt_feedback is not None:
                request_context["validation_feedback"] = attempt_feedback
                request_context["revision_attempt"] = {
                    "attempt_index": attempt_index + 1,
                    "max_attempts": 2,
                    "reason": "previous server profile aggregation response failed schema validation",
                }
            raw_output = self.agent_backend.request(
                role="server_agent",
                action="SERVER.AGGREGATE_CLIENT_PROFILES",
                phase="data_profile",
                input_schema="FedReadyClientResponse@v1",
                output_schema="FedReadyServerState@v1",
                prompt=prompt,
                context=request_context,
                site_id=None,
            )
            api_output = _normalize_agent_strategy_output(raw_output)
            try:
                _require_agent_strategy_output(
                    api_output,
                    action="SERVER.AGGREGATE_CLIENT_PROFILES",
                    expected_applicable_clients=expected_applicable_clients,
                )
                break
            except RuntimeError as exc:
                if attempt_index == 1:
                    raise
                attempt_previous_output = raw_output
                attempt_feedback = {
                    "schema_version": "fedready.server_strategy_feedback.v1",
                    "stage": "server_profile_aggregation_schema_validation",
                    "error": str(exc),
                    "required_fix": render_server_prompt("aggregate_client_profiles_required_fix"),
                }
        if api_output is None:
            raise RuntimeError("SERVER.AGGREGATE_CLIENT_PROFILES failed without output")
        output = _merge_output_template(output_template, api_output)
        output = _with_prepared_split_policy_in_strategy_output(output)
        output["site_wise_info"] = site_wise_info
        return AgentTurn(prompt=prompt, output=output, guardrail=decision.as_policy())

    def _agent_output(
        self,
        *,
        role: str,
        action: str,
        phase: str,
        input_schema: str,
        output_schema: str,
        prompt: str,
        context: dict[str, Any],
        output_template: dict[str, Any],
        allow_template_passthrough: bool = True,
    ) -> dict[str, Any]:
        output = self.agent_backend.request(
            role=role,
            action=action,
            phase=phase,
            input_schema=input_schema,
            output_schema=output_schema,
            prompt=prompt,
            context={
                **context,
                "output_template": output_template,
                "output_template_instruction": render_server_prompt("default_output_template_instruction"),
            },
            site_id=None,
        )
        if not allow_template_passthrough:
            _raise_if_template_passthrough(action=action, output=output, template=output_template)
        return _merge_output_template(output_template, output)


class ClientAgent:
    """Client-side agent that interprets parser output against a server inquiry."""

    def __init__(self, client_id: str, agent_backend: Any) -> None:
        self.client_id = client_id
        self.agent_backend = _require_configured_agent_backend(agent_backend)

    def authorize_inquiry(self, *, inquiry: dict[str, Any]) -> GuardrailDecision:
        if inquiry.get("schema_version") != "fedready.client_inquiry.v1":
            return GuardrailDecision(
                allowed=False,
                rule_id="client.task_inquiry.v1",
                reason_code="INVALID_SCHEMA",
                reason="Client inquiry payload did not match fedready.client_inquiry.v1.",
            )
        return GuardrailAgent(
            party_role="client",
            party_id=self.client_id,
            agent_backend=self.agent_backend,
        ).inspect(
            direction="incoming",
            check=GuardrailCheck(
                role="client_agent",
                source="server_agent_via_flare",
                channel="flare_task",
                action="CLIENT.REPORT_DATA_PROFILE",
                phase="data_profile",
                input_schema="FedReadyClientInquiry@v1",
                output_schema="FedReadyClientResponse@v1",
            ),
            payload=inquiry,
            counterpart="server_agent",
        )

    def authorize_extraction(self, *, policy: dict[str, Any]) -> GuardrailDecision:
        if policy.get("schema_version") != "fedready.site_extraction_policy.v1":
            return GuardrailDecision(
                allowed=False,
                rule_id="client.extract_data.v1",
                reason_code="INVALID_SCHEMA",
                reason="Extraction policy did not match fedready.site_extraction_policy.v1.",
            )
        if policy.get("client_id") != self.client_id:
            return GuardrailDecision(
                allowed=False,
                rule_id="client.extract_data.v1",
                reason_code="WRONG_CLIENT_POLICY",
                reason="Extraction policy client_id did not match the local client.",
            )
        return GuardrailAgent(
            party_role="client",
            party_id=self.client_id,
            agent_backend=self.agent_backend,
        ).inspect(
            direction="incoming",
            check=GuardrailCheck(
                role="client_agent",
                source="server_agent_via_flare",
                channel="flare_task",
                action="CLIENT.EXTRACT_DATA",
                phase="data_harmonization",
                input_schema="FedReadySiteExtractionPolicy@v1",
                output_schema="FedReadyClientExtractionResponse@v1",
            ),
            payload=policy,
            counterpart="server_agent",
        )

    def authorize_outgoing_profile_response(self, *, response: dict[str, Any]) -> GuardrailDecision:
        return GuardrailAgent(
            party_role="client",
            party_id=self.client_id,
            agent_backend=self.agent_backend,
        ).inspect(
            direction="outgoing",
            check=GuardrailCheck(
                role="client_agent",
                source="server_agent_via_flare",
                channel="flare_task",
                action="CLIENT.REPORT_DATA_PROFILE",
                phase="data_profile",
                input_schema="FedReadyClientInquiry@v1",
                output_schema="FedReadyClientResponse@v1",
            ),
            payload=response,
            counterpart="server_agent",
        )

    def authorize_outgoing_extraction_response(self, *, response: dict[str, Any]) -> GuardrailDecision:
        return GuardrailAgent(
            party_role="client",
            party_id=self.client_id,
            agent_backend=self.agent_backend,
        ).inspect(
            direction="outgoing",
            check=GuardrailCheck(
                role="client_agent",
                source="server_agent_via_flare",
                channel="flare_task",
                action="CLIENT.EXTRACT_DATA",
                phase="data_harmonization",
                input_schema="FedReadySiteExtractionPolicy@v1",
                output_schema="FedReadyClientExtractionResponse@v1",
            ),
            payload=response,
            counterpart="server_agent",
        )

    def prepare_training_site(
        self,
        *,
        training_plan: dict[str, Any],
        dataset_summary: dict[str, Any],
        client_config_template: dict[str, Any],
    ) -> AgentTurn:
        decision = require_allowed(
            GuardrailCheck(
                role="client_agent",
                source="server_agent_via_flare",
                channel="local_training_workspace",
                action="CLIENT.PREPARE_TRAINING_SITE",
                phase="fl_training",
                input_schema="FedReadyTrainingPlan@v1",
                output_schema="FedReadyClientTrainingConfig@v1",
            )
        )
        prompt = render_client_prompt("prepare_training_site", client_id=self.client_id)
        output = dict(client_config_template)
        output = self._agent_output(
            role="client_agent",
            action="CLIENT.PREPARE_TRAINING_SITE",
            phase="fl_training",
            input_schema="FedReadyTrainingPlan@v1",
            output_schema="FedReadyClientTrainingConfig@v1",
            prompt=prompt,
            context={
                "training_plan": training_plan,
                "dataset_summary": dataset_summary,
                "output_template": client_config_template,
                "output_template_instruction": render_client_prompt(
                    "prepare_training_site_output_template_instruction"
                ),
            },
        )
        return AgentTurn(prompt=prompt, output=output, guardrail=decision.as_policy())

    def _agent_output(
        self,
        *,
        role: str,
        action: str,
        phase: str,
        input_schema: str,
        output_schema: str,
        prompt: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        return self.agent_backend.request(
            role=role,
            action=action,
            phase=phase,
            input_schema=input_schema,
            output_schema=output_schema,
            prompt=prompt,
            context=context,
            site_id=self.client_id,
        )

    def answer_inquiry(self, *, inquiry: dict[str, Any], parsed_profile: dict[str, Any]) -> AgentTurn:
        prompt = render_client_prompt(
            "report_data_profile",
            client_id=self.client_id,
            inquiry_message_json=json.dumps(inquiry["message"], sort_keys=True),
        )
        local_summary = _client_profile_summary_for_model(parsed_profile)
        shareable_summary = _client_profile_summary_for_server(parsed_profile)
        output_template = {
            "data": "agent_decision_required",
            "fit": {},
            "meta": shareable_summary,
        }
        output = self.agent_backend.request(
            role="client_agent",
            site_id=self.client_id,
            action="CLIENT.REPORT_DATA_PROFILE",
            phase="data_profile",
            input_schema="FedReadyClientInquiry@v1",
            output_schema="FedReadyClientResponse@v1",
            prompt=prompt,
            context={
                "inquiry": inquiry,
                "parsed_profile_summary": local_summary,
                "output_template": output_template,
                "output_template_instruction": render_client_prompt("profile_output_template_instruction"),
            },
        )
        _require_client_profile_output(output, client_id=self.client_id)
        if output.get("data") == "applicable":
            output = {**output, "meta": shareable_summary}
        return AgentTurn(prompt=prompt, output=output)

    def implement_local_adapter(
        self,
        *,
        policy: dict[str, Any],
        extraction_result: dict[str, Any],
        adapter_context: dict[str, Any],
        previous_output: dict[str, Any] | None = None,
        validation_feedback: dict[str, Any] | None = None,
    ) -> AgentTurn:
        decision = require_allowed(
            GuardrailCheck(
                role="client_agent",
                source="client_local_extractor",
                channel="local_adapter_workspace",
                action="CLIENT.IMPLEMENT_LOCAL_ADAPTER",
                phase="data_harmonization",
                input_schema="FedReadyLocalAdapterRequest@v1",
                output_schema="FedReadyLocalAdapterSpec@v1",
            )
        )
        local_adapter_requirements = render_client_prompt("local_adapter_implementation_requirements")
        local_adapter_cli_skeleton = render_client_prompt("local_adapter_cli_skeleton")
        base_prompt = render_client_prompt(
            "implement_local_adapter_base",
            client_id=self.client_id,
            local_adapter_requirements=local_adapter_requirements,
            local_adapter_cli_skeleton=local_adapter_cli_skeleton,
        )
        revision_prompt = render_client_prompt("implement_local_adapter_revision")
        prompt = base_prompt + (revision_prompt if validation_feedback is not None else "")
        output_template = {
            "schema_version": "fedready.local_adapter_spec.v1",
            "client_id": self.client_id,
            "status": "agent_response_required",
            "source_label_type": _adapter_source_label_type(policy=policy, extraction_result=extraction_result),
            "adapter_kind": "client_local_label_adapter",
            "manifest_path": None,
            "script_path": None,
            "preflight_record_count": None,
            "record_count": None,
            "runtime_validation": None,
            "reason": "Live client adapter agent must implement, fail, or mark the site unfeasible.",
            "safe_to_share": False,
        }
        attempt_previous_output = previous_output
        attempt_feedback = validation_feedback
        last_error: RuntimeError | None = None
        output: dict[str, Any] | None = None
        max_agent_attempts = DEFAULT_LOCAL_ADAPTER_MAX_ATTEMPTS
        for attempt_index in range(max_agent_attempts):
            prompt = base_prompt + (revision_prompt if attempt_feedback is not None else "")
            request_context = {
                "policy": _summarize_generated_materializer_payload(policy),
                "local_datalist_request": extraction_result,
                "adapter_context": adapter_context,
                "adapter_cli_skeleton": local_adapter_cli_skeleton,
                "code_workspace": adapter_context.get("adapter_workspace"),
                "code_workspace_allowed_for_agent_writes": bool(adapter_context.get("adapter_workspace")),
                "output_template": output_template,
                "output_template_instruction": render_client_prompt("local_adapter_output_template_instruction"),
            }
            if attempt_previous_output is not None:
                request_context["previous_agent_output"] = _safe_local_adapter_revision_output(attempt_previous_output)
            if attempt_feedback is not None:
                request_context["validation_feedback"] = attempt_feedback
                request_context["revision_attempt"] = {
                    "attempt_index": attempt_index + 1,
                    "max_attempts": max_agent_attempts,
                    "reason": "previous adapter response failed local validation",
                }
            try:
                api_output = self.agent_backend.request(
                    role="client_agent",
                    site_id=self.client_id,
                    action="CLIENT.IMPLEMENT_LOCAL_ADAPTER",
                    phase="data_harmonization",
                    input_schema="FedReadyLocalAdapterRequest@v1",
                    output_schema="FedReadyLocalAdapterSpec@v1",
                    prompt=prompt,
                    context=request_context,
                )
            except Exception as exc:
                last_error = RuntimeError(_short_error(exc))
                if attempt_index == max_agent_attempts - 1:
                    raise
                attempt_feedback = _local_adapter_backend_failure_feedback(
                    client_id=self.client_id,
                    adapter_context=adapter_context,
                    error=exc,
                )
                continue
            _raise_if_template_passthrough(
                action="CLIENT.IMPLEMENT_LOCAL_ADAPTER",
                output=api_output,
                template=output_template,
            )
            output = _normalize_local_adapter_output(api_output, client_id=self.client_id)
            try:
                _require_local_adapter_agent_output(output, client_id=self.client_id)
                break
            except RuntimeError as exc:
                last_error = exc
                if attempt_index == max_agent_attempts - 1:
                    raise
                attempt_previous_output = output
                attempt_feedback = {
                    "schema_version": "fedready.local_adapter_feedback.v1",
                    "stage": "agent_output_schema_validation",
                    "client_id": self.client_id,
                    "error": _short_error(exc),
                    "required_fix": render_client_prompt("local_adapter_schema_required_fix"),
                }
        if output is None:
            raise RuntimeError(
                f"CLIENT.IMPLEMENT_LOCAL_ADAPTER for {self.client_id} failed without output"
            ) from last_error
        return AgentTurn(prompt=prompt, output=output, guardrail=decision.as_policy())

    def report_extraction_result(
        self,
        *,
        policy: dict[str, Any],
        extraction_result: dict[str, Any],
        execution_summary: dict[str, Any],
    ) -> AgentTurn:
        prompt = render_client_prompt("extract_data_report", client_id=self.client_id)
        output = _generic_extraction_report(extraction_result=extraction_result)
        return AgentTurn(prompt=prompt, output=output)

    def review_extraction_visual_qc(
        self,
        *,
        policy: dict[str, Any],
        extraction_result: dict[str, Any],
        qc_context: dict[str, Any],
    ) -> AgentTurn:
        decision = require_allowed(
            GuardrailCheck(
                role="client_agent",
                source="client_local_extractor",
                channel="local_visual_qc",
                action="CLIENT.VISUAL_QC_EXTRACTION",
                phase="data_harmonization",
                input_schema="FedReadyExtractionQCBundle@v1",
                output_schema="FedReadyExtractionQCDecision@v1",
            )
        )
        output = _live_visual_qc_template(client_id=self.client_id, qc_context=qc_context)
        prompt = render_client_prompt("visual_qc_extraction", client_id=self.client_id)
        artifacts = _visual_qc_artifacts(qc_context)
        if qc_context.get("review_satisfied") is True:
            output = _adapter_preflight_visual_qc_decision(
                client_id=self.client_id,
                qc_context=qc_context,
            )
        elif qc_context.get("review_required") is False or not artifacts:
            output = _normalize_visual_qc_output(output, qc_context=qc_context)
        elif len(artifacts) > 1:
            output = self._review_extraction_visual_qc_samples(
                policy=policy,
                extraction_result=extraction_result,
                qc_context=qc_context,
                output=output,
                prompt=prompt,
                artifacts=artifacts,
            )
        else:
            api_output = self._request_visual_qc_backend(
                policy=policy,
                extraction_result=extraction_result,
                qc_context=qc_context,
                output=output,
            )
            output = _normalize_visual_qc_output(_merge_output_template(output, api_output), qc_context=qc_context)
        return AgentTurn(prompt=prompt, output=output, guardrail=decision.as_policy())

    def _review_extraction_visual_qc_samples(
        self,
        *,
        policy: dict[str, Any],
        extraction_result: dict[str, Any],
        qc_context: dict[str, Any],
        output: dict[str, Any],
        prompt: str,
        artifacts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        sample_outputs: list[dict[str, Any]] = []
        for artifact in artifacts:
            single_qc_context = dict(qc_context)
            single_qc_context["sample_count"] = 1
            single_qc_context["artifacts"] = [artifact]
            single_output = _live_visual_qc_template(client_id=self.client_id, qc_context=single_qc_context)
            api_output = self._request_visual_qc_backend(
                policy=policy,
                extraction_result=extraction_result,
                qc_context=single_qc_context,
                output=single_output,
            )
            sample_outputs.append(
                _normalize_visual_qc_output(
                    _merge_output_template(single_output, api_output),
                    qc_context=single_qc_context,
                )
            )
        return _aggregate_visual_qc_sample_outputs(
            base_output=output, sample_outputs=sample_outputs, qc_context=qc_context
        )

    def _request_visual_qc_backend(
        self,
        *,
        policy: dict[str, Any],
        extraction_result: dict[str, Any],
        qc_context: dict[str, Any],
        output: dict[str, Any],
    ) -> dict[str, Any]:
        _visual_qc_backend_mode()
        local_vlm_output = _request_local_vlm_visual_qc(
            client_id=self.client_id,
            policy=policy,
            extraction_result=extraction_result,
            qc_context=qc_context,
            output_template=output,
        )
        if local_vlm_output is None:
            raise RuntimeError("local visual QC backend returned no decision")
        return local_vlm_output


SENSITIVE_FIELD_NAMES = {
    "local_data_path",
    "local_data_paths",
    "filename",
    "filenames",
    "file_path",
    "file_paths",
    "image_path",
    "image_paths",
    "mask_path",
    "mask_paths",
    "raw_image",
    "raw_images",
    "raw_mask",
    "raw_masks",
    "patient_id",
    "patient_ids",
    "sample_id",
    "sample_ids",
    "per_sample_record",
    "per_sample_records",
    "credential",
    "credentials",
    "api_key",
    "token",
    "secret",
}
SENSITIVE_REQUEST_TERMS = {
    "local_data_path",
    "local path",
    "local paths",
    "filename",
    "filenames",
    "raw image",
    "raw images",
    "raw mask",
    "raw masks",
    "patient id",
    "patient ids",
    "sample id",
    "sample ids",
    "per-sample",
    "private records",
    "other client",
    "other clients",
    "credential",
    "credentials",
    "api key",
    "secret",
    "/secret",
}
OVERREACHING_REQUEST_TERMS = {
    "ignore privacy",
    "ignore all privacy",
    "ignore all fedready privacy",
    "bypass privacy",
    "disable privacy",
    "override privacy",
    "ignore guardrail",
    "ignore guardrails",
    "bypass guardrail",
    "exfiltrate",
}
REQUEST_COMMAND_TERMS = {
    "return",
    "send",
    "provide",
    "give",
    "list",
    "dump",
    "print",
    "show",
    "reveal",
    "expose",
}
GUARDRAIL_SKIP_KEYS = {
    "redacted",
    "redactions",
    "server_response_redaction",
    "do_not_return_to_server",
    "forbidden_actions",
    "safe_to_share",
}
GUARDRAIL_ALLOWED_RUNTIME_PATH_SUFFIXES = {
    ("extraction_config", "output_root"),
}


def _guardrail_local_audit(*, payload: dict[str, Any], direction: str) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "schema_version": "fedready.guardrail_local_audit.v1",
        "direction": direction,
        "must_deny": False,
        "reason_code": None,
        "unsafe_key_paths": [],
        "local_path_value_paths": [],
        "semantic_warning_paths": [],
        "semantic_warning_terms": [],
    }
    _guardrail_scan_value(payload, path=[], audit=audit)
    return audit


def _guardrail_scan_value(value: Any, *, path: list[str], audit: dict[str, Any]) -> None:
    if _guardrail_should_skip_path(path):
        return
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = [*path, key_text]
            if key_text.lower() in SENSITIVE_FIELD_NAMES and not _guardrail_should_skip_path(child_path):
                audit["unsafe_key_paths"].append(".".join(child_path))
            _guardrail_scan_value(child, path=child_path, audit=audit)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _guardrail_scan_value(child, path=[*path, str(index)], audit=audit)
        return
    if isinstance(value, str):
        if _looks_like_local_path(value):
            audit["local_path_value_paths"].append(".".join(path))
        terms = _semantic_guardrail_warning_terms(value)
        if terms:
            audit["semantic_warning_paths"].append(".".join(path))
            for term in terms:
                if term not in audit["semantic_warning_terms"]:
                    audit["semantic_warning_terms"].append(term)


def _guardrail_should_skip_path(path: list[str]) -> bool:
    lowered = [part.lower() for part in path]
    if set(lowered) & GUARDRAIL_SKIP_KEYS:
        return True
    return any(tuple(lowered[-len(suffix) :]) == suffix for suffix in GUARDRAIL_ALLOWED_RUNTIME_PATH_SUFFIXES)


def _looks_like_local_path(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    without_urls = re.sub(r"\b[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>]+", "", text)
    return any(
        re.search(pattern, without_urls)
        for pattern in (
            r"(?<![A-Za-z0-9_.-])/(?:[^\s\"'<>]+)",
            r"(?<![A-Za-z0-9_.-])~/(?:[^\s\"'<>]+)",
            r"(?<![A-Za-z0-9_.-])[A-Za-z]:[\\/](?:[^\s\"'<>]+)",
        )
    )


def _redact_local_paths_in_text(value: str) -> str:
    urls: dict[str, str] = {}

    def preserve_url(match: re.Match[str]) -> str:
        marker = f"__FEDREADY_URL_{len(urls)}__"
        urls[marker] = match.group(0)
        return marker

    redacted = re.sub(r"\b[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>]+", preserve_url, value)
    for pattern in (
        r"(?<![A-Za-z0-9_.-])/(?:[^\s\"'<>]+)",
        r"(?<![A-Za-z0-9_.-])~/(?:[^\s\"'<>]+)",
        r"(?<![A-Za-z0-9_.-])[A-Za-z]:[\\/](?:[^\s\"'<>]+)",
    ):
        redacted = re.sub(pattern, "[redacted-local-path]", redacted)
    for marker, url in urls.items():
        redacted = redacted.replace(marker, url)
    return redacted


def _semantic_guardrail_warning_terms(value: str) -> list[str]:
    lowered = value.lower()
    terms = [term for term in sorted(OVERREACHING_REQUEST_TERMS) if term in lowered]
    sensitive_terms = [term for term in sorted(SENSITIVE_REQUEST_TERMS) if term in lowered]
    command_present = any(
        re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", lowered) for term in REQUEST_COMMAND_TERMS
    )
    if sensitive_terms and command_present:
        terms.extend(term for term in sensitive_terms if term not in terms)
    return terms


def _guardrail_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return _guardrail_redact_value(payload, path=[])


def _guardrail_redact_value(value: Any, *, path: list[str]) -> Any:
    if isinstance(value, dict):
        if value.get("schema_version") == "fedready.generated_data_materializer.v1":
            return _generated_materializer_guardrail_payload(value)
        redacted: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if key_text.lower() in SENSITIVE_FIELD_NAMES:
                redacted[key_text] = "[redacted-sensitive-field]"
            else:
                redacted[key_text] = _guardrail_redact_value(child, path=[*path, key_text])
        return redacted
    if isinstance(value, list):
        return [_guardrail_redact_value(item, path=[*path, str(index)]) for index, item in enumerate(value)]
    if isinstance(value, str) and _looks_like_local_path(value):
        return _redact_local_paths_in_text(value)
    return value


def _generated_materializer_guardrail_payload(materializer: dict[str, Any]) -> dict[str, Any]:
    """Preserve server-supplied executable source for client boundary review."""

    return json.loads(json.dumps(materializer, default=str))


def _guardrail_redaction_decision(
    *,
    payload: dict[str, Any],
    redacted_fields: tuple[str, ...],
    rule_id: str,
) -> GuardrailDecision:
    effective_redacted_fields = _guardrail_normalize_redaction_paths(payload, redacted_fields)
    if not effective_redacted_fields:
        return GuardrailDecision(
            allowed=True,
            rule_id=rule_id,
            reason_code=None,
            reason="Guardrail found no applicable optional fields to redact.",
        )
    try:
        redacted_payload = _guardrail_apply_redactions(payload, effective_redacted_fields)
    except ValueError as exc:
        return GuardrailDecision(
            allowed=False,
            rule_id=rule_id,
            reason_code="INVALID_GUARDRAIL_REDACTION",
            reason=f"Guardrail could not apply the requested field redaction ({type(exc).__name__}).",
        )
    return GuardrailDecision(
        allowed=True,
        rule_id=rule_id,
        reason_code="REDACTED_SENSITIVE_OUTPUT",
        reason="Guardrail removed sensitive or unnecessary local metadata before FLARE transmission.",
        redacted_fields=effective_redacted_fields,
    )


def _guardrail_normalize_redaction_paths(payload: dict[str, Any], redacted_fields: tuple[str, ...]) -> tuple[str, ...]:
    """Keep only optional paths that exist in the payload actually sent over FLARE."""
    normalized: list[str] = []
    for field_path in redacted_fields:
        candidate = field_path.removeprefix("payload_for_review.")
        if candidate in normalized or not _guardrail_path_exists(payload, candidate):
            continue
        normalized.append(candidate)
    return tuple(normalized)


def _guardrail_path_exists(payload: dict[str, Any], field_path: str) -> bool:
    parts = field_path.split(".")
    if not parts or any(not part for part in parts):
        return False
    current: Any = payload
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False
    return True


def _guardrail_apply_redactions(payload: dict[str, Any], redacted_fields: tuple[str, ...]) -> dict[str, Any]:
    redacted = json.loads(json.dumps(payload, default=str))
    protected_top_level = {
        "schema_version",
        "client_id",
        "data",
        "status",
        "counts",
        "verification",
        "visual_qc",
        "fit",
        "meta",
    }
    ordered_fields = sorted(
        dict.fromkeys(redacted_fields),
        key=lambda value: value.count("."),
        reverse=True,
    )
    for field_path in ordered_fields:
        parts = field_path.split(".")
        if not parts or any(not part for part in parts):
            raise ValueError(f"invalid redaction path {field_path!r}")
        if len(parts) == 1 and parts[0] in protected_top_level:
            raise ValueError(f"required top-level field cannot be redacted: {field_path}")
        parent: Any = redacted
        for part in parts[:-1]:
            if isinstance(parent, dict) and part in parent:
                parent = parent[part]
            elif isinstance(parent, list) and part.isdigit() and int(part) < len(parent):
                parent = parent[int(part)]
            else:
                raise ValueError(f"redaction path does not exist: {field_path}")
        leaf = parts[-1]
        if isinstance(parent, dict) and leaf in parent:
            del parent[leaf]
        elif isinstance(parent, list) and leaf.isdigit() and int(leaf) < len(parent):
            parent[int(leaf)] = "[redacted-by-guardrail]"
        else:
            raise ValueError(f"redaction path does not exist: {field_path}")
    if redacted.get("schema_version") != payload.get("schema_version"):
        raise ValueError("redaction changed schema_version")
    if "client_id" in payload and redacted.get("client_id") != payload.get("client_id"):
        raise ValueError("redaction changed client_id")
    return redacted


def _normalize_guardrail_agent_output(
    output: dict[str, Any],
    *,
    baseline_decision: GuardrailDecision,
    rule_id: str,
) -> GuardrailDecision:
    if not isinstance(output, dict):
        return GuardrailDecision(
            allowed=False,
            rule_id=rule_id,
            reason_code="INVALID_GUARDRAIL_OUTPUT",
            reason="Guardrail agent returned a non-object decision.",
        )
    normalized = dict(output)
    if normalized.get("schema_version") == "FedReadyGuardrailDecision@v1":
        normalized["schema_version"] = GUARDRAIL_DECISION_OUTPUT_SCHEMA
    if normalized.get("schema_version") != GUARDRAIL_DECISION_OUTPUT_SCHEMA:
        return GuardrailDecision(
            allowed=False,
            rule_id=rule_id,
            reason_code="INVALID_GUARDRAIL_OUTPUT",
            reason="Guardrail agent returned an invalid decision schema.",
        )
    allowed = normalized.get("allowed")
    if not isinstance(allowed, bool):
        return GuardrailDecision(
            allowed=False,
            rule_id=rule_id,
            reason_code="INVALID_GUARDRAIL_OUTPUT",
            reason="Guardrail agent decision missing boolean allowed field.",
        )
    reason_code = normalized.get("reason_code")
    if reason_code is not None and not isinstance(reason_code, str):
        reason_code = "INVALID_GUARDRAIL_REASON_CODE"
    reason = normalized.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = baseline_decision.reason if allowed == baseline_decision.allowed else "Guardrail agent decision."
    raw_redacted_fields = normalized.get("redacted_fields", baseline_decision.redacted_fields)
    if not isinstance(raw_redacted_fields, (list, tuple)) or not all(
        isinstance(value, str) and value for value in raw_redacted_fields
    ):
        return GuardrailDecision(
            allowed=False,
            rule_id=rule_id,
            reason_code="INVALID_GUARDRAIL_REDACTION",
            reason="Guardrail agent returned invalid redacted_fields.",
        )
    return GuardrailDecision(
        allowed=allowed,
        rule_id=str(normalized.get("allow_list_rule") or normalized.get("rule_id") or rule_id),
        reason_code=reason_code,
        reason=reason,
        redacted_fields=tuple(dict.fromkeys(raw_redacted_fields)),
    )


def _client_profile_summary_for_model(parsed_profile: dict[str, Any]) -> dict[str, Any]:
    labels = parsed_profile.get("labels") if isinstance(parsed_profile.get("labels"), dict) else {}
    dataset = parsed_profile.get("dataset") if isinstance(parsed_profile.get("dataset"), dict) else {}
    case_counts = parsed_profile.get("case_counts") if isinstance(parsed_profile.get("case_counts"), dict) else {}
    return {
        "dataset": _copy_present_keys(dataset, ["display_name", "domain", "purpose", "source_type", "license"]),
        "labels": _copy_present_keys(
            labels,
            [
                "label_type",
                "label_source",
                "label_meanings",
                "declared_shareable_concepts",
                "declared_shareable_value_meanings",
                "label_vocabulary_shareable",
                "label_vocabulary_privacy",
                "annotation_label_meanings",
                "annotation_label_columns",
                "annotation_label_values",
                "annotation_label_value_cardinality",
                "annotation_label_value_counts",
                "class_labels",
                "class_label_cardinality",
                "class_label_counts",
                "mask_image_count",
                "annotation_file_count",
                "annotation_has_segmentation_hint",
            ],
        ),
        "case_counts": _copy_present_keys(
            case_counts,
            ["total", "total_primary_images", "train", "val", "validation", "test", "unknown"],
        ),
    }


def _client_profile_summary_for_server(parsed_profile: dict[str, Any]) -> dict[str, Any]:
    """Return the client-owned allow-list for the cross-site profile response."""

    summary = _client_profile_summary_for_model(parsed_profile)
    labels = dict(summary.get("labels", {}))
    labels.pop("label_source", None)
    return {**summary, "labels": labels}


def _require_client_profile_output(output: dict[str, Any], *, client_id: str) -> None:
    if not isinstance(output, dict):
        raise RuntimeError(f"CLIENT.REPORT_DATA_PROFILE for {client_id} returned non-object output")
    data = output.get("data")
    if data == "not applicable":
        return
    if data != "applicable":
        raise RuntimeError(
            f"CLIENT.REPORT_DATA_PROFILE for {client_id} returned invalid data status {data!r}; "
            "expected applicable or not applicable"
        )
    fit = output.get("fit")
    meta = output.get("meta")
    if not isinstance(fit, dict):
        raise RuntimeError(f"CLIENT.REPORT_DATA_PROFILE for {client_id} applicable output missing fit object")
    if not isinstance(meta, dict):
        raise RuntimeError(f"CLIENT.REPORT_DATA_PROFILE for {client_id} applicable output missing meta object")


def _compact_site_profile_summary(site_wise_info: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for client_id, response in sorted(site_wise_info.items()):
        if not isinstance(response, dict):
            summary[client_id] = {"data": "invalid_response"}
            continue
        item: dict[str, Any] = {"data": response.get("data")}
        if response.get("data") == "applicable":
            fit = response.get("fit") if isinstance(response.get("fit"), dict) else {}
            meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
            labels = meta.get("labels") if isinstance(meta.get("labels"), dict) else {}
            dataset = meta.get("dataset") if isinstance(meta.get("dataset"), dict) else {}
            image_dimensions = meta.get("image_dimensions") if isinstance(meta.get("image_dimensions"), dict) else {}
            item["fit"] = _copy_present_keys(
                fit,
                [
                    "applicable",
                    "task_type",
                    "label_type",
                    "matched_terms",
                    "reasons",
                    "summary",
                    "evidence",
                    "uncertainties",
                ],
            )
            item["dataset"] = _copy_present_keys(
                dataset,
                ["display_name", "domain", "purpose", "source_type", "license"],
            )
            item["labels"] = _copy_present_keys(
                labels,
                [
                    "label_type",
                    "label_meanings",
                    "label_vocabulary_shareable",
                    "label_vocabulary_privacy",
                    "annotation_label_meanings",
                    "annotation_label_columns",
                    "annotation_label_values",
                    "annotation_label_value_cardinality",
                    "annotation_label_value_counts",
                    "class_labels",
                    "class_label_cardinality",
                    "class_label_counts",
                    "mask_image_count",
                    "annotation_file_count",
                    "annotation_has_segmentation_hint",
                ],
            )
            item["case_counts"] = _copy_present_keys(
                meta.get("case_counts") if isinstance(meta.get("case_counts"), dict) else {},
                ["total", "train", "val", "validation", "test"],
            )
            item["image_dimensions"] = _copy_present_keys(
                image_dimensions,
                ["available", "channels", "common_sizes", "width", "height"],
            )
            warnings = meta.get("warnings")
            if isinstance(warnings, list):
                item["warning_count"] = len(warnings)
        summary[client_id] = item
    return summary


def _copy_present_keys(source: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: source[key] for key in keys if key in source}


def _normalize_agent_strategy_output(output: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(output, dict):
        return output
    if isinstance(output.get("extraction_strategy"), dict):
        return _with_prepared_split_policy_in_strategy_output(output)
    server_state = output.get("server_state")
    if isinstance(server_state, dict) and isinstance(server_state.get("extraction_strategy"), dict):
        normalized = dict(server_state)
        if "schema_version" not in normalized and isinstance(output.get("schema_version"), str):
            normalized["schema_version"] = output["schema_version"]
        return _with_prepared_split_policy_in_strategy_output(normalized)
    return output


def _prepared_data_split_policy() -> dict[str, Any]:
    policy = render_server_prompt_object("prepared_data_split_policy")
    if not isinstance(policy, dict):
        raise TypeError("prepared_data_split_policy prompt entry must be an object")
    return json.loads(json.dumps(policy, sort_keys=True))


def _generated_contract_with_prepared_split_policy(contract: Any) -> Any:
    if not isinstance(contract, dict):
        return contract
    normalized = dict(contract)
    if isinstance(normalized.get("split_policy"), dict) or isinstance(normalized.get("split_rule"), dict):
        return normalized
    normalized["split_policy"] = _prepared_data_split_policy()
    return normalized


def _with_prepared_split_policy_in_strategy_output(output: dict[str, Any]) -> dict[str, Any]:
    strategy = output.get("extraction_strategy")
    if not isinstance(strategy, dict) or not isinstance(strategy.get("generated_data_contract"), dict):
        return output
    normalized_strategy = dict(strategy)
    normalized_strategy["generated_data_contract"] = _generated_contract_with_prepared_split_policy(
        strategy["generated_data_contract"]
    )
    normalized = dict(output)
    normalized["extraction_strategy"] = normalized_strategy
    return normalized


def _require_agent_strategy_output(
    output: dict[str, Any],
    *,
    action: str,
    expected_applicable_clients: list[str] | None = None,
) -> None:
    if not isinstance(output, dict):
        raise RuntimeError(f"{action} agent output must be a JSON object")
    if output.get("schema_version") != "fedready.server_site_summary.v1":
        raise RuntimeError(f"{action} agent output missing schema_version=fedready.server_site_summary.v1")
    strategy = output.get("extraction_strategy")
    if not isinstance(strategy, dict):
        raise RuntimeError(f"{action} agent output missing extraction_strategy; refusing template fallback")
    if strategy.get("schema_version") != "fedready.task_extraction_strategy.v1":
        raise RuntimeError(f"{action} extraction_strategy missing schema_version=fedready.task_extraction_strategy.v1")
    required_keys = [
        "applicable_clients",
        "per_site_label_mapping",
        "image_rule",
        "label_rule",
        "split_rule",
        "visual_qc_rule",
        "next_step",
    ]
    missing = [key for key in required_keys if key not in strategy]
    if missing:
        raise RuntimeError(
            f"{action} extraction_strategy missing required keys: {', '.join(missing)}; refusing template fallback"
        )
    per_site = strategy.get("per_site_label_mapping")
    if not isinstance(per_site, dict):
        raise RuntimeError(f"{action} extraction_strategy.per_site_label_mapping must be an object")
    applicable = _string_list_set(strategy.get("applicable_clients"), field="applicable_clients", action=action)
    if expected_applicable_clients is not None and applicable != set(expected_applicable_clients):
        raise RuntimeError(
            f"{action} extraction_strategy applicable_clients does not match client decisions; "
            f"expected={sorted(expected_applicable_clients)}, actual={sorted(applicable)}"
        )
    count = strategy.get("applicable_client_count")
    if isinstance(count, int) and count != len(applicable):
        raise RuntimeError(
            f"{action} extraction_strategy applicable_client_count={count} does not match "
            f"applicable_clients={len(applicable)}"
        )
    missing_mappings = applicable - set(per_site)
    if missing_mappings:
        raise RuntimeError(
            f"{action} extraction_strategy missing per_site_label_mapping for: {sorted(missing_mappings)}"
        )
    for client_id in sorted(applicable):
        site_mapping = per_site.get(client_id)
        if not isinstance(site_mapping, dict):
            raise RuntimeError(f"{action} extraction_strategy mapping for {client_id} must be an object")
    for field in ("image_rule", "label_rule", "split_rule", "visual_qc_rule", "next_step"):
        if not isinstance(strategy.get(field), dict):
            raise RuntimeError(f"{action} extraction_strategy.{field} must be an object")
    generated_contract = strategy.get("generated_data_contract")
    if isinstance(generated_contract, dict):
        _require_generated_data_contract(generated_contract, action=action)


def _require_generated_data_contract(contract: dict[str, Any], *, action: str) -> None:
    errors = generated_data_contract_validation_errors(contract)
    if errors:
        raise RuntimeError(f"{action} extraction_strategy.generated_data_contract is invalid: " + "; ".join(errors))


def _string_list_set(value: Any, *, field: str, action: str) -> set[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuntimeError(f"{action} extraction_strategy.{field} must be a list of strings")
    return set(value)


def _raise_if_template_passthrough(*, action: str, output: dict[str, Any], template: dict[str, Any]) -> None:
    if output == template:
        raise RuntimeError(f"{action} backend returned output_template unchanged; refusing template fallback")


def _normalize_training_code_output(output: dict[str, Any], *, code_workspace: str) -> dict[str, Any]:
    if not isinstance(output, dict):
        return output
    normalized = dict(output)
    workspace = Path(code_workspace).resolve()

    if not isinstance(normalized.get("package_dir"), str) or not normalized["package_dir"].strip():
        package = normalized.get("package")
        files = normalized.get("files")
        package_root = None
        if isinstance(package, dict):
            package_root = package.get("root") or package.get("name")
        if not package_root and isinstance(files, dict):
            package_root_file = files.get("package_root")
            if isinstance(package_root_file, str) and package_root_file.strip():
                package_root = Path(package_root_file).parts[0]
        if isinstance(package_root, str) and package_root.strip():
            candidate = Path(package_root)
            if not candidate.is_absolute():
                candidate = workspace / candidate
            normalized["package_dir"] = str(candidate.resolve())

    if not isinstance(normalized.get("entry_script"), str) or not normalized["entry_script"].strip():
        script = None
        entry_script = normalized.get("entry_script")
        if isinstance(entry_script, dict):
            for key in ("path", "script", "entry_script", "file", "relative_path"):
                value = entry_script.get(key)
                if isinstance(value, str) and value.strip():
                    script = value
                    break
        if script is None:
            entry_point = normalized.get("entry_point")
            script = entry_point.get("script") if isinstance(entry_point, dict) else None
        if isinstance(script, str) and script.strip():
            normalized["entry_script"] = script

    if not isinstance(normalized.get("model_class_path"), str) or not normalized["model_class_path"].strip():
        framework = normalized.get("framework")
        model_class = framework.get("model_class") if isinstance(framework, dict) else None
        if not model_class:
            model_info = normalized.get("model_class")
            if isinstance(model_info, dict):
                module_name = model_info.get("module")
                class_name = model_info.get("class")
                if isinstance(module_name, str) and isinstance(class_name, str):
                    model_class = f"{module_name}.{class_name}"
        if isinstance(model_class, str) and model_class.strip():
            normalized["model_class_path"] = model_class

    if not isinstance(normalized.get("framework"), str):
        framework = normalized.get("framework")
        if isinstance(framework, dict) and isinstance(framework.get("name"), str):
            normalized["framework"] = framework["name"]

    reference_selection = normalized.get("reference_example_selection")
    if isinstance(reference_selection, dict):
        if not isinstance(normalized.get("selected_reference_example_path"), str):
            for key in ("selected_reference_example_path", "path", "url", "source", "example_path", "example_url"):
                value = reference_selection.get(key)
                if isinstance(value, str) and value.strip():
                    normalized["selected_reference_example_path"] = value
                    break
        if not isinstance(normalized.get("reference_api_evidence"), list) or not normalized["reference_api_evidence"]:
            for key in ("reference_api_evidence", "api_evidence", "api_observations", "evidence"):
                value = reference_selection.get(key)
                if isinstance(value, list) and value:
                    normalized["reference_api_evidence"] = value
                    break
    selected_reference = normalized.get("selected_reference_example_path")
    if (
        isinstance(selected_reference, str)
        and selected_reference.strip()
        and normalized.get("base_reference") == "agent_selected_nvflare_example"
    ):
        normalized["base_reference"] = selected_reference

    if "metric_artifacts" not in normalized:
        metrics_output = normalized.get("metrics_output")
        artifact = metrics_output.get("artifact") if isinstance(metrics_output, dict) else None
        if isinstance(artifact, str) and artifact.strip():
            normalized["metric_artifacts"] = [artifact]
    return normalized


def _reject_reference_training_code_output(output: dict[str, Any], *, action: str) -> None:
    if output.get("deterministic_ground_truth") is True:
        raise RuntimeError(f"{action} returned reference/baseline code from a live backend")


def _adapter_source_label_type(*, policy: dict[str, Any], extraction_result: dict[str, Any]) -> str | None:
    site_mapping = policy.get("site_label_mapping")
    if isinstance(site_mapping, dict) and isinstance(site_mapping.get("source_label_type"), str):
        return site_mapping["source_label_type"]
    action_required = extraction_result.get("agent_action_required")
    if isinstance(action_required, dict) and isinstance(action_required.get("source_label_type"), str):
        return action_required["source_label_type"]
    capability_gap = extraction_result.get("capability_gap")
    if isinstance(capability_gap, dict) and isinstance(capability_gap.get("source_label_type"), str):
        return capability_gap["source_label_type"]
    if isinstance(extraction_result.get("source_label_type"), str):
        return extraction_result["source_label_type"]
    return None


def _normalize_local_adapter_output(output: dict[str, Any], *, client_id: str) -> dict[str, Any]:
    if not isinstance(output, dict):
        raise RuntimeError(f"CLIENT.IMPLEMENT_LOCAL_ADAPTER for {client_id} returned non-object output")
    nested = output.get("local_adapter")
    if isinstance(nested, dict):
        normalized = dict(nested)
    else:
        normalized = dict(output)
    if normalized.get("schema_version") == "FedReadyLocalAdapterSpec@v1":
        normalized["schema_version"] = "fedready.local_adapter_spec.v1"
    if "status" not in normalized and isinstance(normalized.get("adapter_status"), str):
        normalized["status"] = normalized["adapter_status"]
    if "manifest_path" not in normalized and isinstance(normalized.get("adapter_manifest_path"), str):
        normalized["manifest_path"] = normalized["adapter_manifest_path"]
    if "script_path" not in normalized and isinstance(normalized.get("adapter_script_path"), str):
        normalized["script_path"] = normalized["adapter_script_path"]
    adapter = normalized.get("adapter")
    if isinstance(adapter, dict):
        if "manifest_path" not in normalized and isinstance(adapter.get("manifest_path"), str):
            normalized["manifest_path"] = adapter["manifest_path"]
        if "script_path" not in normalized and isinstance(adapter.get("script_path"), str):
            normalized["script_path"] = adapter["script_path"]
    manifest = normalized.get("manifest")
    if isinstance(manifest, dict):
        if "manifest_path" not in normalized and isinstance(manifest.get("path"), str):
            normalized["manifest_path"] = manifest["path"]
        if "record_count" not in normalized and isinstance(manifest.get("record_count"), int):
            normalized["record_count"] = manifest["record_count"]
    if (
        "status" not in normalized
        and isinstance(normalized.get("manifest_path"), str)
        and isinstance(normalized.get("script_path"), str)
        and isinstance(normalized.get("record_count"), int)
        and normalized["record_count"] > 0
    ):
        normalized["status"] = "implemented"
    normalized.setdefault("schema_version", "fedready.local_adapter_spec.v1")
    normalized.setdefault("client_id", client_id)
    normalized.setdefault("adapter_kind", "client_local_label_adapter")
    normalized.setdefault("safe_to_share", False)
    return normalized


def _safe_local_adapter_revision_output(output: dict[str, Any]) -> dict[str, Any]:
    """Give both live backends the same path-free prior-attempt summary."""

    return {
        "schema_version": output.get("schema_version"),
        "client_id": output.get("client_id"),
        "status": output.get("status"),
        "source_label_type": output.get("source_label_type"),
        "adapter_kind": output.get("adapter_kind"),
        "record_count": output.get("record_count"),
        "manifest_path_present": bool(output.get("manifest_path") or output.get("adapter_manifest_path")),
        "script_path_present": bool(output.get("script_path")),
        "safe_to_share": True,
    }


def _local_adapter_backend_failure_feedback(
    *,
    client_id: str,
    adapter_context: dict[str, Any],
    error: Exception,
) -> dict[str, Any]:
    broker_diagnostic = None
    workspace = adapter_context.get("adapter_workspace")
    if isinstance(workspace, str) and workspace.strip():
        state_path = Path(workspace) / "fedready_adapter_broker_state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = None
        if isinstance(state, dict) and isinstance(state.get("diagnostic"), str):
            broker_diagnostic = state["diagnostic"][:2000]
    return {
        "schema_version": "fedready.local_adapter_feedback.v1",
        "stage": "agent_harness_execution",
        "client_id": client_id,
        "error": _short_error(error),
        "broker_diagnostic": _redact_local_paths_in_text(broker_diagnostic) if broker_diagnostic else None,
        "required_fix": render_client_prompt("local_adapter_backend_failure_required_fix"),
    }


def _require_local_adapter_agent_output(output: dict[str, Any], *, client_id: str) -> None:
    if output.get("schema_version") != "fedready.local_adapter_spec.v1":
        raise RuntimeError(
            f"CLIENT.IMPLEMENT_LOCAL_ADAPTER for {client_id} missing " "schema_version=fedready.local_adapter_spec.v1"
        )
    if output.get("client_id") not in {None, client_id}:
        raise RuntimeError(f"CLIENT.IMPLEMENT_LOCAL_ADAPTER response client_id did not match {client_id}")
    status = output.get("status")
    allowed = {"implemented", "unfeasible", "failed"}
    if status == "not_implemented":
        raise RuntimeError(
            f"CLIENT.IMPLEMENT_LOCAL_ADAPTER for {client_id} returned not_implemented; " "refusing template fallback"
        )
    if status not in allowed:
        raise RuntimeError(
            f"CLIENT.IMPLEMENT_LOCAL_ADAPTER for {client_id} returned invalid status {status!r}; "
            "expected implemented, unfeasible, or failed"
        )
    if status == "implemented":
        manifest_path = output.get("manifest_path")
        script_path = output.get("script_path")
        record_count = output.get("record_count")
        if not isinstance(manifest_path, str) or not manifest_path.strip():
            raise RuntimeError(f"CLIENT.IMPLEMENT_LOCAL_ADAPTER for {client_id} missing manifest_path")
        if not isinstance(script_path, str) or not script_path.strip():
            raise RuntimeError(f"CLIENT.IMPLEMENT_LOCAL_ADAPTER for {client_id} missing script_path")
        if not isinstance(record_count, int) or record_count <= 0:
            raise RuntimeError(f"CLIENT.IMPLEMENT_LOCAL_ADAPTER for {client_id} missing positive record_count")
    elif not isinstance(output.get("reason"), str) or not output["reason"].strip():
        raise RuntimeError(f"CLIENT.IMPLEMENT_LOCAL_ADAPTER for {client_id} status {status!r} missing reason")


def _generic_training_code_spec_template() -> dict[str, Any]:
    return {
        "schema_version": "fedready.training_code_spec.v1",
        "status": "requires_agent_implementation",
        "package_dir": None,
        "entry_script": None,
        "model_class_path": None,
        "model_args": {},
        "mock_record_template": None,
        "task_script_args_template": GENERIC_TRAINING_TASK_ARGS_TEMPLATE,
        "framework": None,
        "base_reference": NVFLARE_EXAMPLE_SELECTION_ID,
        "selected_reference_example_path": None,
        "reference_api_evidence": [],
        "reference_example_selection": {
            "base_example": NVFLARE_EXAMPLE_SELECTION_ID,
            "selection_prompt": NVFLARE_EXAMPLE_SELECTION_PROMPT,
            "required_output_fields": ["selected_reference_example_path", "reference_api_evidence"],
        },
        "metric_artifacts": ["fedready_training_metrics.jsonl"],
        "safe_to_share": False,
        "reason": "Task-dependent training code must be implemented by the server agent at run time.",
    }


def _generic_data_materializer_spec_template(*, generated_contract: dict[str, Any]) -> dict[str, Any]:
    sample_manifest = generated_contract.get("sample_manifest") if isinstance(generated_contract, dict) else None
    sample_manifest_format = (
        generated_contract.get("sample_manifest_format") if isinstance(generated_contract, dict) else None
    )
    record_type = generated_contract.get("record_type") if isinstance(generated_contract, dict) else None
    return {
        "schema_version": DATA_MATERIALIZER_SPEC_OUTPUT_SCHEMA,
        "status": "requires_agent_implementation",
        "record_type": record_type,
        "sample_manifest": sample_manifest,
        "sample_manifest_format": sample_manifest_format,
        "package_dir": None,
        "entry_script": None,
        "interface": {
            "cli": [
                "--adapter-manifest",
                "--output-dir",
                "--data-contract",
                "--policy",
                "--report-path",
            ],
            "report_schema": "fedready.generated_materializer_report.v1",
        },
        "safe_to_share": False,
        "reason": "Generated-contract materializer must be implemented by the server agent at run time.",
    }


def _normalize_data_materializer_output(output: dict[str, Any], *, code_workspace: str) -> dict[str, Any]:
    if not isinstance(output, dict):
        raise RuntimeError("SERVER.IMPLEMENT_DATA_MATERIALIZER returned non-object output")
    nested = output.get("data_materializer")
    if isinstance(nested, dict):
        output = dict(nested)
    normalized = dict(output)
    normalized.setdefault("schema_version", DATA_MATERIALIZER_SPEC_OUTPUT_SCHEMA)
    normalized.setdefault("status", "implemented")
    workspace = Path(code_workspace).resolve()
    package_dir = normalized.get("package_dir")
    if not isinstance(package_dir, str) or not package_dir.strip():
        package_dir = str(workspace)
    package_path = Path(package_dir).expanduser()
    if not package_path.is_absolute():
        if package_path.parts and package_path.parts[0] in {"code_workspace", "materializer_workspace"}:
            package_path = Path(*package_path.parts[1:])
        package_path = workspace / package_path
    package_path = package_path.resolve()
    normalized["package_dir"] = str(package_path)
    entry_script = normalized.get("entry_script")
    if isinstance(entry_script, dict):
        entry_script = entry_script.get("path")
    if not isinstance(entry_script, str) or not entry_script.strip():
        candidates = ["materializer.py", "fedready_generated_materializer.py"]
        for candidate in candidates:
            if (package_path / candidate).is_file():
                entry_script = candidate
                break
    if isinstance(entry_script, str) and entry_script.strip():
        resolved_entry = _resolve_data_materializer_entry_path(
            entry_script=entry_script,
            package_path=package_path,
            workspace=workspace,
        )
        if resolved_entry is not None:
            entry_script = resolved_entry.relative_to(package_path).as_posix()
    normalized["entry_script"] = entry_script
    return normalized


def _resolve_data_materializer_entry_path(*, entry_script: str, package_path: Path, workspace: Path) -> Path | None:
    entry_path = Path(entry_script).expanduser()
    candidates: list[Path] = []
    if entry_path.is_absolute():
        candidates.append(entry_path)
    else:
        candidates.append(package_path / entry_path)
        parts = entry_path.parts
        if parts and parts[0] in {"code_workspace", "materializer_workspace"}:
            candidates.append(workspace / Path(*parts[1:]))
        candidates.append(workspace / entry_path)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved == package_path or package_path not in resolved.parents:
            continue
        if resolved.is_file():
            return resolved
    return None


def _require_data_materializer_agent_output(output: dict[str, Any]) -> None:
    if output.get("schema_version") != DATA_MATERIALIZER_SPEC_OUTPUT_SCHEMA:
        raise RuntimeError(
            "SERVER.IMPLEMENT_DATA_MATERIALIZER missing " f"schema_version={DATA_MATERIALIZER_SPEC_OUTPUT_SCHEMA}"
        )
    if output.get("status") != "implemented":
        raise RuntimeError("SERVER.IMPLEMENT_DATA_MATERIALIZER did not return status=implemented")
    package_dir = output.get("package_dir")
    entry_script = output.get("entry_script")
    if not isinstance(package_dir, str) or not package_dir.strip():
        raise RuntimeError("SERVER.IMPLEMENT_DATA_MATERIALIZER missing package_dir")
    if not isinstance(entry_script, str) or not entry_script.strip():
        raise RuntimeError("SERVER.IMPLEMENT_DATA_MATERIALIZER missing entry_script")
    package_path = Path(package_dir).expanduser().resolve()
    if not package_path.is_dir():
        raise RuntimeError("SERVER.IMPLEMENT_DATA_MATERIALIZER package_dir is not a directory")
    resolved_entry = _resolve_data_materializer_entry_path(
        entry_script=entry_script,
        package_path=package_path,
        workspace=package_path.parent,
    )
    if resolved_entry is None:
        raise RuntimeError("SERVER.IMPLEMENT_DATA_MATERIALIZER entry_script does not exist")


def _generic_extraction_round_summary(*, task: dict[str, Any], extraction_results: dict[str, Any]) -> dict[str, Any]:
    extracted = {
        client_id: result
        for client_id, result in extraction_results.items()
        if isinstance(result, dict) and result.get("data") == "extracted"
    }
    screened_out = {
        client_id: result
        for client_id, result in extraction_results.items()
        if isinstance(result, dict) and result.get("data") in {"screened out", "no valid cases"}
    }
    failed = {
        client_id: result
        for client_id, result in extraction_results.items()
        if not (isinstance(result, dict) and result.get("data") in {"extracted", "screened out", "no valid cases"})
    }
    total_samples = 0
    for result in extracted.values():
        counts = result.get("counts", {})
        if isinstance(counts, dict) and isinstance(counts.get("total"), int):
            total_samples += counts["total"]
    qc_required = {
        client_id: result for client_id, result in extracted.items() if _visual_qc_required_for_training_result(result)
    }
    qc_not_required = {client_id: result for client_id, result in extracted.items() if client_id not in qc_required}
    qc_passed = {
        client_id: result for client_id, result in qc_required.items() if _visual_qc_ready_for_training(result)
    }
    qc_failed = {
        client_id: result
        for client_id, result in qc_required.items()
        if isinstance(result.get("visual_qc"), dict) and result["visual_qc"].get("status") == "failed"
    }
    qc_pending = {
        client_id: result
        for client_id, result in qc_required.items()
        if client_id not in qc_passed and client_id not in qc_failed
    }
    ready_to_continue = bool(extracted) and len(qc_passed) == len(qc_required) and not qc_failed
    return {
        "schema_version": "fedready.extraction_round_summary.v1",
        "task": _task_text_for_contract(task),
        "extracted_client_count": len(extracted),
        "screened_out_client_count": len(screened_out),
        "failed_client_count": len(failed),
        "extracted_clients": sorted(extracted),
        "screened_out_clients": sorted(screened_out),
        "failed_clients": sorted(failed),
        "total_extracted_samples": total_samples,
        "visual_qc": {
            "required_client_count": len(qc_required),
            "not_required_client_count": len(qc_not_required),
            "passed_client_count": len(qc_passed),
            "pending_client_count": len(qc_pending),
            "failed_client_count": len(qc_failed),
            "passed_clients": sorted(qc_passed),
            "pending_clients": sorted(qc_pending),
            "failed_clients": sorted(qc_failed),
            "not_required_clients": sorted(qc_not_required),
            "ready_definition": "required QC has status=passed and selected_transform matches extracted label_orientation.selected_transform",
        },
        "extraction_results": extraction_results,
        "next_step": {
            "action": (
                "collect_post_extraction_statistics" if ready_to_continue else "resolve_visual_qc_before_training"
            ),
            "reason": (
                "all extracted clients passed visual QC"
                if ready_to_continue
                else (
                    "no clients produced extracted data"
                    if not extracted
                    else "one or more extracted clients still need passing visual QC before training"
                )
            ),
        },
    }


def _visual_qc_ready_for_training(result: dict[str, Any]) -> bool:
    return visual_qc_result_ready_for_training(result)


def _visual_qc_required_for_training_result(result: dict[str, Any]) -> bool:
    visual_qc = result.get("visual_qc") if isinstance(result.get("visual_qc"), dict) else {}
    if visual_qc.get("review_required") is False:
        return False
    if result.get("record_type") == CLASSIFICATION:
        return False
    extraction = result.get("extraction") if isinstance(result.get("extraction"), dict) else {}
    if extraction.get("classification_storage") is not None:
        return False
    if extraction.get("sample_manifest_format") == "monai_decathlon_datalist_json":
        return False
    if visual_qc.get("status") == "not_performed":
        reason = str(visual_qc.get("reason") or "").lower()
        if "not applicable" in reason or "not required" in reason or "classification" in reason:
            return False
    return True


def _generic_extraction_report(*, extraction_result: dict[str, Any]) -> dict[str, Any]:
    return extraction_result


def _adapter_preflight_visual_qc_decision(*, client_id: str, qc_context: dict[str, Any]) -> dict[str, Any]:
    """Reuse the digest-bound local guardrail verdict without a second VLM pass."""

    output = _live_visual_qc_template(client_id=client_id, qc_context=qc_context)
    sample_count = max(1, int(output.get("sample_count") or 0))
    output.update(
        {
            "status": "passed",
            "passed": True,
            "reviewed": True,
            "review_required": True,
            "visual_qc_owner": "adapter_preflight_local_guardrail",
            "reviewed_sample_count": 1,
            "selected_transform": "as_is",
            "selected_transform_counts": {"as_is": 1},
            "consensus_required_count": 1,
            "consensus_reached": True,
            "issues": [],
            "reason": "Digest-bound adapter preflight raw and label review passed locally.",
            "recommendation": "continue_with_extracted_records",
            "sample_count": sample_count,
        }
    )
    return output


def _task_text_for_contract(task: dict[str, Any]) -> str:
    value = task.get("task")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return json.dumps(task, sort_keys=True)


def _live_client_inquiry_template(*, task: dict[str, Any], client_ids: list[str]) -> dict[str, Any]:
    task_text = _task_text_for_contract(task)
    inquiry = render_server_prompt_object("live_client_inquiry_template", task_text=task_text)
    if not isinstance(inquiry, dict):
        raise TypeError("live_client_inquiry_template prompt entry must be an object")
    inquiry = dict(inquiry)
    inquiry["clients"] = [{"client_id": client_id} for client_id in client_ids]
    return inquiry


def _require_live_client_inquiry_output(output: dict[str, Any]) -> None:
    if not isinstance(output, dict):
        raise RuntimeError("SERVER.DEFINE_PROFILE_REQUEST agent output must be a JSON object")
    if output.get("schema_version") != "fedready.client_inquiry.v1":
        raise RuntimeError("SERVER.DEFINE_PROFILE_REQUEST missing schema_version=fedready.client_inquiry.v1")
    if not isinstance(output.get("intent"), dict) or not output["intent"]:
        raise RuntimeError("SERVER.DEFINE_PROFILE_REQUEST missing non-empty intent object")
    if not isinstance(output.get("message"), dict):
        raise RuntimeError("SERVER.DEFINE_PROFILE_REQUEST missing message object")
    clients = output.get("clients")
    if not isinstance(clients, list) or not all(isinstance(client, dict) for client in clients):
        raise RuntimeError("SERVER.DEFINE_PROFILE_REQUEST clients must be a list of objects")


def _live_server_site_summary_template(
    *,
    task: dict[str, Any],
    applicable_client_ids: list[str],
) -> dict[str, Any]:
    task_text = _task_text_for_contract(task)
    strategy_body = {
        "applicable_clients": list(applicable_client_ids),
        "per_site_label_mapping": {
            client_id: {"preparation": "client_agent_local_manifest"} for client_id in applicable_client_ids
        },
        "image_rule": {
            "source_selection": "client_agent_determined_from_local_evidence",
            "training_transforms": "record generic requirements; apply transforms at training time",
        },
        "label_rule": {
            "task_target": "derive from task intent and client-local semantic evidence",
            "multiple_annotations": "select the task-relevant annotation; fail closed if ambiguous",
            "output": "canonical task-aligned label referenced by the local manifest",
        },
        "split_rule": {
            "preserve_existing_splits": True,
            "otherwise": "create reproducible client-local splits before sampling",
            "do_not_export_sample_ids": True,
        },
        "visual_qc_rule": {
            "required": "required for spatial image/label outputs; not applicable to scalar image-level classification labels",
            "purpose": "verify local image/label correspondence and task alignment using the appropriate local evidence for the task",
            "if_ambiguous": "request additional local samples or fail closed",
        },
        "next_step": {
            "action": "dispatch_client_local_manifest_preparation",
            "reason": "client agents own deploy-time local data interpretation",
        },
    }
    digest_source = json.dumps(strategy_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": "fedready.server_site_summary.v1",
        "task": task_text,
        "extraction_strategy": {
            "schema_version": "fedready.task_extraction_strategy.v1",
            "strategy_digest": f"sha256:{hashlib.sha256(digest_source).hexdigest()}",
            "applicable_client_count": len(applicable_client_ids),
            **strategy_body,
        },
    }


def _generated_materializer_agent_summary(materializer: Any) -> dict[str, Any] | None:
    if not isinstance(materializer, dict):
        return None
    summary: dict[str, Any] = {}
    for key in (
        "schema_version",
        "status",
        "record_type",
        "sample_manifest",
        "sample_manifest_format",
        "entry_script",
        "source_digest",
    ):
        value = materializer.get(key)
        if value is not None:
            summary[key] = value
    source_files = materializer.get("source_files")
    if isinstance(source_files, list):
        summary["source_file_count"] = len(source_files)
        summary["source_files_redacted"] = True
    return summary


def _summarize_generated_materializer_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summarized = json.loads(json.dumps(payload, default=str))
    materializer_summary = _generated_materializer_agent_summary(summarized.get("generated_data_materializer"))
    if materializer_summary is not None:
        summarized["generated_data_materializer"] = materializer_summary
    policies = summarized.get("policies")
    if isinstance(policies, dict):
        summarized_policies: dict[str, Any] = {}
        for client_id, policy in policies.items():
            if isinstance(policy, dict):
                policy_copy = dict(policy)
                policy_summary = _generated_materializer_agent_summary(policy_copy.get("generated_data_materializer"))
                if policy_summary is not None:
                    policy_copy["generated_data_materializer"] = policy_summary
                summarized_policies[client_id] = policy_copy
            else:
                summarized_policies[client_id] = policy
        summarized["policies"] = summarized_policies
    return summarized


def _restore_generated_materializer_payload(
    output: dict[str, Any],
    *,
    materializer: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(materializer, dict):
        return output
    restored = dict(output)
    restored["generated_data_materializer"] = materializer
    policies = restored.get("policies")
    if isinstance(policies, dict):
        restored_policies: dict[str, Any] = {}
        for client_id, policy in policies.items():
            if isinstance(policy, dict):
                policy_copy = dict(policy)
                policy_copy["generated_data_materializer"] = materializer
                restored_policies[client_id] = policy_copy
            else:
                restored_policies[client_id] = policy
        restored["policies"] = restored_policies
    return restored


def _live_extraction_dispatch_template(
    *,
    task: dict[str, Any],
    extraction_strategy: dict[str, Any],
    target_client_ids: list[str],
    extraction_config: dict[str, Any],
) -> dict[str, Any]:
    per_site = extraction_strategy.get("per_site_label_mapping")
    per_site = per_site if isinstance(per_site, dict) else {}
    generated_contract = _generated_contract_with_prepared_split_policy(
        extraction_strategy.get("generated_data_contract")
    )
    policies: dict[str, Any] = {}
    for client_id in target_client_ids:
        site_mapping = per_site.get(client_id)
        policies[client_id] = {
            "schema_version": "fedready.site_extraction_policy.v1",
            "client_id": client_id,
            "task": _task_text_for_contract(task),
            "applicable": True,
            "strategy_digest": extraction_strategy.get("strategy_digest"),
            "image_rule": extraction_strategy.get("image_rule", {}),
            "label_rule": extraction_strategy.get("label_rule", {}),
            "split_rule": extraction_strategy.get("split_rule", {}),
            "visual_qc_rule": extraction_strategy.get("visual_qc_rule") or _default_visual_qc_rule(),
            "generated_data_contract": generated_contract if isinstance(generated_contract, dict) else None,
            "generated_data_materializer": (
                extraction_strategy.get("generated_data_materializer")
                if isinstance(extraction_strategy.get("generated_data_materializer"), dict)
                else None
            ),
            "site_label_mapping": site_mapping if isinstance(site_mapping, dict) else {},
            "extraction_config": dict(extraction_config),
            "local_adapter_rule": render_server_prompt_object("local_adapter_rule"),
        }
    return {
        "schema_version": "fedready.extraction_dispatch.v1",
        "task": _task_text_for_contract(task),
        "strategy_digest": extraction_strategy.get("strategy_digest"),
        "generated_data_materializer": (
            extraction_strategy.get("generated_data_materializer")
            if isinstance(extraction_strategy.get("generated_data_materializer"), dict)
            else None
        ),
        "target_clients": [{"client_id": client_id} for client_id in target_client_ids],
        "policies": policies,
    }


def _default_visual_qc_rule() -> dict[str, Any]:
    return {
        "required": True,
        "sample_count": 3,
        "reviewer": "client_local_vlm",
        "purpose": "verify local image/label correspondence and task alignment before training use",
        "if_ambiguous": "sample_additional_cases_or_hold_client_for_review",
        "if_failed": "return qc failure and require local adapter correction",
    }


def _lock_live_local_datalist_contract(
    output: dict[str, Any],
    *,
    target_client_ids: list[str],
) -> dict[str, Any]:
    policies = output.get("policies")
    policies = policies if isinstance(policies, dict) else {}
    locked_policies: dict[str, Any] = {}
    for client_id in target_client_ids:
        policy = policies.get(client_id)
        if not isinstance(policy, dict):
            raise RuntimeError(f"SERVER.DISPATCH_EXTRACTION_POLICY missing policy for {client_id}")
        locked_policy = dict(policy)
        locked_policy["client_id"] = client_id
        locked_policy["applicable"] = True
        generated_contract = locked_policy.get("generated_data_contract")
        if isinstance(generated_contract, dict):
            locked_policy["generated_data_contract"] = _generated_contract_with_prepared_split_policy(
                generated_contract
            )
        generated_materializer = output.get("generated_data_materializer")
        if isinstance(generated_materializer, dict):
            locked_policy["generated_data_materializer"] = generated_materializer
        locked_policy["local_adapter_rule"] = render_server_prompt_object("local_adapter_rule")
        locked_policies[client_id] = locked_policy
    locked = dict(output)
    locked["target_clients"] = [{"client_id": client_id} for client_id in target_client_ids]
    locked["policies"] = locked_policies
    return locked


def _merge_output_template(output_template: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(output, dict):
        raise TypeError("Agent output must be a JSON object; refusing output-template fallback.")
    merged = dict(output_template)
    for key, value in output.items():
        template_value = output_template.get(key)
        if isinstance(template_value, dict) and isinstance(value, dict):
            merged[key] = _merge_output_template(template_value, value)
        else:
            merged[key] = value
    if isinstance(output_template.get("schema_version"), str):
        merged["schema_version"] = output_template["schema_version"]
    return merged


def _lock_harness_owned_fields(output: dict[str, Any], *, baseline: dict[str, Any]) -> dict[str, Any]:
    locked = dict(output)
    for key, value in baseline.items():
        locked[key] = json.loads(json.dumps(value, default=str))
    return locked


def _lock_extraction_runtime_config(output: dict[str, Any], *, extraction_config: dict[str, Any]) -> dict[str, Any]:
    """Keep workflow-owned extraction settings out of backend control."""
    policies = output.get("policies")
    if not isinstance(policies, dict):
        return output

    locked_output = dict(output)
    locked_policies: dict[str, Any] = {}
    for client_id, policy in policies.items():
        if isinstance(policy, dict):
            locked_policy = dict(policy)
            locked_policy["extraction_config"] = dict(extraction_config)
            locked_policies[client_id] = locked_policy
        else:
            locked_policies[client_id] = policy
    locked_output["policies"] = locked_policies
    return locked_output


def _sanitize_profile_inquiry_intent(intent: Any, *, baseline: Any) -> dict[str, Any]:
    source = intent if isinstance(intent, dict) else baseline if isinstance(baseline, dict) else {}
    sanitized: dict[str, Any] = {}
    for key, value in source.items():
        key_text = str(key)
        value_text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        lowered = f"{key_text} {value_text}".lower()
        if "generated_data_contract" in lowered or key_text == "fail_closed_condition":
            continue
        sanitized[key_text] = value
    task_description = sanitized.get("task_description")
    if not isinstance(task_description, str) or not task_description.strip():
        baseline_task = baseline.get("task_description") if isinstance(baseline, dict) else None
        if isinstance(baseline_task, str) and baseline_task.strip():
            sanitized["task_description"] = baseline_task.strip()
    sanitized["routing_scope"] = "task_family_evidenced_safe_aggregate_metadata_only"
    sanitized["contract_timing"] = render_server_prompt("profile_inquiry_contract_timing")
    return sanitized


def _lock_client_inquiry_policy(output: dict[str, Any], *, baseline: dict[str, Any]) -> dict[str, Any]:
    """Keep workflow-owned screening instructions out of backend control."""
    if not isinstance(output, dict):
        return output
    baseline_message = baseline.get("message")
    if not isinstance(baseline_message, dict):
        return output
    intent = output.get("intent") if isinstance(output.get("intent"), dict) else baseline.get("intent")
    intent = _sanitize_profile_inquiry_intent(intent, baseline=baseline.get("intent"))
    return {
        "schema_version": baseline.get("schema_version"),
        "task": baseline.get("task"),
        "intent": intent,
        "message": {
            "task": baseline_message.get("task"),
            "intent": intent,
            "instructions": list(baseline_message.get("instructions", [])),
        },
        "clients": [dict(client) for client in baseline.get("clients", []) if isinstance(client, dict)],
    }


def _normalize_visual_qc_output(output: dict[str, Any], qc_context: dict[str, Any] | None = None) -> dict[str, Any]:
    allowed_statuses = {"passed", "failed", "needs_more_samples", "not_performed", "needs_agent_visual_review"}
    normalized = dict(output)
    for field in VISUAL_QC_PER_SAMPLE_FIELDS:
        normalized.pop(field, None)
    consensus = _visual_qc_consensus(normalized, qc_context)
    if consensus["expected_sample_count"] > 0:
        normalized["sample_count"] = consensus["expected_sample_count"]
        normalized["reviewed_sample_count"] = consensus["reviewed_sample_count"]
        normalized["consensus_required_count"] = consensus["required_count"]
        normalized["consensus_reached"] = consensus["reached"]
        normalized["selected_transform_counts"] = consensus["counts"]
    if consensus["expected_sample_count"] > 1:
        if consensus["reached"]:
            selected_transform = consensus["selected_transform"]
            normalized["selected_transform"] = selected_transform
            normalized["reviewed"] = True
            if selected_transform == "as_is":
                normalized["status"] = "passed"
                normalized["passed"] = True
                normalized["issues"] = []
                normalized["recommendation"] = "accept_extraction_for_training"
                return normalized
            normalized["status"] = "failed"
            normalized["passed"] = False
            _append_visual_qc_issue(normalized, f"visual QC selected non-as_is transform: {selected_transform}")
            normalized["recommendation"] = f"rerun extraction or adapter with selected_transform={selected_transform}"
            return normalized
        if consensus["reviewed_sample_count"] > 0 or _visual_qc_has_decision(normalized):
            normalized["status"] = "needs_more_samples"
            normalized["passed"] = False
            normalized["selected_transform"] = "undecided"
            _append_visual_qc_issue(
                normalized,
                "visual QC requires >=2/3 transform consensus before passing or selecting a repair transform",
            )
            normalized["recommendation"] = "review_more_visual_qc_samples_before_accepting_extraction"
            return normalized

    selected = normalized.get("selected_transform")
    status = normalized.get("status")
    if selected in {"hflip", "vflip", "rot180"}:
        normalized["status"] = "failed"
        normalized["passed"] = False
        _append_visual_qc_issue(normalized, f"visual QC selected non-as_is transform: {selected}")
        normalized.setdefault("recommendation", f"rerun extraction or adapter with selected_transform={selected}")
        return normalized
    if status not in allowed_statuses:
        if normalized.get("passed") is True and selected == "as_is":
            normalized["status"] = "passed"
        elif selected == "undecided" and normalized.get("reviewed") is True:
            normalized["status"] = "needs_more_samples"
        else:
            normalized["status"] = "needs_agent_visual_review"
    if normalized.get("status") == "passed":
        if selected == "as_is" and normalized.get("reviewed") is True:
            normalized["passed"] = True
        else:
            normalized["status"] = "needs_agent_visual_review"
            normalized["passed"] = False
    elif normalized.get("status") in {"failed", "needs_more_samples", "needs_agent_visual_review"}:
        normalized["passed"] = False
    return normalized


def _live_visual_qc_template(*, client_id: str, qc_context: dict[str, Any]) -> dict[str, Any]:
    artifacts = _visual_qc_artifacts(qc_context)
    sample_count = len(artifacts)
    available = bool(qc_context.get("available")) and sample_count > 0
    review_required = qc_context.get("review_required") is not False
    visual_qc_owner = str(qc_context.get("visual_qc_owner") or "client_local_visual_qc")
    return {
        "schema_version": "fedready.extraction_visual_qc_decision.v1",
        "client_id": client_id,
        "status": "needs_agent_visual_review" if available and review_required else "not_performed",
        "passed": False,
        "reviewed": False,
        "review_required": review_required,
        "visual_qc_owner": visual_qc_owner,
        "sample_count": sample_count,
        "reviewed_sample_count": 0,
        "selected_transform": "undecided",
        "selected_transform_counts": {},
        "consensus_required_count": max(1, (2 * sample_count + 2) // 3) if sample_count else 0,
        "consensus_reached": False,
        "issues": [] if available or not review_required else ["no_local_qc_artifacts"],
        "reason": (
            "Visual QC is not required by the active generated contract."
            if not review_required
            else (
                "Live visual QC requires an explicit local review decision."
                if available
                else "No local QC artifacts are available for visual review."
            )
        ),
        "recommendation": (
            "continue_without_visual_qc"
            if not review_required
            else ("review_local_qc_artifacts" if available else "do_not_use_for_training_until_qc_artifacts_exist")
        ),
        "safe_to_share": True,
    }


def _aggregate_visual_qc_sample_outputs(
    *,
    base_output: dict[str, Any],
    sample_outputs: list[dict[str, Any]],
    qc_context: dict[str, Any],
) -> dict[str, Any]:
    if base_output.get("deterministic_ground_truth") is True or any(
        sample.get("deterministic_ground_truth") is True for sample in sample_outputs
    ):
        raise RuntimeError("local visual QC received deterministic_ground_truth output")
    counts: dict[str, int] = {}
    reviewed_count = 0
    issues: list[str] = []
    local_vlm_reviews: list[dict[str, Any]] = []
    for sample_output in sample_outputs:
        selected = _visual_qc_sample_vote(sample_output)
        if selected in VISUAL_QC_ALLOWED_SELECTIONS:
            counts[str(selected)] = counts.get(str(selected), 0) + 1
        if sample_output.get("reviewed") is True:
            reviewed_count += 1
        local_vlm_review = sample_output.get("local_vlm_review")
        if isinstance(local_vlm_review, dict):
            local_vlm_reviews.append(local_vlm_review)
    output = dict(base_output)
    output.update(
        {
            "reviewed": reviewed_count > 0,
            "reviewed_sample_count": reviewed_count,
            "selected_transform_counts": counts,
            "issues": issues,
        }
    )
    if local_vlm_reviews:
        output["local_vlm_review"] = {
            "schema_version": LOCAL_VLM_QC_SCHEMA,
            "backend": "local_vlm",
            "status": (
                "completed"
                if all(review.get("status") == "completed" for review in local_vlm_reviews)
                else "partial_or_failed"
            ),
            "reviewed_sample_count": len(local_vlm_reviews),
            "sample_reviews": local_vlm_reviews,
            "calls_logged_client_local": True,
        }

    normalized = _normalize_visual_qc_output(output, qc_context=qc_context)
    if local_vlm_reviews:
        normalized["reason"] = (
            f"Local VLM reviewed {len(local_vlm_reviews)} samples; consensus selected "
            f"{normalized.get('selected_transform', 'undecided')}."
        )
    return normalized


def _visual_qc_sample_vote(sample_output: dict[str, Any]) -> str | None:
    selected = sample_output.get("selected_transform")
    if selected in VISUAL_QC_TRANSFORMS:
        return str(selected)
    counts = _extract_visual_qc_transform_counts(sample_output)
    decision_counts = {transform: count for transform, count in counts.items() if transform in VISUAL_QC_TRANSFORMS}
    if decision_counts:
        top_count = max(decision_counts.values())
        top_transforms = [transform for transform, count in decision_counts.items() if count == top_count]
        if len(top_transforms) == 1 and top_count > 0:
            return top_transforms[0]
    if selected == "undecided":
        return "undecided"
    return None


def _visual_qc_consensus(output: dict[str, Any], qc_context: dict[str, Any] | None) -> dict[str, Any]:
    counts = _extract_visual_qc_transform_counts(output)
    expected_sample_count = _expected_visual_qc_sample_count(output, qc_context, counts)
    selected = output.get("selected_transform")
    if not counts and expected_sample_count <= 1 and selected in VISUAL_QC_TRANSFORMS:
        counts = {str(selected): 1}
        expected_sample_count = max(expected_sample_count, 1)
    reviewed_sample_count = _visual_qc_reviewed_sample_count(output, counts)
    if counts:
        reviewed_sample_count = max(reviewed_sample_count, sum(counts.values()))
    required_count = _visual_qc_required_vote_count(expected_sample_count)
    decision_counts = {transform: count for transform, count in counts.items() if transform in VISUAL_QC_TRANSFORMS}
    selected_transform = "undecided"
    reached = False
    if decision_counts and required_count > 0:
        top_count = max(decision_counts.values())
        top_transforms = [transform for transform, count in decision_counts.items() if count == top_count]
        if len(top_transforms) == 1 and top_count >= required_count:
            selected_transform = top_transforms[0]
            reached = True
    return {
        "counts": {selection: counts.get(selection, 0) for selection in sorted(VISUAL_QC_ALLOWED_SELECTIONS)},
        "expected_sample_count": expected_sample_count,
        "reviewed_sample_count": reviewed_sample_count,
        "required_count": required_count,
        "reached": reached,
        "selected_transform": selected_transform,
    }


def _extract_visual_qc_transform_counts(output: dict[str, Any]) -> dict[str, int]:
    for field in VISUAL_QC_COUNT_FIELDS:
        value = output.get(field)
        if isinstance(value, dict):
            counts = _coerce_visual_qc_counts(value)
            if counts:
                return counts
    for field in VISUAL_QC_TRANSFORM_LIST_FIELDS:
        value = output.get(field)
        if isinstance(value, list):
            counts = _count_visual_qc_selections(value)
            if counts:
                return counts
    for field in VISUAL_QC_PER_SAMPLE_FIELDS:
        value = output.get(field)
        if isinstance(value, list):
            counts = _count_visual_qc_sample_decisions(value)
            if counts:
                return counts
    return {}


def _coerce_visual_qc_counts(value: dict[Any, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, count_value in value.items():
        transform = str(key)
        if transform not in VISUAL_QC_ALLOWED_SELECTIONS:
            continue
        count = _coerce_nonnegative_int(count_value)
        if count is not None and count > 0:
            counts[transform] = counts.get(transform, 0) + count
    return counts


def _count_visual_qc_selections(values: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value in VISUAL_QC_ALLOWED_SELECTIONS:
            counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _count_visual_qc_sample_decisions(values: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        selected = value.get("selected_transform")
        if selected in VISUAL_QC_ALLOWED_SELECTIONS:
            counts[str(selected)] = counts.get(str(selected), 0) + 1
    return counts


def _expected_visual_qc_sample_count(
    output: dict[str, Any],
    qc_context: dict[str, Any] | None,
    counts: dict[str, int],
) -> int:
    candidates: list[int] = []
    for value in (output.get("sample_count"), output.get("reviewed_sample_count")):
        count = _coerce_nonnegative_int(value)
        if count is not None and count > 0:
            candidates.append(count)
    if isinstance(qc_context, dict):
        count = _coerce_nonnegative_int(qc_context.get("sample_count"))
        if count is not None and count > 0:
            candidates.append(count)
        artifacts = qc_context.get("artifacts")
        if isinstance(artifacts, list) and artifacts:
            candidates.append(len(artifacts))
    if counts:
        candidates.append(sum(counts.values()))
    return max(candidates) if candidates else 0


def _visual_qc_reviewed_sample_count(output: dict[str, Any], counts: dict[str, int]) -> int:
    for field in ("reviewed_sample_count", "reviewed_count"):
        count = _coerce_nonnegative_int(output.get(field))
        if count is not None:
            return count
    if counts:
        return sum(counts.values())
    if output.get("reviewed") is True and output.get("selected_transform") in VISUAL_QC_ALLOWED_SELECTIONS:
        return 1
    return 0


def _visual_qc_required_vote_count(sample_count: int) -> int:
    if sample_count <= 0:
        return 0
    return max(1, (2 * sample_count + 2) // 3)


def _visual_qc_has_decision(output: dict[str, Any]) -> bool:
    return (
        output.get("status") in {"passed", "failed"}
        or output.get("passed") is True
        or output.get("selected_transform") in VISUAL_QC_TRANSFORMS
    )


def _append_visual_qc_issue(output: dict[str, Any], message: str) -> None:
    issues = output.get("issues")
    if isinstance(issues, list):
        if message not in issues:
            issues.append(message)
    else:
        output["issues"] = [message]


def _coerce_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float) and value.is_integer():
        return max(0, int(value))
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return None
        return max(0, parsed)
    return None


def _visual_qc_artifacts(qc_context: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = qc_context.get("artifacts") if isinstance(qc_context, dict) else None
    if not isinstance(artifacts, list):
        return []
    return [artifact for artifact in artifacts if isinstance(artifact, dict)]


def _request_local_vlm_visual_qc(
    *,
    client_id: str,
    policy: dict[str, Any],
    extraction_result: dict[str, Any],
    qc_context: dict[str, Any],
    output_template: dict[str, Any],
) -> dict[str, Any] | None:
    if not _should_use_local_vlm_visual_qc():
        return None

    model = _local_vlm_model_name()
    base_url = os.environ.get("FEDREADY_VISION_AGENT_API_BASE_URL", LOCAL_VISION_API_BASE_URL).strip()
    api_key_env = os.environ.get("FEDREADY_VISION_AGENT_API_KEY_ENV", "FEDREADY_LOCAL_VISION_API_KEY").strip()
    max_tokens = _env_positive_int("FEDREADY_VISUAL_QC_VLM_MAX_TOKENS", 512)

    if not model:
        return _local_vlm_qc_failure(
            client_id=client_id,
            output_template=output_template,
            reason="FEDREADY_VISION_AGENT_MODEL is not configured for local visual QC",
            model=model,
            base_url=base_url,
        )
    if not _is_local_vlm_base_url(base_url):
        return _local_vlm_qc_failure(
            client_id=client_id,
            output_template=output_template,
            reason=f"Refusing to send visual QC image to non-local endpoint: {base_url}",
            model=model,
            base_url=base_url,
        )

    artifacts = _visual_qc_artifacts(qc_context)
    artifact = artifacts[0] if artifacts else {}
    image_path = _local_vlm_qc_image_path(artifact)
    if image_path is None:
        return _local_vlm_qc_failure(
            client_id=client_id,
            output_template=output_template,
            reason="local visual QC requires a readable candidate_sheet_path artifact",
            model=model,
            base_url=base_url,
        )
    reference_image_path = task_example_image_path_for_task(
        policy.get("task") if isinstance(policy.get("task"), str) else None
    )
    if reference_image_path is None:
        return _local_vlm_qc_failure(
            client_id=client_id,
            output_template=output_template,
            reason="canonical task_example reference is unavailable or failed integrity validation",
            model=model,
            base_url=base_url,
        )

    prompt = _local_vlm_visual_qc_prompt(
        policy=policy,
        extraction_result=extraction_result,
        qc_context=qc_context,
    )
    log_path = _local_vlm_qc_log_path(image_path)
    try:
        response_text, call_record = _query_local_vlm_image(
            base_url=base_url,
            model=model,
            image_path=image_path,
            prompt=prompt,
            api_key_env=api_key_env,
            max_tokens=max_tokens,
            log_path=log_path,
            reference_image_path=reference_image_path,
        )
        selected_transform, reason = _parse_local_vlm_visual_qc_response(response_text)
        return _local_vlm_qc_decision(
            client_id=client_id,
            output_template=output_template,
            selected_transform=selected_transform,
            reason=reason,
            model=model,
            base_url=base_url,
            call_record=call_record,
        )
    except Exception as exc:
        return _local_vlm_qc_failure(
            client_id=client_id,
            output_template=output_template,
            reason=f"local VLM visual QC call failed: {_short_error(exc)}",
            model=model,
            base_url=base_url,
        )


def _should_use_local_vlm_visual_qc() -> bool:
    return _visual_qc_backend_mode() == "local_vlm"


def _visual_qc_backend_mode() -> str:
    mode = os.environ.get("FEDREADY_VISUAL_QC_BACKEND", "").strip().lower().replace("-", "_")
    if mode in {"", "local_vlm", "vlm", "local"}:
        return "local_vlm"
    raise ValueError(
        "FEDREADY_VISUAL_QC_BACKEND supports only 'local_vlm'; agent, disabled, and deterministic "
        "visual-review fallbacks are unsupported"
    )


def _local_vlm_model_name() -> str:
    value = os.environ.get("FEDREADY_VISION_AGENT_MODEL")
    if value is None:
        return DEFAULT_LOCAL_VISION_MODEL
    return value.strip()


def _local_vlm_visual_qc_prompt(
    *,
    policy: dict[str, Any],
    extraction_result: dict[str, Any],
    qc_context: dict[str, Any],
) -> str:
    task = _safe_short_json(policy.get("task") or policy.get("task_description") or "the requested task")
    label_rule = _safe_short_json(policy.get("label_rule", {}), limit=1200)
    site_mapping = _safe_short_json(policy.get("site_label_mapping", {}), limit=1600)
    extraction_orientation = _safe_short_json(extraction_result.get("label_orientation", {}), limit=800)
    candidates = qc_context.get("transform_candidates", sorted(VISUAL_QC_TRANSFORMS))
    return render_client_prompt(
        "local_vlm_visual_qc",
        task=task,
        candidates=", ".join(str(item) for item in candidates),
        label_rule=label_rule,
        site_mapping=site_mapping,
        extraction_orientation=extraction_orientation,
    )


def _query_local_vlm_image(
    *,
    base_url: str,
    model: str,
    image_path: Path,
    prompt: str,
    api_key_env: str,
    max_tokens: int,
    log_path: Path,
    reference_image_path: Path | None = None,
    supplemental_image_paths: list[tuple[str, Path]] | None = None,
    transport_max_long_side: int = LOCAL_VLM_TRANSPORT_MAX_LONG_SIDE,
) -> tuple[str, dict[str, Any]]:
    for attempt in range(1, LOCAL_VLM_MAX_TRANSPORT_ATTEMPTS + 1):
        try:
            return _query_local_vlm_image_once(
                base_url=base_url,
                model=model,
                image_path=image_path,
                prompt=prompt,
                api_key_env=api_key_env,
                max_tokens=max_tokens,
                log_path=log_path,
                reference_image_path=reference_image_path,
                supplemental_image_paths=supplemental_image_paths,
                transport_max_long_side=transport_max_long_side,
            )
        except Exception as exc:
            if attempt >= LOCAL_VLM_MAX_TRANSPORT_ATTEMPTS or not _is_retryable_local_vlm_error(exc):
                raise
            time.sleep(LOCAL_VLM_RETRY_DELAY_SECONDS * attempt)
    raise AssertionError("local VLM transport retry loop exited unexpectedly")


def _is_retryable_local_vlm_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code == 429 or 500 <= exc.code < 600
    return isinstance(exc, (URLError, TimeoutError, ConnectionError))


def _local_vlm_transport_image(
    image_path: Path,
    source_bytes: bytes,
    *,
    max_long_side: int = LOCAL_VLM_TRANSPORT_MAX_LONG_SIDE,
) -> tuple[bytes, str]:
    source_mime_type = _local_vlm_mime_type(image_path)
    try:
        with Image.open(BytesIO(source_bytes)) as source:
            image = source.convert("RGB")
        if max(image.size) > max_long_side:
            image.thumbnail(
                (max_long_side, max_long_side),
                Image.Resampling.LANCZOS,
            )
        output = BytesIO()
        image.save(
            output,
            format="JPEG",
            quality=LOCAL_VLM_TRANSPORT_JPEG_QUALITY,
            optimize=True,
        )
        return output.getvalue(), "image/jpeg"
    except Exception:
        return source_bytes, source_mime_type


def _query_local_vlm_image_once(
    *,
    base_url: str,
    model: str,
    image_path: Path,
    prompt: str,
    api_key_env: str,
    max_tokens: int,
    log_path: Path,
    reference_image_path: Path | None = None,
    supplemental_image_paths: list[tuple[str, Path]] | None = None,
    transport_max_long_side: int = LOCAL_VLM_TRANSPORT_MAX_LONG_SIDE,
) -> tuple[str, dict[str, Any]]:
    started_at = _utc_now()
    start_time = time.monotonic()
    source_image_bytes = image_path.read_bytes()
    image_bytes, mime_type = _local_vlm_transport_image(
        image_path,
        source_image_bytes,
        max_long_side=transport_max_long_side,
    )
    source_image_mime_type = _local_vlm_mime_type(image_path)
    source_reference_bytes: bytes | None = None
    reference_bytes: bytes | None = None
    reference_mime_type: str | None = None
    reference_source_mime_type: str | None = None
    supplemental_images: list[dict[str, Any]] = []
    for label, supplemental_path in supplemental_image_paths or []:
        normalized_label = str(label).strip()
        if not normalized_label:
            raise ValueError("supplemental local VLM image labels must not be empty")
        source_bytes = supplemental_path.read_bytes()
        transport_bytes, transport_mime_type = _local_vlm_transport_image(
            supplemental_path,
            source_bytes,
            max_long_side=transport_max_long_side,
        )
        supplemental_images.append(
            {
                "label": normalized_label,
                "path": supplemental_path,
                "source_bytes": source_bytes,
                "source_mime_type": _local_vlm_mime_type(supplemental_path),
                "transport_bytes": transport_bytes,
                "transport_mime_type": transport_mime_type,
            }
        )
    if reference_image_path is not None:
        source_reference_bytes = reference_image_path.read_bytes()
        reference_bytes, reference_mime_type = _local_vlm_transport_image(
            reference_image_path,
            source_reference_bytes,
            max_long_side=transport_max_long_side,
        )
        reference_source_mime_type = _local_vlm_mime_type(reference_image_path)
    record: dict[str, Any] = {
        "schema_version": "fedready.local_vlm_call.v1",
        "started_at": started_at,
        "status": "started",
        "base_url": base_url,
        "model": model,
        "prompt": prompt,
        "prompt_digest": _digest_text(prompt),
        "prompt_length": len(prompt),
        "image_path": str(image_path),
        "image_digest": _digest_bytes(source_image_bytes),
        "image_size_bytes": len(source_image_bytes),
        "image_mime_type": source_image_mime_type,
        "transport_image_digest": _digest_bytes(image_bytes),
        "transport_image_size_bytes": len(image_bytes),
        "transport_image_mime_type": mime_type,
        "transport_max_long_side": transport_max_long_side,
        "max_tokens": max_tokens,
        "log_path": str(log_path),
    }
    if supplemental_images:
        record["supplemental_images"] = [
            {
                "label": item["label"],
                "image_path": str(item["path"]),
                "image_digest": _digest_bytes(item["source_bytes"]),
                "image_size_bytes": len(item["source_bytes"]),
                "image_mime_type": item["source_mime_type"],
                "transport_image_digest": _digest_bytes(item["transport_bytes"]),
                "transport_image_size_bytes": len(item["transport_bytes"]),
                "transport_image_mime_type": item["transport_mime_type"],
            }
            for item in supplemental_images
        ]
    if (
        reference_image_path is not None
        and source_reference_bytes is not None
        and reference_bytes is not None
        and reference_mime_type is not None
        and reference_source_mime_type is not None
    ):
        record.update(
            {
                "reference_image_path": str(reference_image_path),
                "reference_image_digest": _digest_bytes(source_reference_bytes),
                "reference_image_size_bytes": len(source_reference_bytes),
                "reference_image_mime_type": reference_source_mime_type,
                "transport_reference_image_digest": _digest_bytes(reference_bytes),
                "transport_reference_image_size_bytes": len(reference_bytes),
                "transport_reference_image_mime_type": reference_mime_type,
            }
        )
    try:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if reference_bytes is not None and reference_mime_type is not None:
            reference_encoded = base64.b64encode(reference_bytes).decode("ascii")
            content.extend(
                [
                    {"type": "text", "text": "Reference example for the requested task:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{reference_mime_type};base64,{reference_encoded}"},
                    },
                ]
            )
        for item in supplemental_images:
            supplemental_encoded = base64.b64encode(item["transport_bytes"]).decode("ascii")
            content.extend(
                [
                    {"type": "text", "text": f"{item['label']}:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{item['transport_mime_type']};base64,{supplemental_encoded}"},
                    },
                ]
            )
        if reference_bytes is not None or supplemental_images:
            content.append({"type": "text", "text": "Candidate preflight QC image to evaluate:"})
        encoded = base64.b64encode(image_bytes).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}})
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if api_key_env:
            api_key = os.environ.get(api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        url = base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"
        request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urlopen(request, timeout=300) as response:
            response_body = response.read().decode("utf-8")
            response_status = response.getcode()
        response_text = _extract_chat_completion_text(response_body)
        record.update(
            {
                "status": "completed",
                "completed_at": _utc_now(),
                "latency_seconds": round(time.monotonic() - start_time, 6),
                "response_status_code": response_status,
                "response_digest": _digest_text(response_body),
                "response_body": response_body,
            }
        )
        _safe_append_jsonl(log_path, record)
        return response_text, record
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "failed_at": _utc_now(),
                "latency_seconds": round(time.monotonic() - start_time, 6),
                "error": {"type": type(exc).__name__, "message": _short_error(exc)},
            }
        )
        _safe_append_jsonl(log_path, record)
        raise


def _extract_chat_completion_text(response_body: str) -> str:
    try:
        response_json = json.loads(response_body)
    except json.JSONDecodeError:
        return response_body
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return response_body
    first = choices[0]
    if not isinstance(first, dict):
        return response_body
    message = first.get("message")
    if not isinstance(message, dict):
        return response_body
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "\n".join(parts)
    return response_body


def _parse_local_vlm_visual_qc_response(response_text: str) -> tuple[str, str]:
    data = _loads_first_json_object(response_text)
    selected: Any = None
    reason: Any = None
    if isinstance(data, dict):
        selected = data.get("selected_transform") or data.get("transform") or data.get("selected")
        reason = data.get("reason") or data.get("visual_evidence") or data.get("evidence")
    normalized = _normalize_local_vlm_transform(selected)
    if normalized is None:
        normalized = _infer_transform_from_text(response_text) or "undecided"
    reason_text = (
        str(reason).strip() if isinstance(reason, str) and reason.strip() else _safe_reason_from_text(response_text)
    )
    return normalized, reason_text


def _local_vlm_qc_decision(
    *,
    client_id: str,
    output_template: dict[str, Any],
    selected_transform: str,
    reason: str,
    model: str,
    base_url: str,
    call_record: dict[str, Any],
) -> dict[str, Any]:
    selected_transform = selected_transform if selected_transform in VISUAL_QC_ALLOWED_SELECTIONS else "undecided"
    reviewed = selected_transform in VISUAL_QC_ALLOWED_SELECTIONS
    passed = selected_transform == "as_is"
    if selected_transform == "undecided":
        status = "needs_more_samples"
        recommendation = "review_more_visual_qc_samples_before_accepting_extraction"
    elif passed:
        status = "passed"
        recommendation = "accept_extraction_for_training"
    else:
        status = "failed"
        recommendation = f"rerun extraction or adapter with selected_transform={selected_transform}"
    counts = {selection: 0 for selection in sorted(VISUAL_QC_ALLOWED_SELECTIONS)}
    counts[selected_transform] = 1
    output = dict(output_template)
    output.update(
        {
            "schema_version": "fedready.extraction_visual_qc_decision.v1",
            "client_id": client_id,
            "status": status,
            "passed": passed,
            "reviewed": reviewed,
            "selected_transform": selected_transform,
            "reviewed_sample_count": 1 if reviewed else 0,
            "selected_transform_counts": counts,
            "consensus_required_count": 1,
            "consensus_reached": selected_transform != "undecided",
            "recommendation": recommendation,
            "reason": reason,
            "local_vlm_review": _local_vlm_review_metadata(model=model, base_url=base_url, call_record=call_record),
        }
    )
    if selected_transform == "undecided":
        _append_visual_qc_issue(output, "local VLM could not select a transform from the visual QC sheet")
    elif selected_transform != "as_is":
        _append_visual_qc_issue(output, f"local VLM selected non_as_is transform: {selected_transform}")
    return output


def _local_vlm_qc_failure(
    *,
    client_id: str,
    output_template: dict[str, Any],
    reason: str,
    model: str,
    base_url: str,
) -> dict[str, Any]:
    output = dict(output_template)
    output.update(
        {
            "schema_version": "fedready.extraction_visual_qc_decision.v1",
            "client_id": client_id,
            "status": "failed",
            "passed": False,
            "reviewed": False,
            "selected_transform": "undecided",
            "reviewed_sample_count": 0,
            "selected_transform_counts": {selection: 0 for selection in sorted(VISUAL_QC_ALLOWED_SELECTIONS)},
            "consensus_required_count": 1,
            "consensus_reached": False,
            "recommendation": "fix_local_vlm_visual_qc_before_training",
            "reason": reason,
            "local_vlm_review": {
                "schema_version": LOCAL_VLM_QC_SCHEMA,
                "backend": "local_vlm",
                "status": "failed",
                "model": model or None,
                "base_url": base_url,
            },
        }
    )
    _append_visual_qc_issue(output, reason)
    return output


def _local_vlm_review_metadata(*, model: str, base_url: str, call_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": LOCAL_VLM_QC_SCHEMA,
        "backend": "local_vlm",
        "status": call_record.get("status"),
        "model": model,
        "base_url": base_url,
        "image_digest": call_record.get("image_digest"),
        "prompt_digest": call_record.get("prompt_digest"),
        "response_digest": call_record.get("response_digest"),
        "image_size_bytes": call_record.get("image_size_bytes"),
        "latency_seconds": call_record.get("latency_seconds"),
        "call_logged_client_local": True,
    }


def _local_vlm_qc_image_path(artifact: dict[str, Any]) -> Path | None:
    value = artifact.get("candidate_sheet_path")
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.exists() or not path.is_file():
        return None
    return path.resolve()


def _local_vlm_qc_log_path(image_path: Path) -> Path:
    env_path = os.environ.get("FEDREADY_VISUAL_QC_LOG_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    return image_path.parent / "local_vlm_qc_calls.jsonl"


def _is_local_vlm_base_url(base_url: str) -> bool:
    try:
        parsed = urlsplit(base_url)
    except ValueError:
        return False
    return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}


def _local_vlm_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    raise ValueError(f"Unsupported image suffix for local VLM visual QC: {suffix or '<none>'}")


def _loads_first_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _normalize_local_vlm_transform(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "as_is": "as_is",
        "asis": "as_is",
        "original": "as_is",
        "none": "as_is",
        "horizontal_flip": "hflip",
        "h_flip": "hflip",
        "hflip": "hflip",
        "left_right_flip": "hflip",
        "vertical_flip": "vflip",
        "v_flip": "vflip",
        "vflip": "vflip",
        "top_bottom_flip": "vflip",
        "rotate_180": "rot180",
        "rotation_180": "rot180",
        "rot180": "rot180",
        "180": "rot180",
        "undecided": "undecided",
        "unknown": "undecided",
        "unclear": "undecided",
    }
    return aliases.get(normalized)


def _infer_transform_from_text(text: str) -> str | None:
    lowered = text.lower()
    matches = [transform for transform in VISUAL_QC_ALLOWED_SELECTIONS if transform in lowered]
    if len(matches) == 1:
        return matches[0]
    return None


def _safe_reason_from_text(text: str) -> str:
    compact = " ".join(text.strip().split())
    if not compact:
        return "local VLM returned an empty visual QC response"
    return compact[:800]


def _safe_short_json(value: Any, *, limit: int = 1000) -> str:
    try:
        text = json.dumps(value, sort_keys=True)
    except TypeError:
        text = str(value)
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text


def _env_positive_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _digest_text(text: str) -> str:
    return _digest_bytes(text.encode("utf-8"))


def _safe_append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    except OSError:
        return


def _short_error(exc: BaseException) -> str:
    message = _redact_local_paths_in_text(str(exc))
    if len(message) > 800:
        return message[:800] + "...<truncated>"
    return message
