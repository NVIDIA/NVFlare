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

"""Server and client agents for FedReady request negotiation.

``ServerAgent`` and ``ClientAgent`` compose prompts, dispatch them to the Codex backend
and validate the returned output. Live runtime code contains only schema templates and
harness-owned summaries; visual-review references are selected from prepared local data
before execution.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fedready.agents.guardrail import (
    ACTION_CHECKS,
    GuardrailAgent,
    GuardrailDecision,
    _redact_local_paths_in_text,
    _require_configured_agent_backend,
    _short_error,
    require_allowed,
)
from fedready.agents.local_adapter import DEFAULT_LOCAL_ADAPTER_MAX_ATTEMPTS, task_example_context
from fedready.agents.training_reference import NVFLARE_EXAMPLE_SELECTION_ID, NVFLARE_EXAMPLE_SELECTION_PROMPT
from fedready.agents.visual_qc import (
    _adapter_preflight_visual_qc_decision,
    _aggregate_visual_qc_sample_outputs,
    _default_visual_qc_rule,
    _live_visual_qc_template,
    _normalize_visual_qc_output,
    _request_local_vlm_visual_qc,
    _visual_qc_artifacts,
    _visual_qc_required_for_training_result,
)
from fedready.data.contracts import available_contract_summaries, generated_data_contract_validation_errors
from fedready.data.qc import visual_qc_result_ready_for_training
from fedready.prompts import render_client_prompt, render_server_prompt, render_server_prompt_object

TRAINING_CODE_SPEC_OUTPUT_SCHEMA = "fedready.training_code_spec.v1"


DATA_MATERIALIZER_SPEC_OUTPUT_SCHEMA = "fedready.data_materializer_spec.v1"


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


class ServerAgent:
    """Server-side agent that expands user tasks and summarizes responses."""

    def __init__(self, agent_backend: Any) -> None:
        self.agent_backend = _require_configured_agent_backend(agent_backend)

    def compose_client_inquiry(self, *, task: dict[str, Any], client_ids: list[str]) -> AgentTurn:
        decision = require_allowed(ACTION_CHECKS["SERVER.DEFINE_PROFILE_REQUEST"])
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
        decision = require_allowed(ACTION_CHECKS["SERVER.DISPATCH_EXTRACTION_POLICY"])
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
        decision = require_allowed(ACTION_CHECKS["SERVER.AGGREGATE_EXTRACTION_RESULTS"])
        # Counts and readiness are verifier-owned facts, with no model decision here.
        output = _generic_extraction_round_summary(task=task, extraction_results=extraction_results)
        return AgentTurn(prompt="", output=output, guardrail=decision.as_policy())

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

        decision = require_allowed(ACTION_CHECKS["SERVER.IMPLEMENT_TRAINING_CODE"])
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

        decision = require_allowed(ACTION_CHECKS["SERVER.IMPLEMENT_TRAINING_CODE"])
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

        decision = require_allowed(ACTION_CHECKS["SERVER.IMPLEMENT_DATA_MATERIALIZER"])
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
                "task_examples": task_example_context(),
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
        decision = require_allowed(ACTION_CHECKS["SERVER.AGGREGATE_CLIENT_PROFILES"])
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
                "task_examples": task_example_context(),
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
        return output


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
            check=ACTION_CHECKS["CLIENT.REPORT_DATA_PROFILE"],
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
            check=ACTION_CHECKS["CLIENT.EXTRACT_DATA"],
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
            check=ACTION_CHECKS["CLIENT.REPORT_DATA_PROFILE"],
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
            check=ACTION_CHECKS["CLIENT.EXTRACT_DATA"],
            payload=response,
            counterpart="server_agent",
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
        decision = require_allowed(ACTION_CHECKS["CLIENT.IMPLEMENT_LOCAL_ADAPTER"])
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
            output = api_output
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

    def review_extraction_visual_qc(
        self,
        *,
        policy: dict[str, Any],
        extraction_result: dict[str, Any],
        qc_context: dict[str, Any],
    ) -> AgentTurn:
        decision = require_allowed(ACTION_CHECKS["CLIENT.VISUAL_QC_EXTRACTION"])
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
            api_output = _request_local_vlm_visual_qc(
                client_id=self.client_id,
                policy=policy,
                extraction_result=extraction_result,
                qc_context=qc_context,
                output_template=output,
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
            api_output = _request_local_vlm_visual_qc(
                client_id=self.client_id,
                policy=policy,
                extraction_result=extraction_result,
                qc_context=single_qc_context,
                output_template=single_output,
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


def _safe_local_adapter_revision_output(output: dict[str, Any]) -> dict[str, Any]:
    """Return a path-free prior-attempt summary for Codex revision."""

    return {
        "schema_version": output.get("schema_version"),
        "client_id": output.get("client_id"),
        "status": output.get("status"),
        "source_label_type": output.get("source_label_type"),
        "adapter_kind": output.get("adapter_kind"),
        "record_count": output.get("record_count"),
        "manifest_path_present": bool(output.get("manifest_path")),
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
    if not isinstance(output, dict):
        raise RuntimeError(f"CLIENT.IMPLEMENT_LOCAL_ADAPTER for {client_id} returned non-object output")
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
    normalized = dict(output)
    package_dir = normalized.get("package_dir")
    if isinstance(package_dir, str) and package_dir.strip():
        package_path = Path(package_dir).expanduser()
        if not package_path.is_absolute():
            package_path = Path(code_workspace) / package_path
        normalized["package_dir"] = str(package_path.resolve())
    return normalized


def _resolve_data_materializer_entry_path(*, entry_script: str, package_path: Path) -> Path | None:
    entry_path = Path(entry_script)
    if entry_path.is_absolute():
        return None
    resolved = (package_path / entry_path).resolve()
    return resolved if resolved.is_relative_to(package_path) and resolved.is_file() else None


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
        client_id: result for client_id, result in qc_required.items() if visual_qc_result_ready_for_training(result)
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
