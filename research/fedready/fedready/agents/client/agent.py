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

"""Client-local authorization, profile responses, adapter requests, and visual review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fedready.agents.client.visual_qc import (
    _adapter_preflight_visual_qc_decision,
    _aggregate_visual_qc_sample_outputs,
    _live_visual_qc_template,
    _normalize_visual_qc_output,
    _request_local_vlm_visual_qc,
    _visual_qc_artifacts,
)
from fedready.agents.common import (
    DEFAULT_LOCAL_ADAPTER_MAX_ATTEMPTS,
    AgentTurn,
    _copy_present_keys,
    _merge_output_template,
    _raise_if_template_passthrough,
    _summarize_generated_materializer_payload,
)
from fedready.agents.guardrail import (
    ACTION_CHECKS,
    GuardrailAgent,
    GuardrailDecision,
    _redact_local_paths_in_text,
    _require_configured_agent_backend,
    _short_error,
    require_allowed,
)
from fedready.prompts import render_client_prompt


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
