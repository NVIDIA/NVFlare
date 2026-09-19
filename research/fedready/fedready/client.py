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

"""Client-side NVFlare executor for FedReady orchestration rounds."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fedready.agents import ClientAgent, _query_local_vlm_image
from fedready.agents.bridge import build_agent_backend
from fedready.agents.local_adapter import (
    DEFAULT_LOCAL_ADAPTER_MAX_ATTEMPTS,
    DEFAULT_LOCAL_ADAPTER_PREFLIGHT_RECORDS,
    AdapterRuntimeResult,
    ClientDataProvenanceSnapshot,
    capture_client_data_provenance,
    ensure_local_adapter_pipeline,
    sanitize_adapter_diagnostic,
    task_example_context_for_task,
)
from fedready.data.contracts import (
    available_contract_summaries,
    generated_contract_box_field,
    generated_contract_field_names,
    generated_contract_label_field,
    generated_contract_record_type,
    normalize_record_type,
    runtime_contract_for_record_type,
)
from fedready.data.extractor import (
    ExtractionConfig,
    _resolve_site_data_path,
    apply_automatic_orientation_repair,
    build_visual_qc_context,
    extract_site_dataset,
    persist_visual_qc_decision,
)
from fedready.data.parser import DataParserConfig, parse_site_dataset
from fedready.flare.channel import (
    CLIENT_RESPONSE_SCHEMA,
    EXTRACTION_RESPONSE_SCHEMA,
    GUARDRAIL_SCHEMA,
    HEADER_ROUND_ACTION,
    LOCAL_EXECUTION_SCHEMA,
    ROUND_EXTRACTION,
    ROUND_PROFILE,
    TASK_QUERY_TASK_NAME,
    FlareMessage,
)
from fedready.flare.channel import current_round_from_shareable as _current_round_from_shareable
from fedready.flare.channel import from_shareable
from fedready.flare.channel import set_fl_ctx_round as _set_fl_ctx_round
from fedready.flare.channel import to_shareable
from fedready.flare.channel import total_rounds_from_shareable as _total_rounds_from_shareable
from fedready.prompts import render_client_prompt, render_client_prompt_object
from fedready.utils.logging import FlowLogger

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal

LIVE_LOCAL_ADAPTER_MAX_ATTEMPTS = DEFAULT_LOCAL_ADAPTER_MAX_ATTEMPTS


class FedReadyTaskQueryExecutor(Executor):
    """Client-side FedReady executor for a bounded task-query job."""

    def __init__(
        self,
        site_meta_path: str,
        project_root: str | None = None,
        output_dir: str = "runs",
        min_count: int = 5,
        max_scan_files: int = 200_000,
        max_image_samples: int = 8,
        histogram_bins: int = 8,
        task_name: str = TASK_QUERY_TASK_NAME,
        total_rounds: int = 2,
        extraction_output_root: str = ExtractionConfig.output_root,
        extraction_output_name: str | None = None,
        extraction_max_samples: int | None = None,
        extraction_overwrite: bool = False,
        extraction_validation_fraction: float = ExtractionConfig.validation_fraction,
        agent_backend: str = "codex",
        agent_timeout_seconds: float = 3600.0,
        agent_poll_interval_seconds: float = 2.0,
    ) -> None:
        super().__init__()
        self.site_meta_path = site_meta_path
        self.project_root = project_root
        self.output_dir = output_dir
        self.min_count = min_count
        self.max_scan_files = max_scan_files
        self.max_image_samples = max_image_samples
        self.histogram_bins = histogram_bins
        self.task_name = task_name
        self.total_rounds = total_rounds
        self.extraction_output_root = extraction_output_root
        self.extraction_output_name = extraction_output_name
        self.extraction_max_samples = extraction_max_samples
        self.extraction_overwrite = extraction_overwrite
        self.extraction_validation_fraction = extraction_validation_fraction
        self.agent_backend = agent_backend
        self.agent_timeout_seconds = agent_timeout_seconds
        self.agent_poll_interval_seconds = agent_poll_interval_seconds

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name != self.task_name:
            return make_reply(ReturnCode.TASK_UNKNOWN)
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        try:
            request_message = from_shareable(shareable)
            current_round = _current_round_from_shareable(shareable)
        except ValueError as exc:
            self.log_exception(fl_ctx, f"FedReady client received malformed FLARE envelope: {exc}")
            return make_reply(ReturnCode.BAD_TASK_DATA)
        try:
            client_id = fl_ctx.get_identity_name() or request_message.site_id
            if not client_id:
                return make_reply(ReturnCode.BAD_PEER_CONTEXT)
            if request_message.site_id is not None and request_message.site_id != client_id:
                return make_reply(ReturnCode.BAD_PEER_CONTEXT)
            _set_fl_ctx_round(fl_ctx, current_round, self.total_rounds)
            if current_round == ROUND_PROFILE:
                response_message = self._execute_inquiry(client_id, request_message, fl_ctx)
            elif current_round == ROUND_EXTRACTION:
                response_message = self._execute_extraction(client_id, request_message, shareable, fl_ctx)
            else:
                return make_reply(ReturnCode.BAD_TASK_DATA)
            return to_shareable(response_message)
        except Exception as exc:  # noqa: BLE001 - executor must return a FLARE reply on local failures.
            self.log_exception(fl_ctx, f"FedReady client executor failed: {exc}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _execute_inquiry(self, client_id: str, request_message: FlareMessage, fl_ctx: FLContext) -> FlareMessage:
        inquiry = request_message.payload
        logger = FlowLogger(Path(self.output_dir) / request_message.session_id)
        client_agent = ClientAgent(
            client_id,
            agent_backend=_build_agent_backend(
                kind=self.agent_backend,
                run_dir=Path(self.output_dir) / request_message.session_id,
                session_id=request_message.session_id,
                timeout_seconds=self.agent_timeout_seconds,
                poll_interval_seconds=self.agent_poll_interval_seconds,
            ),
        )
        receive_prompt = render_client_prompt(
            "client_receive_inquiry",
            client_id=client_id,
            task_name=self.task_name,
            schema_ref=request_message.schema_ref,
        )

        logger.write_client_event(
            client_id=client_id,
            session_id=request_message.session_id,
            phase_id="profile",
            step_id=f"profile.client_receive.{client_id}",
            correlation_id=request_message.correlation_id,
            event_type="CLIENT_REQUEST_RECEIVED",
            payload={"inquiry": inquiry.get("message", {})},
            schema_ref=request_message.schema_ref,
            status="received",
            policy={
                "guardrail_checked": False,
                "decision": "pending",
                "allow_list_rule": None,
                "redactions": [],
            },
            next_step={"action": "check_allow_list", "reason": "client received server inquiry through NVFlare"},
            agent_prompt=receive_prompt,
            flare={"task_name": self.task_name, "task_id": request_message.task_id, "site_id": client_id},
        )

        guardrail = client_agent.authorize_inquiry(inquiry=inquiry)
        logger.write_client_event(
            client_id=client_id,
            session_id=request_message.session_id,
            phase_id="profile",
            step_id=f"profile.client_guardrail.{client_id}",
            correlation_id=request_message.correlation_id,
            event_type="CLIENT_GUARDRAIL_CHECKED",
            payload=guardrail.as_payload(),
            schema_ref=GUARDRAIL_SCHEMA,
            status="completed",
            policy=guardrail.as_policy(),
            next_step={
                "action": "run_local_parser" if guardrail.allowed else "return_denial_via_nvflare",
                "reason": guardrail.reason,
            },
            agent_prompt=receive_prompt,
            flare={"task_name": self.task_name, "task_id": request_message.task_id, "site_id": client_id},
        )
        if not guardrail.allowed:
            return FlareMessage(
                session_id=request_message.session_id,
                correlation_id=request_message.correlation_id,
                schema_ref=CLIENT_RESPONSE_SCHEMA,
                task_id=request_message.task_id,
                site_id=client_id,
                payload=_guardrail_denial_payload(
                    schema_version=CLIENT_RESPONSE_SCHEMA,
                    agent_role="client_agent",
                    decision=guardrail,
                    client_id=client_id,
                ),
            )

        parser_config = DataParserConfig(
            min_count=self.min_count,
            max_scan_files=self.max_scan_files,
            max_image_samples=self.max_image_samples,
            histogram_bins=self.histogram_bins,
        )
        execution_payload = {
            "tool": "fedready.data.parser.parse_site_dataset",
            "client_id": client_id,
            "site_meta": "site-meta.json",
            "parser_config": {
                "min_count": parser_config.min_count,
                "max_scan_files": parser_config.max_scan_files,
                "max_image_samples": parser_config.max_image_samples,
                "histogram_bins": parser_config.histogram_bins,
            },
        }
        logger.write_client_event(
            client_id=client_id,
            session_id=request_message.session_id,
            phase_id="profile",
            step_id=f"profile.client_local_exec_start.{client_id}",
            correlation_id=request_message.correlation_id,
            event_type="CLIENT_LOCAL_EXECUTION_STARTED",
            payload=execution_payload,
            schema_ref="FedReadyLocalExecution@v1",
            status="started",
            policy=guardrail.as_policy(),
            next_step={"action": "parse_site_dataset", "reason": "local aggregate dataset profile is needed"},
            flare={"task_name": self.task_name, "task_id": request_message.task_id, "site_id": client_id},
        )
        parsed_profile = parse_site_dataset(
            self.site_meta_path,
            client_id,
            project_root=self.project_root,
            config=parser_config,
        )
        logger.write_client_event(
            client_id=client_id,
            session_id=request_message.session_id,
            phase_id="profile",
            step_id=f"profile.client_local_exec_done.{client_id}",
            correlation_id=request_message.correlation_id,
            event_type="CLIENT_LOCAL_EXECUTION_COMPLETED",
            payload={
                **execution_payload,
                "result_schema": parsed_profile.get("schema_version"),
                "safe_profile_summary": {
                    "data_type": parsed_profile.get("data_type"),
                    "case_counts": parsed_profile.get("case_counts"),
                    "label_type": parsed_profile.get("labels", {}).get("label_type"),
                    "label_source": parsed_profile.get("labels", {}).get("label_source"),
                    "privacy": parsed_profile.get("privacy"),
                },
            },
            schema_ref="FedReadyLocalExecution@v1",
            status="completed",
            policy=guardrail.as_policy(),
            next_step={"action": "client_agent_interpret_profile", "reason": "parser returned safe aggregate metadata"},
            flare={"task_name": self.task_name, "task_id": request_message.task_id, "site_id": client_id},
        )

        client_turn = client_agent.answer_inquiry(inquiry=inquiry, parsed_profile=parsed_profile)
        outgoing_guardrail = client_agent.authorize_outgoing_profile_response(response=client_turn.output)
        logger.write_client_event(
            client_id=client_id,
            session_id=request_message.session_id,
            phase_id="profile",
            step_id=f"profile.client_outgoing_guardrail.{client_id}",
            correlation_id=request_message.correlation_id,
            event_type="CLIENT_OUTGOING_GUARDRAIL_CHECKED",
            payload=outgoing_guardrail.as_payload(),
            schema_ref=GUARDRAIL_SCHEMA,
            status="completed",
            policy=outgoing_guardrail.as_policy(),
            next_step={
                "action": "return_response_via_nvflare" if outgoing_guardrail.allowed else "return_denial_via_nvflare",
                "reason": outgoing_guardrail.reason,
            },
            agent_prompt=client_turn.prompt,
            flare={"task_name": self.task_name, "task_id": request_message.task_id, "site_id": client_id},
        )
        response_payload = (
            outgoing_guardrail.effective_payload(client_turn.output)
            if outgoing_guardrail.allowed
            else _guardrail_denial_payload(
                schema_version="fedready.client_response.v1",
                agent_role="client_guardrail_agent",
                decision=outgoing_guardrail,
                client_id=client_id,
            )
        )
        response_payload = _bind_payload_client_id(response_payload, client_id=client_id)
        logger.write_client_event(
            client_id=client_id,
            session_id=request_message.session_id,
            phase_id="profile",
            step_id=f"profile.client_output.{client_id}",
            correlation_id=request_message.correlation_id,
            event_type="CLIENT_AGENT_OUTPUT",
            payload=response_payload,
            schema_ref=CLIENT_RESPONSE_SCHEMA,
            status="completed" if outgoing_guardrail.allowed else "denied",
            policy=outgoing_guardrail.as_policy(),
            next_step={
                "action": "return_response_via_nvflare",
                "reason": (
                    "client response sanitized and passed outgoing guardrail"
                    if outgoing_guardrail.redaction_applied
                    else (
                        "client response passed outgoing guardrail"
                        if outgoing_guardrail.allowed
                        else outgoing_guardrail.reason
                    )
                ),
            },
            agent_prompt=client_turn.prompt,
            flare={"task_name": self.task_name, "task_id": request_message.task_id, "site_id": client_id},
        )
        return FlareMessage(
            session_id=request_message.session_id,
            correlation_id=request_message.correlation_id,
            schema_ref=CLIENT_RESPONSE_SCHEMA,
            task_id=request_message.task_id,
            site_id=client_id,
            payload=response_payload,
        )

    def _execute_extraction(
        self,
        client_id: str,
        request_message: FlareMessage,
        shareable: Shareable,
        fl_ctx: FLContext,
    ) -> FlareMessage:
        policy = request_message.payload
        logger = FlowLogger(Path(self.output_dir) / request_message.session_id)
        client_agent = ClientAgent(
            client_id,
            agent_backend=_build_agent_backend(
                kind=self.agent_backend,
                run_dir=Path(self.output_dir) / request_message.session_id,
                session_id=request_message.session_id,
                timeout_seconds=self.agent_timeout_seconds,
                poll_interval_seconds=self.agent_poll_interval_seconds,
            ),
        )
        current_round = _current_round_from_shareable(shareable)
        receive_prompt = render_client_prompt(
            "client_receive_extraction",
            client_id=client_id,
            task_name=self.task_name,
            current_round=current_round,
            schema_ref=request_message.schema_ref,
        )

        logger.write_client_event(
            client_id=client_id,
            session_id=request_message.session_id,
            phase_id="harmonization",
            step_id=f"harmonization.client_receive.{client_id}",
            correlation_id=request_message.correlation_id,
            event_type="CLIENT_EXTRACTION_REQUEST_RECEIVED",
            payload={
                "policy": _safe_policy_summary(policy),
                "current_round": current_round,
                "total_rounds": _total_rounds_from_shareable(shareable, self.total_rounds),
            },
            schema_ref=request_message.schema_ref,
            status="received",
            policy={
                "guardrail_checked": False,
                "decision": "pending",
                "allow_list_rule": None,
                "redactions": [],
            },
            next_step={"action": "check_allow_list", "reason": "client received round-2 policy through NVFlare"},
            agent_prompt=receive_prompt,
            flare=_flare_round_context(self.task_name, request_message, shareable, client_id),
        )

        guardrail = client_agent.authorize_extraction(policy=policy)
        logger.write_client_event(
            client_id=client_id,
            session_id=request_message.session_id,
            phase_id="harmonization",
            step_id=f"harmonization.client_guardrail.{client_id}",
            correlation_id=request_message.correlation_id,
            event_type="CLIENT_EXTRACTION_GUARDRAIL_CHECKED",
            payload=guardrail.as_payload(),
            schema_ref=GUARDRAIL_SCHEMA,
            status="completed",
            policy=guardrail.as_policy(),
            next_step={
                "action": "run_local_extractor" if guardrail.allowed else "return_denial_via_nvflare",
                "reason": guardrail.reason,
            },
            agent_prompt=receive_prompt,
            flare=_flare_round_context(self.task_name, request_message, shareable, client_id),
        )
        if not guardrail.allowed:
            return FlareMessage(
                session_id=request_message.session_id,
                correlation_id=request_message.correlation_id,
                schema_ref=EXTRACTION_RESPONSE_SCHEMA,
                task_id=request_message.task_id,
                site_id=client_id,
                payload=_guardrail_denial_payload(
                    schema_version=EXTRACTION_RESPONSE_SCHEMA,
                    agent_role="client_agent",
                    decision=guardrail,
                    client_id=client_id,
                ),
            )

        config_payload = policy.get("extraction_config", {})
        extractor_config = ExtractionConfig(
            output_root=str(config_payload.get("output_root") or self.extraction_output_root),
            output_name=config_payload.get("output_name") or self.extraction_output_name,
            max_samples=config_payload.get("max_samples", self.extraction_max_samples),
            max_scan_files=self.max_scan_files,
            overwrite=bool(config_payload.get("overwrite", self.extraction_overwrite)),
            validation_fraction=float(config_payload.get("validation_fraction", self.extraction_validation_fraction)),
        )
        execution_payload = {
            "tool": "fedready.data.extractor.extract_site_dataset",
            "client_id": client_id,
            "site_meta": "site-meta.json",
            "current_round": current_round,
            "extraction_config": {
                "output_root": extractor_config.output_root,
                "output_name": extractor_config.output_name,
                "max_samples": extractor_config.max_samples,
                "max_scan_files": extractor_config.max_scan_files,
                "overwrite": extractor_config.overwrite,
                "validation_fraction": extractor_config.validation_fraction,
            },
            "data_binding": "client_agent_local_manifest",
            "policy_digest": policy.get("strategy_digest"),
        }
        logger.write_client_event(
            client_id=client_id,
            session_id=request_message.session_id,
            phase_id="harmonization",
            step_id=f"harmonization.client_local_exec_start.{client_id}",
            correlation_id=request_message.correlation_id,
            event_type="CLIENT_EXTRACTION_EXECUTION_STARTED",
            payload=execution_payload,
            schema_ref=LOCAL_EXECUTION_SCHEMA,
            status="started",
            policy=guardrail.as_policy(),
            next_step={"action": "extract_site_dataset", "reason": "site-specific policy passed allow-list check"},
            flare=_flare_round_context(self.task_name, request_message, shareable, client_id),
        )

        project_root = (
            Path(self.project_root).resolve()
            if self.project_root is not None
            else Path(self.site_meta_path).resolve().parent.parent
        )
        local_data_path = _resolve_site_data_path(
            self.site_meta_path,
            client_id,
            project_root=project_root,
        ).resolve()
        result = _client_local_datalist_request(client_id=client_id, policy=policy)
        if _needs_local_adapter(result):
            adapter_workspace = (
                Path(self.output_dir)
                / request_message.session_id
                / "sites"
                / client_id
                / "local_adapters"
                / str(request_message.task_id)
            )
            adapter_workspace.mkdir(parents=True, exist_ok=True)
            provenance_snapshot = capture_client_data_provenance(local_data_path)
            adapter_manifest_contract = render_client_prompt_object("local_adapter_manifest_contract_policy")
            if not isinstance(adapter_manifest_contract, dict):
                raise TypeError("local_adapter_manifest_contract_policy prompt entry must be an object")
            adapter_manifest_contract = dict(adapter_manifest_contract)
            adapter_manifest_contract["built_in_contracts"] = available_contract_summaries()
            adapter_manifest_contract["server_generated_contract"] = (
                policy.get("generated_data_contract")
                if isinstance(policy.get("generated_data_contract"), dict)
                else None
            )
            adapter_context = {
                "schema_version": "fedready.local_adapter_request_context.v1",
                "client_id": client_id,
                "task": policy.get("task"),
                "source_label_type": (
                    result.get("agent_action_required", {}).get("source_label_type")
                    if isinstance(result.get("agent_action_required"), dict)
                    else policy.get("site_label_mapping", {}).get("source_label_type")
                ),
                "local_data_path": str(local_data_path),
                "adapter_workspace": str(adapter_workspace.resolve()),
                "adapter_manifest_contract": adapter_manifest_contract,
                "task_examples": task_example_context_for_task(
                    policy.get("task") if isinstance(policy.get("task"), str) else None
                ),
                "execution_limits": {
                    "max_records": extractor_config.max_samples,
                    "min_records": self.min_count,
                    "preflight_records": DEFAULT_LOCAL_ADAPTER_PREFLIGHT_RECORDS,
                    "max_samples": extractor_config.max_samples,
                    "validation_fraction": extractor_config.validation_fraction,
                    "split_seed": extractor_config.split_seed,
                    "full_dataset_processing": "deferred_to_extractor_data_dir",
                    "site_conversion_pattern": "one_pattern_per_site_unless_local_metadata_declares_multiple",
                },
                "max_records": extractor_config.max_samples,
                "min_records": self.min_count,
                "server_response_redaction": {
                    "do_not_return_to_server": [
                        "local_data_path",
                        "adapter_workspace",
                        "manifest_path",
                        "script_path",
                        "source filenames",
                        "sample ids",
                        "raw data",
                    ]
                },
            }
            adapter_attempts = LIVE_LOCAL_ADAPTER_MAX_ATTEMPTS
            adapter_feedback: dict[str, Any] | None = None
            previous_adapter_output: dict[str, Any] | None = None
            adapter_failure: dict[str, Any] | None = None
            for adapter_attempt in range(1, adapter_attempts + 1):
                try:
                    adapter_turn = client_agent.implement_local_adapter(
                        policy=policy,
                        extraction_result=result,
                        adapter_context=adapter_context,
                        previous_output=previous_adapter_output,
                        validation_feedback=adapter_feedback,
                    )
                except Exception as exc:  # noqa: BLE001 - report bounded agent exhaustion through FLARE.
                    adapter_failure = _client_local_adapter_failure_result(
                        client_id=client_id,
                        policy=policy,
                        reason_code="CLIENT_LOCAL_ADAPTER_AGENT_FAILED",
                        reason=(
                            "Client-local adapter agent exhausted its bounded attempts without producing "
                            "a validated response."
                        ),
                        backend_kind=self.agent_backend,
                        error_type=type(exc).__name__,
                    )
                    result = adapter_failure
                    break
                logger.write_client_event(
                    client_id=client_id,
                    session_id=request_message.session_id,
                    phase_id="harmonization",
                    step_id=f"harmonization.client_local_adapter.{client_id}.attempt{adapter_attempt}",
                    correlation_id=request_message.correlation_id,
                    event_type="CLIENT_LOCAL_ADAPTER_OUTPUT",
                    payload={
                        **_safe_adapter_output_summary(adapter_turn.output),
                        "attempt": adapter_attempt,
                        "max_attempts": adapter_attempts,
                        "validation_feedback_applied": adapter_feedback is not None,
                    },
                    schema_ref="FedReadyLocalAdapterSpec@v1",
                    status="completed",
                    policy=adapter_turn.guardrail or guardrail.as_policy(),
                    next_step={
                        "action": (
                            "retry_extraction_with_local_adapter"
                            if adapter_turn.output.get("status") in {"implemented", "unfeasible"}
                            else "return_capability_gap_via_nvflare"
                        ),
                        "reason": "client-local adapter agent responded to extractor capability gap",
                    },
                    agent_prompt=adapter_turn.prompt,
                    flare=_flare_round_context(self.task_name, request_message, shareable, client_id),
                )
                adapter_output = dict(adapter_turn.output)
                if adapter_output.get("status") in {"implemented", "unfeasible"}:
                    runtime_error: str | None = None
                    if adapter_output.get("status") == "implemented":
                        runtime_result = _run_shared_local_adapter_pipeline(
                            adapter_workspace=adapter_workspace,
                            local_data_path=local_data_path,
                            task=str(policy.get("task") or "the requested task"),
                            max_records=extractor_config.max_samples,
                            min_records=self.min_count,
                            provenance_snapshot=provenance_snapshot,
                            data_contract=(
                                policy.get("generated_data_contract")
                                if isinstance(policy.get("generated_data_contract"), dict)
                                else None
                            ),
                        )
                        logger.write_client_event(
                            client_id=client_id,
                            session_id=request_message.session_id,
                            phase_id="harmonization",
                            step_id=(f"harmonization.client_local_adapter.{client_id}." f"runtime{adapter_attempt}"),
                            correlation_id=request_message.correlation_id,
                            event_type="CLIENT_LOCAL_ADAPTER_RUNTIME_VALIDATION",
                            payload={
                                "stage": runtime_result.stage,
                                "status": runtime_result.status,
                                "record_count": runtime_result.record_count,
                                "diagnostic": runtime_result.diagnostic,
                                "attempt": adapter_attempt,
                                "max_attempts": adapter_attempts,
                                "backend": self.agent_backend,
                            },
                            schema_ref="FedReadyLocalAdapterRuntimeState@v1",
                            status=runtime_result.status,
                            policy=adapter_turn.guardrail or guardrail.as_policy(),
                            next_step={
                                "action": (
                                    "defer_full_adapter_to_extractor"
                                    if runtime_result.passed
                                    else (
                                        "return_failure_via_nvflare"
                                        if runtime_result.infrastructure_failure
                                        else "revise_client_local_adapter"
                                    )
                                ),
                                "reason": "backend-neutral client-local runtime verdict",
                            },
                            agent_prompt=adapter_turn.prompt,
                            flare=_flare_round_context(
                                self.task_name,
                                request_message,
                                shareable,
                                client_id,
                            ),
                        )
                        if runtime_result.infrastructure_failure:
                            adapter_failure = _client_local_adapter_failure_result(
                                client_id=client_id,
                                policy=policy,
                                reason_code="CLIENT_LOCAL_VLM_UNAVAILABLE",
                                reason=("Required client-local VLM service did not complete " "adapter preflight."),
                                backend_kind=self.agent_backend,
                                error_type="LocalVLMUnavailable",
                                bounded_attempts_exhausted=False,
                            )
                            result = adapter_failure
                            break
                        if runtime_result.passed:
                            adapter_output.update(
                                {
                                    "script_path": str(adapter_workspace / "adapter.py"),
                                    "manifest_path": str(adapter_workspace / "adapter_preflight_manifest.json"),
                                    "preflight_record_count": runtime_result.record_count,
                                    "record_count": runtime_result.record_count,
                                    "reason": _local_adapter_runtime_success_reason(runtime_result),
                                    "runtime_validation": {
                                        "schema_version": "fedready.local_adapter_runtime_attestation.v1",
                                        "status": "passed",
                                        "adapter_sha256": runtime_result.adapter_sha256,
                                        "visual_qc_owner": (
                                            "generated_materializer"
                                            if "deferred_to_generated_materializer_visual_qc"
                                            in runtime_result.issue_codes
                                            else "adapter_preflight_local_guardrail"
                                        ),
                                        "stages": [
                                            "representative_sample_preflight",
                                            "representative_raw_image_review",
                                            "representative_sample_visual_qc",
                                        ],
                                        "full_dataset_deferred_to_extractor": True,
                                        "min_records": self.min_count,
                                        "max_records": extractor_config.max_samples,
                                        "harness_owned": True,
                                    },
                                }
                            )
                        else:
                            runtime_error = f"{runtime_result.stage} failed: {runtime_result.diagnostic}"
                    if runtime_error is None:
                        result = extract_site_dataset(
                            self.site_meta_path,
                            client_id,
                            policy=policy,
                            project_root=self.project_root,
                            config=extractor_config,
                            local_adapter=adapter_output,
                            local_adapter_provenance_snapshot=provenance_snapshot,
                        )
                        if _generated_materializer_owned_failure(result):
                            break
                        validation_error = _adapter_validation_error(
                            client_id=client_id,
                            adapter_output=adapter_output,
                            result=result,
                        )
                    else:
                        validation_error = runtime_error
                        result = {
                            **result,
                            "local_adapter": _safe_adapter_output_summary(adapter_output),
                            "local_adapter_runtime": {
                                "status": "failed",
                                "stage": runtime_result.stage,
                                "diagnostic": runtime_result.diagnostic,
                            },
                        }
                    if validation_error and adapter_output.get("status") == "implemented":
                        if adapter_attempt < adapter_attempts:
                            adapter_feedback = _local_adapter_validation_feedback(
                                client_id=client_id,
                                error=validation_error,
                                adapter_output=adapter_output,
                                result=result,
                                private_roots=(local_data_path, adapter_workspace),
                                data_contract=(
                                    policy.get("generated_data_contract")
                                    if isinstance(policy.get("generated_data_contract"), dict)
                                    else None
                                ),
                            )
                            previous_adapter_output = adapter_output
                            logger.write_client_event(
                                client_id=client_id,
                                session_id=request_message.session_id,
                                phase_id="harmonization",
                                step_id=f"harmonization.client_local_adapter.{client_id}.feedback{adapter_attempt}",
                                correlation_id=request_message.correlation_id,
                                event_type="CLIENT_LOCAL_ADAPTER_VALIDATION_FEEDBACK",
                                payload=adapter_feedback,
                                schema_ref="FedReadyLocalAdapterFeedback@v1",
                                status="needs_revision",
                                policy=adapter_turn.guardrail or guardrail.as_policy(),
                                next_step={
                                    "action": "revise_client_local_adapter",
                                    "reason": "local extractor rejected the adapter output",
                                },
                                agent_prompt=adapter_turn.prompt,
                                flare=_flare_round_context(self.task_name, request_message, shareable, client_id),
                            )
                            continue
                        adapter_failure = _client_local_adapter_failure_result(
                            client_id=client_id,
                            policy=policy,
                            reason_code="CLIENT_LOCAL_ADAPTER_VALIDATION_FAILED",
                            reason=(
                                "Client-local adapter output failed local extraction validation after bounded "
                                "revision attempts."
                            ),
                            backend_kind=self.agent_backend,
                            error_type="AdapterValidationError",
                        )
                        result = adapter_failure
                else:
                    if adapter_turn.output.get("status") == "failed":
                        adapter_failure = _client_local_adapter_failure_result(
                            client_id=client_id,
                            policy=policy,
                            reason_code="CLIENT_LOCAL_ADAPTER_REPORTED_FAILURE",
                            reason=(
                                "Client-local adapter agent reported failure and did not produce a validated "
                                "local datalist."
                            ),
                            backend_kind=self.agent_backend,
                            error_type="AgentReportedFailure",
                        )
                        result = adapter_failure
                    else:
                        result = {**result, "local_adapter": _safe_adapter_output_summary(adapter_turn.output)}
                break
            if adapter_failure is not None:
                logger.write_client_event(
                    client_id=client_id,
                    session_id=request_message.session_id,
                    phase_id="harmonization",
                    step_id=f"harmonization.client_local_adapter.{client_id}.failed",
                    correlation_id=request_message.correlation_id,
                    event_type="CLIENT_LOCAL_ADAPTER_FAILED",
                    payload=adapter_failure,
                    schema_ref="FedReadyClientExtractionResponse@v1",
                    status="failed",
                    policy=guardrail.as_policy(),
                    next_step={
                        "action": "return_failure_via_nvflare",
                        "reason": "bounded client-local adapter work did not produce a validated datalist",
                    },
                    flare=_flare_round_context(self.task_name, request_message, shareable, client_id),
                )
        # Extractor diagnostics are revision-only client data and must not reach reporting agents.
        result.pop("_client_local_diagnostic", None)
        qc_context = build_visual_qc_context(
            extraction_result=result,
            project_root=project_root,
        )
        local_adapter_summary = result.get("local_adapter") if isinstance(result.get("local_adapter"), dict) else {}
        if local_adapter_summary.get("preflight_visual_review_passed") is True:
            qc_context = {
                **qc_context,
                "review_satisfied": True,
                "visual_qc_owner": "adapter_preflight_local_guardrail",
            }
        visual_qc_turn = client_agent.review_extraction_visual_qc(
            policy=policy,
            extraction_result=result,
            qc_context=qc_context,
        )
        if result.get("data") == "extracted":
            result, visual_qc_output = apply_automatic_orientation_repair(
                extraction_result=result,
                decision=visual_qc_turn.output,
                project_root=project_root,
            )
            persist_visual_qc_decision(
                extraction_result=result,
                decision=visual_qc_output,
                project_root=project_root,
            )
            result = {**result, "visual_qc": visual_qc_output}
            logger.write_client_event(
                client_id=client_id,
                session_id=request_message.session_id,
                phase_id="harmonization",
                step_id=f"harmonization.client_visual_qc.{client_id}",
                correlation_id=request_message.correlation_id,
                event_type="CLIENT_ORIENTATION_QC_DECISION",
                payload=visual_qc_output,
                schema_ref="FedReadyExtractionQCDecision@v1",
                status="completed",
                policy=visual_qc_turn.guardrail or guardrail.as_policy(),
                next_step={
                    "action": (
                        "complete_local_extraction"
                        if visual_qc_output.get("status") in {"passed", "not_performed", "needs_agent_visual_review"}
                        else "return_qc_issue_via_nvflare"
                    ),
                    "reason": "client visual QC decision recorded for extracted samples",
                },
                agent_prompt=visual_qc_turn.prompt,
                flare=_flare_round_context(self.task_name, request_message, shareable, client_id),
            )
        result_summary = {
            "data": result.get("data"),
            "counts": result.get("counts"),
            "failed_pairs": result.get("failed_pairs"),
            "verification": result.get("verification"),
            "label_orientation": result.get("label_orientation"),
            "visual_qc": result.get("visual_qc"),
            "visual_qc_artifacts": result.get("visual_qc_artifacts"),
            "agent_action_required": result.get("agent_action_required"),
            "local_adapter": result.get("local_adapter"),
            "source_label_type": result.get("source_label_type"),
            "privacy": result.get("privacy"),
            "warnings": result.get("warnings"),
        }
        logger.write_client_event(
            client_id=client_id,
            session_id=request_message.session_id,
            phase_id="harmonization",
            step_id=f"harmonization.client_local_exec_done.{client_id}",
            correlation_id=request_message.correlation_id,
            event_type="CLIENT_EXTRACTION_EXECUTION_COMPLETED",
            payload={
                **execution_payload,
                "result_schema": result.get("schema_version"),
                "safe_result_summary": result_summary,
            },
            schema_ref=LOCAL_EXECUTION_SCHEMA,
            status="completed",
            policy=guardrail.as_policy(),
            next_step={"action": "client_agent_return_extraction_result", "reason": "extractor completed"},
            flare=_flare_round_context(self.task_name, request_message, shareable, client_id),
        )

        client_turn = client_agent.report_extraction_result(
            policy=policy,
            extraction_result=result,
            execution_summary={
                **execution_payload,
                "result_schema": result.get("schema_version"),
                "safe_result_summary": result_summary,
            },
        )
        extraction_response = _redact_server_visible_extraction_response(client_turn.output)
        outgoing_guardrail = client_agent.authorize_outgoing_extraction_response(response=extraction_response)
        logger.write_client_event(
            client_id=client_id,
            session_id=request_message.session_id,
            phase_id="harmonization",
            step_id=f"harmonization.client_outgoing_guardrail.{client_id}",
            correlation_id=request_message.correlation_id,
            event_type="CLIENT_EXTRACTION_OUTGOING_GUARDRAIL_CHECKED",
            payload=outgoing_guardrail.as_payload(),
            schema_ref=GUARDRAIL_SCHEMA,
            status="completed",
            policy=outgoing_guardrail.as_policy(),
            next_step={
                "action": "return_response_via_nvflare" if outgoing_guardrail.allowed else "return_denial_via_nvflare",
                "reason": outgoing_guardrail.reason,
            },
            agent_prompt=client_turn.prompt,
            flare=_flare_round_context(self.task_name, request_message, shareable, client_id),
        )
        response_payload = (
            outgoing_guardrail.effective_payload(extraction_response)
            if outgoing_guardrail.allowed
            else _guardrail_denial_payload(
                schema_version="fedready.client_extraction_response.v1",
                agent_role="client_guardrail_agent",
                decision=outgoing_guardrail,
                client_id=client_id,
            )
        )
        response_payload = _bind_payload_client_id(response_payload, client_id=client_id)
        logger.write_client_event(
            client_id=client_id,
            session_id=request_message.session_id,
            phase_id="harmonization",
            step_id=f"harmonization.client_output.{client_id}",
            correlation_id=request_message.correlation_id,
            event_type="CLIENT_EXTRACTION_OUTPUT",
            payload=response_payload,
            schema_ref=EXTRACTION_RESPONSE_SCHEMA,
            status="completed" if outgoing_guardrail.allowed else "denied",
            policy=outgoing_guardrail.as_policy(),
            next_step={
                "action": "return_response_via_nvflare",
                "reason": (
                    "client extraction result sanitized and passed outgoing guardrail"
                    if outgoing_guardrail.redaction_applied
                    else (
                        "client extraction result passed outgoing guardrail"
                        if outgoing_guardrail.allowed
                        else outgoing_guardrail.reason
                    )
                ),
            },
            agent_prompt=client_turn.prompt,
            flare=_flare_round_context(self.task_name, request_message, shareable, client_id),
        )
        return FlareMessage(
            session_id=request_message.session_id,
            correlation_id=request_message.correlation_id,
            schema_ref=EXTRACTION_RESPONSE_SCHEMA,
            task_id=request_message.task_id,
            site_id=client_id,
            payload=response_payload,
        )


def _local_adapter_runtime_success_reason(runtime_result: AdapterRuntimeResult) -> str:
    if "deferred_to_generated_materializer_visual_qc" in runtime_result.issue_codes:
        return (
            "sample preflight and client-local raw-image review passed; label-specific visual "
            "QC is deferred to generated materializer-owned artifacts, and full local adapter "
            "execution is deferred to extractor data-dir materialization"
        )
    return (
        "sample preflight and client-local VLM review passed; full local adapter execution "
        "is deferred to extractor data-dir materialization"
    )


def _redact_server_visible_extraction_response(payload: dict[str, Any]) -> dict[str, Any]:
    redacted = json.loads(json.dumps(payload, default=str))
    extraction = redacted.get("extraction") if isinstance(redacted, dict) else None
    if isinstance(extraction, dict) and extraction.get("local_output_path_redacted") is True:
        extraction["output_root"] = "[redacted-output-root]"
    return redacted


def _guardrail_denial_payload(
    *,
    schema_version: str,
    agent_role: str,
    decision: Any,
    client_id: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": schema_version,
        "status": "denied",
        "agent_role": agent_role,
        "reason_code": decision.reason_code,
        "reason": decision.reason,
    }
    if client_id is not None:
        payload["client_id"] = client_id
    return payload


def _bind_payload_client_id(payload: dict[str, Any], *, client_id: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("client response payload must be a JSON object")
    existing = payload.get("client_id")
    if existing not in {None, client_id}:
        raise RuntimeError(f"client response payload client_id did not match {client_id}")
    bound = dict(payload)
    bound["client_id"] = client_id
    return bound


def _safe_policy_summary(policy: dict[str, Any]) -> dict[str, Any]:
    site_mapping = policy.get("site_label_mapping", {})
    return {
        "schema_version": policy.get("schema_version"),
        "client_id": policy.get("client_id"),
        "applicable": policy.get("applicable"),
        "strategy_digest": policy.get("strategy_digest"),
        "source_label_type": site_mapping.get("source_label_type") if isinstance(site_mapping, dict) else None,
        "matched_terms": site_mapping.get("matched_terms") if isinstance(site_mapping, dict) else None,
        "requires_site_local_adaptor": (
            site_mapping.get("requires_site_local_adaptor") if isinstance(site_mapping, dict) else None
        ),
        "local_adapter_rule": policy.get("local_adapter_rule"),
        "extraction_config": policy.get("extraction_config"),
    }


def _client_local_datalist_request(*, client_id: str, policy: dict[str, Any]) -> dict[str, Any]:
    site_mapping = policy.get("site_label_mapping")
    site_mapping = site_mapping if isinstance(site_mapping, dict) else {}
    source_label_type = str(site_mapping.get("source_label_type") or "unknown")
    return {
        "schema_version": "fedready.local_extraction_result.v1",
        "client_id": client_id,
        "data": "local datalist required",
        "policy_digest": policy.get("strategy_digest"),
        "source_label_type": source_label_type,
        "screening": {
            "schema_version": "fedready.local_label_screening.v1",
            "status": "client_local_preparation_required",
            "reason_code": "CLIENT_LOCAL_DATALIST_REQUIRED",
            "reason": render_client_prompt("local_datalist_required_reason"),
            "safe_to_share": True,
        },
        "agent_action_required": {
            "required": True,
            "kind": "client_local_label_adapter",
            "source_label_type": source_label_type,
            "safe_to_share": True,
        },
        "privacy": {
            "safe_to_share": True,
            "redacted": ["local_paths", "filenames", "sample_ids", "raw_data"],
        },
    }


def _client_local_adapter_failure_result(
    *,
    client_id: str,
    policy: dict[str, Any],
    reason_code: str,
    reason: str,
    backend_kind: str,
    error_type: str,
    bounded_attempts_exhausted: bool = True,
) -> dict[str, Any]:
    return {
        "schema_version": "fedready.local_extraction_result.v1",
        "client_id": client_id,
        "data": "failed",
        "status": "failed",
        "policy_digest": policy.get("strategy_digest"),
        "counts": {
            "total": 0,
            "by_split": {"train": 0, "validation": 0, "test": 0},
        },
        "screening": {
            "schema_version": "fedready.local_label_screening.v1",
            "status": "failed",
            "reason_code": reason_code,
            "reason": reason,
            "safe_to_share": True,
        },
        "verification": {
            "schema_version": "fedready.local_extraction_verification.v1",
            "passed": False,
            "extracted_count": 0,
            "reason_code": reason_code,
        },
        "local_adapter": {
            "schema_version": "fedready.local_adapter_spec.v1",
            "client_id": client_id,
            "status": "failed",
            "record_count": 0,
            "reason": reason,
        },
        "failure": {
            "schema_version": "fedready.client_local_failure.v1",
            "stage": "client_local_adapter",
            "reason_code": reason_code,
            "reason": reason,
            "backend": backend_kind,
            "error_type": error_type,
            "bounded_attempts_exhausted": bounded_attempts_exhausted,
            "safe_to_share": True,
        },
        "visual_qc": {
            "schema_version": "fedready.extraction_visual_qc_decision.v1",
            "status": "not_performed",
            "passed": False,
            "reviewed": False,
            "reason": "Visual QC was skipped because client-local data preparation failed.",
            "recommendation": "exclude_client_from_training",
            "safe_to_share": True,
        },
        "privacy": {
            "safe_to_share": True,
            "redacted": [
                "local_paths",
                "filenames",
                "sample_ids",
                "raw_data",
                "agent_exception_message",
            ],
        },
        "warnings": [reason],
    }


def _needs_local_adapter(result: dict[str, Any]) -> bool:
    action = result.get("agent_action_required")
    return (
        isinstance(action, dict)
        and action.get("required") is True
        and action.get("kind") == "client_local_label_adapter"
    )


def _adapter_extraction_verified(result: dict[str, Any]) -> bool:
    verification = result.get("verification") if isinstance(result.get("verification"), dict) else {}
    counts = result.get("counts") if isinstance(result.get("counts"), dict) else {}
    return (
        result.get("data") == "extracted" and verification.get("passed") is True and int(counts.get("total") or 0) > 0
    )


def _local_adapter_runtime_timeout_seconds() -> int:
    raw = os.environ.get("FEDREADY_LOCAL_ADAPTER_RUNTIME_TIMEOUT_SECONDS", "3600")
    try:
        value = int(raw)
    except ValueError:
        value = 3600
    return max(60, value)


def _run_shared_local_adapter_pipeline(
    *,
    adapter_workspace: Path,
    local_data_path: Path,
    task: str,
    max_records: int | None,
    min_records: int,
    provenance_snapshot: ClientDataProvenanceSnapshot,
    data_contract: dict[str, Any] | None = None,
) -> AdapterRuntimeResult:
    """Apply the same harness-owned acceptance pipeline to every live backend."""

    model = os.environ.get("FEDREADY_VISION_AGENT_MODEL", "Qwen/Qwen3-VL-8B-Instruct").strip()
    base_url = os.environ.get(
        "FEDREADY_VISION_AGENT_API_BASE_URL",
        "http://127.0.0.1:8001/v1",
    ).strip()
    api_key_env = os.environ.get(
        "FEDREADY_VISION_AGENT_API_KEY_ENV",
        "FEDREADY_LOCAL_VISION_API_KEY",
    ).strip()
    try:
        max_tokens = int(os.environ.get("FEDREADY_VISUAL_QC_VLM_MAX_TOKENS", "512"))
    except ValueError:
        max_tokens = 512
    return ensure_local_adapter_pipeline(
        workspace=adapter_workspace,
        local_data_path=local_data_path,
        task=task,
        max_records=max_records,
        min_records=min_records,
        timeout_seconds=_local_adapter_runtime_timeout_seconds(),
        local_vlm_model=model,
        local_vlm_base_url=base_url,
        local_vlm_api_key_env=api_key_env,
        local_vlm_max_tokens=max(64, min(max_tokens, 1024)),
        query_image=_query_local_vlm_image,
        provenance_snapshot=provenance_snapshot,
        data_contract=data_contract,
    )


def _local_adapter_full_run_deferred_output(adapter_output: dict[str, Any]) -> bool:
    runtime = adapter_output.get("runtime_validation") if isinstance(adapter_output, dict) else None
    return isinstance(runtime, dict) and runtime.get("full_dataset_deferred_to_extractor") is True


def _adapter_record_count_consistent(adapter_output: dict[str, Any], result: dict[str, Any]) -> bool:
    declared = adapter_output.get("record_count")
    if declared is None:
        return True
    declared_count = _coerce_adapter_count(declared)
    if declared_count is None:
        return False

    full_run_deferred = _local_adapter_full_run_deferred_output(adapter_output)
    local_adapter = result.get("local_adapter") if isinstance(result.get("local_adapter"), dict) else {}
    manifest_count = _coerce_adapter_count(local_adapter.get("record_count"))
    if manifest_count is not None:
        if full_run_deferred:
            return manifest_count > 0
        return declared_count == manifest_count

    counts = result.get("counts") if isinstance(result.get("counts"), dict) else {}
    extracted_count = _coerce_adapter_count(counts.get("total") or 0)
    if extracted_count is None:
        return False
    if full_run_deferred:
        return extracted_count > 0
    verification = result.get("verification") if isinstance(result.get("verification"), dict) else {}
    sample_limited = bool(verification.get("sample_limited")) or verification.get("matches_all_valid_cases") is False
    if sample_limited:
        return declared_count >= extracted_count
    return declared_count == extracted_count


def _adapter_validation_error(
    *,
    client_id: str,
    adapter_output: dict[str, Any],
    result: dict[str, Any],
) -> str | None:
    if adapter_output.get("status") != "implemented":
        return None
    if not _adapter_extraction_verified(result):
        return (
            "Client-local adapter reported implemented but did not produce verified extracted data: "
            f"client_id={client_id}, data={result.get('data')!r}, "
            f"screening={(result.get('screening') or {}).get('reason_code')!r}, "
            f"warnings={result.get('warnings')!r}"
        )
    if not _adapter_record_count_consistent(adapter_output, result):
        return (
            "Client-local adapter record_count was inconsistent with the adapter manifest or extracted sample count: "
            f"client_id={client_id}, adapter_record_count={adapter_output.get('record_count')!r}, "
            f"extracted_count={(result.get('counts') or {}).get('total')!r}"
        )
    return None


def _generated_materializer_owned_failure(result: dict[str, Any]) -> bool:
    screening = result.get("screening") if isinstance(result.get("screening"), dict) else {}
    reason_code = screening.get("reason_code")
    return isinstance(reason_code, str) and (
        reason_code.startswith("GENERATED_MATERIALIZER_") or reason_code.startswith("SERVER_GENERATED_MATERIALIZER_")
    )


def _local_adapter_validation_feedback(
    *,
    client_id: str,
    error: str,
    adapter_output: dict[str, Any],
    result: dict[str, Any],
    private_roots: tuple[Path, ...],
    data_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    screening = result.get("screening") if isinstance(result.get("screening"), dict) else {}
    verification = result.get("verification") if isinstance(result.get("verification"), dict) else {}
    counts = result.get("counts") if isinstance(result.get("counts"), dict) else {}
    warnings = result.get("warnings") if isinstance(result.get("warnings"), list) else []
    safe_error = sanitize_adapter_diagnostic(error, private_roots=private_roots)
    safe_warnings = [sanitize_adapter_diagnostic(str(value), private_roots=private_roots) for value in warnings[:5]]
    client_local_diagnostic = result.get("_client_local_diagnostic")
    if isinstance(client_local_diagnostic, str):
        client_local_diagnostic = sanitize_adapter_diagnostic(
            client_local_diagnostic,
            private_roots=private_roots,
        )
    else:
        client_local_diagnostic = None
    manifest_diagnostic = _local_adapter_manifest_diagnostic(adapter_output, data_contract=data_contract)
    return {
        "schema_version": "fedready.local_adapter_feedback.v1",
        "client_id": client_id,
        "stage": "local_extraction_validation",
        "error": safe_error,
        "adapter_output_summary": _safe_adapter_output_summary(adapter_output),
        "extractor_result_summary": {
            "data": result.get("data"),
            "screening_status": screening.get("status"),
            "screening_reason_code": screening.get("reason_code"),
            "verification_passed": verification.get("passed"),
            "extracted_count": counts.get("total"),
            "warnings": safe_warnings,
            "client_local_diagnostic": client_local_diagnostic,
        },
        "local_manifest_diagnostic": manifest_diagnostic,
        "required_fix": _local_adapter_validation_required_fix(data_contract),
        "safe_to_share": True,
    }


def _local_adapter_manifest_diagnostic(
    adapter_output: dict[str, Any],
    *,
    data_contract: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    manifest_value = adapter_output.get("manifest_path") or adapter_output.get("adapter_manifest_path")
    if not isinstance(manifest_value, str) or not manifest_value.strip():
        return {"manifest_path_present": False}
    manifest_path = Path(manifest_value).expanduser()
    if not manifest_path.exists():
        return {"manifest_path_present": True, "manifest_path_readable": False}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"manifest_path_present": True, "manifest_path_readable": False, "manifest_json_valid": False}
    if not isinstance(manifest, dict):
        return {"manifest_path_present": True, "manifest_path_readable": True, "manifest_json_object": False}
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        return {"manifest_path_present": True, "manifest_path_readable": True, "records_present": False}

    base_dir = manifest_path.parent
    if generated_contract_record_type(data_contract) != "unknown":
        return _generated_contract_manifest_diagnostic(
            records=records,
            base_dir=base_dir,
            data_contract=data_contract,
        )

    record_type = normalize_record_type(manifest.get("record_type"))
    if record_type == "unknown":
        return {
            "manifest_path_present": True,
            "manifest_path_readable": True,
            "record_type_declared": False,
            "diagnosis": "Local adapter manifest must declare a contract record_type before feedback diagnostics can be applied.",
        }
    return _contract_manifest_diagnostic(records=records, base_dir=base_dir, record_type=record_type)


def _contract_manifest_diagnostic(
    *,
    records: list[Any],
    base_dir: Path,
    record_type: str,
) -> dict[str, Any]:
    runtime_contract = runtime_contract_for_record_type(record_type)
    contract = getattr(runtime_contract, "CONTRACT", None)
    required_fields = tuple(getattr(contract, "adapter_record_required_fields", ()) or ())
    if not required_fields:
        return {
            "manifest_path_present": True,
            "manifest_path_readable": True,
            "record_type": record_type,
            "contract_required_fields_available": False,
        }
    path_fields = {
        field
        for field in required_fields
        if field.endswith("_path") or field in {"image", "image_path", "label_source", "label_source_path"}
    }
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            return {
                "manifest_path_present": True,
                "manifest_path_readable": True,
                "record_type": record_type,
                "first_invalid_record_index": index,
                "record_json_object": False,
                "diagnosis": "Local adapter manifest records must be JSON objects.",
            }
        missing_fields = [
            field
            for field in required_fields
            if field not in record
            or record.get(field) is None
            or record.get(field) == ""
            or (isinstance(record.get(field), list) and not record.get(field))
        ]
        unreadable_path_fields: dict[str, dict[str, Any]] = {}
        for field in sorted(path_fields):
            candidate = _local_adapter_feedback_record_path(record.get(field), base_dir)
            exists = candidate.exists() if candidate is not None else False
            if not exists:
                unreadable_path_fields[field] = {
                    "exists": exists,
                    "shape": _local_adapter_feedback_path_shape(record.get(field)),
                }
        if not missing_fields and not unreadable_path_fields:
            continue
        return {
            "manifest_path_present": True,
            "manifest_path_readable": True,
            "record_type": record_type,
            "first_invalid_record_index": index,
            "required_fields": list(required_fields),
            "missing_or_empty_fields": missing_fields,
            "unreadable_path_fields": unreadable_path_fields,
            "diagnosis": (
                "At least one contract manifest record is missing a required field, "
                "has an empty required value, or points a required path field at an unreadable local file."
            ),
        }
    return {
        "manifest_path_present": True,
        "manifest_path_readable": True,
        "record_type": record_type,
        "first_invalid_record_index": None,
        "required_fields": list(required_fields),
    }


def _local_adapter_validation_required_fix(data_contract: dict[str, Any] | None) -> str:
    if generated_contract_record_type(data_contract) != "unknown":
        required_fields = sorted(generated_contract_field_names(data_contract)["required"])
        required_text = ", ".join(required_fields) if required_fields else "the generated-contract required fields"
        return (
            "Revise adapter.py in adapter_workspace, rerun the bounded preflight and local visual QC, and "
            "return status=implemented only when adapter_preflight_manifest.json contains readable "
            "representative path fields, "
            f"all generated-contract required fields ({required_text}), and valid record values. Full adapter "
            "execution must remain deferred to extractor data-dir materialization. If source images live inside "
            "an archive, extract or copy only selected preflight images into adapter_workspace and make the "
            "manifest reference filesystem-readable local copies. Do not fabricate labels, geometry, or "
            "provenance; return status=unfeasible if local evidence cannot satisfy the generated contract."
        )
    return render_client_prompt("local_adapter_validation_required_fix")


def _generated_contract_manifest_diagnostic(
    *,
    records: list[Any],
    base_dir: Path,
    data_contract: dict[str, Any] | None,
) -> dict[str, Any]:
    required_fields = sorted(generated_contract_field_names(data_contract)["required"])
    box_field = generated_contract_box_field(data_contract)
    label_field = generated_contract_label_field(data_contract)
    path_fields = {
        field
        for field in required_fields
        if field.endswith("_path") or field in {"image", "image_path", "label_source", "label_source_path"}
    }
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            return {
                "manifest_path_present": True,
                "manifest_path_readable": True,
                "first_invalid_record_index": index,
                "record_json_object": False,
                "diagnosis": "Generated-contract adapter records must be JSON objects.",
            }
        missing_fields = [
            field
            for field in required_fields
            if field not in record
            or record.get(field) is None
            or record.get(field) == ""
            or (isinstance(record.get(field), list) and not record.get(field))
        ]
        unreadable_path_fields: dict[str, dict[str, Any]] = {}
        for field in sorted(path_fields):
            candidate = _local_adapter_feedback_record_path(record.get(field), base_dir)
            exists = candidate.exists() if candidate is not None else False
            if not exists:
                unreadable_path_fields[field] = {
                    "exists": exists,
                    "shape": _local_adapter_feedback_path_shape(record.get(field)),
                }
        if not missing_fields and not unreadable_path_fields:
            continue
        return {
            "manifest_path_present": True,
            "manifest_path_readable": True,
            "first_invalid_record_index": index,
            "generated_record_type": generated_contract_record_type(data_contract),
            "required_fields": required_fields,
            "missing_or_empty_fields": missing_fields,
            "unreadable_path_fields": unreadable_path_fields,
            "box_field": box_field,
            "label_field": label_field,
            "diagnosis": (
                "At least one generated-contract manifest record is missing a required field, "
                "has an empty required value, or points a required path field at an unreadable local file."
            ),
        }
    return {
        "manifest_path_present": True,
        "manifest_path_readable": True,
        "first_invalid_record_index": None,
        "generated_record_type": generated_contract_record_type(data_contract),
        "required_fields": required_fields,
        "box_field": box_field,
        "label_field": label_field,
    }


def _local_adapter_feedback_record_path(value: Any, base_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def _local_adapter_feedback_path_shape(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {"present": False}
    path = Path(value)
    parts = path.parts
    return {
        "present": True,
        "absolute": path.is_absolute(),
        "component_count": len(parts),
        "suffix": path.suffix.lower(),
        "looks_archive_internal_or_source_relative": (not path.is_absolute() and len(parts) > 2),
    }


def _coerce_adapter_count(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    if count < 0:
        return None
    return count


def _safe_adapter_output_summary(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": output.get("schema_version"),
        "client_id": output.get("client_id"),
        "status": output.get("status"),
        "adapter_kind": output.get("adapter_kind"),
        "source_label_type": output.get("source_label_type"),
        "record_count": output.get("record_count"),
        "manifest_path_redacted": bool(output.get("manifest_path")),
        "script_path_redacted": bool(output.get("script_path")),
        "reason": output.get("reason"),
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


def _flare_round_context(
    task_name: str,
    request_message: FlareMessage,
    shareable: Shareable,
    client_id: str,
) -> dict[str, Any]:
    return {
        "task_name": task_name,
        "task_id": request_message.task_id,
        "site_id": client_id,
        "current_round": _current_round_from_shareable(shareable),
        "total_rounds": _total_rounds_from_shareable(shareable, 2),
        "round_action": shareable.get_header(HEADER_ROUND_ACTION),
    }
