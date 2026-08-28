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

"""Server-side NVFlare controller for AgenticFL orchestration rounds."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from agenticfl.agents import (
    AgentTurn,
    GuardrailAgent,
    GuardrailCheck,
    ServerAgent,
    _lock_extraction_runtime_config,
    _restore_generated_materializer_payload,
)
from agenticfl.agents.bridge import build_agent_backend
from agenticfl.data.extractor import ExtractionConfig
from agenticfl.data.parser import list_client_ids
from agenticfl.flare.channel import (
    AGENTICFL_MESSAGE_KEY,
    CLIENT_INQUIRY_SCHEMA,
    CLIENT_RESPONSE_SCHEMA,
    EXTRACTION_POLICY_SCHEMA,
    EXTRACTION_RESPONSE_SCHEMA,
    EXTRACTION_SUMMARY_SCHEMA,
    GUARDRAIL_SCHEMA,
    HEADER_CORRELATION_ID,
    HEADER_CURRENT_ROUND,
    HEADER_ROUND_ACTION,
    HEADER_SCHEMA_REF,
    HEADER_SESSION_ID,
    HEADER_SITE_ID,
    HEADER_TASK_ID,
    HEADER_TOTAL_ROUNDS,
    ROUND_EXTRACTION,
    ROUND_PROFILE,
    SERVER_STATE_SCHEMA,
    SIMULATION_SCHEMA_VERSION,
    TASK_QUERY_TASK_NAME,
    FlareMessage,
    from_shareable,
)
from agenticfl.flare.channel import round_props as _round_props
from agenticfl.flare.channel import set_fl_ctx_round as _set_fl_ctx_round
from agenticfl.flare.channel import set_round_headers as _set_round_headers
from agenticfl.flare.channel import to_shareable
from agenticfl.utils.logging import FlowLogger, payload_digest, timestamp_utc

from nvflare.apis.controller_spec import Task, TaskCompletionStatus
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal


class AgenticFLTaskQueryController(Controller):
    """Server-side AgenticFL controller for a bounded task-query job."""

    def __init__(
        self,
        site_meta_path: str,
        task: str = "<TASK_DESCRIPTION>",
        project_root: str | None = None,
        output_dir: str = "runs",
        session_id: str | None = None,
        min_count: int = 5,
        max_scan_files: int = 200_000,
        max_image_samples: int = 8,
        histogram_bins: int = 8,
        max_clients: int | None = None,
        task_name: str = TASK_QUERY_TASK_NAME,
        total_rounds: int = 2,
        result_wait_timeout: int = 600,
        wait_time_after_min_received: int = 0,
        extraction_output_root: str = ExtractionConfig.output_root,
        extraction_output_name: str | None = None,
        extraction_max_samples: int | None = None,
        extraction_overwrite: bool = False,
        extraction_validation_fraction: float = ExtractionConfig.validation_fraction,
        agent_backend: str = "codex",
        agent_timeout_seconds: float = 3600.0,
        agent_poll_interval_seconds: float = 2.0,
        client_inquiry_prompt: str | None = None,
        profile_resume_run_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.site_meta_path = site_meta_path
        self.task = task
        self.project_root = project_root
        self.output_dir = output_dir
        self.session_id = session_id
        self.min_count = min_count
        self.max_scan_files = max_scan_files
        self.max_image_samples = max_image_samples
        self.histogram_bins = histogram_bins
        self.max_clients = max_clients
        self.task_name = task_name
        self.total_rounds = total_rounds
        self.result_wait_timeout = result_wait_timeout
        self.wait_time_after_min_received = wait_time_after_min_received
        self.extraction_output_root = extraction_output_root
        self.extraction_output_name = extraction_output_name
        self.extraction_max_samples = extraction_max_samples
        self.extraction_overwrite = extraction_overwrite
        self.extraction_validation_fraction = extraction_validation_fraction
        self.agent_backend = agent_backend
        self.agent_timeout_seconds = agent_timeout_seconds
        self.agent_poll_interval_seconds = agent_poll_interval_seconds
        self.client_inquiry_prompt = client_inquiry_prompt
        self.profile_resume_run_dir = profile_resume_run_dir

    def start_controller(self, fl_ctx: FLContext) -> None:
        self.log_info(fl_ctx, "AgenticFL task-query controller started.")

    def stop_controller(self, fl_ctx: FLContext) -> None:
        self.log_info(fl_ctx, "AgenticFL task-query controller stopped.")

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        if abort_signal.triggered:
            return
        if self.profile_resume_run_dir:
            AgenticFLProfileResumeController.control_flow(self, abort_signal, fl_ctx)
            return

        session = self.session_id or f"{_task_slug(self.task)}_{timestamp_utc().replace(':', '').replace('.', '_')}"
        run_dir = Path(self.output_dir) / session
        logger = FlowLogger(run_dir)
        user_task = {"task": self.task}

        client_ids = _server_visible_clients(self.site_meta_path, self.max_clients)
        client_ids = _filter_available_clients(client_ids, fl_ctx)
        if not client_ids:
            self.system_panic("No available AgenticFL clients matched the server-visible client id list.", fl_ctx)
            return

        agent_backend = _build_agent_backend(
            kind=self.agent_backend,
            run_dir=run_dir,
            session_id=session,
            timeout_seconds=self.agent_timeout_seconds,
            poll_interval_seconds=self.agent_poll_interval_seconds,
        )
        server_agent = ServerAgent(agent_backend)
        server_guardrail = GuardrailAgent(party_role="server", party_id="server", agent_backend=agent_backend)
        _set_fl_ctx_round(fl_ctx, ROUND_PROFILE, self.total_rounds)
        server_turn = server_agent.compose_client_inquiry(task=user_task, client_ids=client_ids)
        client_payload = _client_inquiry_payload(server_turn.output)
        prompt_override_applied = self.client_inquiry_prompt is not None
        if prompt_override_applied:
            client_payload = _client_inquiry_with_prompt_override(client_payload, self.client_inquiry_prompt or "")
        server_event_payload = _server_inquiry_log_payload(
            server_output=server_turn.output,
            client_payload=client_payload,
            prompt_override_applied=prompt_override_applied,
        )
        job_id = fl_ctx.get_job_id() or session
        correlation_id = str(uuid4())
        outgoing_guardrail = server_guardrail.inspect(
            direction="outgoing",
            check=GuardrailCheck(
                role="server_agent",
                source="human_user",
                channel="server_console",
                action="SERVER.DEFINE_PROFILE_REQUEST",
                phase="data_profile",
                input_schema="TaskSpec@v1",
                output_schema="AgenticFLClientInquiry@v1",
            ),
            payload=client_payload,
            counterpart="client_agents",
        )
        request = to_shareable(
            FlareMessage(
                session_id=session,
                correlation_id=correlation_id,
                schema_ref=CLIENT_INQUIRY_SCHEMA,
                task_id=f"{job_id}.{self.task_name}",
                payload=client_payload,
            )
        )
        _set_round_headers(request, current_round=ROUND_PROFILE, total_rounds=self.total_rounds, action="profile")

        logger.write_server_event(
            session_id=session,
            phase_id="profile",
            step_id="profile.server_inquiry.001",
            correlation_id=correlation_id,
            event_type="SERVER_AGENT_OUTPUT",
            payload=server_event_payload,
            schema_ref=CLIENT_INQUIRY_SCHEMA,
            privacy_class="policy_metadata",
            visibility="cross_site_redacted",
            status="created",
            policy=server_turn.guardrail,
            next_step={
                "action": "broadcast_inquiry_via_nvflare_job",
                "reason": "server agent composed task-aware client inquiry",
            },
            agent_prompt=server_turn.prompt,
            flare={
                "job_id": job_id,
                "task_name": self.task_name,
                "targets": client_ids,
                "current_round": ROUND_PROFILE,
                "total_rounds": self.total_rounds,
            },
        )
        logger.write_server_event(
            session_id=session,
            phase_id="profile",
            step_id="profile.server_outgoing_guardrail.001",
            correlation_id=correlation_id,
            event_type="SERVER_OUTGOING_GUARDRAIL_CHECKED",
            payload=outgoing_guardrail.as_payload(),
            schema_ref=GUARDRAIL_SCHEMA,
            privacy_class="policy_metadata",
            visibility="server",
            status="completed",
            policy=outgoing_guardrail.as_policy(),
            next_step={
                "action": "broadcast_inquiry_via_nvflare_job" if outgoing_guardrail.allowed else "block_broadcast",
                "reason": outgoing_guardrail.reason,
            },
            agent_prompt=server_turn.prompt,
            flare={
                "job_id": job_id,
                "task_name": self.task_name,
                "targets": client_ids,
                "current_round": ROUND_PROFILE,
                "total_rounds": self.total_rounds,
            },
        )
        if not outgoing_guardrail.allowed:
            logger.write_artifact(
                "server/requests/client_inquiry_blocked.json",
                {"client_inquiry": client_payload, "guardrail": outgoing_guardrail.as_payload()},
            )
            return

        task = Task(
            name=self.task_name,
            data=request,
            props=_round_props(ROUND_PROFILE, self.total_rounds, "profile"),
            timeout=self.result_wait_timeout,
        )
        self.broadcast_and_wait(
            task=task,
            targets=client_ids,
            min_responses=len(client_ids),
            fl_ctx=fl_ctx,
            wait_time_after_min_received=self.wait_time_after_min_received,
            abort_signal=abort_signal,
        )
        replies = {client_task.client.name: client_task.result for client_task in task.client_tasks}

        site_wise_info: dict[str, Any] = {}
        expected_profile_task_id = f"{job_id}.{self.task_name}"
        for index, client_id in enumerate(client_ids, start=1):
            reply = replies.get(client_id)
            expected_reply = {
                "expected_session_id": session,
                "expected_correlation_id": correlation_id,
                "expected_schema_ref": CLIENT_RESPONSE_SCHEMA,
                "expected_task_id": expected_profile_task_id,
                "expected_site_id": client_id,
            }
            response = _response_from_reply(client_id, reply, **expected_reply)
            message = _message_from_reply(reply, client_id=client_id, **expected_reply)
            incoming_guardrail = server_guardrail.inspect(
                direction="incoming",
                check=GuardrailCheck(
                    role="server_agent",
                    source="client_agent_via_flare",
                    channel="flare_task_result",
                    action="SERVER.AGGREGATE_CLIENT_PROFILES",
                    phase="data_profile",
                    input_schema="AgenticFLClientResponse@v1",
                    output_schema="AgenticFLServerState@v1",
                ),
                payload=response,
                counterpart=client_id,
            )
            logger.write_server_event(
                session_id=session,
                phase_id="profile",
                step_id=f"profile.server_incoming_guardrail.{index:03d}",
                correlation_id=correlation_id,
                event_type="SERVER_INCOMING_GUARDRAIL_CHECKED",
                payload={"client_id": client_id, "guardrail": incoming_guardrail.as_payload()},
                schema_ref=GUARDRAIL_SCHEMA,
                privacy_class="aggregate_stats",
                status="completed",
                policy=incoming_guardrail.as_policy(),
                next_step={
                    "action": "accept_client_response" if incoming_guardrail.allowed else "block_client_response",
                    "reason": incoming_guardrail.reason,
                },
                flare={
                    "job_id": job_id,
                    "task_name": self.task_name,
                    "task_id": message.task_id if message is not None else None,
                    "site_id": client_id,
                    "return_code": reply.get_return_code() if reply is not None else ReturnCode.EMPTY_RESULT,
                    "current_round": ROUND_PROFILE,
                    "total_rounds": self.total_rounds,
                },
            )
            if not incoming_guardrail.allowed:
                response = _server_guardrail_denial_payload(client_id=client_id, decision=incoming_guardrail)
            site_wise_info[client_id] = response
            logger.write_server_event(
                session_id=session,
                phase_id="profile",
                step_id=f"profile.server_receive.{index:03d}",
                correlation_id=correlation_id,
                event_type="CLIENT_AGENT_OUTPUT_RECEIVED",
                payload={"client_id": client_id, "response": response},
                schema_ref=CLIENT_RESPONSE_SCHEMA,
                privacy_class="aggregate_stats",
                status="received" if message is not None else "error",
                next_step={
                    "action": "update_site_wise_info",
                    "reason": "client response received through NVFlare task result",
                },
                flare={
                    "job_id": job_id,
                    "task_name": self.task_name,
                    "task_id": message.task_id if message is not None else None,
                    "site_id": client_id,
                    "return_code": reply.get_return_code() if reply is not None else ReturnCode.EMPTY_RESULT,
                    "current_round": ROUND_PROFILE,
                    "total_rounds": self.total_rounds,
                },
            )

        summary_turn = server_agent.summarize_responses(
            task=user_task,
            inquiry=server_turn.output,
            site_wise_info=site_wise_info,
        )
        extraction_strategy = summary_turn.output["extraction_strategy"]
        server_state = {
            "schema_version": SIMULATION_SCHEMA_VERSION,
            "session_id": session,
            "server_request": user_task,
            "server_inquiry": server_turn.output,
            "client_broadcast_inquiry": client_payload,
            "site_wise_info": site_wise_info,
            "extraction_strategy": extraction_strategy,
            "transport": {
                "kind": "nvflare_job_api_controller_executor",
                "job_id": job_id,
                "task_name": self.task_name,
                "message_key": AGENTICFL_MESSAGE_KEY,
                "headers": [
                    HEADER_SESSION_ID,
                    HEADER_CORRELATION_ID,
                    HEADER_SCHEMA_REF,
                    HEADER_TASK_ID,
                    HEADER_SITE_ID,
                    HEADER_CURRENT_ROUND,
                    HEADER_TOTAL_ROUNDS,
                    HEADER_ROUND_ACTION,
                ],
            },
            "rounds": {
                "total_rounds": self.total_rounds,
                "completed": [ROUND_PROFILE],
                "round_1": {
                    "name": "profile",
                    "target_clients": client_ids,
                    "response_count": len(site_wise_info),
                },
            },
        }
        logger.write_artifact("server/site_wise_info.json", {"site_wise_info": site_wise_info})
        logger.write_artifact("server/decisions/extraction_strategy.json", extraction_strategy)
        logger.write_artifact("server/final_state.json", server_state)
        logger.write_artifact("server/requests/client_inquiry.json", client_payload)

        logger.write_server_event(
            session_id=session,
            phase_id="profile",
            step_id="profile.server_summary.001",
            correlation_id=correlation_id,
            event_type="SERVER_AGENT_OUTPUT",
            payload=summary_turn.output,
            schema_ref=SERVER_STATE_SCHEMA,
            privacy_class="aggregate_stats",
            status="completed",
            policy=summary_turn.guardrail,
            next_step=(
                {"action": "dispatch_extraction_policies", "reason": "round 1 completed and round 2 is enabled"}
                if self.total_rounds >= ROUND_EXTRACTION
                else extraction_strategy["next_step"]
            ),
            agent_prompt=summary_turn.prompt,
            flare={
                "job_id": job_id,
                "task_name": self.task_name,
                "current_round": ROUND_PROFILE,
                "total_rounds": self.total_rounds,
            },
        )

        if self.total_rounds >= ROUND_EXTRACTION:
            self._run_extraction_round(
                logger=logger,
                session=session,
                job_id=job_id,
                user_task=user_task,
                server_agent=server_agent,
                server_guardrail=server_guardrail,
                extraction_strategy=extraction_strategy,
                server_state=server_state,
                fl_ctx=fl_ctx,
                abort_signal=abort_signal,
            )

    def _run_extraction_round(
        self,
        *,
        logger: FlowLogger,
        session: str,
        job_id: str,
        user_task: dict[str, Any],
        server_agent: ServerAgent,
        server_guardrail: GuardrailAgent,
        extraction_strategy: dict[str, Any],
        server_state: dict[str, Any],
        fl_ctx: FLContext,
        abort_signal: Signal,
        dispatch_turn: AgentTurn | None = None,
    ) -> None:
        if abort_signal.triggered:
            return

        _set_fl_ctx_round(fl_ctx, ROUND_EXTRACTION, self.total_rounds)
        applicable_clients = [
            client_id for client_id in extraction_strategy.get("applicable_clients", []) if isinstance(client_id, str)
        ]
        target_clients = _filter_available_clients(applicable_clients, fl_ctx)
        extraction_config = _extraction_config_payload(
            session=session,
            output_root=self.extraction_output_root,
            output_name=self.extraction_output_name,
            max_samples=self.extraction_max_samples,
            overwrite=self.extraction_overwrite,
            validation_fraction=self.extraction_validation_fraction,
        )
        if _strategy_needs_generated_materializer(extraction_strategy):
            materializer_workspace = (
                logger.run_dir
                / "server"
                / "generated_materializers"
                / _safe_materializer_workspace_name(extraction_strategy)
            )
            materializer_workspace.mkdir(parents=True, exist_ok=True)
            materializer_turn = server_agent.implement_data_materializer(
                task=user_task,
                extraction_strategy=extraction_strategy,
                code_workspace=str(materializer_workspace),
            )
            materializer_payload = _pack_generated_data_materializer(materializer_turn.output)
            extraction_strategy = {
                **extraction_strategy,
                "generated_data_materializer": materializer_payload,
            }
            server_state["extraction_strategy"] = extraction_strategy
            logger.write_artifact("server/decisions/extraction_strategy.json", extraction_strategy)
            logger.write_artifact(
                "server/generated_materializers/materializer_spec.json",
                materializer_turn.output,
            )
            logger.write_server_event(
                session_id=session,
                phase_id="harmonization",
                step_id="harmonization.server_materializer.001",
                correlation_id=str(uuid4()),
                event_type="SERVER_AGENT_OUTPUT",
                payload={
                    "schema_version": "agenticfl.generated_data_materializer_event.v1",
                    "status": materializer_payload.get("status"),
                    "record_type": materializer_payload.get("record_type"),
                    "source_file_count": len(materializer_payload.get("source_files") or []),
                    "source_digest": materializer_payload.get("source_digest"),
                },
                schema_ref="AgenticFLDataMaterializerSpec@v1",
                privacy_class="policy_metadata",
                visibility="server",
                status="completed",
                policy=materializer_turn.guardrail,
                next_step={
                    "action": "dispatch_extraction_policies",
                    "reason": "server agent generated the materializer required by the generated data contract",
                },
                agent_prompt=materializer_turn.prompt,
                flare={
                    "job_id": job_id,
                    "task_name": self.task_name,
                    "current_round": ROUND_EXTRACTION,
                    "total_rounds": self.total_rounds,
                },
            )

        if dispatch_turn is None:
            dispatch_turn = server_agent.compose_extraction_dispatch(
                task=user_task,
                extraction_strategy=extraction_strategy,
                target_client_ids=target_clients,
                extraction_config=extraction_config,
            )
        else:
            dispatch_turn = AgentTurn(
                prompt=dispatch_turn.prompt,
                output=_lock_extraction_runtime_config(
                    _restore_generated_materializer_payload(
                        dispatch_turn.output,
                        materializer=(
                            extraction_strategy.get("generated_data_materializer")
                            if isinstance(extraction_strategy.get("generated_data_materializer"), dict)
                            else None
                        ),
                    ),
                    extraction_config=extraction_config,
                ),
                guardrail=dispatch_turn.guardrail,
            )
        policies = dispatch_turn.output["policies"]
        target_clients = [client_id for client_id in target_clients if client_id in policies]
        correlation_id = str(uuid4())

        logger.write_artifact("server/requests/extraction_policies.json", dispatch_turn.output)
        logger.write_server_event(
            session_id=session,
            phase_id="harmonization",
            step_id="harmonization.server_policy.001",
            correlation_id=correlation_id,
            event_type="SERVER_AGENT_OUTPUT",
            payload=dispatch_turn.output,
            schema_ref=EXTRACTION_POLICY_SCHEMA,
            privacy_class="policy_metadata",
            visibility="cross_site_redacted",
            status="created",
            policy=dispatch_turn.guardrail,
            next_step={
                "action": "dispatch_site_specific_extraction_policy_via_nvflare_job",
                "reason": "server agent generated round-2 site policies",
            },
            agent_prompt=dispatch_turn.prompt,
            flare={
                "job_id": job_id,
                "task_name": self.task_name,
                "targets": target_clients,
                "current_round": ROUND_EXTRACTION,
                "total_rounds": self.total_rounds,
            },
        )

        guarded_target_clients: list[str] = []
        blocked_extraction_results: dict[str, Any] = {}
        for index, client_id in enumerate(target_clients, start=1):
            site_policy = policies[client_id]
            outgoing_guardrail = server_guardrail.inspect(
                direction="outgoing",
                check=GuardrailCheck(
                    role="server_agent",
                    source="server_state",
                    channel="flare_task",
                    action="SERVER.DISPATCH_EXTRACTION_POLICY",
                    phase="data_harmonization",
                    input_schema="AgenticFLServerState@v1",
                    output_schema="AgenticFLSiteExtractionPolicy@v1",
                ),
                payload=site_policy,
                counterpart=client_id,
            )
            logger.write_server_event(
                session_id=session,
                phase_id="harmonization",
                step_id=f"harmonization.server_outgoing_guardrail.{index:03d}",
                correlation_id=correlation_id,
                event_type="SERVER_OUTGOING_GUARDRAIL_CHECKED",
                payload={"client_id": client_id, "guardrail": outgoing_guardrail.as_payload()},
                schema_ref=GUARDRAIL_SCHEMA,
                privacy_class="policy_metadata",
                visibility="server",
                status="completed",
                policy=outgoing_guardrail.as_policy(),
                next_step={
                    "action": "dispatch_site_policy" if outgoing_guardrail.allowed else "block_site_policy",
                    "reason": outgoing_guardrail.reason,
                },
                flare={
                    "job_id": job_id,
                    "task_name": self.task_name,
                    "site_id": client_id,
                    "current_round": ROUND_EXTRACTION,
                    "total_rounds": self.total_rounds,
                },
            )
            if outgoing_guardrail.allowed:
                guarded_target_clients.append(client_id)
            else:
                blocked_extraction_results[client_id] = _server_guardrail_denial_payload(
                    client_id=client_id,
                    decision=outgoing_guardrail,
                )
        target_clients = guarded_target_clients

        extraction_tasks: list[tuple[int, str, str, Task]] = []
        for index, client_id in enumerate(target_clients, start=1):
            site_policy = policies[client_id]
            task_id = f"{job_id}.{self.task_name}.round{ROUND_EXTRACTION}.{index:03d}"
            request = to_shareable(
                FlareMessage(
                    session_id=session,
                    correlation_id=correlation_id,
                    schema_ref=EXTRACTION_POLICY_SCHEMA,
                    task_id=task_id,
                    site_id=client_id,
                    payload=site_policy,
                )
            )
            _set_round_headers(
                request,
                current_round=ROUND_EXTRACTION,
                total_rounds=self.total_rounds,
                action="extract",
            )
            task = Task(
                name=self.task_name,
                data=request,
                props=_round_props(ROUND_EXTRACTION, self.total_rounds, "extract"),
                timeout=self.result_wait_timeout,
            )
            self.send(
                task=task,
                targets=[client_id],
                fl_ctx=fl_ctx,
            )
            extraction_tasks.append((index, client_id, task_id, task))

        extraction_results: dict[str, Any] = dict(blocked_extraction_results)
        for index, client_id, task_id, task in extraction_tasks:
            _wait_for_task_completion(self, task=task, fl_ctx=fl_ctx, abort_signal=abort_signal)
            reply = task.client_tasks[0].result if task.client_tasks else None
            expected_reply = {
                "expected_session_id": session,
                "expected_correlation_id": correlation_id,
                "expected_schema_ref": EXTRACTION_RESPONSE_SCHEMA,
                "expected_task_id": task_id,
                "expected_site_id": client_id,
            }
            response = _response_from_reply(client_id, reply, **expected_reply)
            message = _message_from_reply(reply, client_id=client_id, **expected_reply)
            incoming_guardrail = server_guardrail.inspect(
                direction="incoming",
                check=GuardrailCheck(
                    role="server_agent",
                    source="client_agent_via_flare",
                    channel="flare_task_result",
                    action="SERVER.AGGREGATE_EXTRACTION_RESULTS",
                    phase="data_harmonization",
                    input_schema="AgenticFLClientExtractionResponse@v1",
                    output_schema="AgenticFLExtractionRoundSummary@v1",
                ),
                payload=response,
                counterpart=client_id,
            )
            logger.write_server_event(
                session_id=session,
                phase_id="harmonization",
                step_id=f"harmonization.server_incoming_guardrail.{index:03d}",
                correlation_id=correlation_id,
                event_type="SERVER_INCOMING_GUARDRAIL_CHECKED",
                payload={"client_id": client_id, "guardrail": incoming_guardrail.as_payload()},
                schema_ref=GUARDRAIL_SCHEMA,
                privacy_class="aggregate_stats",
                status="completed",
                policy=incoming_guardrail.as_policy(),
                next_step={
                    "action": (
                        "accept_client_extraction_response"
                        if incoming_guardrail.allowed
                        else "block_client_extraction_response"
                    ),
                    "reason": incoming_guardrail.reason,
                },
                flare={
                    "job_id": job_id,
                    "task_name": self.task_name,
                    "task_id": message.task_id if message is not None else None,
                    "site_id": client_id,
                    "return_code": reply.get_return_code() if reply is not None else ReturnCode.EMPTY_RESULT,
                    "current_round": ROUND_EXTRACTION,
                    "total_rounds": self.total_rounds,
                },
            )
            if not incoming_guardrail.allowed:
                response = _server_guardrail_denial_payload(client_id=client_id, decision=incoming_guardrail)
            extraction_results[client_id] = response
            logger.write_server_event(
                session_id=session,
                phase_id="harmonization",
                step_id=f"harmonization.server_receive.{index:03d}",
                correlation_id=correlation_id,
                event_type="CLIENT_EXTRACTION_OUTPUT_RECEIVED",
                payload={"client_id": client_id, "response": response},
                schema_ref=EXTRACTION_RESPONSE_SCHEMA,
                privacy_class="aggregate_stats",
                status="received" if message is not None else "error",
                next_step={
                    "action": "update_extraction_results",
                    "reason": "client extraction result received through NVFlare task result",
                },
                flare={
                    "job_id": job_id,
                    "task_name": self.task_name,
                    "task_id": message.task_id if message is not None else None,
                    "site_id": client_id,
                    "return_code": reply.get_return_code() if reply is not None else ReturnCode.EMPTY_RESULT,
                    "current_round": ROUND_EXTRACTION,
                    "total_rounds": self.total_rounds,
                },
            )

        summary_turn = server_agent.summarize_extraction_results(
            task=user_task,
            extraction_results=extraction_results,
        )
        server_state["extraction_dispatch"] = dispatch_turn.output
        server_state["extraction_results"] = extraction_results
        server_state["extraction_round_summary"] = summary_turn.output
        server_state["rounds"]["completed"].append(ROUND_EXTRACTION)
        server_state["rounds"]["round_2"] = {
            "name": "extraction",
            "target_clients": target_clients,
            "response_count": len(extraction_results),
            "extracted_client_count": summary_turn.output["extracted_client_count"],
            "screened_out_client_count": summary_turn.output["screened_out_client_count"],
            "failed_client_count": summary_turn.output["failed_client_count"],
        }
        logger.write_artifact("server/responses/extraction_results.json", {"extraction_results": extraction_results})
        logger.write_artifact("server/decisions/extraction_round_summary.json", summary_turn.output)
        logger.write_artifact("server/final_state.json", server_state)
        logger.write_server_event(
            session_id=session,
            phase_id="harmonization",
            step_id="harmonization.server_summary.001",
            correlation_id=correlation_id,
            event_type="SERVER_AGENT_OUTPUT",
            payload=summary_turn.output,
            schema_ref=EXTRACTION_SUMMARY_SCHEMA,
            privacy_class="aggregate_stats",
            status="completed",
            policy=summary_turn.guardrail,
            next_step=summary_turn.output["next_step"],
            agent_prompt=summary_turn.prompt,
            flare={
                "job_id": job_id,
                "task_name": self.task_name,
                "current_round": ROUND_EXTRACTION,
                "total_rounds": self.total_rounds,
            },
        )


def _server_guardrail_denial_payload(*, client_id: str | None, decision: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "denied",
        "agent_role": "server_guardrail_agent",
        "reason_code": decision.reason_code,
        "reason": decision.reason,
    }
    if client_id is not None:
        payload["client_id"] = client_id
    return payload


def _client_inquiry_payload(server_output: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": server_output["schema_version"],
        "task": server_output["task"],
        "intent": server_output["intent"],
        "message": server_output["message"],
    }


def _client_inquiry_with_prompt_override(payload: dict[str, Any], prompt: str) -> dict[str, Any]:
    prompt_text = prompt.strip()
    if not prompt_text:
        raise ValueError("client inquiry prompt override must not be empty")
    return {
        **payload,
        "message": {
            "task": payload.get("task"),
            "intent": payload.get("intent"),
            "prompt_override": True,
            "custom_prompt": prompt_text,
            "instructions": [prompt_text],
        },
    }


def _server_inquiry_log_payload(
    *,
    server_output: dict[str, Any],
    client_payload: dict[str, Any],
    prompt_override_applied: bool,
) -> dict[str, Any]:
    if not prompt_override_applied:
        return server_output
    return {
        **client_payload,
        "prompt_override_applied": True,
        "effective_payload": "client_inquiry_sent_via_flare",
        "original_server_agent_message": server_output.get("message"),
    }


def _extraction_config_payload(
    *,
    session: str,
    output_root: str,
    output_name: str | None,
    max_samples: int | None,
    overwrite: bool,
    validation_fraction: float,
) -> dict[str, Any]:
    return {
        "output_root": output_root,
        "output_name": output_name or f"{session}_round2_extracted",
        "max_samples": max_samples,
        "overwrite": overwrite,
        "validation_fraction": validation_fraction,
    }


def _server_visible_clients(site_meta_path: str, max_clients: int | None) -> list[str]:
    redacted = list_client_ids(site_meta_path)
    client_ids = [client["client_id"] for client in redacted["clients"]]
    if max_clients is not None:
        client_ids = client_ids[:max_clients]
    return client_ids


def _filter_available_clients(client_ids: list[str], fl_ctx: FLContext) -> list[str]:
    engine = fl_ctx.get_engine()
    if engine is None:
        return client_ids
    available = {client.name for client in engine.get_clients()}
    if not available:
        return client_ids
    return [client_id for client_id in client_ids if client_id in available]


def _wait_for_task_completion(
    controller: Controller,
    *,
    task: Task,
    fl_ctx: FLContext,
    abort_signal: Signal,
) -> None:
    while task.completion_status is None:
        if abort_signal.triggered:
            controller.cancel_task(task, completion_status=TaskCompletionStatus.ABORTED, fl_ctx=fl_ctx)
            return
        time.sleep(getattr(controller, "_task_check_period", 0.5))


def _response_from_reply(
    client_id: str,
    reply: Shareable | None,
    *,
    expected_session_id: str | None = None,
    expected_correlation_id: str | None = None,
    expected_schema_ref: str | None = None,
    expected_task_id: str | None = None,
    expected_site_id: str | None = None,
) -> dict[str, Any]:
    if reply is None:
        return {"status": "error", "client_id": client_id, "reason_code": ReturnCode.EMPTY_RESULT}
    rc = reply.get_return_code()
    if rc != ReturnCode.OK:
        return {"status": "error", "client_id": client_id, "reason_code": rc}
    try:
        return _validated_reply_message(
            client_id=client_id,
            reply=reply,
            expected_session_id=expected_session_id,
            expected_correlation_id=expected_correlation_id,
            expected_schema_ref=expected_schema_ref,
            expected_task_id=expected_task_id,
            expected_site_id=expected_site_id,
        ).payload
    except Exception as exc:  # noqa: BLE001 - server records malformed client replies as bounded errors.
        return {
            "status": "error",
            "client_id": client_id,
            "reason_code": "MALFORMED_REPLY",
            "reason": f"Malformed FLARE reply ({type(exc).__name__}).",
        }


def _message_from_reply(
    reply: Shareable | None,
    *,
    client_id: str,
    expected_session_id: str | None = None,
    expected_correlation_id: str | None = None,
    expected_schema_ref: str | None = None,
    expected_task_id: str | None = None,
    expected_site_id: str | None = None,
) -> FlareMessage | None:
    if reply is None or reply.get_return_code() != ReturnCode.OK:
        return None
    try:
        return _validated_reply_message(
            client_id=client_id,
            reply=reply,
            expected_session_id=expected_session_id,
            expected_correlation_id=expected_correlation_id,
            expected_schema_ref=expected_schema_ref,
            expected_task_id=expected_task_id,
            expected_site_id=expected_site_id,
        )
    except Exception:
        return None


def _validated_reply_message(
    *,
    client_id: str,
    reply: Shareable,
    expected_session_id: str | None,
    expected_correlation_id: str | None,
    expected_schema_ref: str | None,
    expected_task_id: str | None,
    expected_site_id: str | None,
) -> FlareMessage:
    message = from_shareable(reply)
    _require_matching_value("session_id", message.session_id, expected_session_id)
    _require_matching_value("correlation_id", message.correlation_id, expected_correlation_id)
    _require_matching_value("schema_ref", message.schema_ref, expected_schema_ref)
    _require_matching_value("task_id", message.task_id, expected_task_id)
    if expected_site_id is not None and message.site_id != expected_site_id:
        raise ValueError("FLARE reply site_id did not match expected client")
    payload_client_id = message.payload.get("client_id")
    if payload_client_id != client_id:
        raise ValueError("FLARE reply payload client_id did not match expected client")
    return message


def _require_matching_value(field: str, observed: str, expected: str | None) -> None:
    if expected is not None and observed != expected:
        raise ValueError(f"FLARE reply {field} did not match expected value")


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


def _task_slug(task_text: str) -> str:
    slug = "_".join(token for token in task_text.lower().replace("-", " ").split() if token)
    slug = "".join(char for char in slug if char.isalnum() or char == "_")
    return slug or "task_query"


class AgenticFLProfileResumeController(AgenticFLTaskQueryController):
    """Resume after a fully guarded profile round without rediscovering client data."""

    def __init__(self, *, profile_resume_run_dir: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.profile_resume_run_dir = str(Path(profile_resume_run_dir).expanduser().resolve())

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        if abort_signal.triggered:
            return

        session = self.session_id or f"{_task_slug(self.task)}_{timestamp_utc().replace(':', '').replace('.', '_')}"
        run_dir = Path(self.output_dir) / session
        logger = FlowLogger(run_dir)
        user_task = {"task": self.task}
        client_ids = _server_visible_clients(self.site_meta_path, self.max_clients)
        client_ids = _filter_available_clients(client_ids, fl_ctx)
        if not client_ids:
            self.system_panic("No available AgenticFL clients matched the server-visible client id list.", fl_ctx)
            return

        agent_backend = _build_agent_backend(
            kind=self.agent_backend,
            run_dir=run_dir,
            session_id=session,
            timeout_seconds=self.agent_timeout_seconds,
            poll_interval_seconds=self.agent_poll_interval_seconds,
        )
        server_agent = ServerAgent(agent_backend)
        server_guardrail = GuardrailAgent(party_role="server", party_id="server", agent_backend=agent_backend)
        _set_fl_ctx_round(fl_ctx, ROUND_PROFILE, self.total_rounds)
        try:
            resumed_profile = _load_profile_resume_state(
                profile_run_dir=Path(self.profile_resume_run_dir),
                expected_client_ids=client_ids,
                task=self.task,
            )
        except ValueError:
            partial_profile = _load_partial_profile_resume_state(
                profile_run_dir=Path(self.profile_resume_run_dir),
                expected_client_ids=client_ids,
                task=self.task,
            )
            resumed_profile = AgenticFLProfileResumeController._retry_partial_profile(
                self,
                partial_profile=partial_profile,
                client_ids=client_ids,
                session=session,
                logger=logger,
                user_task=user_task,
                server_agent=server_agent,
                server_guardrail=server_guardrail,
                fl_ctx=fl_ctx,
                abort_signal=abort_signal,
            )
        server_turn = resumed_profile["server_turn"]
        site_wise_info = resumed_profile["site_wise_info"]
        client_payload = _client_inquiry_payload(server_turn.output)
        job_id = fl_ctx.get_job_id() or session
        correlation_id = str(uuid4())
        logger.write_artifact(
            "server/profile_resume.json",
            {
                "schema_version": "agenticfl.profile_resume.v1",
                "source_session_id": resumed_profile["source_session_id"],
                "profile_response_count": len(site_wise_info),
                "profile_digest": resumed_profile["profile_digest"],
                "guards_verified": True,
                "reused_client_count": resumed_profile.get("reused_client_count", len(site_wise_info)),
                "retried_client_count": resumed_profile.get("retried_client_count", 0),
                "retried_clients": resumed_profile.get("retried_clients", []),
            },
        )
        logger.write_server_event(
            session_id=session,
            phase_id="profile",
            step_id="profile.server_resume.001",
            correlation_id=correlation_id,
            event_type="SERVER_PROFILE_RESUMED",
            payload={
                "source_session_id": resumed_profile["source_session_id"],
                "profile_response_count": len(site_wise_info),
                "profile_digest": resumed_profile["profile_digest"],
            },
            schema_ref=SERVER_STATE_SCHEMA,
            privacy_class="aggregate_stats",
            status="completed",
            policy={"guardrail_checked": True, "decision": "allowed", "allow_list_rule": "profile_resume.v1"},
            next_step={
                "action": "aggregate_saved_profile_responses",
                "reason": "all saved client profile responses were previously guard-admitted",
            },
            flare={
                "job_id": job_id,
                "task_name": self.task_name,
                "current_round": ROUND_PROFILE,
                "total_rounds": self.total_rounds,
            },
        )

        summary_turn = resumed_profile["summary_turn"]
        extraction_strategy = summary_turn.output["extraction_strategy"]
        server_state = {
            "schema_version": SIMULATION_SCHEMA_VERSION,
            "session_id": session,
            "server_request": user_task,
            "server_inquiry": server_turn.output,
            "client_broadcast_inquiry": client_payload,
            "site_wise_info": site_wise_info,
            "extraction_strategy": extraction_strategy,
            "profile_resume": {
                "source_session_id": resumed_profile["source_session_id"],
                "profile_digest": resumed_profile["profile_digest"],
                "response_count": len(site_wise_info),
            },
            "transport": {
                "kind": "nvflare_job_api_controller_executor",
                "job_id": job_id,
                "task_name": self.task_name,
                "message_key": AGENTICFL_MESSAGE_KEY,
                "headers": [
                    HEADER_SESSION_ID,
                    HEADER_CORRELATION_ID,
                    HEADER_SCHEMA_REF,
                    HEADER_TASK_ID,
                    HEADER_SITE_ID,
                    HEADER_CURRENT_ROUND,
                    HEADER_TOTAL_ROUNDS,
                    HEADER_ROUND_ACTION,
                ],
            },
            "rounds": {
                "total_rounds": self.total_rounds,
                "completed": [ROUND_PROFILE],
                "round_1": {
                    "name": "profile",
                    "target_clients": client_ids,
                    "response_count": len(site_wise_info),
                    "resumed": True,
                },
            },
        }
        logger.write_artifact("server/site_wise_info.json", {"site_wise_info": site_wise_info})
        logger.write_artifact("server/decisions/extraction_strategy.json", extraction_strategy)
        logger.write_artifact("server/final_state.json", server_state)
        logger.write_artifact("server/requests/client_inquiry.json", client_payload)
        logger.write_server_event(
            session_id=session,
            phase_id="profile",
            step_id="profile.server_summary.001",
            correlation_id=correlation_id,
            event_type="SERVER_AGENT_OUTPUT",
            payload=summary_turn.output,
            schema_ref=SERVER_STATE_SCHEMA,
            privacy_class="aggregate_stats",
            status="completed",
            policy=summary_turn.guardrail,
            next_step=(
                {"action": "dispatch_extraction_policies", "reason": "resumed profile round completed"}
                if self.total_rounds >= ROUND_EXTRACTION
                else extraction_strategy["next_step"]
            ),
            agent_prompt=summary_turn.prompt,
            flare={
                "job_id": job_id,
                "task_name": self.task_name,
                "current_round": ROUND_PROFILE,
                "total_rounds": self.total_rounds,
            },
        )
        if self.total_rounds >= ROUND_EXTRACTION:
            self._run_extraction_round(
                logger=logger,
                session=session,
                job_id=job_id,
                user_task=user_task,
                server_agent=server_agent,
                server_guardrail=server_guardrail,
                extraction_strategy=extraction_strategy,
                server_state=server_state,
                fl_ctx=fl_ctx,
                abort_signal=abort_signal,
                dispatch_turn=resumed_profile.get("dispatch_turn"),
            )

    def _retry_partial_profile(
        self,
        *,
        partial_profile: dict[str, Any],
        client_ids: list[str],
        session: str,
        logger: FlowLogger,
        user_task: dict[str, Any],
        server_agent: ServerAgent,
        server_guardrail: GuardrailAgent,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> dict[str, Any]:
        server_turn = partial_profile["server_turn"]
        client_payload = _client_inquiry_payload(server_turn.output)
        retry_client_ids = partial_profile["retry_client_ids"]
        reused_responses = partial_profile["reused_responses"]
        reused_guards = partial_profile["reused_guards"]
        job_id = fl_ctx.get_job_id() or session
        correlation_id = str(uuid4())

        outgoing_guardrail = server_guardrail.inspect(
            direction="outgoing",
            check=GuardrailCheck(
                role="server_agent",
                source="human_user",
                channel="server_console",
                action="SERVER.DEFINE_PROFILE_REQUEST",
                phase="data_profile",
                input_schema="TaskSpec@v1",
                output_schema="AgenticFLClientInquiry@v1",
            ),
            payload=client_payload,
            counterpart="client_agents",
        )
        logger.write_server_event(
            session_id=session,
            phase_id="profile",
            step_id="profile.server_inquiry.001",
            correlation_id=correlation_id,
            event_type="SERVER_AGENT_OUTPUT",
            payload=server_turn.output,
            schema_ref=CLIENT_INQUIRY_SCHEMA,
            privacy_class="policy_metadata",
            visibility="cross_site_redacted",
            status="resumed",
            policy=server_turn.guardrail,
            next_step={
                "action": "retry_incomplete_profile_clients",
                "reason": "reuse complete guarded profiles and retry only incomplete profile boundaries",
            },
            agent_prompt=server_turn.prompt,
            flare={
                "job_id": job_id,
                "task_name": self.task_name,
                "targets": retry_client_ids,
                "current_round": ROUND_PROFILE,
                "total_rounds": self.total_rounds,
            },
        )
        logger.write_server_event(
            session_id=session,
            phase_id="profile",
            step_id="profile.server_outgoing_guardrail.001",
            correlation_id=correlation_id,
            event_type="SERVER_OUTGOING_GUARDRAIL_CHECKED",
            payload=outgoing_guardrail.as_payload(),
            schema_ref=GUARDRAIL_SCHEMA,
            privacy_class="policy_metadata",
            visibility="server",
            status="completed",
            policy=outgoing_guardrail.as_policy(),
            next_step={
                "action": "retry_incomplete_profile_clients" if outgoing_guardrail.allowed else "block_broadcast",
                "reason": outgoing_guardrail.reason,
            },
            flare={
                "job_id": job_id,
                "task_name": self.task_name,
                "targets": retry_client_ids,
                "current_round": ROUND_PROFILE,
                "total_rounds": self.total_rounds,
            },
        )
        if not outgoing_guardrail.allowed:
            raise RuntimeError("partial profile resume inquiry failed the server outgoing guardrail")

        replies: dict[str, Shareable | None] = {}
        if retry_client_ids:
            request = to_shareable(
                FlareMessage(
                    session_id=session,
                    correlation_id=correlation_id,
                    schema_ref=CLIENT_INQUIRY_SCHEMA,
                    task_id=f"{job_id}.{self.task_name}",
                    payload=client_payload,
                )
            )
            _set_round_headers(
                request,
                current_round=ROUND_PROFILE,
                total_rounds=self.total_rounds,
                action="profile",
            )
            task = Task(
                name=self.task_name,
                data=request,
                props=_round_props(ROUND_PROFILE, self.total_rounds, "profile"),
                timeout=self.result_wait_timeout,
            )
            self.broadcast_and_wait(
                task=task,
                targets=retry_client_ids,
                min_responses=len(retry_client_ids),
                fl_ctx=fl_ctx,
                wait_time_after_min_received=self.wait_time_after_min_received,
                abort_signal=abort_signal,
            )
            replies = {client_task.client.name: client_task.result for client_task in task.client_tasks}

        site_wise_info: dict[str, Any] = {}
        retry_errors: list[str] = []
        expected_profile_task_id = f"{job_id}.{self.task_name}"
        for index, client_id in enumerate(client_ids, start=1):
            if client_id in reused_responses:
                response = reused_responses[client_id]
                guard_record = reused_guards[client_id]
                guard_payload = guard_record["decision"]
                guard_policy = guard_record["policy"]
                message = None
                response_status = "resumed"
            else:
                reply = replies.get(client_id)
                expected_reply = {
                    "expected_session_id": session,
                    "expected_correlation_id": correlation_id,
                    "expected_schema_ref": CLIENT_RESPONSE_SCHEMA,
                    "expected_task_id": expected_profile_task_id,
                    "expected_site_id": client_id,
                }
                response = _response_from_reply(client_id, reply, **expected_reply)
                message = _message_from_reply(reply, client_id=client_id, **expected_reply)
                incoming_guardrail = server_guardrail.inspect(
                    direction="incoming",
                    check=GuardrailCheck(
                        role="server_agent",
                        source="client_agent_via_flare",
                        channel="flare_task_result",
                        action="SERVER.AGGREGATE_CLIENT_PROFILES",
                        phase="data_profile",
                        input_schema="AgenticFLClientResponse@v1",
                        output_schema="AgenticFLServerState@v1",
                    ),
                    payload=response,
                    counterpart=client_id,
                )
                guard_payload = incoming_guardrail.as_payload()
                guard_policy = incoming_guardrail.as_policy()
                if not incoming_guardrail.allowed:
                    response = _server_guardrail_denial_payload(
                        client_id=client_id,
                        decision=incoming_guardrail,
                    )
                if _profile_response_requires_retry(response):
                    retry_errors.append(client_id)
                response_status = "received" if message is not None else "error"

            logger.write_server_event(
                session_id=session,
                phase_id="profile",
                step_id=f"profile.server_incoming_guardrail.{index:03d}",
                correlation_id=correlation_id,
                event_type="SERVER_INCOMING_GUARDRAIL_CHECKED",
                payload={"client_id": client_id, "guardrail": guard_payload},
                schema_ref=GUARDRAIL_SCHEMA,
                privacy_class="aggregate_stats",
                status="completed",
                policy=guard_policy,
                next_step={
                    "action": "accept_client_response",
                    "reason": (
                        "saved response was previously server-guarded"
                        if client_id in reused_responses
                        else "retried response completed server incoming review"
                    ),
                },
                flare={
                    "job_id": job_id,
                    "task_name": self.task_name,
                    "site_id": client_id,
                    "current_round": ROUND_PROFILE,
                    "total_rounds": self.total_rounds,
                },
            )
            site_wise_info[client_id] = response
            logger.write_server_event(
                session_id=session,
                phase_id="profile",
                step_id=f"profile.server_receive.{index:03d}",
                correlation_id=correlation_id,
                event_type="CLIENT_AGENT_OUTPUT_RECEIVED",
                payload={"client_id": client_id, "response": response},
                schema_ref=CLIENT_RESPONSE_SCHEMA,
                privacy_class="aggregate_stats",
                status=response_status,
                next_step={
                    "action": "update_site_wise_info",
                    "reason": (
                        "saved guarded response reused"
                        if client_id in reused_responses
                        else "retried client response received through NVFlare"
                    ),
                },
                flare={
                    "job_id": job_id,
                    "task_name": self.task_name,
                    "task_id": message.task_id if message is not None else None,
                    "site_id": client_id,
                    "current_round": ROUND_PROFILE,
                    "total_rounds": self.total_rounds,
                },
            )

        if retry_errors:
            raise RuntimeError(
                "partial profile resume still has execution errors for: " + ", ".join(sorted(retry_errors))
            )

        summary_turn = server_agent.summarize_responses(
            task=user_task,
            inquiry=server_turn.output,
            site_wise_info=site_wise_info,
        )
        return {
            "server_turn": server_turn,
            "summary_turn": summary_turn,
            "dispatch_turn": None,
            "site_wise_info": site_wise_info,
            "source_session_id": partial_profile["source_session_id"],
            "profile_digest": payload_digest(
                {
                    "inquiry": server_turn.output,
                    "site_wise_info": site_wise_info,
                    "summary": summary_turn.output,
                }
            ),
            "reused_client_count": len(reused_responses),
            "retried_client_count": len(retry_client_ids),
            "retried_clients": retry_client_ids,
        }


def _strategy_needs_generated_materializer(extraction_strategy: dict[str, Any]) -> bool:
    generated_contract = extraction_strategy.get("generated_data_contract")
    if not isinstance(generated_contract, dict):
        return False
    requirement = generated_contract.get("server_materializer_requirement")
    if isinstance(requirement, dict):
        return requirement.get("required") is not False
    return True


def _safe_materializer_workspace_name(extraction_strategy: dict[str, Any]) -> str:
    digest = payload_digest(
        {
            "generated_data_contract": extraction_strategy.get("generated_data_contract"),
            "strategy_digest": extraction_strategy.get("strategy_digest"),
        }
    )
    return "materializer_" + digest.split(":", 1)[-1][:16]


def _resolve_generated_materializer_entry_path(*, entry_script: str, package_root: Path) -> Path | None:
    entry_path = Path(entry_script).expanduser()
    candidates: list[Path] = []
    if entry_path.is_absolute():
        candidates.append(entry_path)
    else:
        candidates.append(package_root / entry_path)
        parts = entry_path.parts
        if parts and parts[0] in {"code_workspace", "materializer_workspace"}:
            candidates.append(package_root.parent / Path(*parts[1:]))
        candidates.append(package_root.parent / entry_path)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved == package_root or package_root not in resolved.parents:
            continue
        if resolved.is_file():
            return resolved
    return None


def _pack_generated_data_materializer(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("schema_version") != "agenticfl.data_materializer_spec.v1":
        raise ValueError("generated data materializer spec has an unexpected schema_version")
    if spec.get("status") != "implemented":
        raise ValueError("generated data materializer spec did not return status=implemented")
    package_dir = spec.get("package_dir")
    entry_script = spec.get("entry_script")
    if not isinstance(package_dir, str) or not isinstance(entry_script, str):
        raise ValueError("generated data materializer spec missing package_dir or entry_script")
    package_root = Path(package_dir).expanduser().resolve()
    if not package_root.is_dir():
        raise ValueError("generated data materializer package_dir is not readable")
    entry_path = _resolve_generated_materializer_entry_path(
        entry_script=entry_script,
        package_root=package_root,
    )
    if entry_path is None:
        raise ValueError("generated data materializer entry_script does not exist")

    source_files: list[dict[str, Any]] = []
    for file_path in sorted(package_root.rglob("*")):
        if not file_path.is_file():
            continue
        if "__pycache__" in file_path.parts:
            continue
        relative = file_path.relative_to(package_root).as_posix()
        if relative.startswith(".") or "/." in relative:
            continue
        if file_path.stat().st_size > 512_000:
            raise ValueError(f"generated data materializer file is too large: {relative}")
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"generated data materializer file is not UTF-8 text: {relative}") from exc
        source_files.append(
            {
                "path": relative,
                "content": content,
                "sha256": payload_digest(content),
            }
        )
    entry_relative = entry_path.relative_to(package_root).as_posix()
    if not any(entry["path"] == entry_relative for entry in source_files):
        raise ValueError("generated data materializer entry_script was not packaged")
    payload = {
        "schema_version": "agenticfl.generated_data_materializer.v1",
        "status": "implemented",
        "record_type": spec.get("record_type"),
        "sample_manifest": spec.get("sample_manifest"),
        "sample_manifest_format": spec.get("sample_manifest_format"),
        "entry_script": entry_relative,
        "interface": spec.get("interface"),
        "implementation_notes": spec.get("implementation_notes"),
        "source_files": source_files,
        "source_file_count": len(source_files),
        "safe_to_share": False,
    }
    payload["source_digest"] = payload_digest(
        {
            "entry_script": payload["entry_script"],
            "source_files": [{"path": item["path"], "sha256": item["sha256"]} for item in payload["source_files"]],
        }
    )
    return payload


def _profile_response_requires_retry(response: Any, *, client_id: str | None = None) -> bool:
    if not isinstance(response, dict):
        return True
    if client_id is not None and response.get("client_id") != client_id:
        return True
    if response.get("status") == "error":
        return True
    if response.get("status") == "denied":
        return False
    return response.get("data") not in {"applicable", "not applicable"}


def _load_partial_profile_resume_state(
    *,
    profile_run_dir: Path,
    expected_client_ids: list[str],
    task: str,
) -> dict[str, Any]:
    action_dir = profile_run_dir.expanduser().resolve() / "server" / "rounds" / "round_01_profile" / "actions"
    inquiry_events = _read_jsonl_objects(action_dir / "01_server_agent_output.jsonl")
    response_events = _read_jsonl_objects(action_dir / "03_client_agent_output_received.jsonl")
    guard_events = _read_jsonl_objects(action_dir / "02_server_incoming_guardrail_checked.jsonl")

    inquiry_candidates = [
        event
        for event in inquiry_events
        if isinstance(event.get("agent_output"), dict)
        and event["agent_output"].get("schema_version") == "agenticfl.client_inquiry.v1"
    ]
    if not inquiry_candidates:
        raise ValueError(f"profile resume source has no server inquiry event: {profile_run_dir}")
    inquiry_event = inquiry_candidates[-1]
    inquiry = inquiry_event.get("agent_output")
    if not isinstance(inquiry, dict) or inquiry.get("task") != task:
        raise ValueError("profile resume source task does not match the requested task")

    expected = set(expected_client_ids)
    responses: dict[str, dict[str, Any]] = {}
    for event in response_events:
        agent_output = event.get("agent_output")
        if not isinstance(agent_output, dict):
            continue
        client_id = agent_output.get("client_id")
        response = agent_output.get("response")
        if not isinstance(client_id, str) or not isinstance(response, dict):
            raise ValueError("profile resume source contains an invalid client profile response")
        if client_id not in expected:
            raise ValueError(f"profile resume source contains an unexpected response for {client_id}")
        if client_id in responses:
            raise ValueError(f"profile resume source contains duplicate response for {client_id}")
        responses[client_id] = response

    guards: dict[str, dict[str, Any]] = {}
    for event in guard_events:
        agent_output = event.get("agent_output")
        if not isinstance(agent_output, dict):
            continue
        client_id = agent_output.get("client_id")
        decision = agent_output.get("guardrail")
        if not isinstance(client_id, str) or not isinstance(decision, dict):
            raise ValueError("profile resume source contains an invalid server guard decision")
        if client_id not in expected:
            raise ValueError(f"profile resume source contains an unexpected guard decision for {client_id}")
        if client_id in guards:
            raise ValueError(f"profile resume source contains duplicate guard decision for {client_id}")
        policy = event.get("policy")
        guards[client_id] = {
            "decision": decision,
            "policy": (
                policy
                if isinstance(policy, dict)
                else {
                    "guardrail_checked": True,
                    "decision": "allowed" if decision.get("allowed") is True else "denied",
                    "allow_list_rule": decision.get("allow_list_rule"),
                }
            ),
        }

    reusable_responses: dict[str, dict[str, Any]] = {}
    reusable_guards: dict[str, dict[str, Any]] = {}
    retry_client_ids: list[str] = []
    for client_id in expected_client_ids:
        response = responses.get(client_id)
        guard = guards.get(client_id)
        if (
            response is not None
            and guard is not None
            and guard["decision"].get("allowed") is True
            and not _profile_response_requires_retry(response, client_id=client_id)
        ):
            reusable_responses[client_id] = response
            reusable_guards[client_id] = guard
        else:
            retry_client_ids.append(client_id)

    source_policy = inquiry_event.get("policy")
    return {
        "server_turn": AgentTurn(
            prompt=str(inquiry_event.get("agent_prompt") or "resume saved profile inquiry"),
            output=inquiry,
            guardrail=source_policy if isinstance(source_policy, dict) else None,
        ),
        "reused_responses": reusable_responses,
        "reused_guards": reusable_guards,
        "retry_client_ids": retry_client_ids,
        "source_session_id": str(inquiry_event.get("session_id") or profile_run_dir.name),
        "partial_profile_digest": payload_digest(
            {
                "inquiry": inquiry,
                "reused_responses": reusable_responses,
                "retry_client_ids": retry_client_ids,
            }
        ),
    }


def _load_profile_resume_state(
    *,
    profile_run_dir: Path,
    expected_client_ids: list[str],
    task: str,
) -> dict[str, Any]:
    action_dir = profile_run_dir.expanduser().resolve() / "server" / "rounds" / "round_01_profile" / "actions"
    inquiry_events = _read_jsonl_objects(action_dir / "01_server_agent_output.jsonl")
    response_events = _read_jsonl_objects(action_dir / "03_client_agent_output_received.jsonl")
    guard_events = _read_jsonl_objects(action_dir / "02_server_incoming_guardrail_checked.jsonl")
    inquiry_candidates = [
        event
        for event in inquiry_events
        if isinstance(event.get("agent_output"), dict)
        and event["agent_output"].get("schema_version") == "agenticfl.client_inquiry.v1"
    ]
    if not inquiry_candidates:
        raise ValueError(f"profile resume source has no server inquiry event: {profile_run_dir}")
    inquiry_event = inquiry_candidates[-1]
    inquiry = inquiry_event.get("agent_output")
    if not isinstance(inquiry, dict) or inquiry.get("task") != task:
        raise ValueError("profile resume source task does not match the requested task")

    summary_candidates = [
        event
        for event in inquiry_events
        if isinstance(event.get("agent_output"), dict)
        and event["agent_output"].get("schema_version") == "agenticfl.server_site_summary.v1"
    ]
    if not summary_candidates:
        raise ValueError(f"profile resume source has no server profile summary: {profile_run_dir}")
    summary_event = summary_candidates[-1]
    summary = summary_event.get("agent_output")
    if not isinstance(summary, dict) or summary.get("task") != task:
        raise ValueError("profile resume source summary task does not match the requested task")

    responses: dict[str, dict[str, Any]] = {}
    for event in response_events:
        agent_output = event.get("agent_output")
        if not isinstance(agent_output, dict):
            continue
        client_id = agent_output.get("client_id")
        response = agent_output.get("response")
        if not isinstance(client_id, str) or not isinstance(response, dict):
            raise ValueError("profile resume source contains an invalid client profile response")
        if client_id in responses:
            raise ValueError(f"profile resume source contains duplicate response for {client_id}")
        responses[client_id] = response

    allowed_guards: set[str] = set()
    for event in guard_events:
        agent_output = event.get("agent_output")
        if not isinstance(agent_output, dict):
            continue
        client_id = agent_output.get("client_id")
        decision = agent_output.get("guardrail")
        if not isinstance(client_id, str) or not isinstance(decision, dict):
            raise ValueError("profile resume source contains an invalid server guard decision")
        if decision.get("allowed") is not True:
            raise ValueError(f"profile resume source has no allowed inbound guard decision for {client_id}")
        allowed_guards.add(client_id)

    expected = set(expected_client_ids)
    if set(responses) != expected:
        raise ValueError("profile resume source responses do not exactly match the selected client ids")
    if allowed_guards != expected:
        raise ValueError("profile resume source guard decisions do not exactly match the selected client ids")
    ordered_responses = {client_id: responses[client_id] for client_id in expected_client_ids}
    if summary.get("site_wise_info") != ordered_responses:
        raise ValueError("profile resume source summary does not match guarded client responses")
    if not isinstance(summary.get("extraction_strategy"), dict):
        raise ValueError("profile resume source summary is missing extraction_strategy")

    dispatch_turn = None
    dispatch_path = (
        profile_run_dir.expanduser().resolve()
        / "server"
        / "rounds"
        / "round_02_harmonization"
        / "actions"
        / "01_server_agent_output.jsonl"
    )
    if dispatch_path.exists():
        dispatch_events = _read_jsonl_objects(dispatch_path)
        dispatch_candidates = [
            event
            for event in dispatch_events
            if isinstance(event.get("agent_output"), dict)
            and event["agent_output"].get("schema_version") == "agenticfl.extraction_dispatch.v1"
        ]
        if dispatch_candidates:
            dispatch_event = dispatch_candidates[-1]
            dispatch = dispatch_event["agent_output"]
            strategy_digest = summary["extraction_strategy"].get("strategy_digest")
            if dispatch.get("strategy_digest") != strategy_digest:
                raise ValueError("profile resume source dispatch does not match extraction strategy")
            dispatch_policy = dispatch_event.get("policy")
            dispatch_turn = AgentTurn(
                prompt=str(dispatch_event.get("agent_prompt") or "resume saved extraction dispatch"),
                output=dispatch,
                guardrail=dispatch_policy if isinstance(dispatch_policy, dict) else None,
            )
    source_policy = inquiry_event.get("policy")
    summary_policy = summary_event.get("policy")
    return {
        "server_turn": AgentTurn(
            prompt=str(inquiry_event.get("agent_prompt") or "resume saved profile inquiry"),
            output=inquiry,
            guardrail=source_policy if isinstance(source_policy, dict) else None,
        ),
        "summary_turn": AgentTurn(
            prompt=str(summary_event.get("agent_prompt") or "resume saved profile summary"),
            output=summary,
            guardrail=summary_policy if isinstance(summary_policy, dict) else None,
        ),
        "dispatch_turn": dispatch_turn,
        "site_wise_info": ordered_responses,
        "source_session_id": str(inquiry_event.get("session_id") or profile_run_dir.name),
        "profile_digest": payload_digest({"inquiry": inquiry, "site_wise_info": ordered_responses, "summary": summary}),
    }


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"profile resume source is missing required artifact: {path}") from exc
    events: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"profile resume source contains invalid JSONL: {path}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"profile resume source contains a non-object event: {path}")
        events.append(value)
    return events
