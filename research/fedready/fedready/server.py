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

"""Server-side NVFlare controller for FedReady orchestration rounds."""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

from fedready.agents.bridge import CodexWorkerBackend
from fedready.agents.guardrail import ACTION_CHECKS, GuardrailAgent
from fedready.agents.roles import ServerAgent
from fedready.data.extractor import ExtractionConfig
from fedready.data.parser import list_client_ids
from fedready.flare.channel import (
    CLIENT_INQUIRY_SCHEMA,
    CLIENT_RESPONSE_SCHEMA,
    EXTRACTION_POLICY_SCHEMA,
    EXTRACTION_RESPONSE_SCHEMA,
    EXTRACTION_SUMMARY_SCHEMA,
    FEDREADY_MESSAGE_KEY,
    GUARDRAIL_SCHEMA,
    HEADER_CORRELATION_ID,
    HEADER_CURRENT_ROUND,
    HEADER_ROUND_ACTION,
    HEADER_SCHEMA_REF,
    HEADER_SESSION_ID,
    HEADER_SITE_ID,
    HEADER_TASK_ID,
    ROUND_EXTRACTION,
    ROUND_PROFILE,
    SERVER_STATE_SCHEMA,
    SIMULATION_SCHEMA_VERSION,
    TASK_QUERY_TASK_NAME,
    FlareMessage,
    from_shareable,
)
from fedready.flare.channel import round_props as _round_props
from fedready.flare.channel import set_fl_ctx_round as _set_fl_ctx_round
from fedready.flare.channel import set_round_headers as _set_round_headers
from fedready.flare.channel import to_shareable
from fedready.utils.logging import FlowLogger, payload_digest

from nvflare.apis.controller_spec import Task, TaskCompletionStatus
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal


class FedReadyTaskQueryController(Controller):
    """Server-side FedReady controller for a bounded task-query job."""

    def __init__(
        self,
        site_meta_path: str,
        session_id: str,
        task: str = "<TASK_DESCRIPTION>",
        output_dir: str = "runs",
        max_clients: int | None = None,
        task_name: str = TASK_QUERY_TASK_NAME,
        result_wait_timeout: int = 600,
        wait_time_after_min_received: int = 0,
        extraction_output_root: str = ExtractionConfig.output_root,
        extraction_output_name: str | None = None,
        extraction_max_samples: int | None = None,
        extraction_overwrite: bool = False,
        extraction_validation_fraction: float = ExtractionConfig.validation_fraction,
        agent_timeout_seconds: float = 3600.0,
    ) -> None:
        super().__init__()
        self.site_meta_path = site_meta_path
        self.task = task
        self.output_dir = output_dir
        self.session_id = session_id
        self.max_clients = max_clients
        self.task_name = task_name
        self.result_wait_timeout = result_wait_timeout
        self.wait_time_after_min_received = wait_time_after_min_received
        self.extraction_output_root = extraction_output_root
        self.extraction_output_name = extraction_output_name
        self.extraction_max_samples = extraction_max_samples
        self.extraction_overwrite = extraction_overwrite
        self.extraction_validation_fraction = extraction_validation_fraction
        self.agent_timeout_seconds = agent_timeout_seconds

    def start_controller(self, fl_ctx: FLContext) -> None:
        self.log_info(fl_ctx, "FedReady task-query controller started.")

    def stop_controller(self, fl_ctx: FLContext) -> None:
        self.log_info(fl_ctx, "FedReady task-query controller stopped.")

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        if abort_signal.triggered:
            return

        session = self.session_id
        run_dir = Path(self.output_dir) / session
        logger = FlowLogger(run_dir)
        user_task = {"task": self.task}

        client_ids = _server_visible_clients(self.site_meta_path, self.max_clients)
        client_ids = _filter_available_clients(client_ids, fl_ctx)
        if not client_ids:
            self.system_panic("No available FedReady clients matched the server-visible client id list.", fl_ctx)
            return

        agent_backend = CodexWorkerBackend.from_env(
            run_dir=run_dir,
            session_id=session,
            timeout_seconds=self.agent_timeout_seconds,
        )
        server_agent = ServerAgent(agent_backend)
        server_guardrail = GuardrailAgent(party_role="server", party_id="server", agent_backend=agent_backend)
        _set_fl_ctx_round(fl_ctx, ROUND_PROFILE)
        server_turn = server_agent.compose_client_inquiry(task=user_task, client_ids=client_ids)
        client_payload = _client_inquiry_payload(server_turn.output)
        job_id = fl_ctx.get_job_id() or session
        correlation_id = str(uuid4())
        outgoing_guardrail = server_guardrail.inspect(
            direction="outgoing",
            check=ACTION_CHECKS["SERVER.DEFINE_PROFILE_REQUEST"],
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
        _set_round_headers(request, current_round=ROUND_PROFILE, action="profile")

        write_event = partial(logger.write_server_event, session_id=session, phase_id="profile")
        write_event(
            step_id="profile.server_inquiry.001",
            correlation_id=correlation_id,
            event_type="SERVER_AGENT_OUTPUT",
            payload=server_turn.output,
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
            },
        )
        write_event(
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
            props=_round_props(ROUND_PROFILE, "profile"),
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
            response, message = _read_reply(client_id, reply, **expected_reply)
            incoming_guardrail = server_guardrail.inspect(
                direction="incoming",
                check=ACTION_CHECKS["SERVER.AGGREGATE_CLIENT_PROFILES"],
                payload=response,
                counterpart=client_id,
            )
            write_event(
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
                },
            )
            if not incoming_guardrail.allowed:
                response = _server_guardrail_denial_payload(client_id=client_id, decision=incoming_guardrail)
            site_wise_info[client_id] = response
            write_event(
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
                "message_key": FEDREADY_MESSAGE_KEY,
                "headers": [
                    HEADER_SESSION_ID,
                    HEADER_CORRELATION_ID,
                    HEADER_SCHEMA_REF,
                    HEADER_TASK_ID,
                    HEADER_SITE_ID,
                    HEADER_CURRENT_ROUND,
                    HEADER_ROUND_ACTION,
                ],
            },
            "rounds": {
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

        write_event(
            step_id="profile.server_summary.001",
            correlation_id=correlation_id,
            event_type="SERVER_AGENT_OUTPUT",
            payload=summary_turn.output,
            schema_ref=SERVER_STATE_SCHEMA,
            privacy_class="aggregate_stats",
            status="completed",
            policy=summary_turn.guardrail,
            next_step={"action": "dispatch_extraction_policies", "reason": "profile round completed"},
            agent_prompt=summary_turn.prompt,
            flare={
                "job_id": job_id,
                "task_name": self.task_name,
                "current_round": ROUND_PROFILE,
            },
        )

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
    ) -> None:
        if abort_signal.triggered:
            return

        write_event = partial(logger.write_server_event, session_id=session, phase_id="harmonization")
        _set_fl_ctx_round(fl_ctx, ROUND_EXTRACTION)
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
            write_event(
                step_id="harmonization.server_materializer.001",
                correlation_id=str(uuid4()),
                event_type="SERVER_AGENT_OUTPUT",
                payload={
                    "schema_version": "fedready.generated_data_materializer_event.v1",
                    "status": materializer_payload.get("status"),
                    "record_type": materializer_payload.get("record_type"),
                    "source_file_count": len(materializer_payload.get("source_files") or []),
                    "source_digest": materializer_payload.get("source_digest"),
                },
                schema_ref="FedReadyDataMaterializerSpec@v1",
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
                },
            )

        dispatch_turn = server_agent.compose_extraction_dispatch(
            task=user_task,
            extraction_strategy=extraction_strategy,
            target_client_ids=target_clients,
            extraction_config=extraction_config,
        )
        policies = dispatch_turn.output["policies"]
        target_clients = [client_id for client_id in target_clients if client_id in policies]
        correlation_id = str(uuid4())

        logger.write_artifact("server/requests/extraction_policies.json", dispatch_turn.output)
        write_event(
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
            },
        )

        guarded_target_clients: list[str] = []
        blocked_extraction_results: dict[str, Any] = {}
        for index, client_id in enumerate(target_clients, start=1):
            site_policy = policies[client_id]
            outgoing_guardrail = server_guardrail.inspect(
                direction="outgoing",
                check=ACTION_CHECKS["SERVER.DISPATCH_EXTRACTION_POLICY"],
                payload=site_policy,
                counterpart=client_id,
            )
            write_event(
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
                action="extract",
            )
            task = Task(
                name=self.task_name,
                data=request,
                props=_round_props(ROUND_EXTRACTION, "extract"),
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
            response, message = _read_reply(client_id, reply, **expected_reply)
            incoming_guardrail = server_guardrail.inspect(
                direction="incoming",
                check=ACTION_CHECKS["SERVER.AGGREGATE_EXTRACTION_RESULTS"],
                payload=response,
                counterpart=client_id,
            )
            write_event(
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
                },
            )
            if not incoming_guardrail.allowed:
                response = _server_guardrail_denial_payload(client_id=client_id, decision=incoming_guardrail)
            extraction_results[client_id] = response
            write_event(
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
        write_event(
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


def _read_reply(
    client_id: str,
    reply: Shareable | None,
    *,
    expected_session_id: str | None = None,
    expected_correlation_id: str | None = None,
    expected_schema_ref: str | None = None,
    expected_task_id: str | None = None,
    expected_site_id: str | None = None,
) -> tuple[dict[str, Any], FlareMessage | None]:
    if reply is None:
        return {"status": "error", "client_id": client_id, "reason_code": ReturnCode.EMPTY_RESULT}, None
    rc = reply.get_return_code()
    if rc != ReturnCode.OK:
        return {"status": "error", "client_id": client_id, "reason_code": rc}, None
    try:
        message = _validated_reply_message(
            client_id=client_id,
            reply=reply,
            expected_session_id=expected_session_id,
            expected_correlation_id=expected_correlation_id,
            expected_schema_ref=expected_schema_ref,
            expected_task_id=expected_task_id,
            expected_site_id=expected_site_id,
        )
        return message.payload, message
    except Exception as exc:  # noqa: BLE001 - server records malformed client replies as bounded errors.
        return {
            "status": "error",
            "client_id": client_id,
            "reason_code": "MALFORMED_REPLY",
            "reason": f"Malformed FLARE reply ({type(exc).__name__}).",
        }, None


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
    if spec.get("schema_version") != "fedready.data_materializer_spec.v1":
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
        "schema_version": "fedready.generated_data_materializer.v1",
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
