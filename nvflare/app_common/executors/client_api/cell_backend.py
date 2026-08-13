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

"""Shared CJ-side Cell protocol machinery for out-of-process Client API modes."""

import threading
import time
from collections import OrderedDict
from typing import Optional, Tuple

from nvflare.apis.fl_constant import FLContextKey, FLMetaKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.utils.analytix_utils import create_analytic_dxo
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext, ClientAPIBackendSpec
from nvflare.client.cell.defs import CHANNEL, MsgKey, ResultState, Topic
from nvflare.client.config import ConfigKey
from nvflare.client.decomposers import register_framework_decomposers
from nvflare.fuel.data_event.utils import get_scope_property
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply as make_cell_reply
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.security.logging import secure_format_traceback

_RESULT_AUTHORITY_LIMIT = 256


class CellSession:
    """Protocol state shared by owned and attached trainer sessions."""

    def __init__(self, trainer_fqcn: str, session_id: Optional[str] = None):
        self.trainer_fqcn = trainer_fqcn
        self.session_id = session_id
        self.ready = threading.Event()
        self.result_source_live = threading.Event()
        self.result_accepted = threading.Event()
        self.result_source_task_id: Optional[str] = None
        self._activity_lock = threading.Lock()
        self._last_peer_activity: Optional[float] = None

    def touch(self) -> None:
        with self._activity_lock:
            self._last_peer_activity = time.monotonic()

    def silent_for(self) -> Optional[float]:
        with self._activity_lock:
            last = self._last_peer_activity
        return None if last is None else max(0.0, time.monotonic() - last)

    # Preserve the external-process backend's established terminology.
    touch_peer_activity = touch
    peer_silent_for = silent_for


class CellTask:
    """Correlation state for the one task admitted by a backend."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.result_ready = threading.Event()
        self.result: Optional[Shareable] = None
        self.result_id: Optional[str] = None
        self.accepted_attempt_id: Optional[str] = None
        self.result_receiver_ids = ()


class CellBackendBase(ClientAPIBackendSpec):
    """Common Cell setup and trainer-to-CJ protocol callbacks.

    Subclasses own rendezvous and liveness. Attach additionally enables result
    attempt correlation so lost acceptance replies can be recovered safely.
    """

    result_attempts = False

    def __init__(self):
        super().__init__()
        self.logger = get_obj_logger(self)
        self._context: Optional[ClientAPIBackendContext] = None
        self._engine = None
        self._cell = None
        self._cj_fqcn: Optional[str] = None
        self._job_id: Optional[str] = None
        self._site_name: Optional[str] = None
        self._secure_mode = False
        self._protocol_secure = False
        self._site_auth_token: Optional[str] = None
        self._site_auth_token_signature: Optional[str] = None
        self._owned_pass_through_routes: set[Tuple[str, str]] = set()
        self._current_task: Optional[CellTask] = None
        self._task_lock = threading.Lock()
        self._execute_gate = threading.Lock()
        self._result_authority = OrderedDict()
        self._closed = False
        self._finalized = False

    def _initialize_cell(
        self,
        context: ClientAPIBackendContext,
        fl_ctx: FLContext,
        mode: str,
        pass_through_routes: Tuple[Tuple[str, str], ...] = (),
        delegate_site_auth: bool = False,
    ) -> None:
        self._context = context
        self._engine = fl_ctx.get_engine()
        if self._engine is None:
            raise RuntimeError("no engine available in fl_ctx")
        self._cell = self._engine.get_cell()
        if self._cell is None:
            raise RuntimeError(f"no Cell available from the engine: {mode} mode requires the CJ cell")
        self._cj_fqcn = self._cell.get_fqcn()
        self._job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID)
        self._site_name = fl_ctx.get_identity_name()
        if not self._job_id or not self._site_name:
            raise RuntimeError("job id/site name not available in fl_ctx")

        self._secure_mode = bool(fl_ctx.get_prop(FLContextKey.SECURE_MODE, False))
        if self._secure_mode and delegate_site_auth:
            self._site_auth_token = self._get_site_auth_value(FLMetaKey.AUTH_TOKEN)
            self._site_auth_token_signature = self._get_site_auth_value(FLMetaKey.AUTH_TOKEN_SIGNATURE)
        for route in pass_through_routes:
            if route not in self._cell.decode_pass_through_topics:
                self._owned_pass_through_routes.add(route)
            self._cell.decode_pass_through_topics.add(route)

        self._register_common_protocol_cbs()
        register_framework_decomposers(context.params_exchange_format, context.server_expected_format, self.logger)
        context.executor.set_analytics_fire_fed_event(True)

    def _get_site_auth_value(self, key: str) -> str:
        value = get_scope_property(scope_name=self._site_name, key=key)
        if not isinstance(value, str) or not value or value == "NA":
            raise RuntimeError(f"secure Client API session cannot delegate missing site credential {key!r}")
        return value

    def _session_security_payload(self) -> dict:
        payload = {MsgKey.SECURE_MODE: self._secure_mode}
        if self._secure_mode:
            payload.update(
                {
                    MsgKey.AUTH_TOKEN: self._site_auth_token,
                    MsgKey.AUTH_TOKEN_SIGNATURE: self._site_auth_token_signature,
                }
            )
        return payload

    def _register_common_protocol_cbs(self) -> None:
        self._cell.register_request_cb(channel=CHANNEL, topic=Topic.RESULT_READY, cb=self._handle_result_ready)
        self._cell.register_request_cb(
            channel=CHANNEL, topic=Topic.RESULT_SOURCE_SETTLED, cb=self._handle_result_source_settled
        )
        if self.result_attempts:
            self._cell.register_request_cb(channel=CHANNEL, topic=Topic.RESULT_STATUS, cb=self._handle_result_status)
        self._cell.register_request_cb(channel=CHANNEL, topic=Topic.LOG, cb=self._handle_log)
        self._cell.register_request_cb(channel=CHANNEL, topic=Topic.HEARTBEAT, cb=self._handle_heartbeat)

    def _get_protocol_session(self) -> Optional[CellSession]:
        raise NotImplementedError

    def _validate_session_msg(self, request, payload) -> Tuple[Optional[CellSession], Optional[str]]:
        session = self._get_protocol_session()
        if (
            session is None
            or not session.ready.is_set()
            or session.session_id is None
            or getattr(session, "error", None)
        ):
            return None, "no active trainer session"
        origin = request.get_header(MessageHeaderKey.ORIGIN) or ""
        if origin != session.trainer_fqcn:
            return None, f"unexpected origin {origin!r}"
        if payload.get(MsgKey.SESSION_ID) != session.session_id:
            return None, "stale or unknown session id"
        session.touch()
        return session, None

    def _handle_result_ready(self, request):
        if self._closed:
            return self._protocol_reply(Topic.RESULT_REJECTED, **{MsgKey.REASON: "backend is closed"})
        payload = request.payload
        if not isinstance(payload, dict):
            return make_cell_reply(CellReturnCode.INVALID_REQUEST, error="RESULT_READY payload must be a dict")
        session, reason = self._validate_session_msg(request, payload)
        if reason:
            self.logger.warning(f"rejecting RESULT_READY: {reason}")
            return self._protocol_reply(Topic.RESULT_REJECTED, **{MsgKey.REASON: reason})

        task_id = payload.get(MsgKey.TASK_ID)
        result = payload.get(MsgKey.RESULT)
        result_id = payload.get(MsgKey.RESULT_ID)
        attempt_id = payload.get(MsgKey.ATTEMPT_ID)
        if self.result_attempts and not all(isinstance(v, str) and v for v in (task_id, result_id, attempt_id)):
            return self._protocol_reply(
                Topic.RESULT_REJECTED,
                **{MsgKey.REASON: "missing result correlation fields"},
            )

        with self._task_lock:
            if self._closed:
                return self._protocol_reply(Topic.RESULT_REJECTED, **{MsgKey.REASON: "backend is closed"})
            if self.result_attempts:
                authority = self._result_authority.get(task_id)
                if authority is not None:
                    accepted_result_id, accepted_attempt_id = authority
                    if accepted_result_id == result_id:
                        return self._protocol_reply(
                            Topic.RESULT_ACCEPTED,
                            **{
                                MsgKey.RESULT_ID: result_id,
                                MsgKey.ACCEPTED_ATTEMPT_ID: accepted_attempt_id,
                            },
                        )
                    return self._protocol_reply(
                        Topic.RESULT_REJECTED,
                        **{MsgKey.REASON: "another result is canonical"},
                    )

            task = self._current_task
            if task is None or task.task_id != task_id:
                reason = f"no current task matching task_id {task_id!r}"
                self.logger.warning(f"rejecting RESULT_READY: {reason}")
                return self._protocol_reply(
                    Topic.RESULT_REJECTED,
                    **{MsgKey.REASON: reason},
                )
            if task.result is not None:
                return self._protocol_reply(
                    Topic.RESULT_REJECTED,
                    **{MsgKey.REASON: "a result was already accepted for this task"},
                )
            if not isinstance(result, Shareable):
                return self._protocol_reply(
                    Topic.RESULT_REJECTED,
                    **{MsgKey.REASON: "invalid result envelope: Shareable result required"},
                )
            task.result = result
            task.result_id = result_id
            task.accepted_attempt_id = attempt_id
            if self.result_attempts:
                self._remember_result_authority(task_id, result_id, attempt_id)
            session.result_source_live.set()
            session.result_accepted.set()
            session.result_source_task_id = task_id
            self._on_result_accepted(session, task, result)
            task.result_ready.set()

        fields = {}
        if self.result_attempts:
            fields = {MsgKey.RESULT_ID: result_id, MsgKey.ACCEPTED_ATTEMPT_ID: attempt_id}
        return self._protocol_reply(Topic.RESULT_ACCEPTED, **fields)

    def _handle_result_status(self, request):
        payload = request.payload
        if not isinstance(payload, dict):
            return make_cell_reply(CellReturnCode.INVALID_REQUEST, error="RESULT_STATUS payload must be a dict")
        _, reason = self._validate_session_msg(request, payload)
        if reason:
            return self._protocol_reply(Topic.ERROR, **{MsgKey.REASON: reason})
        task_id = payload.get(MsgKey.TASK_ID)
        result_id = payload.get(MsgKey.RESULT_ID)
        with self._task_lock:
            authority = self._result_authority.get(task_id)
        if authority is None or authority[0] != result_id:
            return self._protocol_reply(Topic.RESULT_STATUS, **{MsgKey.RESULT_STATE: ResultState.UNKNOWN})
        return self._protocol_reply(
            Topic.RESULT_STATUS,
            **{
                MsgKey.RESULT_STATE: ResultState.ACCEPTED,
                MsgKey.ACCEPTED_ATTEMPT_ID: authority[1],
            },
        )

    def _handle_result_source_settled(self, request):
        payload = request.payload
        if not isinstance(payload, dict):
            return make_cell_reply(CellReturnCode.INVALID_REQUEST, error="RESULT_SOURCE_SETTLED payload must be a dict")
        session, reason = self._validate_session_msg(request, payload)
        if reason:
            return self._protocol_reply(Topic.ERROR, **{MsgKey.REASON: reason})
        task_id = payload.get(MsgKey.TASK_ID)
        with self._task_lock:
            if not task_id or session.result_source_task_id != task_id:
                return self._protocol_reply(
                    Topic.ERROR,
                    **{MsgKey.REASON: f"no live result source matching task_id {task_id!r}"},
                )
            session.result_source_live.clear()
            session.result_source_task_id = None
            self._on_result_source_settled(session, task_id)
            if self.result_attempts:
                self._trim_result_authority()
        return self._protocol_reply(Topic.RESULT_SOURCE_SETTLED, **{MsgKey.TASK_ID: task_id})

    def _on_result_accepted(self, session: CellSession, task: CellTask, result: Shareable) -> None:
        """Hook for mode-specific ownership of an accepted result source."""

    def _on_result_source_settled(self, session: CellSession, task_id: str) -> None:
        """Hook for mode-specific retirement of accepted result-source state."""

    def _remember_result_authority(self, task_id: str, result_id: str, attempt_id: str) -> None:
        self._result_authority[task_id] = (result_id, attempt_id)
        self._result_authority.move_to_end(task_id)
        self._trim_result_authority()

    def _trim_result_authority(self) -> None:
        session = self._get_protocol_session()
        if session is not None and session.result_source_live.is_set():
            return
        while len(self._result_authority) > _RESULT_AUTHORITY_LIMIT:
            self._result_authority.popitem(last=False)

    def _handle_log(self, request):
        if self._closed:
            return None
        try:
            payload = request.payload
            if not isinstance(payload, dict):
                self.logger.error(f"invalid LOG data format, expecting Dict, but got {type(payload)}")
                return None
            _, reason = self._validate_session_msg(request, payload)
            if reason:
                self.logger.warning(f"dropping LOG data: {reason}")
                return None
            record = {k: v for k, v in payload.items() if k != MsgKey.SESSION_ID}
            if "key" in record:
                record["tag"] = record.pop("key")
            dxo = create_analytic_dxo(**record)
            with self._engine.new_context() as fl_ctx:
                self._context.executor.fire_log_analytics(fl_ctx, dxo)
        except Exception:
            self.logger.error(f"failed to process trainer LOG data: {secure_format_traceback()}")
        return None

    def _handle_heartbeat(self, request):
        if self._closed:
            return self._protocol_reply(Topic.ERROR, **{MsgKey.REASON: "backend is closed"})
        payload = request.payload
        if not isinstance(payload, dict):
            return make_cell_reply(CellReturnCode.INVALID_REQUEST, error="HEARTBEAT payload must be a dict")
        session, reason = self._validate_session_msg(request, payload)
        if reason:
            self.logger.warning(f"rejecting HEARTBEAT: {reason}")
            return self._protocol_reply(Topic.ERROR, **{MsgKey.REASON: reason})
        # Only attach uses heartbeat source-liveness for result-attempt authority.
        # External-process teardown owns its result barrier and must not have that
        # already-shipped behavior changed by a delayed heartbeat.
        if self.result_attempts:
            source_live = payload.get(MsgKey.RESULT_SOURCE_LIVE)
            if source_live is True:
                session.result_source_live.set()
            elif source_live is False:
                session.result_source_live.clear()
                session.result_source_task_id = None
                with self._task_lock:
                    self._trim_result_authority()
        return self._protocol_reply(Topic.HEARTBEAT, **{MsgKey.SESSION_ID: session.session_id})

    def _send_abort(self, session: Optional[CellSession], reason: str) -> None:
        if session is None or session.session_id is None:
            return
        try:
            self._cell.fire_and_forget(
                channel=CHANNEL,
                topic=Topic.ABORT,
                targets=[session.trainer_fqcn],
                message=new_cell_message({}, {MsgKey.SESSION_ID: session.session_id, MsgKey.REASON: reason}),
                optional=True,
                secure=self._protocol_secure,
            )
        except Exception:
            self.logger.error(secure_format_traceback())

    def _task_exchange_config(self) -> dict:
        context = self._context
        return {
            ConfigKey.TRAIN_WITH_EVAL: context.train_with_evaluation,
            # Keep bootstrap/session control payloads independent of Python enum
            # registration. JSON happened to serialize these str enums as their
            # values in launched mode, but FOBS preserves their Python type and
            # rejects them in a bare attached trainer.
            ConfigKey.EXCHANGE_FORMAT: context.params_exchange_format.value,
            ConfigKey.SERVER_EXPECTED_FORMAT: context.server_expected_format.value,
            ConfigKey.TRANSFER_TYPE: context.params_transfer_type.value,
            ConfigKey.TRAIN_TASK_NAME: context.train_task_name,
            ConfigKey.EVAL_TASK_NAME: context.evaluate_task_name,
            ConfigKey.SUBMIT_MODEL_TASK_NAME: context.submit_model_task_name,
            ConfigKey.LAUNCH_ONCE: self._launch_once_config(),
        }

    def _launch_once_config(self) -> bool:
        return self._context.launch_once

    @staticmethod
    def _protocol_reply(reply_topic: str, **fields):
        body = {MsgKey.REPLY_TOPIC: reply_topic}
        body.update(fields)
        return make_cell_reply(CellReturnCode.OK, body=body)

    def _disable_pass_through(self) -> None:
        if self._cell is not None:
            for route in self._owned_pass_through_routes:
                self._cell.decode_pass_through_topics.discard(route)
        self._owned_pass_through_routes.clear()
        self._site_auth_token = None
        self._site_auth_token_signature = None
