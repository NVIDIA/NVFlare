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

"""Non-owning Client API backend for an externally started trainer."""

import threading
import time
import uuid
from typing import Optional, Tuple

from nvflare.apis.fl_constant import FLMetaKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext
from nvflare.app_common.executors.client_api.cell_backend import CellBackendBase, CellSession, CellTask
from nvflare.client.cell.attach import make_attach_trainer_fqcn, validate_attach_transport
from nvflare.client.cell.defs import CHANNEL, PROTOCOL_VERSION, MsgKey, TaskState, Topic
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.f3.streaming.download_service import DownloadService
from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.fuel.utils.fobs.decomposers.via_downloader import RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY
from nvflare.security.logging import secure_format_traceback

_CONTROL_RETRY_INTERVAL = 0.5
_CONTROL_ATTEMPT_TIMEOUT = 2.0
_RESULT_POLL_INTERVAL = 0.25
_SESSION_MONITOR_INTERVAL = 0.2
_DEFAULT_ATTACH_TASK_TIMEOUT = 600.0


class _AttachCancelSignal(Signal):
    """Signal that also observes task abort, backend close, and session loss."""

    def __init__(self, abort_signal: Signal, backend: "AttachBackend", session_id: str):
        super().__init__()
        self._abort_signal = abort_signal
        self._backend = backend
        self._session_id = session_id

    @property
    def triggered(self):
        if super().triggered:
            return True
        if self._abort_signal.triggered:
            self.trigger("task aborted")
        elif self._backend._closed:
            self.trigger("backend closed")
        elif not self._backend._session_matches(self._session_id):
            self.trigger("attach session lost")
        return super().triggered


class _AttachedTrainerSession(CellSession):
    def __init__(self, trainer_fqcn: str):
        super().__init__(trainer_fqcn, uuid.uuid4().hex)
        self.error: Optional[str] = None


class AttachBackend(CellBackendBase):
    """Runs the Cell task protocol without owning the external trainer process."""

    result_attempts = True

    def __init__(self):
        super().__init__()
        self._trainer_fqcn: Optional[str] = None
        self._session: Optional[_AttachedTrainerSession] = None
        self._session_lock = threading.Lock()
        self._session_stop = threading.Event()
        self._session_thread: Optional[threading.Thread] = None
        self._attach_deadline: Optional[float] = None

    # ------------------------------------------------------------------ lifecycle

    def initialize(self, context: ClientAPIBackendContext, fl_ctx: FLContext) -> None:
        if not context.attach_id:
            raise ValueError("attach mode requires attach_id")
        if context.heartbeat_timeout == 0 and context.result_wait_timeout is None:
            raise ValueError(
                "attach mode requires heartbeat_timeout > 0 or a finite result_wait_timeout "
                "because it does not own a trainer process for liveness detection"
            )
        try:
            self._initialize_cell(context, fl_ctx, "attach")
            self._trainer_fqcn = make_attach_trainer_fqcn(self._site_name, context.attach_id)

            timeout = context.attach_timeout
            self._attach_deadline = None if timeout is None else time.monotonic() + timeout
            with self._session_lock:
                self._session = _AttachedTrainerSession(self._trainer_fqcn)
            self._session_thread = threading.Thread(
                target=self._session_loop,
                name=f"client_api_attach_{context.attach_id}",
                daemon=True,
            )
            self._session_thread.start()
            context.executor.log_info(
                fl_ctx,
                f"waiting for attached trainer {self._trainer_fqcn}",
            )
        except BaseException:
            self._unwind()
            raise

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        executor = self._context.executor
        if self._closed:
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        if not self._execute_gate.acquire(blocking=False):
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            executor.log_error(fl_ctx, f"an attach task is already active; rejecting concurrent task {task_name!r}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        try:
            session = self._wait_for_session(abort_signal)
            if session is None:
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                reason = self._session_error() or "attach timeout expired"
                executor.log_error(fl_ctx, f"cannot run {task_name!r}: {reason}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            return self._run_task(session, task_name, shareable, fl_ctx, abort_signal)
        except Exception:
            executor.log_error(fl_ctx, secure_format_traceback())
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        finally:
            self._execute_gate.release()

    def finalize(self, fl_ctx: FLContext) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._closed = True
        self._session_stop.set()
        session = self._get_session()
        if session is not None and session.ready.is_set():
            self._request_shutdown(session)
        thread = self._session_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=_CONTROL_ATTEMPT_TIMEOUT + _SESSION_MONITOR_INTERVAL)
        self._disable_pass_through()

    # ------------------------------------------------------------------ session

    def _session_loop(self) -> None:
        while not self._session_stop.is_set():
            session = self._get_session()
            if session is None:
                return
            if not session.ready.is_set():
                if session.error:
                    return
                if self._deadline_expired():
                    session.error = f"trainer did not attach within attach_timeout={self._context.attach_timeout}s"
                    return
                self._try_session_open(session)
                self._session_stop.wait(_CONTROL_RETRY_INTERVAL)
                continue

            liveness_error = session.error or self._liveness_error(session)
            if liveness_error is None:
                self._session_stop.wait(_SESSION_MONITOR_INTERVAL)
                continue

            reason = liveness_error
            with self._task_lock:
                active_task = self._current_task is not None
            if active_task:
                session.error = reason
                self._session_stop.wait(_SESSION_MONITOR_INTERVAL)
                continue
            if not self._context.allow_reconnect:
                session.error = reason
                return
            with self._session_lock:
                if self._session is session and not self._closed:
                    self._session = _AttachedTrainerSession(self._trainer_fqcn)
                    # Reconnect gets a fresh bound and a fresh timeout budget.
                    timeout = self._context.attach_timeout
                    self._attach_deadline = None if timeout is None else time.monotonic() + timeout

    def _try_session_open(self, session: _AttachedTrainerSession) -> None:
        payload = {
            MsgKey.SESSION_ID: session.session_id,
            MsgKey.ATTACH_ID: self._context.attach_id,
            MsgKey.JOB_ID: self._job_id,
            MsgKey.SITE_NAME: self._site_name,
            MsgKey.TRAINER_FQCN: session.trainer_fqcn,
            MsgKey.PROTOCOL_VERSION: PROTOCOL_VERSION,
            MsgKey.RANK: "0",
            MsgKey.HEARTBEAT_INTERVAL: self._context.heartbeat_interval,
            MsgKey.HEARTBEAT_TIMEOUT: self._context.heartbeat_timeout,
            MsgKey.TASK_EXCHANGE: self._task_exchange_config(),
            MsgKey.MEMORY_GC_ROUNDS: self._context.memory_gc_rounds,
            MsgKey.CUDA_EMPTY_CACHE: self._context.cuda_empty_cache,
        }
        try:
            reply = self._cell.send_request(
                channel=CHANNEL,
                topic=Topic.SESSION_OPEN,
                target=session.trainer_fqcn,
                request=new_cell_message({}, payload),
                timeout=_CONTROL_ATTEMPT_TIMEOUT,
                optional=True,
            )
        except Exception:
            self.logger.debug(f"SESSION_OPEN to {session.trainer_fqcn} not delivered")
            return
        if reply is None or reply.get_header(MessageHeaderKey.RETURN_CODE) != CellReturnCode.OK:
            return
        body = reply.payload
        if not isinstance(body, dict):
            session.error = "invalid SESSION_OPEN reply payload"
            return
        topic = body.get(MsgKey.REPLY_TOPIC)
        if topic == Topic.SESSION_REJECTED:
            session.error = str(body.get(MsgKey.REASON) or "trainer rejected SESSION_OPEN")
            return
        if topic != Topic.SESSION_ACCEPTED or body.get(MsgKey.SESSION_ID) != session.session_id:
            session.error = f"invalid SESSION_OPEN reply topic/session: {topic!r}"
            return
        try:
            validate_attach_transport(
                body.get(MsgKey.CONNECT_URL),
                body.get(MsgKey.CONNECTION_SECURITY),
                self._context.allow_insecure_attach,
            )
        except ValueError as e:
            session.error = str(e)
            self.logger.error(session.error)
            # The trainer has already committed this session. Release it so a
            # policy rejection never leaves flare.init() waiting indefinitely.
            self._request_shutdown(session)
            return
        session.touch()
        session.ready.set()
        self.logger.info(
            f"attached trainer session established: fqcn={session.trainer_fqcn} session_id={session.session_id}"
        )

    def _wait_for_session(self, abort_signal: Signal) -> Optional[_AttachedTrainerSession]:
        while not self._closed and not abort_signal.triggered:
            session = self._get_session()
            if session is None or session.error:
                return None
            if session.ready.wait(_RESULT_POLL_INTERVAL):
                return session
            if self._deadline_expired():
                return None
        return None

    def _get_session(self) -> Optional[_AttachedTrainerSession]:
        with self._session_lock:
            return self._session

    def _get_protocol_session(self) -> Optional[_AttachedTrainerSession]:
        return self._get_session()

    def _session_matches(self, session_id: str) -> bool:
        session = self._get_session()
        return bool(session and session.ready.is_set() and not session.error and session.session_id == session_id)

    def _session_error(self) -> Optional[str]:
        session = self._get_session()
        return None if session is None else session.error

    def _deadline_expired(self) -> bool:
        return self._attach_deadline is not None and time.monotonic() >= self._attach_deadline

    def _liveness_error(self, session: _AttachedTrainerSession) -> Optional[str]:
        timeout = self._context.heartbeat_timeout
        if timeout <= 0:
            return None
        silent_for = session.silent_for()
        if silent_for is not None and silent_for > timeout:
            return f"attached trainer heartbeat timed out after {silent_for:.1f}s (timeout={timeout}s)"
        return None

    # ------------------------------------------------------------------ task execution

    def _run_task(
        self,
        session: _AttachedTrainerSession,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        task = CellTask(uuid.uuid4().hex)
        shareable.set_header(FLMetaKey.JOB_ID, fl_ctx.get_job_id())
        shareable.set_header(FLMetaKey.SITE_NAME, fl_ctx.get_identity_name())
        with self._task_lock:
            self._current_task = task
        try:
            accepted, reason = self._deliver_task(session, task, task_name, shareable, abort_signal)
            if not accepted:
                if abort_signal.triggered:
                    self._send_abort(session, f"task {task_name!r} aborted")
                    return make_reply(ReturnCode.TASK_ABORTED)
                self._context.executor.log_error(fl_ctx, f"trainer did not accept task {task_name!r}: {reason}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            deadline = (
                None
                if self._context.result_wait_timeout is None
                else time.monotonic() + self._context.result_wait_timeout
            )
            while not task.result_ready.wait(_RESULT_POLL_INTERVAL):
                if abort_signal.triggered:
                    self._send_abort(session, f"task {task_name!r} aborted")
                    return make_reply(ReturnCode.TASK_ABORTED)
                if not self._session_matches(session.session_id):
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
                if deadline is not None and time.monotonic() >= deadline:
                    self._send_abort(session, f"task {task_name!r} result timed out")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            with self._task_lock:
                result = task.result
            if not isinstance(result, Shareable):
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
            if current_round is not None:
                result.set_header(AppConstants.CURRENT_ROUND, current_round)
            return result
        finally:
            with self._task_lock:
                if self._current_task is task:
                    self._current_task = None

    def _deliver_task(
        self,
        session: _AttachedTrainerSession,
        task: CellTask,
        task_name: str,
        shareable: Shareable,
        abort_signal: Signal,
    ) -> Tuple[bool, Optional[str]]:
        timeout = self._context.task_wait_timeout
        if timeout is None:
            timeout = _DEFAULT_ATTACH_TASK_TIMEOUT
        deadline = time.monotonic() + timeout
        first_attempt = True
        last_reason = None
        while time.monotonic() < deadline and not abort_signal.triggered:
            if not first_attempt:
                state = self._query_task_status(
                    session,
                    task.task_id,
                    timeout=min(_CONTROL_ATTEMPT_TIMEOUT, max(0.0, deadline - time.monotonic())),
                )
                if state and state != TaskState.UNKNOWN:
                    return True, None
            first_attempt = False
            attempt_id = uuid.uuid4().hex
            waiters = []

            def _on_transaction_created(transaction):
                waiters.append(DownloadService.get_transfer_waiter(transaction.tx_id))

            request = new_cell_message(
                {},
                {
                    MsgKey.SESSION_ID: session.session_id,
                    MsgKey.TASK_ID: task.task_id,
                    MsgKey.ATTEMPT_ID: attempt_id,
                    MsgKey.TASK_NAME: task_name,
                    MsgKey.MODEL: shareable,
                },
            )
            cancel = _AttachCancelSignal(abort_signal, self, session.session_id)
            remaining = max(0.0, deadline - time.monotonic())
            # Preserve a small part of the absolute delivery budget for
            # TASK_STATUS recovery if the acceptance reply is lost.
            status_reserve = min(_CONTROL_ATTEMPT_TIMEOUT, remaining / 2)
            request_timeout = max(0.0, remaining - status_reserve)
            try:
                reply = self._cell.send_request(
                    channel=CHANNEL,
                    topic=Topic.TASK_READY,
                    target=session.trainer_fqcn,
                    request=request,
                    timeout=request_timeout,
                    abort_signal=cancel,
                    receiver_ids=(session.trainer_fqcn,),
                    fobs_ctx_props={
                        FOBSContextKey.STREAM_PROGRESS_CB: lambda **_kwargs: None,
                        RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY: _on_transaction_created,
                    },
                )
                last_reason, terminal = self._check_task_accepted(reply)
                if last_reason is None:
                    return True, None
                if terminal:
                    self._delete_transfers(waiters)
                    return False, last_reason
            except Exception as e:
                last_reason = str(e)
            state = self._query_task_status(
                session,
                task.task_id,
                timeout=min(_CONTROL_ATTEMPT_TIMEOUT, max(0.0, deadline - time.monotonic())),
            )
            if state and state != TaskState.UNKNOWN:
                return True, None
            self._delete_transfers(waiters)
            if cancel.triggered or not self._session_matches(session.session_id):
                break
            self._session_stop.wait(min(_CONTROL_RETRY_INTERVAL, max(0.0, deadline - time.monotonic())))
        return False, last_reason or "task delivery timed out"

    def _query_task_status(
        self,
        session: _AttachedTrainerSession,
        task_id: str,
        timeout: float = _CONTROL_ATTEMPT_TIMEOUT,
    ) -> Optional[str]:
        if timeout <= 0:
            return None
        try:
            reply = self._cell.send_request(
                channel=CHANNEL,
                topic=Topic.TASK_STATUS,
                target=session.trainer_fqcn,
                request=new_cell_message(
                    {},
                    {MsgKey.SESSION_ID: session.session_id, MsgKey.TASK_ID: task_id},
                ),
                timeout=timeout,
                optional=True,
            )
        except Exception:
            return None
        if reply is None or reply.get_header(MessageHeaderKey.RETURN_CODE) != CellReturnCode.OK:
            return None
        body = reply.payload
        return body.get(MsgKey.TASK_STATE) if isinstance(body, dict) else None

    @staticmethod
    def _check_task_accepted(reply) -> Tuple[Optional[str], bool]:
        if reply is None:
            return "no reply from trainer", False
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
        if rc != CellReturnCode.OK:
            return f"cell-level failure delivering TASK_READY: {rc}", False
        body = reply.payload
        if not isinstance(body, dict):
            return f"invalid TASK_READY reply payload: {body!r}", True
        topic = body.get(MsgKey.REPLY_TOPIC)
        if topic == Topic.TASK_ACCEPTED:
            return None, False
        reason = body.get(MsgKey.REASON)
        if topic == Topic.TASK_FAILED:
            return f"trainer rejected TASK_READY: {reason}", True
        return f"invalid TASK_READY reply topic {topic!r}: {reason}", True

    @staticmethod
    def _delete_transfers(waiters) -> None:
        for waiter in waiters:
            try:
                DownloadService.delete_transaction(waiter.transaction_id)
            except Exception:
                pass

    def _request_shutdown(self, session: _AttachedTrainerSession) -> None:
        try:
            reply = self._cell.send_request(
                channel=CHANNEL,
                topic=Topic.SHUTDOWN,
                target=session.trainer_fqcn,
                request=new_cell_message(
                    {},
                    {MsgKey.SESSION_ID: session.session_id, MsgKey.REASON: "job ended"},
                ),
                timeout=_CONTROL_ATTEMPT_TIMEOUT,
                optional=True,
            )
            if reply is not None and isinstance(reply.payload, dict):
                if reply.payload.get(MsgKey.RESULT_SOURCE_LIVE) is False:
                    session.result_source_live.clear()
        except Exception:
            self.logger.debug("attached trainer did not acknowledge SHUTDOWN")

    def _launch_once_config(self) -> bool:
        return True

    def _unwind(self) -> None:
        self._closed = True
        self._session_stop.set()
        self._disable_pass_through()
