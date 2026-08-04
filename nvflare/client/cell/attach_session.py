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

"""Attach-only trainer session protocol used by :class:`CellClientAPI`."""

import atexit
import os
import threading
import time
import uuid
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional

from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.client.cell.attach import make_attach_trainer_fqcn, validate_attach_profile
from nvflare.client.cell.attach_rendezvous import AttachEndpointKey, wait_for_attach_endpoint
from nvflare.client.cell.bootstrap import BootstrapKey
from nvflare.client.cell.defs import CHANNEL, PROTOCOL_VERSION, MsgKey, ResultState, TaskState, Topic
from nvflare.client.config import ConfigKey, ExchangeFormat
from nvflare.client.decomposers import register_framework_decomposers
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.utils import make_reply as make_cell_reply
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.drivers.net_utils import enhance_credential_info
from nvflare.fuel.f3.streaming.download_service import DownloadService
from nvflare.fuel.f3.streaming.stream_const import STREAM_CHANNEL, STREAM_DATA_TOPIC, StreamHeaderKey
from nvflare.fuel.f3.streaming.transfer_progress import TransferProgressState
from nvflare.fuel.utils.fobs.decomposers.via_downloader import RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY

if TYPE_CHECKING:
    from nvflare.apis.shareable import Shareable
    from nvflare.client.cell.api import CellClientAPI

_RESULT_STATUS_TIMEOUT = 2.0
_RESULT_SEND_ATTEMPTS = 2
_AMBIGUOUS_RESULT_POLL_INTERVAL = 0.25
_TASK_LEDGER_LIMIT = 256


class TrainerSessionError(RuntimeError):
    """The trainer's Client API session ended or could not be established."""


class _UncertainResultReply(RuntimeError):
    """RESULT_READY may have committed even though its acknowledgement is unavailable."""


class AttachTrainerSession:
    """Own attach rendezvous, retry, and deduplication state for one trainer API."""

    def __init__(self, api: "CellClientAPI"):
        self._api = api
        config = api._config
        self.attach_id = config[BootstrapKey.ATTACH_ID]
        cj_fqcn = config.get(BootstrapKey.CJ_FQCN)
        self.trainer_fqcn = make_attach_trainer_fqcn(cj_fqcn, self.attach_id) if cj_fqcn else None
        connect_url = config.get(BootstrapKey.CONNECT_URL)
        # A direct network profile is validated immediately. A shared-file
        # profile resolves the CJ-owned listener through its rendezvous record
        # when init() starts, so a trainer may be started before the job.
        self.connection_security = (
            validate_attach_profile(connect_url, config.get(BootstrapKey.CONNECTION_SECURITY)) if connect_url else None
        )
        self._wait_deadline: Optional[float] = None
        self._opened = threading.Event()
        self._closed = threading.Event()
        self._task_states = OrderedDict()
        self._task_attempts = {}
        self._task_sequences = {}
        self._highest_task_sequence = 0
        self._evicted_task_sequence = 0
        self._retryable_task = None
        self._current_result_id: Optional[str] = None
        self._cleanup_registered = False

    def prepare_connection(self) -> str:
        timeout = self._api._config.get(BootstrapKey.JOB_WAIT_TIMEOUT)
        self._wait_deadline = None if timeout is None else time.monotonic() + timeout
        connect_url = self._api._config.get(BootstrapKey.CONNECT_URL)
        if connect_url:
            self._api._cj_fqcn = self._api._config[BootstrapKey.CJ_FQCN]
            self._api._trainer_fqcn = self.trainer_fqcn
            return connect_url

        record = wait_for_attach_endpoint(
            root_dir=self._api._config[BootstrapKey.RENDEZVOUS_DIR],
            site_name=self._api._site_name,
            attach_id=self.attach_id,
            timeout=self._remaining_wait_timeout(),
            stop_event=self._closed,
        )
        self._api._cj_fqcn = record[AttachEndpointKey.CJ_FQCN]
        self.trainer_fqcn = record[AttachEndpointKey.TRAINER_FQCN]
        self._api._trainer_fqcn = self.trainer_fqcn
        connect_url = record[AttachEndpointKey.CONNECT_URL]
        connection_security = record[AttachEndpointKey.CONNECTION_SECURITY]
        self.connection_security = validate_attach_profile(connect_url, connection_security)
        self._api._config[BootstrapKey.CONNECT_URL] = connect_url
        self._api._config[BootstrapKey.CONNECTION_SECURITY] = connection_security
        return connect_url

    def _remaining_wait_timeout(self) -> Optional[float]:
        if self._wait_deadline is None:
            return None
        return max(0.0, self._wait_deadline - time.monotonic())

    def cell_security(self) -> tuple[bool, dict]:
        if self.connection_security is None:
            raise RuntimeError("attach connection was not resolved before Cell construction")
        credentials = {DriverParams.CONNECTION_SECURITY.value: self.connection_security}
        if self.connection_security != ConnectionSecurity.CLEAR:
            ca_cert = self._api._config.get(BootstrapKey.CA_CERT)
            if not ca_cert:
                raise RuntimeError(
                    f"attach profile using {self.connection_security!r} requires {BootstrapKey.CA_CERT!r}"
                )
            credentials[DriverParams.CA_CERT.value] = ca_cert
        if self.connection_security == ConnectionSecurity.MTLS:
            enhance_credential_info(credentials)
            missing = [
                param.value
                for param in (DriverParams.CA_CERT, DriverParams.CLIENT_CERT, DriverParams.CLIENT_KEY)
                if not os.path.isfile(credentials.get(param.value, ""))
                or not os.access(credentials[param.value], os.R_OK)
            ]
            if missing:
                raise RuntimeError(
                    "mTLS attach requires readable ca_cert, client_cert, and client_key files; "
                    f"missing or unreadable: {', '.join(missing)}"
                )
        return self.connection_security != ConnectionSecurity.CLEAR, credentials

    def register_callbacks(self, cell) -> None:
        cell.register_request_cb(channel=CHANNEL, topic=Topic.SESSION_OPEN, cb=self._handle_session_open)
        cell.register_request_cb(channel=CHANNEL, topic=Topic.TASK_STATUS, cb=self._handle_task_status)

    def install_pre_decode_guard(self, cell) -> None:
        """Reject unauthorized attach streams from headers before FOBS decode."""
        core_cell = getattr(cell, "core_cell", None)
        if core_cell is None or not hasattr(core_cell, "set_message_interceptor"):
            raise RuntimeError("attach requires a Cell message interceptor for pre-decode origin authorization")
        core_cell.set_message_interceptor(self._pre_decode_guard)

    def _pre_decode_guard(self, message):
        channel = message.get_header(MessageHeaderKey.CHANNEL, "")
        topic = message.get_header(MessageHeaderKey.TOPIC, "")
        if channel == STREAM_CHANNEL and topic == STREAM_DATA_TOPIC:
            channel = message.get_header(StreamHeaderKey.CHANNEL, "")
            topic = message.get_header(StreamHeaderKey.TOPIC, "")
        if channel != CHANNEL:
            return None

        origin = message.get_header(MessageHeaderKey.ORIGIN) or ""
        with self._api._lock:
            bound_origin = self._api._cj_fqcn
        if bound_origin:
            authorized = origin == bound_origin
        else:
            path = FQCN.split(origin) if isinstance(origin, str) else []
            authorized = topic == Topic.SESSION_OPEN and len(path) == 2 and path[0] == self._api._site_name
        if authorized:
            return None
        return make_cell_reply(
            CellReturnCode.AUTHENTICATION_ERROR,
            error=f"attach message {topic!r} from unauthorized origin {origin!r}",
        )

    def wait_for_open(self) -> None:
        timeout = self._remaining_wait_timeout()
        if not self._opened.wait(timeout):
            configured = self._api._config.get(BootstrapKey.JOB_WAIT_TIMEOUT)
            raise TrainerSessionError(f"no SESSION_OPEN received within job_wait_timeout={configured}s")
        if not self._api._session_id:
            raise TrainerSessionError("SESSION_OPEN wait was interrupted before a session was established")

    def register_cleanup(self) -> None:
        if not self._cleanup_registered:
            atexit.register(self.cleanup)
            self._cleanup_registered = True

    def close(self) -> None:
        if self._cleanup_registered:
            atexit.unregister(self.cleanup)
            self._cleanup_registered = False
        self._closed.set()
        self._opened.set()

    def cleanup(self) -> None:
        """Best-effort interpreter cleanup without preempting a live result source."""
        with self._api._lock:
            if self._api._result_send_active:
                return
        self._api.shutdown()

    def mark_task_delivered(self, task_id: str) -> None:
        with self._api._lock:
            self._task_states[task_id] = TaskState.DELIVERED

    def mark_result_publishing(self, task_id: str) -> str:
        with self._api._lock:
            if self._task_states.get(task_id) == TaskState.COMPLETE:
                raise TrainerSessionError(f"a result was already published for task {task_id!r}")
            self._task_states[task_id] = TaskState.RESULT_PUBLISHING
            if not self._current_result_id:
                self._current_result_id = uuid.uuid4().hex
            return self._current_result_id

    def mark_task_complete(self, task_id: str) -> None:
        with self._api._lock:
            self._task_states[task_id] = TaskState.COMPLETE
            self._trim_task_ledger()

    def clear_result(self) -> None:
        self._current_result_id = None

    def reserve_task(self, task_id, attempt_id, task_sequence):
        """Reserve a logical task, returning an idempotent reply for a duplicate."""
        if not isinstance(task_id, str) or not task_id:
            return self._api._reply(Topic.TASK_FAILED, **{MsgKey.REASON: "TASK_READY requires task_id"})
        if not isinstance(attempt_id, str) or not attempt_id:
            return self._api._reply(
                Topic.TASK_FAILED,
                **{MsgKey.TASK_ID: task_id, MsgKey.REASON: "TASK_READY requires attempt_id"},
            )
        if not isinstance(task_sequence, int) or isinstance(task_sequence, bool) or task_sequence <= 0:
            return self._api._reply(
                Topic.TASK_FAILED,
                **{MsgKey.TASK_ID: task_id, MsgKey.REASON: "TASK_READY requires a positive integer task_seq"},
            )
        with self._api._lock:
            known_state = self._task_states.get(task_id)
            known_attempt = self._task_attempts.get(task_id)
            known_sequence = self._task_sequences.get(task_id)
            if known_state is None:
                retryable = self._retryable_task == (task_id, task_sequence)
                if task_sequence <= self._evicted_task_sequence or (
                    task_sequence <= self._highest_task_sequence and not retryable
                ):
                    return self._api._reply(
                        Topic.TASK_FAILED,
                        **{
                            MsgKey.TASK_ID: task_id,
                            MsgKey.REASON: (
                                f"stale task_seq {task_sequence}; " f"watermark={self._evicted_task_sequence}"
                            ),
                        },
                    )
                self._highest_task_sequence = max(self._highest_task_sequence, task_sequence)
                if retryable:
                    self._retryable_task = None
                # Reservation prevents concurrent duplicate conversion, but it
                # is not an acceptance claim. TASK_STATUS must remain UNKNOWN
                # until conversion succeeds and the task is actually queued.
                self._task_states[task_id] = TaskState.UNKNOWN
                self._task_states.move_to_end(task_id)
                self._task_attempts[task_id] = attempt_id
                self._task_sequences[task_id] = task_sequence
                return None
            if known_sequence != task_sequence:
                return self._api._reply(
                    Topic.TASK_FAILED,
                    **{
                        MsgKey.TASK_ID: task_id,
                        MsgKey.REASON: (
                            f"task_seq mismatch for task {task_id!r}: "
                            f"expected {known_sequence}, got {task_sequence}"
                        ),
                    },
                )
        if known_state == TaskState.UNKNOWN:
            return self._api._reply(
                Topic.TASK_STATUS,
                **{
                    MsgKey.TASK_ID: task_id,
                    MsgKey.TASK_SEQ: known_sequence,
                    MsgKey.TASK_STATE: TaskState.UNKNOWN,
                },
            )
        return self._api._reply(
            Topic.TASK_ACCEPTED,
            **{
                MsgKey.TASK_ID: task_id,
                MsgKey.TASK_SEQ: known_sequence,
                MsgKey.ATTEMPT_ID: known_attempt,
                MsgKey.TASK_STATE: known_state,
            },
        )

    def commit_reserved_task_locked(self, task_id: str, attempt_id: str) -> None:
        """Publish QUEUED after queue insertion; caller must hold the API lock."""
        state = self._task_states.get(task_id)
        if state != TaskState.UNKNOWN or self._task_attempts.get(task_id) != attempt_id:
            raise TrainerSessionError(f"task reservation changed before queue commit for {task_id!r}")
        self._task_states[task_id] = TaskState.QUEUED

    def forget_reserved_task(self, task_id, attempt_id) -> None:
        with self._api._lock:
            if self._task_attempts.get(task_id) == attempt_id:
                task_sequence = self._task_sequences.get(task_id)
                self._task_states.pop(task_id, None)
                self._task_attempts.pop(task_id, None)
                self._task_sequences.pop(task_id, None)
                if task_sequence is not None:
                    self._retryable_task = (task_id, task_sequence)

    def _trim_task_ledger(self) -> None:
        while len(self._task_states) > _TASK_LEDGER_LIMIT:
            task_id, state = next(iter(self._task_states.items()))
            if state != TaskState.COMPLETE:
                return
            self._task_states.pop(task_id, None)
            self._task_attempts.pop(task_id, None)
            task_sequence = self._task_sequences.pop(task_id, None)
            if task_sequence is not None:
                self._evicted_task_sequence = max(self._evicted_task_sequence, task_sequence)

    def publish_result(
        self,
        task_id: str,
        result_id: str,
        shareable: "Shareable",
        source_receiver_ids,
        fobs_ctx_props: dict,
    ) -> None:
        """Publish one logical result with status recovery for a lost acceptance reply."""
        api = self._api
        last_error = None
        attempt_waiters = {}
        for _ in range(_RESULT_SEND_ATTEMPTS):
            attempt_id = uuid.uuid4().hex
            waiters = []
            attempt_waiters[attempt_id] = waiters

            def _on_transaction_created(transaction, attempt_waiters=waiters):
                waiter = DownloadService.get_transfer_waiter(transaction.tx_id)
                attempt_waiters.append(waiter)
                api._add_result_transfer_waiter(waiter)

            def _has_pending_attempt_transfer(attempt_waiters=waiters):
                return any(not waiter.done() for waiter in attempt_waiters)

            attempt_fobs_ctx = dict(fobs_ctx_props)
            attempt_fobs_ctx[RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY] = _on_transaction_created
            request = new_cell_message(
                {MessageHeaderKey.PASS_THROUGH: True},
                {
                    MsgKey.SESSION_ID: api._session_id,
                    MsgKey.TASK_ID: task_id,
                    MsgKey.RESULT_ID: result_id,
                    MsgKey.ATTEMPT_ID: attempt_id,
                    MsgKey.RESULT: shareable,
                },
            )
            try:
                reply = api._cell.send_request(
                    channel=CHANNEL,
                    topic=Topic.RESULT_READY,
                    target=api._cj_fqcn,
                    request=request,
                    timeout=30.0,
                    abort_signal=api._result_abort_signal,
                    progress_wait_cb=_has_pending_attempt_transfer,
                    num_receivers=len(source_receiver_ids) if source_receiver_ids else 1,
                    receiver_ids=source_receiver_ids,
                    fobs_ctx_props=attempt_fobs_ctx,
                )
                accepted_attempt_id = self._accepted_result_attempt(reply, result_id)
            except _UncertainResultReply as e:
                last_error = e
                accepted_attempt_id = self._accepted_result_attempt_from_status(task_id, result_id)
                if not accepted_attempt_id:
                    session_end = api._result_publication_end_reason()
                    if session_end:
                        raise TrainerSessionError(session_end) from e
            except TrainerSessionError:
                raise
            except Exception as e:
                # A transport exception does not prove that the CJ failed to
                # canonicalize this attempt. Preserve its lazy sources while
                # resolving authority through RESULT_STATUS or a duplicate send.
                last_error = e
                accepted_attempt_id = self._accepted_result_attempt_from_status(task_id, result_id)
                if not accepted_attempt_id:
                    session_end = api._result_publication_end_reason()
                    if session_end:
                        raise TrainerSessionError(session_end) from e
            if accepted_attempt_id:
                self._keep_canonical_attempt(accepted_attempt_id, attempt_waiters)
                return
        accepted_attempt_id = self._resolve_ambiguous_result(task_id, result_id, attempt_waiters)
        if accepted_attempt_id:
            self._keep_canonical_attempt(accepted_attempt_id, attempt_waiters)
            return
        raise TrainerSessionError(f"result publication failed after status recovery: {last_error}") from last_error

    def _resolve_ambiguous_result(self, task_id: str, result_id: str, attempt_waiters: dict) -> Optional[str]:
        """Keep candidate lazy sources live until authority or delivery is proven."""
        all_waiters = [waiter for waiters in attempt_waiters.values() for waiter in waiters]
        if not all_waiters:
            return None

        while True:
            accepted_attempt_id = self._accepted_result_attempt_from_status(task_id, result_id)
            if accepted_attempt_id:
                return accepted_attempt_id

            for attempt_id, waiters in attempt_waiters.items():
                if waiters and all(waiter.done() for waiter in waiters):
                    outcomes = [waiter.wait(timeout=0) for waiter in waiters]
                    if all(
                        outcome is not None and outcome.status == TransferProgressState.COMPLETED
                        for outcome in outcomes
                    ):
                        return attempt_id

            if all(waiter.done() for waiter in all_waiters):
                return None
            session_end = self._api._result_publication_end_reason()
            if session_end:
                raise TrainerSessionError(session_end)
            time.sleep(_AMBIGUOUS_RESULT_POLL_INTERVAL)

    @staticmethod
    def _accepted_result_attempt(reply, result_id: str) -> str:
        if reply is None:
            raise _UncertainResultReply("no reply to RESULT_READY from the CJ")
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
        if rc != CellReturnCode.OK:
            raise _UncertainResultReply(f"cell-level failure on RESULT_READY: {rc}")
        body = reply.payload
        if not isinstance(body, dict):
            raise TrainerSessionError(f"invalid RESULT_READY reply payload: {body!r}")
        topic = body.get(MsgKey.REPLY_TOPIC)
        if topic == Topic.RESULT_REJECTED:
            raise TrainerSessionError(f"result was rejected by the CJ: {body.get(MsgKey.REASON)}")
        if topic != Topic.RESULT_ACCEPTED:
            raise TrainerSessionError(f"invalid RESULT_READY reply topic {topic!r}")
        reply_result_id = body.get(MsgKey.RESULT_ID)
        if reply_result_id != result_id:
            raise TrainerSessionError(
                f"RESULT_ACCEPTED result id mismatch: expected {result_id!r}, got {reply_result_id!r}"
            )
        accepted_attempt_id = body.get(MsgKey.ACCEPTED_ATTEMPT_ID)
        if not isinstance(accepted_attempt_id, str) or not accepted_attempt_id:
            raise TrainerSessionError("RESULT_ACCEPTED carried no accepted attempt id")
        return accepted_attempt_id

    def _accepted_result_attempt_from_status(self, task_id: str, result_id: str) -> Optional[str]:
        api = self._api
        try:
            reply = api._cell.send_request(
                channel=CHANNEL,
                topic=Topic.RESULT_STATUS,
                target=api._cj_fqcn,
                request=new_cell_message(
                    {},
                    {
                        MsgKey.SESSION_ID: api._session_id,
                        MsgKey.TASK_ID: task_id,
                        MsgKey.RESULT_ID: result_id,
                    },
                ),
                timeout=_RESULT_STATUS_TIMEOUT,
                optional=True,
            )
        except Exception:
            return None
        if reply is None or reply.get_header(MessageHeaderKey.RETURN_CODE) != CellReturnCode.OK:
            return None
        body = reply.payload
        if not isinstance(body, dict) or body.get(MsgKey.REPLY_TOPIC) != Topic.RESULT_STATUS:
            return None
        state = body.get(MsgKey.RESULT_STATE)
        if state == ResultState.REJECTED:
            raise TrainerSessionError(f"result was rejected by the CJ: {body.get(MsgKey.REASON)}")
        if state != ResultState.ACCEPTED:
            return None
        accepted_attempt_id = body.get(MsgKey.ACCEPTED_ATTEMPT_ID)
        if not isinstance(accepted_attempt_id, str) or not accepted_attempt_id:
            raise TrainerSessionError("accepted RESULT_STATUS carried no accepted attempt id")
        return accepted_attempt_id

    def _keep_canonical_attempt(self, accepted_attempt_id: str, attempt_waiters: dict) -> None:
        canonical_waiters = attempt_waiters.get(accepted_attempt_id)
        if canonical_waiters is None:
            raise TrainerSessionError(f"CJ selected unknown result attempt {accepted_attempt_id!r}")
        for attempt_id, waiters in attempt_waiters.items():
            if attempt_id != accepted_attempt_id:
                self._api._delete_result_transfers(waiters)
        self._api._replace_result_transfer_waiters(canonical_waiters)

    def _handle_session_open(self, request):
        payload = request.payload
        if not isinstance(payload, dict):
            return make_cell_reply(CellReturnCode.INVALID_REQUEST, error="SESSION_OPEN payload must be a dict")
        api = self._api
        origin = request.get_header(MessageHeaderKey.ORIGIN) or ""
        session_id = payload.get(MsgKey.SESSION_ID)
        rejection = self._validate_open(origin, session_id, payload)
        if rejection:
            return api._reply(
                Topic.SESSION_REJECTED,
                **{MsgKey.SESSION_ID: session_id, MsgKey.REASON: rejection},
            )

        with api._lock:
            if api._session_id:
                if api._session_id != session_id or api._cj_fqcn != origin:
                    return api._reply(
                        Topic.SESSION_REJECTED,
                        **{
                            MsgKey.SESSION_ID: session_id,
                            MsgKey.REASON: "trainer is already bound to another CJ/session",
                        },
                    )
            else:
                runtime, rejection = self._runtime_settings(payload)
                if rejection:
                    return api._reply(
                        Topic.SESSION_REJECTED,
                        **{MsgKey.SESSION_ID: session_id, MsgKey.REASON: rejection},
                    )
                try:
                    register_framework_decomposers(
                        runtime["task_exchange"].get(ConfigKey.EXCHANGE_FORMAT, ExchangeFormat.RAW),
                        runtime["task_exchange"].get(ConfigKey.SERVER_EXPECTED_FORMAT, ExchangeFormat.NUMPY),
                        api.logger,
                    )
                except Exception as e:
                    return api._reply(
                        Topic.SESSION_REJECTED,
                        **{
                            MsgKey.SESSION_ID: session_id,
                            MsgKey.REASON: f"failed to configure trainer runtime: {e}",
                        },
                    )
                try:
                    api._install_site_auth_headers(
                        secure_mode=runtime["secure_mode"],
                        auth_token=runtime["auth_token"],
                        token_signature=runtime["auth_token_signature"],
                    )
                except TrainerSessionError as e:
                    return api._reply(
                        Topic.SESSION_REJECTED,
                        **{MsgKey.SESSION_ID: session_id, MsgKey.REASON: str(e)},
                    )
                api._cj_fqcn = origin
                api._session_id = session_id
                api._job_id = payload.get(MsgKey.JOB_ID)
                api._heartbeat_interval = runtime["heartbeat_interval"]
                api._heartbeat_timeout = runtime["heartbeat_timeout"]
                api._task_exchange = runtime["task_exchange"]
                api._launch_once = True
                api._memory_gc_rounds = runtime["memory_gc_rounds"]
                api._cuda_empty_cache = bool(payload.get(MsgKey.CUDA_EMPTY_CACHE, False))
                api._note_cj_activity()
                self._opened.set()

        return api._reply(
            Topic.SESSION_ACCEPTED,
            **{
                MsgKey.SESSION_ID: session_id,
                MsgKey.PROTOCOL_VERSION: PROTOCOL_VERSION,
            },
        )

    def _validate_open(self, origin: str, session_id, payload: dict) -> Optional[str]:
        api = self._api
        if not origin:
            return "SESSION_OPEN has no CJ origin"
        if not isinstance(session_id, str) or not session_id:
            return "SESSION_OPEN has no session id"
        if payload.get(MsgKey.ATTACH_ID) != self.attach_id:
            return "attach id mismatch"
        if payload.get(MsgKey.SITE_NAME) != api._site_name:
            return "site name mismatch"
        job_id = payload.get(MsgKey.JOB_ID)
        if not isinstance(job_id, str) or not job_id or len(FQCN.split(job_id)) != 1 or FQCN.validate(job_id):
            return "SESSION_OPEN has invalid job id"
        expected_origin = FQCN.join([api._site_name, job_id])
        if origin != expected_origin:
            return f"CJ origin mismatch: expected {expected_origin!r}, got {origin!r}"
        if api._cj_fqcn and origin != api._cj_fqcn:
            return f"CJ origin does not match rendezvous endpoint {api._cj_fqcn!r}"
        if payload.get(MsgKey.TRAINER_FQCN) != self.trainer_fqcn:
            return "trainer FQCN mismatch"
        if payload.get(MsgKey.PROTOCOL_VERSION) != PROTOCOL_VERSION:
            return f"unsupported protocol version {payload.get(MsgKey.PROTOCOL_VERSION)!r} (expect {PROTOCOL_VERSION})"
        if str(payload.get(MsgKey.RANK)) != "0" or str(api._rank) != "0":
            return "only rank 0 may bind an attach session"
        return None

    def _runtime_settings(self, payload: dict) -> tuple[Optional[dict], Optional[str]]:
        try:
            heartbeat_interval = self._api._valid_heartbeat_number(
                MsgKey.HEARTBEAT_INTERVAL,
                payload.get(MsgKey.HEARTBEAT_INTERVAL),
                positive=True,
            )
            heartbeat_timeout = self._api._valid_heartbeat_number(
                MsgKey.HEARTBEAT_TIMEOUT,
                payload.get(MsgKey.HEARTBEAT_TIMEOUT),
                positive=False,
            )
            if 0 < heartbeat_timeout <= heartbeat_interval:
                raise TrainerSessionError(
                    f"invalid heartbeat policy: interval {heartbeat_interval} must be less than timeout {heartbeat_timeout}"
                )
            task_exchange = payload.get(MsgKey.TASK_EXCHANGE)
            if not isinstance(task_exchange, dict):
                raise TrainerSessionError("SESSION_OPEN task_exchange must be a dict")
            secure_mode = payload.get(MsgKey.SECURE_MODE)
            if type(secure_mode) is not bool:
                raise TrainerSessionError("SESSION_OPEN secure_mode must be a bool")
            memory_gc_rounds = payload.get(MsgKey.MEMORY_GC_ROUNDS, 0)
            if not isinstance(memory_gc_rounds, int) or isinstance(memory_gc_rounds, bool) or memory_gc_rounds < 0:
                raise TrainerSessionError("SESSION_OPEN memory_gc_rounds must be an integer >= 0")
        except TrainerSessionError as e:
            return None, str(e)
        return {
            "heartbeat_interval": heartbeat_interval,
            "heartbeat_timeout": heartbeat_timeout,
            "secure_mode": secure_mode,
            "auth_token": payload.get(MsgKey.AUTH_TOKEN),
            "auth_token_signature": payload.get(MsgKey.AUTH_TOKEN_SIGNATURE),
            "task_exchange": dict(task_exchange),
            "memory_gc_rounds": memory_gc_rounds,
        }, None

    def _handle_task_status(self, request):
        payload = request.payload
        if not isinstance(payload, dict):
            return make_cell_reply(CellReturnCode.INVALID_REQUEST, error="TASK_STATUS payload must be a dict")
        reject_reason = self._api._validate_cj_control(request, payload)
        if reject_reason:
            return self._api._reply(Topic.ERROR, **{MsgKey.REASON: reject_reason})
        task_id = payload.get(MsgKey.TASK_ID)
        with self._api._lock:
            state = self._task_states.get(task_id, TaskState.UNKNOWN)
            attempt_id = self._task_attempts.get(task_id)
        return self._api._reply(
            Topic.TASK_STATUS,
            **{
                MsgKey.TASK_ID: task_id,
                MsgKey.TASK_STATE: state,
                MsgKey.ACCEPTED_ATTEMPT_ID: attempt_id,
            },
        )
