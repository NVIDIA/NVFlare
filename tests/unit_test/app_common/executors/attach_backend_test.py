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

"""Tests for the non-owning Client API Attach backend."""

import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.executors.client_api.attach_backend import AttachBackend
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext
from nvflare.client.cell.defs import CHANNEL, MsgKey, TaskState, Topic
from nvflare.client.config import ConfigKey
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, MessageType
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.registry import Registry
from nvflare.fuel.f3.cellnet.utils import make_reply as make_cell_reply
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.streaming.download_service import OBJ_DOWNLOADER_CHANNEL
from nvflare.fuel.f3.streaming.stream_const import STREAM_CHANNEL, STREAM_DATA_TOPIC, StreamHeaderKey

CJ_FQCN = "site-1.job-1"
TRAINER_FQCN = "site-1.-client_api_trainer_a"


class _NoOpStats:
    def increment(self, *args, **kwargs):
        pass


def _accepted(topic, **fields):
    return make_cell_reply(CellReturnCode.OK, body={MsgKey.REPLY_TOPIC: topic, **fields})


class FakeCell:
    def __init__(
        self,
        connect_url="grpc://127.0.0.1:9000",
        listener_scheme="grpcs",
        listener_connection_security="mtls",
        listener_params=None,
        attach_config=False,
    ):
        self.decode_pass_through_topics = set()
        self.cbs = {}
        self.sent = []
        self.sent_security = []
        self.fired = []
        self.fired_security = []
        self.connect_url = connect_url
        self.deliver_session = True
        self.connected = True
        self.lose_task_reply = False
        self.deliver_result = True
        self.deliver_result_on_status = False
        self.task_reply_topic = Topic.TASK_ACCEPTED
        self.task_status_state = TaskState.QUEUED
        self.task_serialization_error = None
        self.hold_lost_task_reply_until_cancel = False
        self.pending_task = None
        self.task_ready_count = 0
        self.session_open_count = 0
        self.session_open_failures = 0
        self.shutdown_source_live = False
        params = listener_params or {
            DriverParams.SCHEME.value: listener_scheme,
            DriverParams.CONNECTION_SECURITY.value: listener_connection_security,
            DriverParams.URL.value: connect_url,
        }
        communicator = MagicMock()
        communicator.start_listener.return_value = ("attach-listener", connect_url, params)
        configurator = MagicMock()
        configurator.get_config.return_value = (
            {
                "client_api_attach": {
                    "scheme": listener_scheme,
                    "resources": {
                        key: value
                        for key, value in params.items()
                        if key not in (DriverParams.SCHEME.value, DriverParams.URL.value)
                    },
                }
            }
            if attach_config
            else {}
        )
        self.core_cell = SimpleNamespace(
            communicator=communicator,
            comm_configurator=configurator,
            identity_resolver=SimpleNamespace(exact_identity_map={}),
            send_request=self._send_control_request,
            add_incoming_filter=MagicMock(),
        )

    def register_request_cb(self, channel, topic, cb):
        assert channel == CHANNEL
        self.cbs[topic] = cb

    def get_fqcn(self):
        return "site-1.job-1"

    def is_cell_connected(self, target):
        return self.connected

    def send_request(self, channel, topic, target, request, timeout=None, **kwargs):
        if topic == Topic.SESSION_OPEN:
            if kwargs.get("secure"):
                return self._send_control_request(channel, topic, target, request, timeout=timeout, **kwargs)
            raise AssertionError("clear SESSION_OPEN must use the CoreCell control path")
        self.sent.append((topic, target, request.payload))
        self.sent_security.append((topic, bool(kwargs.get("secure", False))))
        if topic == Topic.TASK_READY:
            self.task_ready_count += 1
            if self.task_serialization_error:
                raise self.task_serialization_error
            # Real Cell.send_request sets this only after local FOBS encoding
            # succeeds and before any transport work begins.
            request.set_header(StreamHeaderKey.PAYLOAD_ENCODING, "fobs")
            task = request.payload
            self.pending_task = (target, task)
            if self.task_reply_topic == Topic.TASK_FAILED:
                return _accepted(Topic.TASK_FAILED, **{MsgKey.REASON: "unsupported task"})
            if self.deliver_result:
                self._deliver_result(target, task)
            if self.lose_task_reply:
                if self.hold_lost_task_reply_until_cancel:
                    deadline = time.monotonic() + 1.0
                    while time.monotonic() < deadline and not kwargs["abort_signal"].triggered:
                        time.sleep(0.001)
                    assert kwargs["abort_signal"].triggered
                raise RuntimeError("TASK_ACCEPTED reply lost")
            return _accepted(Topic.TASK_ACCEPTED)
        if topic == Topic.TASK_STATUS:
            if self.deliver_result_on_status and self.pending_task:
                self.deliver_result_on_status = False
                self._deliver_result(*self.pending_task)
            return _accepted(Topic.TASK_STATUS, **{MsgKey.TASK_STATE: self.task_status_state})
        if topic == Topic.SHUTDOWN:
            return make_cell_reply(
                CellReturnCode.OK,
                body={MsgKey.RESULT_SOURCE_LIVE: self.shutdown_source_live},
            )
        return make_cell_reply(CellReturnCode.OK)

    def _send_control_request(self, channel, topic, target, request, timeout=None, **kwargs):
        self.sent.append((topic, target, request.payload))
        self.sent_security.append((topic, bool(kwargs.get("secure", False))))
        assert topic == Topic.SESSION_OPEN
        self.session_open_count += 1
        if not self.deliver_session or self.session_open_count <= self.session_open_failures:
            return make_cell_reply(CellReturnCode.COMM_ERROR)
        return _accepted(
            Topic.SESSION_ACCEPTED,
            **{
                MsgKey.SESSION_ID: request.payload[MsgKey.SESSION_ID],
                MsgKey.CONNECT_URL: self.connect_url,
                MsgKey.CONNECTION_SECURITY: "clear",
            },
        )

    def fire_and_forget(self, channel, topic, targets, message, **kwargs):
        self.fired.append((topic, tuple(targets), message.payload))
        self.fired_security.append((topic, bool(kwargs.get("secure", False))))

    def deliver(self, topic, origin, payload):
        return self.cbs[topic](new_cell_message({MessageHeaderKey.ORIGIN: origin}, payload))

    def _deliver_result(self, origin, task):
        result_reply = self.deliver(
            Topic.RESULT_READY,
            origin,
            {
                MsgKey.SESSION_ID: task[MsgKey.SESSION_ID],
                MsgKey.TASK_ID: task[MsgKey.TASK_ID],
                MsgKey.RESULT_ID: uuid.uuid4().hex,
                MsgKey.ATTEMPT_ID: uuid.uuid4().hex,
                MsgKey.RESULT: Shareable({"answer": 42}),
            },
        )
        assert result_reply.payload[MsgKey.REPLY_TOPIC] == Topic.RESULT_ACCEPTED


def _fl_ctx(cell, secure_mode=True):
    engine = MagicMock()
    engine.get_cell.return_value = cell
    engine.new_context.return_value.__enter__.return_value = FLContext()
    fl_ctx = FLContext()
    fl_ctx.put(ReservedKey.ENGINE, engine, private=True, sticky=False)
    fl_ctx.put(ReservedKey.RUN_NUM, "job-1", private=False, sticky=False)
    fl_ctx.put(ReservedKey.IDENTITY_NAME, "site-1", private=False, sticky=False)
    fl_ctx.put(FLContextKey.CURRENT_JOB_ID, "job-1", private=False, sticky=False)
    fl_ctx.put(FLContextKey.SECURE_MODE, secure_mode, private=True, sticky=False)
    return fl_ctx


def _context(**overrides):
    values = {
        "executor": MagicMock(),
        "attach_id": "trainer_a",
        "attach_timeout": 1.0,
        "heartbeat_interval": 5.0,
        "heartbeat_timeout": 30.0,
        "task_wait_timeout": 2.0,
        "result_wait_timeout": 2.0,
    }
    values.update(overrides)
    return ClientAPIBackendContext(**values)


def _wait_ready(backend, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        session = backend._get_session()
        if session and session.ready.is_set():
            return session
        time.sleep(0.01)
    raise AssertionError("attach session was not established")


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_attach_session_executes_task_and_finalize_only_closes_protocol():
    cell = FakeCell()
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)
    backend.initialize(_context(), fl_ctx)
    session = _wait_ready(backend)

    result = backend.execute("train", Shareable(), fl_ctx, Signal())
    assert session.result_source_live.is_set()
    heartbeat = cell.deliver(
        Topic.HEARTBEAT,
        TRAINER_FQCN,
        {MsgKey.SESSION_ID: session.session_id, MsgKey.RESULT_SOURCE_LIVE: False},
    )
    assert heartbeat.payload[MsgKey.REPLY_TOPIC] == Topic.HEARTBEAT
    assert not session.result_source_live.is_set()
    backend.finalize(fl_ctx)

    assert session.trainer_fqcn == TRAINER_FQCN
    assert result["answer"] == 42
    task_ready = next(payload for topic, _, payload in cell.sent if topic == Topic.TASK_READY)
    assert task_ready[MsgKey.TASK_SEQ] == 1
    assert any(topic == Topic.SESSION_OPEN for topic, _, _ in cell.sent)
    assert any(topic == Topic.SHUTDOWN for topic, _, _ in cell.sent)
    protected_topics = (Topic.SESSION_OPEN, Topic.TASK_READY, Topic.SHUTDOWN)
    assert all(secure for topic, secure in cell.sent_security if topic in protected_topics)
    assert not backend._session_thread.is_alive()
    cell.core_cell.communicator.remove_connector.assert_not_called()
    assert TRAINER_FQCN not in cell.core_cell.identity_resolver.exact_identity_map


def test_cp_routed_trainer_does_not_mutate_cj_identity_map():
    cell = FakeCell()
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)

    backend.initialize(_context(), fl_ctx)

    assert TRAINER_FQCN not in cell.core_cell.identity_resolver.exact_identity_map
    backend.finalize(fl_ctx)
    assert TRAINER_FQCN not in cell.core_cell.identity_resolver.exact_identity_map


def test_session_open_task_exchange_uses_wire_primitive_values():
    cell = FakeCell()
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)
    backend.initialize(_context(), fl_ctx)
    _wait_ready(backend)

    session_open = next(payload for topic, _, payload in cell.sent if topic == Topic.SESSION_OPEN)
    task_exchange = session_open[MsgKey.TASK_EXCHANGE]

    assert MsgKey.SECURE_MODE not in session_open
    assert MsgKey.AUTH_TOKEN not in session_open
    assert MsgKey.AUTH_TOKEN_SIGNATURE not in session_open
    assert cell.decode_pass_through_topics == set()
    for key in (
        ConfigKey.EXCHANGE_FORMAT,
        ConfigKey.SERVER_EXPECTED_FORMAT,
        ConfigKey.TRANSFER_TYPE,
    ):
        assert type(task_exchange[key]) is str

    backend.finalize(fl_ctx)


def test_attach_timeout_returns_error_without_hanging():
    cell = FakeCell()
    cell.deliver_session = False
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)
    backend.initialize(_context(attach_timeout=0.05), fl_ctx)

    result = backend.execute("train", Shareable(), fl_ctx, Signal())
    backend.finalize(fl_ctx)

    assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION


def test_session_open_retries_core_control_path_after_target_unreachable():
    cell = FakeCell()
    cell.session_open_failures = 1
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)

    backend.initialize(_context(), fl_ctx)
    session = _wait_ready(backend)
    backend.finalize(fl_ctx)

    assert session.ready.is_set()
    assert cell.session_open_count >= 2


def test_lost_task_acceptance_uses_status_without_redelivery():
    cell = FakeCell()
    cell.lose_task_reply = True
    cell.deliver_result = False
    cell.deliver_result_on_status = True
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)
    backend.initialize(_context(), fl_ctx)
    _wait_ready(backend)

    result = backend.execute("train", Shareable(), fl_ctx, Signal())
    backend.finalize(fl_ctx)

    assert result["answer"] == 42
    assert cell.task_ready_count == 1
    assert any(topic == Topic.TASK_STATUS for topic, _, _ in cell.sent)


def test_accepted_result_recovers_lost_task_confirmation_without_status_or_redelivery():
    cell = FakeCell()
    cell.lose_task_reply = True
    cell.hold_lost_task_reply_until_cancel = True
    cell.task_status_state = TaskState.UNKNOWN
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)
    backend.initialize(_context(task_wait_timeout=600.0), fl_ctx)
    _wait_ready(backend)

    start = time.monotonic()
    result = backend.execute("train", Shareable(), fl_ctx, Signal())
    elapsed = time.monotonic() - start
    backend.finalize(fl_ctx)

    assert result["answer"] == 42
    assert elapsed < 1.0
    assert cell.task_ready_count == 1
    assert not any(topic == Topic.TASK_STATUS for topic, _, _ in cell.sent)


def test_run_task_uses_already_accepted_result_when_delivery_reports_failure(monkeypatch):
    cell = FakeCell()
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)
    backend.initialize(_context(), fl_ctx)
    session = _wait_ready(backend)

    def accept_result_then_report_failure(_session, task, *_args):
        task.result = Shareable({"answer": 42})
        task.result_ready.set()
        return False, "TASK_ACCEPTED confirmation lost"

    monkeypatch.setattr(backend, "_deliver_task", accept_result_then_report_failure)
    result = backend.execute("train", Shareable(), fl_ctx, Signal())
    backend.finalize(fl_ctx)

    assert session.ready.is_set()
    assert result["answer"] == 42


def test_semantic_task_rejection_is_not_retried():
    cell = FakeCell()
    cell.task_reply_topic = Topic.TASK_FAILED
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)
    backend.initialize(_context(), fl_ctx)
    _wait_ready(backend)

    result = backend.execute("unsupported", Shareable(), fl_ctx, Signal())
    backend.finalize(fl_ctx)

    assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert cell.task_ready_count == 1
    assert not any(topic == Topic.TASK_STATUS for topic, _, _ in cell.sent)


def test_local_task_serialization_error_is_not_retried_or_probed():
    cell = FakeCell()
    cell.task_serialization_error = ValueError("unsupported task payload")
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)
    backend.initialize(_context(task_wait_timeout=600.0), fl_ctx)
    _wait_ready(backend)

    result = backend.execute("train", Shareable(), fl_ctx, Signal())
    backend.finalize(fl_ctx)

    assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert cell.task_ready_count == 1
    assert not any(topic == Topic.TASK_STATUS for topic, _, _ in cell.sent)


def test_session_loss_is_terminal_when_reconnect_is_disabled():
    cell = FakeCell()
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)
    backend.initialize(_context(heartbeat_interval=0.01, heartbeat_timeout=0.05), fl_ctx)
    session = _wait_ready(backend)

    with session._activity_lock:
        session._last_peer_activity = time.monotonic() - 1.0

    assert _wait_until(lambda: bool(session.error))
    assert backend._get_session() is session
    backend.finalize(fl_ctx)


def test_reconnect_uses_fresh_session_and_rejects_stale_traffic():
    cell = FakeCell()
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)
    backend.initialize(
        _context(allow_reconnect=True, heartbeat_interval=0.01, heartbeat_timeout=0.05),
        fl_ctx,
    )
    first = _wait_ready(backend)

    with first._activity_lock:
        first._last_peer_activity = time.monotonic() - 1.0

    assert _wait_until(lambda: backend._get_session() is not first and backend._get_session().ready.is_set())
    second = backend._get_session()
    assert second.session_id != first.session_id

    stale = cell.deliver(
        Topic.HEARTBEAT,
        TRAINER_FQCN,
        {MsgKey.SESSION_ID: first.session_id},
    )
    assert stale.payload[MsgKey.REPLY_TOPIC] == Topic.ERROR
    assert "session id" in stale.payload[MsgKey.REASON]
    backend.finalize(fl_ctx)


def test_reconnect_waits_for_accepted_result_source_to_be_released():
    cell = FakeCell()
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)
    backend.initialize(
        _context(allow_reconnect=True, heartbeat_interval=0.01, heartbeat_timeout=0.05),
        fl_ctx,
    )
    first = _wait_ready(backend)

    result = backend.execute("train", Shareable(), fl_ctx, Signal())
    assert result["answer"] == 42
    assert first.result_source_live.is_set()
    assert backend._current_task is None

    with first._activity_lock:
        first._last_peer_activity = time.monotonic() - 1.0
    time.sleep(0.15)
    assert backend._get_session() is first
    assert first.error is None

    heartbeat = cell.deliver(
        Topic.HEARTBEAT,
        TRAINER_FQCN,
        {MsgKey.SESSION_ID: first.session_id, MsgKey.RESULT_SOURCE_LIVE: False},
    )
    assert heartbeat.payload[MsgKey.REPLY_TOPIC] == Topic.HEARTBEAT
    with first._activity_lock:
        first._last_peer_activity = time.monotonic() - 1.0

    assert _wait_until(lambda: backend._get_session() is not first and backend._get_session().ready.is_set())
    backend.finalize(fl_ctx)


def test_reconnect_retires_accepted_result_source_after_confirmed_disconnect():
    cell = FakeCell()
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)
    backend.initialize(
        _context(allow_reconnect=True, heartbeat_interval=0.01, heartbeat_timeout=0.05),
        fl_ctx,
    )
    first = _wait_ready(backend)

    result = backend.execute("train", Shareable(), fl_ctx, Signal())
    assert result["answer"] == 42
    assert first.result_source_live.is_set()

    cell.connected = False
    with first._activity_lock:
        first._last_peer_activity = time.monotonic() - 1.0

    assert _wait_until(lambda: backend._get_session() is not first and backend._get_session().ready.is_set())
    assert not first.result_source_live.is_set()
    backend.finalize(fl_ctx)


def test_reconnect_preserves_accepted_result_source_during_transient_disconnect():
    cell = FakeCell()
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)
    backend.initialize(
        _context(allow_reconnect=True, heartbeat_interval=0.01, heartbeat_timeout=0.5),
        fl_ctx,
    )
    first = _wait_ready(backend)

    result = backend.execute("train", Shareable(), fl_ctx, Signal())
    assert result["answer"] == 42
    cell.connected = False
    with first._activity_lock:
        first._last_peer_activity = time.monotonic() - 1.0

    assert _wait_until(lambda: first.result_source_disconnect_since is not None)
    cell.connected = True
    time.sleep(0.6)

    assert backend._get_session() is first
    assert first.result_source_live.is_set()
    assert first.result_source_disconnect_since is None
    heartbeat = cell.deliver(
        Topic.HEARTBEAT,
        TRAINER_FQCN,
        {MsgKey.SESSION_ID: first.session_id, MsgKey.RESULT_SOURCE_LIVE: False},
    )
    assert heartbeat.payload[MsgKey.REPLY_TOPIC] == Topic.HEARTBEAT
    backend.finalize(fl_ctx)


def test_attach_rejects_heartbeat_disabled_session_before_shutdown_can_be_lost():
    cell = FakeCell()
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)

    with pytest.raises(ValueError, match="heartbeat_timeout > 0.*SHUTDOWN"):
        backend.initialize(_context(heartbeat_timeout=0.0, result_wait_timeout=2.0), fl_ctx)

    assert not any(topic == Topic.SESSION_OPEN for topic, _, _ in cell.sent)
    assert not any(topic == Topic.SHUTDOWN for topic, _, _ in cell.sent)


def test_reconnect_does_not_replay_task_interrupted_by_session_loss():
    cell = FakeCell()
    cell.deliver_result = False
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)
    backend.initialize(
        _context(allow_reconnect=True, heartbeat_interval=0.01, heartbeat_timeout=0.05),
        fl_ctx,
    )
    first = _wait_ready(backend)
    result_box = {}
    execution = threading.Thread(
        target=lambda: result_box.setdefault("result", backend.execute("train", Shareable(), fl_ctx, Signal()))
    )
    execution.start()
    assert _wait_until(lambda: cell.task_ready_count == 1)

    with first._activity_lock:
        first._last_peer_activity = time.monotonic() - 1.0

    execution.join(timeout=2.0)
    assert not execution.is_alive()
    assert result_box["result"].get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert _wait_until(lambda: backend._get_session() is not first and backend._get_session().ready.is_set())
    assert cell.task_ready_count == 1
    backend.finalize(fl_ctx)


def test_network_attach_uses_existing_cp_route_without_a_cj_listener():
    cell = FakeCell(attach_config=False)
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell, secure_mode=True)

    backend.initialize(_context(), fl_ctx)
    assert _wait_ready(backend).ready.is_set()
    cell.core_cell.communicator.start_listener.assert_not_called()
    assert any(topic == Topic.SESSION_OPEN for topic, _, _ in cell.sent)
    assert (Topic.SESSION_OPEN, True) in cell.sent_security
    backend.finalize(fl_ctx)


def test_secure_network_attach_guard_covers_streamed_and_direct_protocol_before_decode():
    cell = FakeCell(attach_config=False)
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell, secure_mode=True)
    backend.initialize(_context(), fl_ctx)
    _wait_ready(backend)
    cell.core_cell.add_incoming_filter.assert_has_calls(
        [
            call(channel=STREAM_CHANNEL, topic=STREAM_DATA_TOPIC, cb=backend._protocol_guard.check),
            call(channel=CHANNEL, topic="*", cb=backend._protocol_guard.check),
            call(channel=OBJ_DOWNLOADER_CHANNEL, topic="*", cb=backend._protocol_guard.check),
        ]
    )

    clear = new_cell_message(
        {
            MessageHeaderKey.DESTINATION: CJ_FQCN,
            MessageHeaderKey.ORIGIN: TRAINER_FQCN,
            MessageHeaderKey.CHANNEL: STREAM_CHANNEL,
            MessageHeaderKey.TOPIC: STREAM_DATA_TOPIC,
            StreamHeaderKey.CHANNEL: CHANNEL,
            StreamHeaderKey.TOPIC: Topic.RESULT_READY,
        },
        MagicMock(name="undecoded_result"),
    )
    rejected = backend._secure_protocol_guard(clear)
    clear.set_header(MessageHeaderKey.SECURE, True)
    claimed_secure = backend._secure_protocol_guard(clear)
    clear.set_header(MessageHeaderKey.ENCRYPTED, True)
    foreign_trainer = new_cell_message(
        {
            MessageHeaderKey.DESTINATION: CJ_FQCN,
            MessageHeaderKey.ORIGIN: "site-1.-client_api_other",
            MessageHeaderKey.SECURE: True,
            MessageHeaderKey.ENCRYPTED: True,
            MessageHeaderKey.CHANNEL: STREAM_CHANNEL,
            MessageHeaderKey.TOPIC: STREAM_DATA_TOPIC,
            StreamHeaderKey.CHANNEL: CHANNEL,
            StreamHeaderKey.TOPIC: Topic.RESULT_READY,
        },
        MagicMock(name="other_undecoded_result"),
    )
    direct_clear = new_cell_message(
        {
            MessageHeaderKey.DESTINATION: CJ_FQCN,
            MessageHeaderKey.ORIGIN: TRAINER_FQCN,
            MessageHeaderKey.CHANNEL: CHANNEL,
            MessageHeaderKey.TOPIC: Topic.RESULT_READY,
        },
        MagicMock(name="direct_undecoded_result"),
    )
    direct_clear_download = new_cell_message(
        {
            MessageHeaderKey.DESTINATION: CJ_FQCN,
            MessageHeaderKey.ORIGIN: TRAINER_FQCN,
            MessageHeaderKey.CHANNEL: OBJ_DOWNLOADER_CHANNEL,
            MessageHeaderKey.TOPIC: "download",
        },
        MagicMock(name="direct_undecoded_download"),
    )

    assert rejected.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.AUTHENTICATION_ERROR
    assert claimed_secure.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.AUTHENTICATION_ERROR
    assert backend._secure_protocol_guard(clear) is None
    assert (
        backend._secure_protocol_guard(foreign_trainer).get_header(MessageHeaderKey.RETURN_CODE)
        == CellReturnCode.AUTHENTICATION_ERROR
    )
    assert (
        backend._secure_protocol_guard(direct_clear).get_header(MessageHeaderKey.RETURN_CODE)
        == CellReturnCode.AUTHENTICATION_ERROR
    )
    assert (
        backend._secure_protocol_guard(direct_clear_download).get_header(MessageHeaderKey.RETURN_CODE)
        == CellReturnCode.AUTHENTICATION_ERROR
    )

    other_owner = object()
    backend._protocol_guard.claim(other_owner, "site-1.-client_api_other", "site-1")
    assert backend._secure_protocol_guard(foreign_trainer) is None
    backend._protocol_guard.release(other_owner)
    backend.finalize(fl_ctx)


def test_secure_attach_adversaries_stop_in_core_cell_before_application_decode():
    cell = FakeCell(attach_config=False)
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell, secure_mode=True)
    backend.initialize(_context(), fl_ctx)
    _wait_ready(backend)

    core = CoreCell.__new__(CoreCell)
    core.my_info = SimpleNamespace(fqcn=CJ_FQCN)
    core.logger = MagicMock()
    core.received_msg_counter_pool = _NoOpStats()
    core.in_filter_reg = Registry()
    core.message_interceptor = None
    core._stats_category = MagicMock(return_value="test")
    core._process_request = MagicMock(side_effect=AssertionError("application decode was reached"))
    for filter_call in cell.core_cell.add_incoming_filter.call_args_list:
        core.add_incoming_filter(**filter_call.kwargs)

    common_headers = {
        MessageHeaderKey.MSG_TYPE: MessageType.REQ,
        MessageHeaderKey.DESTINATION: CJ_FQCN,
        MessageHeaderKey.REPLY_EXPECTED: False,
    }
    foreign_stream = new_cell_message(
        {
            **common_headers,
            MessageHeaderKey.ORIGIN: "site-1.-client_api_foreign",
            MessageHeaderKey.CHANNEL: STREAM_CHANNEL,
            MessageHeaderKey.TOPIC: STREAM_DATA_TOPIC,
            MessageHeaderKey.SECURE: True,
            MessageHeaderKey.ENCRYPTED: True,
            StreamHeaderKey.CHANNEL: CHANNEL,
            StreamHeaderKey.TOPIC: Topic.RESULT_READY,
        },
        MagicMock(name="foreign_stream_payload"),
    )
    direct_clear = new_cell_message(
        {
            **common_headers,
            MessageHeaderKey.ORIGIN: TRAINER_FQCN,
            MessageHeaderKey.CHANNEL: CHANNEL,
            MessageHeaderKey.TOPIC: Topic.RESULT_READY,
        },
        MagicMock(name="direct_clear_payload"),
    )

    foreign_reply = CoreCell._process_received_msg(core, Endpoint("site-1"), None, foreign_stream)
    direct_reply = CoreCell._process_received_msg(core, Endpoint("site-1"), None, direct_clear)

    assert foreign_reply.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.AUTHENTICATION_ERROR
    assert direct_reply.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.AUTHENTICATION_ERROR
    core._process_request.assert_not_called()
    backend.finalize(fl_ctx)


def test_secure_attach_guard_shares_claims_across_concurrent_backends():
    cell = FakeCell(attach_config=False)
    fl_ctx = _fl_ctx(cell, secure_mode=True)
    first = AttachBackend()
    second = AttachBackend()
    first.initialize(_context(attach_id="trainer_a"), fl_ctx)
    second.initialize(_context(attach_id="trainer_b"), fl_ctx)
    _wait_ready(first)
    _wait_ready(second)

    guard = first._protocol_guard
    assert guard is second._protocol_guard
    assert cell.core_cell.add_incoming_filter.call_count == 3

    def protected(origin):
        return new_cell_message(
            {
                MessageHeaderKey.DESTINATION: CJ_FQCN,
                MessageHeaderKey.ORIGIN: origin,
                MessageHeaderKey.CHANNEL: CHANNEL,
                MessageHeaderKey.TOPIC: Topic.HEARTBEAT,
                MessageHeaderKey.SECURE: True,
                MessageHeaderKey.ENCRYPTED: True,
            },
            MagicMock(name="undecoded_heartbeat"),
        )

    assert guard(protected("site-1.-client_api_trainer_a")) is None
    assert guard(protected("site-1.-client_api_trainer_b")) is None
    rejected = guard(protected("site-1.-client_api_foreign"))
    assert rejected.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.AUTHENTICATION_ERROR

    first.finalize(fl_ctx)
    assert guard(protected("site-1.-client_api_trainer_b")) is None
    second.finalize(fl_ctx)
    assert guard(protected("site-1.-client_api_foreign")) is None


def test_dedicated_network_attach_listener_is_rejected():
    cell = FakeCell(listener_scheme="grpc", listener_connection_security="clear", attach_config=True)
    backend = AttachBackend()

    with pytest.raises(ValueError, match="network Attach trainers must connect through"):
        backend.initialize(_context(), _fl_ctx(cell, secure_mode=False))

    cell.core_cell.communicator.start_listener.assert_not_called()
    assert not any(topic == Topic.SESSION_OPEN for topic, _, _ in cell.sent)


def _shared_file_listener_params(tmp_path: Path) -> dict:
    root_dir = tmp_path / "cellnet"
    listener_dir = root_dir / "lst_12345678"
    conns_dir = listener_dir / "conns"
    conns_dir.mkdir(parents=True)
    marker = listener_dir / ".nvf_file_transport"
    marker.touch()
    root_dir.chmod(0o770)
    listener_dir.chmod(0o770)
    conns_dir.chmod(0o770)
    marker.chmod(0o660)
    return {
        DriverParams.URL.value: f"shared-file://0{listener_dir}",
        DriverParams.SCHEME.value: "shared-file",
        DriverParams.CONNECTION_SECURITY.value: "clear",
        "root_dir": str(root_dir),
    }


def test_shared_file_attach_listener_does_not_require_insecure_opt_in(tmp_path):
    params = _shared_file_listener_params(tmp_path)
    cell = FakeCell(
        connect_url=params[DriverParams.URL.value],
        listener_scheme="shared-file",
        listener_connection_security="clear",
        listener_params=params,
        attach_config=True,
    )
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell, secure_mode=True)
    context = _context()

    backend.initialize(context, fl_ctx)

    assert _wait_ready(backend).ready.is_set()
    context.executor.log_warning.assert_not_called()
    session_open = next(payload for topic, _, payload in cell.sent if topic == Topic.SESSION_OPEN)
    assert MsgKey.SECURE_MODE not in session_open
    assert MsgKey.AUTH_TOKEN not in session_open
    assert MsgKey.AUTH_TOKEN_SIGNATURE not in session_open
    assert cell.decode_pass_through_topics == set()
    assert (Topic.SESSION_OPEN, False) in cell.sent_security
    backend.finalize(fl_ctx)


def test_world_accessible_shared_file_attach_listener_is_rejected(tmp_path):
    params = _shared_file_listener_params(tmp_path)
    listener_dir = Path(params[DriverParams.URL.value].removeprefix("shared-file://0"))
    listener_dir.chmod(0o777)
    cell = FakeCell(
        connect_url=params[DriverParams.URL.value],
        listener_scheme="shared-file",
        listener_connection_security="clear",
        listener_params=params,
        attach_config=True,
    )
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell, secure_mode=False)

    try:
        backend.initialize(_context(allow_insecure_attach=True), fl_ctx)
    except ValueError as e:
        assert "shared-file attach requires" in str(e)
    else:
        raise AssertionError("a world-accessible shared-file route must be rejected even with the insecure opt-in")
    assert not any(topic == Topic.SESSION_OPEN for topic, _, _ in cell.sent)


def test_missing_attach_listener_config_selects_cp_route():
    cell = FakeCell(attach_config=False)
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)

    backend.initialize(_context(), fl_ctx)
    assert _wait_ready(backend).ready.is_set()
    cell.core_cell.communicator.start_listener.assert_not_called()
    backend.finalize(fl_ctx)


def test_finalize_unblocks_pending_result_wait():
    cell = FakeCell()
    cell.deliver_result = False
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)
    backend.initialize(
        _context(heartbeat_interval=5.0, heartbeat_timeout=30.0, result_wait_timeout=None),
        fl_ctx,
    )
    _wait_ready(backend)

    result_box = {}
    execution = threading.Thread(
        target=lambda: result_box.setdefault("result", backend.execute("train", Shareable(), fl_ctx, Signal()))
    )
    execution.start()
    assert _wait_until(lambda: cell.task_ready_count == 1)

    backend.finalize(fl_ctx)
    execution.join(timeout=2.0)

    assert not execution.is_alive()
    assert result_box["result"].get_return_code() == ReturnCode.EXECUTION_EXCEPTION


def test_finalize_keeps_attach_route_until_accepted_result_source_disconnects():
    cell = FakeCell()
    cell.shutdown_source_live = True
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)
    backend.initialize(_context(heartbeat_interval=0.01, heartbeat_timeout=0.05), fl_ctx)
    session = _wait_ready(backend)
    session.result_source_live.set()

    finalizer = threading.Thread(target=lambda: backend.finalize(fl_ctx))
    finalizer.start()
    assert _wait_until(lambda: any(topic == Topic.SHUTDOWN for topic, _, _ in cell.sent))
    assert finalizer.is_alive()
    cell.core_cell.communicator.remove_connector.assert_not_called()

    cell.connected = False
    finalizer.join(timeout=2.0)

    assert not finalizer.is_alive()
    cell.core_cell.communicator.remove_connector.assert_not_called()
