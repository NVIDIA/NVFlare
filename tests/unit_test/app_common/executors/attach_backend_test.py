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
from unittest.mock import MagicMock

from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.executors.client_api.attach_backend import AttachBackend
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext
from nvflare.client.cell.defs import CHANNEL, MsgKey, TaskState, Topic
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply as make_cell_reply
from nvflare.fuel.f3.cellnet.utils import new_cell_message

TRAINER_FQCN = "site-1.-client_api_trainer_a"


def _accepted(topic, **fields):
    return make_cell_reply(CellReturnCode.OK, body={MsgKey.REPLY_TOPIC: topic, **fields})


class FakeCell:
    def __init__(self, connect_url="grpc://127.0.0.1:9000"):
        self.decode_pass_through_topics = set()
        self.decode_pass_through_relay_topics = set()
        self.cbs = {}
        self.sent = []
        self.fired = []
        self.connect_url = connect_url
        self.deliver_session = True
        self.connected = True
        self.lose_task_reply = False
        self.deliver_result = True
        self.task_reply_topic = Topic.TASK_ACCEPTED
        self.task_ready_count = 0

    def register_request_cb(self, channel, topic, cb):
        assert channel == CHANNEL
        self.cbs[topic] = cb

    def get_fqcn(self):
        return "site-1.job-1"

    def is_cell_connected(self, target):
        return self.connected

    def send_request(self, channel, topic, target, request, timeout=None, **kwargs):
        self.sent.append((topic, target, request.payload))
        if topic == Topic.SESSION_OPEN:
            if not self.deliver_session:
                return make_cell_reply(CellReturnCode.COMM_ERROR)
            return _accepted(
                Topic.SESSION_ACCEPTED,
                **{
                    MsgKey.SESSION_ID: request.payload[MsgKey.SESSION_ID],
                    MsgKey.CONNECT_URL: self.connect_url,
                    MsgKey.CONNECTION_SECURITY: "clear",
                },
            )
        if topic == Topic.TASK_READY:
            self.task_ready_count += 1
            task = request.payload
            if self.task_reply_topic == Topic.TASK_FAILED:
                return _accepted(Topic.TASK_FAILED, **{MsgKey.REASON: "unsupported task"})
            if not self.deliver_result:
                return _accepted(Topic.TASK_ACCEPTED)
            result_reply = self.deliver(
                Topic.RESULT_READY,
                target,
                {
                    MsgKey.SESSION_ID: task[MsgKey.SESSION_ID],
                    MsgKey.TASK_ID: task[MsgKey.TASK_ID],
                    MsgKey.RESULT_ID: uuid.uuid4().hex,
                    MsgKey.ATTEMPT_ID: uuid.uuid4().hex,
                    MsgKey.RESULT: Shareable({"answer": 42}),
                },
            )
            assert result_reply.payload[MsgKey.REPLY_TOPIC] == Topic.RESULT_ACCEPTED
            if self.lose_task_reply:
                raise RuntimeError("TASK_ACCEPTED reply lost")
            return _accepted(Topic.TASK_ACCEPTED)
        if topic == Topic.TASK_STATUS:
            return _accepted(Topic.TASK_STATUS, **{MsgKey.TASK_STATE: TaskState.QUEUED})
        if topic == Topic.SHUTDOWN:
            return make_cell_reply(CellReturnCode.OK, body={MsgKey.RESULT_SOURCE_LIVE: False})
        return make_cell_reply(CellReturnCode.OK)

    def fire_and_forget(self, channel, topic, targets, message, **kwargs):
        self.fired.append((topic, tuple(targets), message.payload))

    def deliver(self, topic, origin, payload):
        return self.cbs[topic](new_cell_message({MessageHeaderKey.ORIGIN: origin}, payload))


def _fl_ctx(cell):
    engine = MagicMock()
    engine.get_cell.return_value = cell
    engine.new_context.return_value.__enter__.return_value = FLContext()
    fl_ctx = FLContext()
    fl_ctx.put(ReservedKey.ENGINE, engine, private=True, sticky=False)
    fl_ctx.put(ReservedKey.RUN_NUM, "job-1", private=False, sticky=False)
    fl_ctx.put(ReservedKey.IDENTITY_NAME, "site-1", private=False, sticky=False)
    fl_ctx.put(FLContextKey.CURRENT_JOB_ID, "job-1", private=False, sticky=False)
    fl_ctx.put(FLContextKey.SECURE_MODE, False, private=True, sticky=False)
    return fl_ctx


def _context(**overrides):
    values = {
        "executor": MagicMock(),
        "attach_id": "trainer_a",
        "attach_timeout": 1.0,
        "heartbeat_timeout": 0.0,
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
    assert any(topic == Topic.SESSION_OPEN for topic, _, _ in cell.sent)
    assert any(topic == Topic.SHUTDOWN for topic, _, _ in cell.sent)
    assert not backend._session_thread.is_alive()


def test_attach_timeout_returns_error_without_hanging():
    cell = FakeCell()
    cell.deliver_session = False
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)
    backend.initialize(_context(attach_timeout=0.05), fl_ctx)

    result = backend.execute("train", Shareable(), fl_ctx, Signal())
    backend.finalize(fl_ctx)

    assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION


def test_lost_task_acceptance_uses_status_without_redelivery():
    cell = FakeCell()
    cell.lose_task_reply = True
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)
    backend.initialize(_context(), fl_ctx)
    _wait_ready(backend)

    result = backend.execute("train", Shareable(), fl_ctx, Signal())
    backend.finalize(fl_ctx)

    assert result["answer"] == 42
    assert cell.task_ready_count == 1
    assert any(topic == Topic.TASK_STATUS for topic, _, _ in cell.sent)


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


def test_cleartext_non_loopback_route_requires_explicit_opt_in():
    cell = FakeCell(connect_url="grpc://10.20.30.40:9000")
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)
    backend.initialize(_context(), fl_ctx)

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not backend._session_error():
        time.sleep(0.01)
    assert "cleartext non-loopback" in backend._session_error()
    backend.finalize(fl_ctx)

    allowed = AttachBackend()
    allowed.initialize(_context(allow_insecure_attach=True), fl_ctx)
    assert _wait_ready(allowed).ready.is_set()
    allowed.finalize(fl_ctx)
