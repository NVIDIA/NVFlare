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
from unittest.mock import MagicMock

from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.executors.client_api.attach_backend import AttachBackend
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext
from nvflare.client.cell.defs import CHANNEL, MsgKey, TaskState, Topic
from nvflare.client.config import ConfigKey
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply as make_cell_reply
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.f3.drivers.driver_params import DriverParams

CJ_FQCN = "site-1.job-1"
TRAINER_FQCN = f"{CJ_FQCN}.-client_api_trainer_a"


def _accepted(topic, **fields):
    return make_cell_reply(CellReturnCode.OK, body={MsgKey.REPLY_TOPIC: topic, **fields})


class FakeCell:
    def __init__(
        self,
        connect_url="grpc://127.0.0.1:9000",
        listener_scheme="grpcs",
        listener_connection_security="mtls",
        listener_params=None,
        attach_config=True,
    ):
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
        )

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
            raise AssertionError("SESSION_OPEN must use the CoreCell control path")
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
            return make_cell_reply(
                CellReturnCode.OK,
                body={MsgKey.RESULT_SOURCE_LIVE: self.shutdown_source_live},
            )
        return make_cell_reply(CellReturnCode.OK)

    def _send_control_request(self, channel, topic, target, request, timeout=None, **kwargs):
        self.sent.append((topic, target, request.payload))
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

    def deliver(self, topic, origin, payload):
        return self.cbs[topic](new_cell_message({MessageHeaderKey.ORIGIN: origin}, payload))


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
    task_ready = next(payload for topic, _, payload in cell.sent if topic == Topic.TASK_READY)
    assert task_ready[MsgKey.TASK_SEQ] == 1
    assert any(topic == Topic.SESSION_OPEN for topic, _, _ in cell.sent)
    assert any(topic == Topic.SHUTDOWN for topic, _, _ in cell.sent)
    assert not backend._session_thread.is_alive()
    cell.core_cell.communicator.remove_connector.assert_called_once_with("attach-listener")
    assert TRAINER_FQCN not in cell.core_cell.identity_resolver.exact_identity_map


def test_mtls_listener_binds_trainer_fqcn_to_site_certificate_identity():
    cell = FakeCell()
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)

    backend.initialize(_context(), fl_ctx)

    assert cell.core_cell.identity_resolver.exact_identity_map[TRAINER_FQCN] == "site-1"
    backend.finalize(fl_ctx)
    assert TRAINER_FQCN not in cell.core_cell.identity_resolver.exact_identity_map


def test_conflicting_trainer_transport_identity_fails_before_listener_or_session():
    cell = FakeCell()
    cell.core_cell.identity_resolver.exact_identity_map[TRAINER_FQCN] = "other-site"
    backend = AttachBackend()

    try:
        backend.initialize(_context(), _fl_ctx(cell))
    except ValueError as e:
        assert "already bound" in str(e)
    else:
        raise AssertionError("a conflicting transport identity must be rejected")

    cell.core_cell.communicator.start_listener.assert_not_called()
    assert not any(topic == Topic.SESSION_OPEN for topic, _, _ in cell.sent)


def test_session_open_task_exchange_uses_wire_primitive_values():
    cell = FakeCell()
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)
    backend.initialize(_context(), fl_ctx)
    _wait_ready(backend)

    session_open = next(payload for topic, _, payload in cell.sent if topic == Topic.SESSION_OPEN)
    task_exchange = session_open[MsgKey.TASK_EXCHANGE]

    assert session_open[MsgKey.RESULT_RELAY] is True
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


def test_clear_attach_listener_requires_explicit_opt_in_before_session_open():
    cell = FakeCell(listener_scheme="grpc", listener_connection_security="clear")
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell, secure_mode=False)

    try:
        backend.initialize(_context(), fl_ctx)
    except ValueError as e:
        assert "CJ-owned attach listener" in str(e)
    else:
        raise AssertionError("non-secure attach must require explicit opt-in")
    assert not any(topic == Topic.SESSION_OPEN for topic, _, _ in cell.sent)

    allowed = AttachBackend()
    allowed_context = _context(allow_insecure_attach=True)
    allowed.initialize(allowed_context, fl_ctx)
    assert _wait_ready(allowed).ready.is_set()
    allowed_context.executor.log_warning.assert_called_once()
    allowed.finalize(fl_ctx)


def test_secure_site_still_requires_mtls_attach_listener_before_session_open():
    cell = FakeCell(listener_scheme="grpc", listener_connection_security="mtls")
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell, secure_mode=True)

    try:
        backend.initialize(_context(), fl_ctx)
    except ValueError as e:
        assert "CJ-owned attach listener" in str(e)
    else:
        raise AssertionError("a secure-mode site with a clear attach listener must be rejected")
    assert not any(topic == Topic.SESSION_OPEN for topic, _, _ in cell.sent)

    allowed = AttachBackend()
    allowed_context = _context(allow_insecure_attach=True)
    allowed.initialize(allowed_context, fl_ctx)
    assert _wait_ready(allowed).ready.is_set()
    allowed_context.executor.log_warning.assert_called_once()
    allowed.finalize(fl_ctx)


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
    )
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell, secure_mode=False)
    context = _context()

    backend.initialize(context, fl_ctx)

    assert _wait_ready(backend).ready.is_set()
    context.executor.log_warning.assert_not_called()
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


def test_missing_attach_listener_config_fails_before_session_open():
    cell = FakeCell(attach_config=False)
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)

    try:
        backend.initialize(_context(), fl_ctx)
    except ValueError as e:
        assert "client_api_attach" in str(e)
    else:
        raise AssertionError("attach must require a dedicated listener configuration")
    assert not any(topic == Topic.SESSION_OPEN for topic, _, _ in cell.sent)


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


def test_finalize_keeps_attach_listener_until_accepted_result_source_disconnects():
    cell = FakeCell()
    cell.shutdown_source_live = True
    backend = AttachBackend()
    fl_ctx = _fl_ctx(cell)
    backend.initialize(_context(), fl_ctx)
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
    cell.core_cell.communicator.remove_connector.assert_called_once_with("attach-listener")
