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

"""Tests for the trainer-side Cell engine (CellClientAPI).

Drives the trainer's half of the external_process protocol against a fake CJ cell: init()'s
HELLO handshake, direct Cell Shareable tasks/results, result transaction progress, the
batch-loop is_running() semantics, and ABORT/SHUTDOWN session ends."""

import os
import signal as process_signal
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.apis.fl_constant import FLMetaKey
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.client.cell import api as cell_api
from nvflare.client.cell import attach_session as attach_session_module
from nvflare.client.cell.api import CellClientAPI, TrainerSessionError
from nvflare.client.cell.attach_rendezvous import AttachEndpointPublisher
from nvflare.client.cell.bootstrap import (
    ATTACH_EXECUTION_MODE,
    BOOTSTRAP_SCHEMA_VERSION,
    EXTERNAL_PROCESS_EXECUTION_MODE,
    BootstrapKey,
    read_bootstrap_config,
    write_bootstrap_config,
)
from nvflare.client.cell.defs import CHANNEL, PROTOCOL_VERSION, MsgKey, ResultState, TaskState, Topic
from nvflare.client.config import ConfigKey, ExchangeFormat
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply as make_cell_reply
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.f3.streaming.stream_const import STREAM_CHANNEL, STREAM_DATA_TOPIC, StreamHeaderKey
from nvflare.fuel.f3.streaming.transfer_progress import TransferProgressState
from nvflare.fuel.utils.fobs import FOBSContextKey

CJ_FQCN = "site-1.job-1"
TRAINER_FQCN = "site-1.job-1.client_api_trainer_1"
SESSION_ID = "session-abc"
ATTACH_ID = "trainer_a"
ATTACH_TRAINER_FQCN = "site-1.-client_api_trainer_a"


def _hello_accepted_reply(heartbeat_interval=0.05, heartbeat_timeout=0.0, secure_mode=False):
    security = {MsgKey.SECURE_MODE: secure_mode}
    if secure_mode:
        security.update(
            {
                MsgKey.AUTH_TOKEN: "site-auth-token",
                MsgKey.AUTH_TOKEN_SIGNATURE: "site-auth-signature",
            }
        )
    return make_cell_reply(
        CellReturnCode.OK,
        body={
            MsgKey.REPLY_TOPIC: Topic.HELLO_ACCEPTED,
            MsgKey.SESSION_ID: SESSION_ID,
            MsgKey.JOB_ID: "job-1",
            MsgKey.SITE_NAME: "site-1",
            MsgKey.HEARTBEAT_INTERVAL: heartbeat_interval,
            MsgKey.HEARTBEAT_TIMEOUT: heartbeat_timeout,
            **security,
        },
    )


def _result_accepted_reply():
    return make_cell_reply(CellReturnCode.OK, body={MsgKey.REPLY_TOPIC: Topic.RESULT_ACCEPTED})


class FakeCell:
    """The CJ cell as seen from the trainer: records the trainer's outbound requests/messages
    and lets a test deliver CJ->trainer control messages (TASK_READY/ABORT/SHUTDOWN)."""

    def __init__(self):
        self.fqcn = TRAINER_FQCN
        self.started = False
        self.stopped = False
        self.stop_calls = 0
        self.cbs = {}
        self.requests = []  # (topic, target, payload)
        self.request_messages = []
        self.request_kwargs = []
        self.fired = []  # (topic, targets, payload)
        self.on_request = None
        self.fobs_context = {}
        self.heartbeat_interval = 0.05
        self.heartbeat_timeout = 0.0
        self.secure_mode = False

    def get_fqcn(self):
        return self.fqcn

    def start(self):
        self.started = True

    def stop(self):
        self.stop_calls += 1
        self.stopped = True

    def register_request_cb(self, channel, topic, cb):
        assert channel == CHANNEL
        self.cbs[topic] = cb

    def update_fobs_context(self, props):
        self.fobs_context.update(props)

    def send_request(self, channel, topic, target, request, timeout=None, **kwargs):
        self.requests.append((topic, target, request.payload))
        self.request_messages.append(request)
        self.request_kwargs.append(kwargs)
        if self.on_request is not None:
            return self.on_request(topic, target, request)
        if topic == Topic.HELLO:
            return _hello_accepted_reply(self.heartbeat_interval, self.heartbeat_timeout, self.secure_mode)
        if topic == Topic.HEARTBEAT:
            return make_cell_reply(
                CellReturnCode.OK,
                body={MsgKey.REPLY_TOPIC: Topic.HEARTBEAT, MsgKey.SESSION_ID: SESSION_ID},
            )
        if topic == Topic.RESULT_READY:
            return _result_accepted_reply()
        return make_cell_reply(CellReturnCode.OK)

    def fire_and_forget(self, channel, topic, targets, message, **kwargs):
        self.fired.append((topic, tuple(targets), message.payload))

    def deliver(self, topic, origin, payload):
        return self.cbs[topic](new_cell_message({MessageHeaderKey.ORIGIN: origin}, payload))


class AttachFakeCell(FakeCell):
    def __init__(self):
        super().__init__()
        self.core_cell = SimpleNamespace(message_interceptor=None)

        def _set_message_interceptor(cb, *args, **kwargs):
            self.core_cell.message_interceptor = cb
            self.core_cell.message_interceptor_args = args
            self.core_cell.message_interceptor_kwargs = kwargs

        self.core_cell.set_message_interceptor = _set_message_interceptor
        self.fqcn = ATTACH_TRAINER_FQCN
        self.session_open_payload = {
            MsgKey.SESSION_ID: SESSION_ID,
            MsgKey.ATTACH_ID: ATTACH_ID,
            MsgKey.JOB_ID: "job-1",
            MsgKey.SITE_NAME: "site-1",
            MsgKey.TRAINER_FQCN: ATTACH_TRAINER_FQCN,
            MsgKey.PROTOCOL_VERSION: PROTOCOL_VERSION,
            MsgKey.RANK: "0",
            MsgKey.HEARTBEAT_INTERVAL: 0.05,
            MsgKey.HEARTBEAT_TIMEOUT: 0.0,
            MsgKey.TASK_EXCHANGE: {
                ConfigKey.TRAIN_TASK_NAME: "train",
                ConfigKey.EVAL_TASK_NAME: "validate",
                ConfigKey.SUBMIT_MODEL_TASK_NAME: "submit_model",
                ConfigKey.LAUNCH_ONCE: True,
            },
            MsgKey.MEMORY_GC_ROUNDS: 0,
            MsgKey.CUDA_EMPTY_CACHE: False,
        }
        self.session_reply = None
        self.open_on_start = True

    def start(self):
        super().start()
        if self.open_on_start:
            self.session_reply = self.deliver(Topic.SESSION_OPEN, CJ_FQCN, self.session_open_payload)


@pytest.fixture
def bootstrap_path(tmp_path):
    path = str(tmp_path / "bootstrap.json")
    write_bootstrap_config(
        path,
        {
            BootstrapKey.SCHEMA_VERSION: BOOTSTRAP_SCHEMA_VERSION,
            BootstrapKey.EXECUTION_MODE: EXTERNAL_PROCESS_EXECUTION_MODE,
            BootstrapKey.CONNECT_URL: "tcp://127.0.0.1:12345",
            BootstrapKey.CJ_FQCN: CJ_FQCN,
            BootstrapKey.TRAINER_FQCN: TRAINER_FQCN,
            BootstrapKey.LAUNCH_TOKEN: "the-token",
            BootstrapKey.JOB_ID: "job-1",
            BootstrapKey.SITE_NAME: "site-1",
            BootstrapKey.TASK_EXCHANGE: {
                ConfigKey.TRAIN_TASK_NAME: "train",
                ConfigKey.EVAL_TASK_NAME: "validate",
                ConfigKey.SUBMIT_MODEL_TASK_NAME: "submit_model",
                ConfigKey.LAUNCH_ONCE: True,
            },
            BootstrapKey.MEMORY_GC_ROUNDS: 3,
            BootstrapKey.CUDA_EMPTY_CACHE: True,
        },
    )
    return path


@pytest.fixture
def env(bootstrap_path, monkeypatch):
    cell = FakeCell()
    monkeypatch.setattr(cell_api, "Cell", MagicMock(return_value=cell))
    # Each real trainer is a dedicated process, but these tests construct many APIs in
    # one pytest process. Observe the F3 cleanup call without permanently shutting down
    # the process-global streaming executors used by later tests.
    cell.shutdown_f3_streaming = MagicMock()
    monkeypatch.setattr(cell_api, "_shutdown_f3_streaming", cell.shutdown_f3_streaming)
    cell.auth_filter = MagicMock()
    monkeypatch.setattr(cell_api, "set_add_auth_headers_filters", cell.auth_filter)
    return cell


@pytest.fixture
def attach_bootstrap_path(tmp_path):
    path = str(tmp_path / "attach.json")
    write_bootstrap_config(
        path,
        {
            BootstrapKey.SCHEMA_VERSION: BOOTSTRAP_SCHEMA_VERSION,
            BootstrapKey.EXECUTION_MODE: ATTACH_EXECUTION_MODE,
            BootstrapKey.ATTACH_ID: ATTACH_ID,
            BootstrapKey.SITE_NAME: "site-1",
            BootstrapKey.CONNECT_URL: "grpc://127.0.0.1:12345",
            BootstrapKey.CONNECTION_SECURITY: "clear",
            BootstrapKey.JOB_WAIT_TIMEOUT: 1.0,
        },
    )
    return path


@pytest.fixture
def attach_env(attach_bootstrap_path, monkeypatch):
    cell = AttachFakeCell()
    cell_ctor = MagicMock(return_value=cell)
    monkeypatch.setattr(cell_api, "Cell", cell_ctor)
    cell.cell_ctor = cell_ctor
    cell.shutdown_f3_streaming = MagicMock()
    monkeypatch.setattr(cell_api, "_shutdown_f3_streaming", cell.shutdown_f3_streaming)
    cell.auth_filter = MagicMock()
    monkeypatch.setattr(cell_api, "set_add_auth_headers_filters", cell.auth_filter)
    return cell


def _init_api(bootstrap_path, env, rank="0"):
    api = CellClientAPI(bootstrap_file=bootstrap_path)
    api.init(rank=rank)
    return api


def _set_launch_once(bootstrap_path, launch_once):
    config = read_bootstrap_config(bootstrap_path)
    config[BootstrapKey.TASK_EXCHANGE][ConfigKey.LAUNCH_ONCE] = launch_once
    write_bootstrap_config(bootstrap_path, config)


def _set_formats(bootstrap_path, params_exchange_format, server_expected_format):
    config = read_bootstrap_config(bootstrap_path)
    exchange = config[BootstrapKey.TASK_EXCHANGE]
    exchange[ConfigKey.EXCHANGE_FORMAT] = params_exchange_format
    exchange[ConfigKey.SERVER_EXPECTED_FORMAT] = server_expected_format
    write_bootstrap_config(bootstrap_path, config)


def _set_secure_mode(bootstrap_path, secure_mode):
    config = read_bootstrap_config(bootstrap_path)
    config[BootstrapKey.SECURE_MODE] = secure_mode
    write_bootstrap_config(bootstrap_path, config)


def _deliver_task(env, task_name="train", task_id=None, model=None, result_receiver_ids=None):
    task_id = task_id or uuid.uuid4().hex
    if model is None:
        model = FLModel(params={"w": [1.0]}, params_type=ParamsType.FULL)
    shareable = FLModelUtils.to_shareable(model)
    if result_receiver_ids is not None:
        shareable.set_header(FOBSContextKey.RECEIVER_IDS, result_receiver_ids)
    payload = {
        MsgKey.SESSION_ID: SESSION_ID,
        MsgKey.TASK_ID: task_id,
        MsgKey.TASK_NAME: task_name,
        MsgKey.MODEL: shareable,
    }
    reply = env.deliver(Topic.TASK_READY, CJ_FQCN, payload)
    return task_id, reply


def _wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _send_and_capture_error(api, model, errors):
    try:
        api.send(model)
    except BaseException as e:
        errors.append(e)


def _init_and_capture_error(api, errors):
    try:
        api.init()
    except BaseException as e:
        errors.append(e)


def _deliver_attach_task(env, task_id="task-1", attempt_id="attempt-1", result_receiver_ids=None):
    shareable = FLModelUtils.to_shareable(FLModel(params={"w": [1.0]}, params_type=ParamsType.FULL))
    if result_receiver_ids is not None:
        shareable.set_header(FOBSContextKey.RECEIVER_IDS, result_receiver_ids)
    return env.deliver(
        Topic.TASK_READY,
        CJ_FQCN,
        {
            MsgKey.SESSION_ID: SESSION_ID,
            MsgKey.TASK_ID: task_id,
            MsgKey.TASK_SEQ: 1,
            MsgKey.ATTEMPT_ID: attempt_id,
            MsgKey.TASK_NAME: "train",
            MsgKey.MODEL: shareable,
        },
    )


class TestAttachMode:
    def test_init_connects_to_cp_and_discovers_dynamic_cj(self, attach_bootstrap_path, attach_env):
        api = _init_api(attach_bootstrap_path, attach_env)

        assert api._cj_fqcn == CJ_FQCN
        assert api._job_id == "job-1"
        assert api._trainer_fqcn == ATTACH_TRAINER_FQCN
        assert attach_env.session_reply.payload[MsgKey.REPLY_TOPIC] == Topic.SESSION_ACCEPTED
        kwargs = attach_env.cell_ctor.call_args.kwargs
        assert kwargs["fqcn"] == ATTACH_TRAINER_FQCN
        assert kwargs["parent_url"] == "grpc://127.0.0.1:12345"
        assert kwargs["secure"] is False
        assert kwargs["credentials"] == {}
        assert kwargs["parent_resources"] == {"connection_security": "clear"}
        assert kwargs["auth_identity_map"] == {"site-1": "site-1"}
        assert MsgKey.CONNECT_URL not in attach_env.session_reply.payload
        assert MsgKey.CONNECTION_SECURITY not in attach_env.session_reply.payload
        assert attach_env.core_cell.message_interceptor == api._attach._pre_decode_guard
        api.shutdown()
        assert not attach_env.shutdown_f3_streaming.called

    def test_shared_file_profile_discovers_cj_owned_listener(self, attach_bootstrap_path, attach_env):
        rendezvous_dir = str(Path(attach_bootstrap_path).parent)
        listener_dir = Path(rendezvous_dir) / "lst_12345678"
        listener_dir.mkdir(mode=0o770)
        (listener_dir / "conns").mkdir(mode=0o770)
        marker = listener_dir / ".nvf_file_transport"
        marker.touch(mode=0o660)
        marker.chmod(0o660)
        config = read_bootstrap_config(attach_bootstrap_path)
        del config[BootstrapKey.CONNECT_URL]
        del config[BootstrapKey.CONNECTION_SECURITY]
        config.pop(BootstrapKey.CJ_FQCN, None)
        config[BootstrapKey.RENDEZVOUS_DIR] = rendezvous_dir
        write_bootstrap_config(attach_bootstrap_path, config)
        publisher = AttachEndpointPublisher(rendezvous_dir, "site-1", ATTACH_ID)
        publisher.publish(
            cj_fqcn=CJ_FQCN,
            trainer_fqcn=ATTACH_TRAINER_FQCN,
            connect_url=f"shared-file://0{rendezvous_dir}/lst_12345678",
            connection_security="clear",
        )

        try:
            api = _init_api(attach_bootstrap_path, attach_env)
            kwargs = attach_env.cell_ctor.call_args.kwargs
            assert kwargs["parent_url"] == f"shared-file://0{rendezvous_dir}/lst_12345678"
            assert kwargs["secure"] is False
            assert api._cj_fqcn == CJ_FQCN
            api.shutdown()
        finally:
            publisher.close()

    def test_shutdown_interrupts_unbounded_shared_file_rendezvous_wait(self, attach_bootstrap_path, attach_env):
        config = read_bootstrap_config(attach_bootstrap_path)
        del config[BootstrapKey.CONNECT_URL]
        del config[BootstrapKey.CONNECTION_SECURITY]
        config.pop(BootstrapKey.CJ_FQCN, None)
        config[BootstrapKey.RENDEZVOUS_DIR] = str(Path(attach_bootstrap_path).parent)
        config[BootstrapKey.JOB_WAIT_TIMEOUT] = None
        write_bootstrap_config(attach_bootstrap_path, config)
        api = CellClientAPI(bootstrap_file=attach_bootstrap_path)
        errors = []
        initializer = threading.Thread(target=lambda: _init_and_capture_error(api, errors))
        initializer.start()
        time.sleep(0.05)

        api.shutdown()
        initializer.join(timeout=1.0)

        assert not initializer.is_alive()
        assert len(errors) == 1
        assert "rendezvous wait was stopped" in str(errors[0])

    def test_pre_decode_guard_rejects_foreign_origins_without_touching_payload(self, attach_bootstrap_path):
        config = read_bootstrap_config(attach_bootstrap_path)
        del config[BootstrapKey.CONNECT_URL]
        del config[BootstrapKey.CONNECTION_SECURITY]
        config.pop(BootstrapKey.CJ_FQCN, None)
        config[BootstrapKey.RENDEZVOUS_DIR] = str(Path(attach_bootstrap_path).parent)
        write_bootstrap_config(attach_bootstrap_path, config)
        api = CellClientAPI(bootstrap_file=attach_bootstrap_path)
        canary = MagicMock(name="lazy_payload_canary")

        def _stream_message(origin, topic):
            return new_cell_message(
                {
                    MessageHeaderKey.CHANNEL: STREAM_CHANNEL,
                    MessageHeaderKey.TOPIC: STREAM_DATA_TOPIC,
                    MessageHeaderKey.ORIGIN: origin,
                    StreamHeaderKey.CHANNEL: CHANNEL,
                    StreamHeaderKey.TOPIC: topic,
                },
                canary,
            )

        cross_site_open = api._attach._pre_decode_guard(_stream_message("site-2.job-1", Topic.SESSION_OPEN))
        premature_task = api._attach._pre_decode_guard(_stream_message(CJ_FQCN, Topic.TASK_READY))
        same_site_open = api._attach._pre_decode_guard(_stream_message(CJ_FQCN, Topic.SESSION_OPEN))

        assert cross_site_open.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.AUTHENTICATION_ERROR
        assert premature_task.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.AUTHENTICATION_ERROR
        assert same_site_open is None
        assert canary.mock_calls == []

        api._cj_fqcn = CJ_FQCN
        foreign_after_binding = api._attach._pre_decode_guard(_stream_message("site-1.other-job", Topic.SESSION_OPEN))
        bound_task = api._attach._pre_decode_guard(_stream_message(CJ_FQCN, Topic.TASK_READY))
        assert foreign_after_binding.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.AUTHENTICATION_ERROR
        assert bound_task is None
        assert canary.mock_calls == []

    def test_session_open_is_idempotent_and_rejects_second_cj(self, attach_bootstrap_path, attach_env):
        api = _init_api(attach_bootstrap_path, attach_env)

        duplicate = attach_env.deliver(Topic.SESSION_OPEN, CJ_FQCN, attach_env.session_open_payload)
        foreign = attach_env.deliver(Topic.SESSION_OPEN, "site-1.other-job", attach_env.session_open_payload)

        assert duplicate.payload[MsgKey.REPLY_TOPIC] == Topic.SESSION_ACCEPTED
        assert foreign.payload[MsgKey.REPLY_TOPIC] == Topic.SESSION_REJECTED
        assert api._cj_fqcn == CJ_FQCN

    def test_rejected_session_open_does_not_poison_waiting_init(self, attach_bootstrap_path, attach_env):
        attach_env.open_on_start = False
        api = CellClientAPI(bootstrap_file=attach_bootstrap_path)
        init_errors = []

        def _init():
            try:
                api.init(rank="0")
            except BaseException as e:
                init_errors.append(e)

        init_thread = threading.Thread(target=_init)
        init_thread.start()
        assert _wait_until(lambda: Topic.SESSION_OPEN in attach_env.cbs)

        invalid_opens = []
        for key, value in (
            (MsgKey.ATTACH_ID, "other"),
            (MsgKey.SITE_NAME, "other-site"),
            (MsgKey.TRAINER_FQCN, "site-1.-client_api_other"),
            (MsgKey.PROTOCOL_VERSION, PROTOCOL_VERSION + 1),
            (MsgKey.RANK, "1"),
            (MsgKey.HEARTBEAT_INTERVAL, 0),
            (MsgKey.TASK_EXCHANGE, "not-a-dict"),
            (MsgKey.MEMORY_GC_ROUNDS, -1),
        ):
            payload = dict(attach_env.session_open_payload)
            payload[key] = value
            invalid_opens.append((CJ_FQCN, payload))
        missing_session = dict(attach_env.session_open_payload)
        missing_session.pop(MsgKey.SESSION_ID)
        invalid_job = dict(attach_env.session_open_payload)
        invalid_job[MsgKey.JOB_ID] = "job.with.extra.segment"
        invalid_opens.extend(
            [
                ("", dict(attach_env.session_open_payload)),
                ("site-2.job-1", dict(attach_env.session_open_payload)),
                ("site-1.other-job", dict(attach_env.session_open_payload)),
                (CJ_FQCN, missing_session),
                (CJ_FQCN, invalid_job),
            ]
        )

        for origin, payload in invalid_opens:
            reply = attach_env.deliver(Topic.SESSION_OPEN, origin, payload)
            assert reply.payload[MsgKey.REPLY_TOPIC] == Topic.SESSION_REJECTED
            assert api._session_id is None
            assert not api._attach._opened.is_set()
            assert init_thread.is_alive()

        accepted = attach_env.deliver(Topic.SESSION_OPEN, CJ_FQCN, attach_env.session_open_payload)
        init_thread.join(timeout=1.0)

        assert accepted.payload[MsgKey.REPLY_TOPIC] == Topic.SESSION_ACCEPTED
        assert not init_thread.is_alive()
        assert init_errors == []
        assert api._session_id == SESSION_ID
        api.shutdown()

    def test_attach_materializes_result_at_cj_without_site_auth(self, attach_bootstrap_path, attach_env):
        api = _init_api(attach_bootstrap_path, attach_env)
        try:
            _deliver_attach_task(attach_env, result_receiver_ids=["server.job"])
            api.receive()

            def _on_request(topic, target, request):
                if topic == Topic.RESULT_READY:
                    return make_cell_reply(
                        CellReturnCode.OK,
                        body={
                            MsgKey.REPLY_TOPIC: Topic.RESULT_ACCEPTED,
                            MsgKey.RESULT_ID: request.payload[MsgKey.RESULT_ID],
                            MsgKey.ACCEPTED_ATTEMPT_ID: request.payload[MsgKey.ATTEMPT_ID],
                        },
                    )
                return make_cell_reply(CellReturnCode.OK)

            attach_env.on_request = _on_request
            api.send(FLModel(params={"w": [2.0]}, params_type=ParamsType.FULL))

            result_request = [m for m in attach_env.request_messages if MsgKey.RESULT in m.payload][0]
            result_kwargs = attach_env.request_kwargs[attach_env.request_messages.index(result_request)]
            assert attach_env.cell_ctor.call_args.kwargs["secure"] is False
            attach_env.auth_filter.assert_not_called()
            assert result_request.get_header(MessageHeaderKey.PASS_THROUGH) is not True
            assert result_kwargs["receiver_ids"] == (CJ_FQCN,)
            assert result_kwargs["num_receivers"] == 1
        finally:
            api.shutdown()

    def test_decomposer_failure_does_not_half_bind_session(self, attach_bootstrap_path, attach_env, monkeypatch):
        attach_env.open_on_start = False
        api = CellClientAPI(bootstrap_file=attach_bootstrap_path)
        init_errors = []

        def _init():
            try:
                api.init(rank="0")
            except BaseException as e:
                init_errors.append(e)

        init_thread = threading.Thread(target=_init)
        init_thread.start()
        assert _wait_until(lambda: Topic.SESSION_OPEN in attach_env.cbs)

        register_decomposers = MagicMock(side_effect=RuntimeError("torch is unavailable"))
        monkeypatch.setattr(attach_session_module, "register_framework_decomposers", register_decomposers)
        rejected = attach_env.deliver(Topic.SESSION_OPEN, CJ_FQCN, attach_env.session_open_payload)

        assert rejected.payload[MsgKey.REPLY_TOPIC] == Topic.SESSION_REJECTED
        assert "torch is unavailable" in rejected.payload[MsgKey.REASON]
        # A rejected open must not bind the dynamic job Cell.
        assert api._cj_fqcn is None
        assert api._session_id is None
        assert not api._attach._opened.is_set()
        assert init_thread.is_alive()

        register_decomposers.side_effect = None
        accepted = attach_env.deliver(Topic.SESSION_OPEN, CJ_FQCN, attach_env.session_open_payload)
        init_thread.join(timeout=1.0)

        assert accepted.payload[MsgKey.REPLY_TOPIC] == Topic.SESSION_ACCEPTED
        assert not init_thread.is_alive()
        assert init_errors == []
        assert api._cj_fqcn == CJ_FQCN
        assert api._session_id == SESSION_ID
        api.shutdown()

    def test_secure_profile_requires_and_passes_mtls_credentials(self, attach_bootstrap_path, attach_env):
        profile_dir = Path(attach_bootstrap_path).parent
        ca_cert = profile_dir / "rootCA.pem"
        client_cert = profile_dir / "client.crt"
        client_key = profile_dir / "client.key"
        for path in (ca_cert, client_cert, client_key):
            path.write_text("test credential", encoding="utf-8")

        config = read_bootstrap_config(attach_bootstrap_path)
        config[BootstrapKey.CONNECT_URL] = "grpcs://site.example:9000"
        config[BootstrapKey.CONNECTION_SECURITY] = "mtls"
        config[BootstrapKey.CA_CERT] = str(ca_cert)
        config[BootstrapKey.SECURE_MODE] = True
        write_bootstrap_config(attach_bootstrap_path, config)

        api = _init_api(attach_bootstrap_path, attach_env)

        kwargs = attach_env.cell_ctor.call_args.kwargs
        assert kwargs["secure"] is True
        assert kwargs["credentials"] == {
            "ca_cert": str(ca_cert),
            "client_cert": str(client_cert),
            "client_key": str(client_key),
        }
        assert kwargs["parent_resources"] == {"connection_security": "mtls"}
        api.shutdown()

    def test_secure_cell_uses_site_credentials_over_clear_cp_transport(self, attach_bootstrap_path, attach_env):
        profile_dir = Path(attach_bootstrap_path).parent
        ca_cert = profile_dir / "rootCA.pem"
        client_cert = profile_dir / "client.crt"
        client_key = profile_dir / "client.key"
        for path in (ca_cert, client_cert, client_key):
            path.write_text("test credential", encoding="utf-8")

        config = read_bootstrap_config(attach_bootstrap_path)
        config[BootstrapKey.SECURE_MODE] = True
        config[BootstrapKey.CA_CERT] = str(ca_cert)
        config[BootstrapKey.AUTH_IDENTITY] = "custom-site-cn"
        write_bootstrap_config(attach_bootstrap_path, config)

        api = _init_api(attach_bootstrap_path, attach_env)

        kwargs = attach_env.cell_ctor.call_args.kwargs
        assert kwargs["secure"] is True
        assert kwargs["credentials"] == {
            "ca_cert": str(ca_cert),
            "client_cert": str(client_cert),
            "client_key": str(client_key),
        }
        assert kwargs["parent_resources"] == {"connection_security": "clear"}
        assert kwargs["auth_identity_map"] == {"site-1": "custom-site-cn"}
        attach_env.auth_filter.assert_not_called()

        claimed_secure = new_cell_message(
            {
                MessageHeaderKey.CHANNEL: STREAM_CHANNEL,
                MessageHeaderKey.TOPIC: STREAM_DATA_TOPIC,
                MessageHeaderKey.ORIGIN: CJ_FQCN,
                MessageHeaderKey.SECURE: True,
                StreamHeaderKey.CHANNEL: CHANNEL,
                StreamHeaderKey.TOPIC: Topic.TASK_READY,
            },
            MagicMock(name="undecoded_task"),
        )
        rejected = api._attach._pre_decode_guard(claimed_secure)
        claimed_secure.set_header(MessageHeaderKey.ENCRYPTED, True)
        assert rejected.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.AUTHENTICATION_ERROR
        assert api._attach._pre_decode_guard(claimed_secure) is None

        def _accept_result(topic, _target, request):
            assert topic == Topic.RESULT_READY
            return make_cell_reply(
                CellReturnCode.OK,
                body={
                    MsgKey.REPLY_TOPIC: Topic.RESULT_ACCEPTED,
                    MsgKey.RESULT_ID: request.payload[MsgKey.RESULT_ID],
                    MsgKey.ACCEPTED_ATTEMPT_ID: request.payload[MsgKey.ATTEMPT_ID],
                },
            )

        attach_env.on_request = _accept_result
        _deliver_attach_task(attach_env)
        model = api.receive(timeout=0.1)
        api.send(FLModel(params=model.params, params_type=ParamsType.FULL))
        result_index = next(i for i, (topic, _, _) in enumerate(attach_env.requests) if topic == Topic.RESULT_READY)
        assert attach_env.request_kwargs[result_index]["secure"] is True
        api.shutdown()

    def test_secure_profile_rejects_missing_mtls_client_credentials(self, attach_bootstrap_path, attach_env):
        ca_cert = Path(attach_bootstrap_path).parent / "rootCA.pem"
        ca_cert.write_text("test credential", encoding="utf-8")
        config = read_bootstrap_config(attach_bootstrap_path)
        config[BootstrapKey.CONNECT_URL] = "grpcs://site.example:9000"
        config[BootstrapKey.CONNECTION_SECURITY] = "mtls"
        config[BootstrapKey.CA_CERT] = str(ca_cert)
        config[BootstrapKey.SECURE_MODE] = True
        write_bootstrap_config(attach_bootstrap_path, config)

        api = CellClientAPI(bootstrap_file=attach_bootstrap_path)
        with pytest.raises(RuntimeError, match=r"missing or unreadable: client_cert, client_key"):
            api.init(rank="0")

        attach_env.cell_ctor.assert_not_called()

    def test_profile_rejects_bare_ca_tls_before_cell_construction(self, attach_bootstrap_path, attach_env):
        ca_cert = Path(attach_bootstrap_path).parent / "rootCA.pem"
        ca_cert.write_text("test credential", encoding="utf-8")
        config = read_bootstrap_config(attach_bootstrap_path)
        config[BootstrapKey.CONNECT_URL] = "grpcs://site.example:9000"
        config[BootstrapKey.CONNECTION_SECURITY] = "tls"
        config[BootstrapKey.CA_CERT] = str(ca_cert)
        write_bootstrap_config(attach_bootstrap_path, config)

        with pytest.raises(ValueError, match="bare-CA TLS attach is not supported"):
            CellClientAPI(bootstrap_file=attach_bootstrap_path)

        attach_env.cell_ctor.assert_not_called()

    def test_duplicate_task_is_queued_once_and_status_is_recoverable(self, attach_bootstrap_path, attach_env):
        api = _init_api(attach_bootstrap_path, attach_env)

        first = _deliver_attach_task(attach_env)
        with api._liveness_lock:
            api._last_cj_activity = None
        duplicate = _deliver_attach_task(attach_env, attempt_id="attempt-2")
        status = attach_env.deliver(
            Topic.TASK_STATUS,
            CJ_FQCN,
            {MsgKey.SESSION_ID: SESSION_ID, MsgKey.TASK_ID: "task-1"},
        )

        assert first.payload[MsgKey.REPLY_TOPIC] == Topic.TASK_ACCEPTED
        assert duplicate.payload[MsgKey.REPLY_TOPIC] == Topic.TASK_ACCEPTED
        assert status.payload[MsgKey.TASK_STATE] == TaskState.QUEUED
        assert api._last_cj_activity is not None
        assert api._task_queue.qsize() == 1
        assert api.receive().params == {"w": [1.0]}
        assert api._attach._task_states["task-1"] == TaskState.DELIVERED

    def test_task_status_stays_unknown_until_conversion_and_queueing_succeed(
        self, attach_bootstrap_path, attach_env, monkeypatch
    ):
        api = _init_api(attach_bootstrap_path, attach_env)
        conversion_started = threading.Event()
        release_conversion = threading.Event()

        def fail_conversion(_shareable):
            conversion_started.set()
            assert release_conversion.wait(timeout=2.0)
            raise ValueError("conversion failed")

        monkeypatch.setattr(cell_api.FLModelUtils, "from_shareable", fail_conversion)
        replies = []
        delivery = threading.Thread(target=lambda: replies.append(_deliver_attach_task(attach_env)))
        delivery.start()
        assert conversion_started.wait(timeout=1.0)

        status = attach_env.deliver(
            Topic.TASK_STATUS,
            CJ_FQCN,
            {MsgKey.SESSION_ID: SESSION_ID, MsgKey.TASK_ID: "task-1"},
        )
        duplicate = _deliver_attach_task(attach_env, attempt_id="attempt-2")
        assert status.payload[MsgKey.TASK_STATE] == TaskState.UNKNOWN
        assert duplicate.payload[MsgKey.REPLY_TOPIC] == Topic.TASK_STATUS
        assert duplicate.payload[MsgKey.TASK_STATE] == TaskState.UNKNOWN
        assert api._task_queue.empty()

        release_conversion.set()
        delivery.join(timeout=1.0)
        assert not delivery.is_alive()
        assert replies[0].payload[MsgKey.REPLY_TOPIC] == Topic.TASK_FAILED
        assert api._task_queue.empty()

        final_status = attach_env.deliver(
            Topic.TASK_STATUS,
            CJ_FQCN,
            {MsgKey.SESSION_ID: SESSION_ID, MsgKey.TASK_ID: "task-1"},
        )
        assert final_status.payload[MsgKey.TASK_STATE] == TaskState.UNKNOWN

    def test_lost_result_acceptance_is_recovered_by_result_status(self, attach_bootstrap_path, attach_env):
        api = _init_api(attach_bootstrap_path, attach_env)
        _deliver_attach_task(attach_env)
        assert api.receive() is not None
        accepted_result_id = None
        accepted_attempt_id = None

        def _on_request(topic, target, request):
            nonlocal accepted_attempt_id, accepted_result_id
            if topic == Topic.RESULT_READY:
                accepted_result_id = request.payload[MsgKey.RESULT_ID]
                accepted_attempt_id = request.payload[MsgKey.ATTEMPT_ID]
                raise RuntimeError("acceptance reply lost")
            if topic == Topic.RESULT_STATUS:
                assert request.payload[MsgKey.RESULT_ID] == accepted_result_id
                return make_cell_reply(
                    CellReturnCode.OK,
                    body={
                        MsgKey.REPLY_TOPIC: Topic.RESULT_STATUS,
                        MsgKey.RESULT_STATE: ResultState.ACCEPTED,
                        MsgKey.ACCEPTED_ATTEMPT_ID: accepted_attempt_id,
                    },
                )
            return make_cell_reply(CellReturnCode.OK)

        attach_env.on_request = _on_request
        api.send(FLModel(params={"w": [2.0]}, params_type=ParamsType.FULL))

        assert accepted_result_id
        assert api._attach._task_states["task-1"] == TaskState.COMPLETE

    def test_lost_result_acceptance_is_recovered_after_routine_shutdown(
        self, attach_bootstrap_path, attach_env, monkeypatch
    ):
        api = _init_api(attach_bootstrap_path, attach_env)
        _deliver_attach_task(attach_env)
        assert api.receive() is not None

        waiter = MagicMock()
        waiter.transaction_id = "canonical-tx"
        waiter.done.return_value = True
        waiter.wait.return_value = SimpleNamespace(
            status=TransferProgressState.COMPLETED,
            reason="all_receivers_succeeded",
        )
        monkeypatch.setattr(cell_api.DownloadService, "get_transfer_waiter", lambda tx_id: waiter)
        delete_transaction = MagicMock()
        monkeypatch.setattr(cell_api.DownloadService, "delete_transaction", delete_transaction)
        accepted_result_id = None
        accepted_attempt_id = None

        def _on_request(topic, target, request):
            nonlocal accepted_attempt_id, accepted_result_id
            if topic == Topic.RESULT_READY:
                accepted_result_id = request.payload[MsgKey.RESULT_ID]
                accepted_attempt_id = request.payload[MsgKey.ATTEMPT_ID]
                index = attach_env.request_messages.index(request)
                tx_created = attach_env.request_kwargs[index]["fobs_ctx_props"][
                    cell_api.RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY
                ]
                tx_created(SimpleNamespace(tx_id="canonical-tx"))
                attach_env.deliver(
                    Topic.SHUTDOWN,
                    CJ_FQCN,
                    {MsgKey.SESSION_ID: SESSION_ID},
                )
                raise RuntimeError("acceptance reply lost")
            if topic == Topic.RESULT_STATUS:
                assert request.payload[MsgKey.RESULT_ID] == accepted_result_id
                return make_cell_reply(
                    CellReturnCode.OK,
                    body={
                        MsgKey.REPLY_TOPIC: Topic.RESULT_STATUS,
                        MsgKey.RESULT_STATE: ResultState.ACCEPTED,
                        MsgKey.ACCEPTED_ATTEMPT_ID: accepted_attempt_id,
                    },
                )
            return make_cell_reply(CellReturnCode.OK)

        attach_env.on_request = _on_request
        api.send(FLModel(params={"w": [2.0]}, params_type=ParamsType.FULL))

        delete_transaction.assert_not_called()
        waiter.wait.assert_called_once()
        assert attach_env.stopped
        assert api._attach._task_states["task-1"] == TaskState.COMPLETE

    def test_uncertain_result_preserves_canonical_attempt_and_cancels_only_duplicate(
        self, attach_bootstrap_path, attach_env, monkeypatch
    ):
        api = _init_api(attach_bootstrap_path, attach_env)
        _deliver_attach_task(attach_env)
        assert api.receive() is not None

        waiters = {}
        for tx_id in ("tx-1", "tx-2"):
            waiter = MagicMock()
            waiter.transaction_id = tx_id
            waiter.done.return_value = True
            waiter.wait.return_value = SimpleNamespace(
                status=TransferProgressState.COMPLETED,
                reason="all_receivers_succeeded",
            )
            waiters[tx_id] = waiter
        monkeypatch.setattr(cell_api.DownloadService, "get_transfer_waiter", lambda tx_id: waiters[tx_id])
        delete_transaction = MagicMock()
        monkeypatch.setattr(cell_api.DownloadService, "delete_transaction", delete_transaction)

        attempts = []

        def _on_request(topic, target, request):
            if topic == Topic.RESULT_READY:
                attempts.append(request.payload[MsgKey.ATTEMPT_ID])
                index = attach_env.request_messages.index(request)
                tx_created = attach_env.request_kwargs[index]["fobs_ctx_props"][
                    cell_api.RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY
                ]
                tx_created(SimpleNamespace(tx_id=f"tx-{len(attempts)}"))
                if len(attempts) == 1:
                    raise RuntimeError("acceptance reply lost")
                return make_cell_reply(
                    CellReturnCode.OK,
                    body={
                        MsgKey.REPLY_TOPIC: Topic.RESULT_ACCEPTED,
                        MsgKey.RESULT_ID: request.payload[MsgKey.RESULT_ID],
                        MsgKey.ACCEPTED_ATTEMPT_ID: attempts[0],
                    },
                )
            if topic == Topic.RESULT_STATUS:
                return None
            return make_cell_reply(CellReturnCode.OK)

        attach_env.on_request = _on_request
        api.send(FLModel(params={"w": [2.0]}, params_type=ParamsType.FULL))

        assert len(attempts) == 2
        delete_transaction.assert_called_once_with("tx-2")
        waiters["tx-1"].wait.assert_called()
        waiters["tx-2"].wait.assert_not_called()

    def test_all_control_replies_lost_keeps_lazy_source_until_receiver_confirmation(
        self, attach_bootstrap_path, attach_env, monkeypatch
    ):
        api = _init_api(attach_bootstrap_path, attach_env)
        _deliver_attach_task(attach_env)
        assert api.receive() is not None

        canonical_complete = threading.Event()
        completed_outcome = SimpleNamespace(
            status=TransferProgressState.COMPLETED,
            reason="all_receivers_succeeded",
        )
        canonical_waiter = MagicMock()
        canonical_waiter.transaction_id = "tx-1"
        canonical_waiter.done.side_effect = canonical_complete.is_set
        canonical_waiter.wait.side_effect = lambda timeout=None: (
            completed_outcome if canonical_complete.is_set() else None
        )
        duplicate_waiter = MagicMock()
        duplicate_waiter.transaction_id = "tx-2"
        duplicate_waiter.done.return_value = False
        duplicate_waiter.wait.return_value = None
        waiters = {"tx-1": canonical_waiter, "tx-2": duplicate_waiter}
        monkeypatch.setattr(cell_api.DownloadService, "get_transfer_waiter", lambda tx_id: waiters[tx_id])
        delete_transaction = MagicMock()
        monkeypatch.setattr(cell_api.DownloadService, "delete_transaction", delete_transaction)
        monkeypatch.setattr(attach_session_module, "_AMBIGUOUS_RESULT_POLL_INTERVAL", 0.01)

        attempts = []

        def _on_request(topic, target, request):
            if topic == Topic.RESULT_READY:
                attempts.append(request.payload[MsgKey.ATTEMPT_ID])
                index = attach_env.request_messages.index(request)
                tx_created = attach_env.request_kwargs[index]["fobs_ctx_props"][
                    cell_api.RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY
                ]
                tx_created(SimpleNamespace(tx_id=f"tx-{len(attempts)}"))
                raise RuntimeError("acceptance reply lost")
            if topic == Topic.RESULT_STATUS:
                return None
            return make_cell_reply(CellReturnCode.OK)

        attach_env.on_request = _on_request
        errors = []
        sender = threading.Thread(
            target=lambda: _send_and_capture_error(
                api,
                FLModel(params={"w": [2.0]}, params_type=ParamsType.FULL),
                errors,
            )
        )
        sender.start()
        assert _wait_until(lambda: len(attempts) == 2)
        time.sleep(0.03)
        assert sender.is_alive()
        delete_transaction.assert_not_called()

        canonical_complete.set()
        sender.join(timeout=1.0)

        assert not sender.is_alive()
        assert errors == []
        delete_transaction.assert_called_once_with("tx-2")
        canonical_waiter.wait.assert_called()

    def test_second_send_for_completed_attach_task_raises_even_without_cache_clear(
        self, attach_bootstrap_path, attach_env
    ):
        api = _init_api(attach_bootstrap_path, attach_env)
        _deliver_attach_task(attach_env)
        assert api.receive() is not None

        def _on_request(topic, target, request):
            if topic == Topic.RESULT_READY:
                return make_cell_reply(
                    CellReturnCode.OK,
                    body={
                        MsgKey.REPLY_TOPIC: Topic.RESULT_ACCEPTED,
                        MsgKey.RESULT_ID: request.payload[MsgKey.RESULT_ID],
                        MsgKey.ACCEPTED_ATTEMPT_ID: request.payload[MsgKey.ATTEMPT_ID],
                    },
                )
            return make_cell_reply(CellReturnCode.OK)

        attach_env.on_request = _on_request
        result = FLModel(params={"w": [2.0]}, params_type=ParamsType.FULL)
        api.send(result, clear_cache=False)

        with pytest.raises(TrainerSessionError, match="already published"):
            api.send(result, clear_cache=False)
        assert [topic for topic, _, _ in attach_env.requests].count(Topic.RESULT_READY) == 1

    def test_completed_task_ledger_is_bounded_with_stale_watermark(self, attach_bootstrap_path, attach_env):
        api = _init_api(attach_bootstrap_path, attach_env)

        for index in range(300):
            task_id = f"task-{index}"
            assert api._attach.reserve_task(task_id, f"attempt-{index}", index + 1) is None
            api._attach.mark_task_complete(task_id)

        duplicate = api._attach.reserve_task("task-0", "delayed-attempt", 1)
        assert duplicate.payload[MsgKey.REPLY_TOPIC] == Topic.TASK_FAILED
        assert "stale task_seq" in duplicate.payload[MsgKey.REASON]
        assert len(api._attach._task_states) == 256

    def test_lifecycle_cleanup_defers_while_result_source_is_live(self, attach_bootstrap_path, attach_env):
        api = _init_api(attach_bootstrap_path, attach_env)
        api._result_send_active = True

        api._attach.cleanup()

        assert not attach_env.stopped
        api._result_send_active = False
        api._attach.cleanup()
        assert attach_env.stopped
        assert not attach_env.shutdown_f3_streaming.called


def test_shutdown_f3_streaming_is_ordered_and_safe_to_repeat(monkeypatch):
    calls = []
    monkeypatch.setattr(cell_api.DownloadService, "shutdown", lambda: calls.append("download"))
    monkeypatch.setattr(cell_api.reliable_retry_scheduler, "shutdown", lambda: calls.append("retry"))
    monkeypatch.setattr(cell_api, "stream_shutdown", lambda: calls.append("stream"))

    cell_api._shutdown_f3_streaming()
    cell_api._shutdown_f3_streaming()

    assert calls == ["download", "retry", "stream", "download", "retry", "stream"]


def test_shutdown_f3_streaming_attempts_every_stage_and_can_retry(monkeypatch):
    calls = []
    fail_download_once = True

    def shutdown_download():
        nonlocal fail_download_once
        calls.append("download")
        if fail_download_once:
            fail_download_once = False
            raise RuntimeError("download failed")

    monkeypatch.setattr(cell_api.DownloadService, "shutdown", shutdown_download)
    monkeypatch.setattr(cell_api.reliable_retry_scheduler, "shutdown", lambda: calls.append("retry"))
    monkeypatch.setattr(cell_api, "stream_shutdown", lambda: calls.append("stream"))

    with pytest.raises(RuntimeError, match="download service"):
        cell_api._shutdown_f3_streaming()
    assert calls == ["download", "retry", "stream"]

    cell_api._shutdown_f3_streaming()
    assert calls == ["download", "retry", "stream", "download", "retry", "stream"]


def test_trainer_session_error_is_runtime_error():
    assert issubclass(TrainerSessionError, RuntimeError)


class TestInit:
    def test_external_owner_watchdog_terminates_group_when_cj_disappears(self, bootstrap_path, monkeypatch):
        config = read_bootstrap_config(bootstrap_path)
        config[BootstrapKey.CJ_PID] = 424242
        write_bootstrap_config(bootstrap_path, config)
        api = CellClientAPI(bootstrap_file=bootstrap_path)
        terminated = threading.Event()
        monkeypatch.setattr(cell_api, "_OWNER_WATCHDOG_INTERVAL", 0.001)
        monkeypatch.setattr(api, "_owner_process_alive", lambda: False)
        monkeypatch.setattr(api, "_terminate_orphaned_process_group", terminated.set)

        api._start_owner_watchdog()

        assert terminated.wait(1.0)
        api._stop_owner_watchdog()

    @pytest.mark.skipif(os.name != "posix", reason="process-group signals are POSIX")
    def test_orphan_termination_escalates_ignored_sigterm(self, bootstrap_path, monkeypatch):
        api = CellClientAPI(bootstrap_file=bootstrap_path)
        signals = []
        exits = []
        monkeypatch.setattr(cell_api.os, "getpgrp", lambda: 1234)
        monkeypatch.setattr(cell_api.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)))
        monkeypatch.setattr(cell_api.time, "sleep", lambda _timeout: None)
        monkeypatch.setattr(cell_api.os, "_exit", exits.append)

        api._terminate_orphaned_process_group()

        assert signals == [(1234, process_signal.SIGTERM), (1234, process_signal.SIGKILL)]
        assert exits == [1]

    def test_init_does_hello_handshake(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        try:
            assert env.started
            hello = [r for r in env.requests if r[0] == Topic.HELLO][0]
            _, target, payload = hello
            assert target == CJ_FQCN
            assert payload[MsgKey.TRAINER_FQCN] == TRAINER_FQCN
            assert payload[MsgKey.PROOF] == "the-token"
            assert payload[MsgKey.PROTOCOL_VERSION] == PROTOCOL_VERSION
            assert payload[MsgKey.JOB_ID] == "job-1"
            assert payload[MsgKey.RANK] == "0"
            assert api._session_id == SESSION_ID
            assert api._memory_gc_rounds == 3
            assert api._cuda_empty_cache is True
            for topic in (Topic.TASK_READY, Topic.ABORT, Topic.SHUTDOWN):
                assert topic in env.cbs
        finally:
            api.shutdown()

    def test_init_raises_on_hello_rejected(self, bootstrap_path, env):
        env.on_request = lambda topic, target, request: make_cell_reply(
            CellReturnCode.OK, body={MsgKey.REPLY_TOPIC: Topic.HELLO_REJECTED, MsgKey.REASON: "bad token"}
        )
        api = CellClientAPI(bootstrap_file=bootstrap_path)
        with pytest.raises(TrainerSessionError, match="bad token"):
            api.init(rank="0")
        assert env.stopped, "a failed HELLO must stop the cell"

    def test_secure_init_rejects_missing_accepted_credential_without_binding_session(self, bootstrap_path, env):
        _set_secure_mode(bootstrap_path, True)
        reply = _hello_accepted_reply(secure_mode=True)
        reply.payload.pop(MsgKey.AUTH_TOKEN_SIGNATURE)
        env.on_request = lambda _topic, _target, _request: reply
        api = CellClientAPI(bootstrap_file=bootstrap_path)

        with pytest.raises(TrainerSessionError, match="no site auth token signature"):
            api.init(rank="0")

        assert api._session_id is None
        assert env.stopped

    def test_non_secure_init_accepts_protocol_v1_reply_without_secure_mode(self, bootstrap_path, env):
        reply = _hello_accepted_reply(secure_mode=False)
        reply.payload.pop(MsgKey.SECURE_MODE)
        env.on_request = lambda _topic, _target, _request: reply
        api = CellClientAPI(bootstrap_file=bootstrap_path)

        try:
            api.init(rank="0")

            assert api._session_id == SESSION_ID
            assert api._secure_mode is False
            env.auth_filter.assert_not_called()
        finally:
            api.shutdown()

    def test_secure_init_rejects_protocol_v1_reply_without_secure_mode(self, bootstrap_path, env):
        _set_secure_mode(bootstrap_path, True)
        reply = _hello_accepted_reply(secure_mode=True)
        reply.payload.pop(MsgKey.SECURE_MODE)
        env.on_request = lambda _topic, _target, _request: reply
        api = CellClientAPI(bootstrap_file=bootstrap_path)

        with pytest.raises(TrainerSessionError, match="secure_mode disagrees"):
            api.init(rank="0")

        assert api._session_id is None
        assert env.stopped

    def test_init_stops_retrying_hello_at_overall_deadline(self, bootstrap_path, env, monkeypatch):
        monkeypatch.setattr(cell_api, "_HELLO_TIMEOUT", 0.03)
        monkeypatch.setattr(cell_api, "_HELLO_RETRY_INTERVAL", 0.005)
        clock = [0.0]

        def monotonic():
            now = clock[0]
            clock[0] += 0.005
            return now

        monkeypatch.setattr(cell_api, "time", SimpleNamespace(monotonic=monotonic, sleep=MagicMock()))
        env.on_request = lambda topic, target, request: make_cell_reply(CellReturnCode.TIMEOUT)
        api = CellClientAPI(bootstrap_file=bootstrap_path)
        try:
            with pytest.raises(TrainerSessionError, match="no HELLO reply"):
                api.init(rank="0")

            assert len([request for request in env.requests if request[0] == Topic.HELLO]) > 1
            assert env.stopped
        finally:
            api.shutdown()

    def test_init_can_retry_after_failed_hello(self, bootstrap_path, monkeypatch):
        first_cell = FakeCell()
        first_cell.on_request = lambda topic, target, request: make_cell_reply(
            CellReturnCode.OK, body={MsgKey.REPLY_TOPIC: Topic.HELLO_REJECTED, MsgKey.REASON: "bad token"}
        )
        second_cell = FakeCell()
        second_cell.heartbeat_interval = 0.01
        second_cell.heartbeat_timeout = 0.2
        monkeypatch.setattr(cell_api, "Cell", MagicMock(side_effect=[first_cell, second_cell]))
        monkeypatch.setattr(cell_api, "_shutdown_f3_streaming", MagicMock())
        api = CellClientAPI(bootstrap_file=bootstrap_path)
        try:
            with pytest.raises(TrainerSessionError, match="bad token"):
                api.init(rank="0")

            api.init(rank="0")

            assert first_cell.stopped
            assert second_cell.started
            assert api._heartbeat_thread is not None and api._heartbeat_thread.is_alive()
        finally:
            api.shutdown()

    def test_secure_init_retry_installs_auth_filters_on_replacement_cell(self, bootstrap_path, monkeypatch):
        _set_secure_mode(bootstrap_path, True)
        first_cell = FakeCell()
        first_cell.secure_mode = True
        second_cell = FakeCell()
        second_cell.secure_mode = True
        auth_filter = MagicMock()
        monkeypatch.setattr(cell_api, "Cell", MagicMock(side_effect=[first_cell, second_cell]))
        monkeypatch.setattr(cell_api, "set_add_auth_headers_filters", auth_filter)
        monkeypatch.setattr(cell_api, "_shutdown_f3_streaming", MagicMock())
        api = CellClientAPI(bootstrap_file=bootstrap_path)
        start_heartbeat = MagicMock(side_effect=[RuntimeError("heartbeat start failed"), None])
        monkeypatch.setattr(api, "_start_heartbeat", start_heartbeat)

        try:
            with pytest.raises(RuntimeError, match="heartbeat start failed"):
                api.init(rank="0")

            api.init(rank="0")

            assert first_cell.stopped
            assert second_cell.started
            assert [entry.args[0] for entry in auth_filter.call_args_list] == [first_cell, second_cell]
            for entry in auth_filter.call_args_list:
                assert entry.kwargs == {
                    "client_name": "site-1",
                    "auth_token": "site-auth-token",
                    "token_signature": "site-auth-signature",
                }
        finally:
            api.shutdown()

    def test_shutdown_before_init_keeps_api_closed(self, bootstrap_path, env):
        api = CellClientAPI(bootstrap_file=bootstrap_path)

        api.shutdown()
        api.init(rank="0")

        cell_api.Cell.assert_not_called()
        assert api._closed is True

    def test_shutdown_waits_for_in_flight_init_then_stops_cell(self, bootstrap_path, monkeypatch):
        cell = FakeCell()
        start_entered = threading.Event()
        allow_start = threading.Event()

        def blocked_start():
            start_entered.set()
            assert allow_start.wait(1.0)
            cell.started = True

        cell.start = blocked_start
        monkeypatch.setattr(cell_api, "Cell", MagicMock(return_value=cell))
        monkeypatch.setattr(cell_api, "_shutdown_f3_streaming", MagicMock())
        api = CellClientAPI(bootstrap_file=bootstrap_path)
        init_thread = threading.Thread(target=api.init, kwargs={"rank": "0"})
        shutdown_thread = threading.Thread(target=api.shutdown)

        init_thread.start()
        assert start_entered.wait(0.5)
        shutdown_thread.start()
        assert shutdown_thread.is_alive()
        allow_start.set()
        init_thread.join(timeout=1.0)
        shutdown_thread.join(timeout=1.0)

        assert not init_thread.is_alive()
        assert not shutdown_thread.is_alive()
        assert cell.stopped
        assert api._closed is True

    def test_init_stops_partially_started_cell_before_retry(self, bootstrap_path, monkeypatch):
        first_cell = FakeCell()
        first_cell.start = MagicMock(side_effect=RuntimeError("start failed"))
        second_cell = FakeCell()
        monkeypatch.setattr(cell_api, "Cell", MagicMock(side_effect=[first_cell, second_cell]))
        monkeypatch.setattr(cell_api, "_shutdown_f3_streaming", MagicMock())
        api = CellClientAPI(bootstrap_file=bootstrap_path)
        try:
            with pytest.raises(RuntimeError, match="start failed"):
                api.init(rank="0")

            api.init(rank="0")

            assert first_cell.stopped
            assert second_cell.started
        finally:
            api.shutdown()

    def test_repeated_control_rank_init_reuses_session_and_heartbeat(self, bootstrap_path, env):
        env.heartbeat_interval = 0.01
        env.heartbeat_timeout = 0.2
        api = CellClientAPI(bootstrap_file=bootstrap_path)
        try:
            api.init(rank="0")
            heartbeat_thread = api._heartbeat_thread

            api.init(rank="0")

            cell_api.Cell.assert_called_once()
            assert len([request for request in env.requests if request[0] == Topic.HELLO]) == 1
            assert api._heartbeat_thread is heartbeat_thread
            assert heartbeat_thread is not None and heartbeat_thread.is_alive()
        finally:
            api.shutdown()

    def test_repeated_passive_rank_init_does_not_open_control_session(self, bootstrap_path, env):
        api = CellClientAPI(bootstrap_file=bootstrap_path)
        try:
            api.init(rank="1")
            api.init(rank="0")

            assert api._rank == "1"
            assert api._is_control_rank is False
            cell_api.Cell.assert_not_called()
            assert not env.started
        finally:
            api.shutdown()

    def test_non_control_rank_has_passive_api(self, bootstrap_path, env):
        api = CellClientAPI(bootstrap_file=bootstrap_path)
        try:
            api.init(rank="1")
            # rank != 0 opens no session (rank contract): no cell built, receive None, not running
            assert not env.started
            assert api.receive() is None
            assert api.is_running() is False
            with pytest.raises(RuntimeError, match="only rank 0 can call log"):
                api.log("accuracy", 0.9, AnalyticsDataType.SCALAR)
        finally:
            api.shutdown()

    def test_non_control_rank_send_is_noop(self, bootstrap_path, env):
        api = CellClientAPI(bootstrap_file=bootstrap_path)
        model = FLModel(params={"w": [1.0]})
        try:
            api.init(rank="1")

            api.send(model)

            assert model.params == {"w": [1.0]}
            cell_api.Cell.assert_not_called()
            assert env.requests == []
        finally:
            api.shutdown()


class TestReceiveSend:
    def test_receive_gets_direct_cell_shareable(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        try:
            task_id, accepted = _deliver_task(env, task_name="train")
            assert accepted.payload[MsgKey.REPLY_TOPIC] == Topic.TASK_ACCEPTED

            model = api.receive()
            assert isinstance(model, FLModel)
            assert model.params == {"w": [1.0]}
            assert api.get_task_name() == "train"
            assert api.is_train() is True and api.is_evaluate() is False
            # Cell decoded the Shareable before invoking TASK_READY; there is no second
            # payload protocol or acknowledgement.
        finally:
            api.shutdown()

    def test_clear_removes_all_task_scoped_state(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        try:
            _deliver_task(env, result_receiver_ids=("server",))
            api.receive()

            api.clear()

            assert api._result_receiver_ids is None
            assert api.is_train() is False
            assert api.is_evaluate() is False
            assert api.is_submit_model() is False
            with pytest.raises(RuntimeError, match="no current task"):
                api.get_task_name()
            with pytest.raises(RuntimeError, match='"receive" needs to be called'):
                api.send(FLModel(params={"w": [2.0]}))
        finally:
            api.shutdown()

    def test_declared_pytorch_conversion_runs_at_receive_send_boundary(self, bootstrap_path, env):
        torch = pytest.importorskip("torch")
        import numpy as np

        _set_formats(bootstrap_path, ExchangeFormat.PYTORCH, ExchangeFormat.NUMPY)
        api = _init_api(bootstrap_path, env)
        try:
            _deliver_task(
                env,
                model=FLModel(params={"w": np.asarray([1.0, 2.0])}, params_type=ParamsType.FULL),
            )
            received = api.receive()
            assert isinstance(received.params["w"], torch.Tensor)

            api.send(FLModel(params={"w": received.params["w"] + 1}), clear_cache=False)

            result_payload = [p for t, _, p in env.requests if t == Topic.RESULT_READY][0]
            wire_model = FLModelUtils.from_shareable(result_payload[MsgKey.RESULT])
            assert isinstance(wire_model.params["w"], np.ndarray)
            np.testing.assert_array_equal(wire_model.params["w"], np.asarray([2.0, 3.0]))
        finally:
            api.shutdown()

    def test_task_is_queued_before_task_accepted(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        try:
            _, accepted = _deliver_task(env)
            assert accepted.payload[MsgKey.REPLY_TOPIC] == Topic.TASK_ACCEPTED
            assert api._task_queue.qsize() == 1
            assert env.fobs_context[FOBSContextKey.ABORT_SIGNAL] is api._abort_signal
        finally:
            api.shutdown()

    def test_invalid_direct_task_is_rejected(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        try:
            reply = env.deliver(
                Topic.TASK_READY,
                CJ_FQCN,
                {
                    MsgKey.SESSION_ID: SESSION_ID,
                    MsgKey.TASK_ID: "bad-task",
                    MsgKey.TASK_NAME: "train",
                    MsgKey.MODEL: {},
                },
            )
            assert reply.payload[MsgKey.REPLY_TOPIC] == Topic.TASK_FAILED
            assert api.receive(timeout=0.01) is None
        finally:
            api.shutdown()

    def test_send_publishes_pass_through_result_without_counting_cj_as_receiver(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        try:
            task_id, _ = _deliver_task(env)
            api.receive()

            api.send(FLModel(params={"w": [2.0]}, params_type=ParamsType.FULL))

            result_ready = [r for r in env.requests if r[0] == Topic.RESULT_READY][0]
            _, target, payload = result_ready
            assert target == CJ_FQCN
            assert payload[MsgKey.SESSION_ID] == SESSION_ID
            assert payload[MsgKey.TASK_ID] == task_id
            assert isinstance(payload[MsgKey.RESULT], Shareable)
            result_request = [m for m in env.request_messages if MsgKey.RESULT in m.payload][0]
            result_kwargs = env.request_kwargs[env.request_messages.index(result_request)]
            assert result_request.get_header(MessageHeaderKey.PASS_THROUGH) is True
            assert result_kwargs["receiver_ids"] is None
            assert result_kwargs["num_receivers"] == 1
            assert api._closed is False
            assert env.stopped is False
        finally:
            api.shutdown()

    def test_send_preserves_declared_ultimate_result_receivers(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        try:
            _deliver_task(env, result_receiver_ids=["server.job", "peer.job"])
            api.receive()

            api.send(FLModel(params={"w": [2.0]}, params_type=ParamsType.FULL))

            result_request = [m for m in env.request_messages if MsgKey.RESULT in m.payload][0]
            result_kwargs = env.request_kwargs[env.request_messages.index(result_request)]
            assert result_kwargs["receiver_ids"] == ("server.job", "peer.job")
            assert result_kwargs["num_receivers"] == 2
        finally:
            api.shutdown()

    def test_secure_mode_send_keeps_pass_through_and_ultimate_receivers(self, bootstrap_path, env):
        _set_secure_mode(bootstrap_path, True)
        env.secure_mode = True
        api = _init_api(bootstrap_path, env)
        try:
            _deliver_task(env, result_receiver_ids=["server.job"])
            api.receive()

            api.send(FLModel(params={"w": [2.0]}, params_type=ParamsType.FULL))

            result_request = [m for m in env.request_messages if MsgKey.RESULT in m.payload][0]
            result_kwargs = env.request_kwargs[env.request_messages.index(result_request)]
            assert result_request.get_header(MessageHeaderKey.PASS_THROUGH) is True
            env.auth_filter.assert_called_once_with(
                env,
                client_name="site-1",
                auth_token="site-auth-token",
                token_signature="site-auth-signature",
            )
            assert result_kwargs["receiver_ids"] == ("server.job",)
            assert result_kwargs["num_receivers"] == 1
        finally:
            api.shutdown()

    def test_per_task_send_closes_cell_after_result_acceptance(self, bootstrap_path, env):
        _set_launch_once(bootstrap_path, False)
        api = _init_api(bootstrap_path, env)
        assert api._launch_once is False
        _deliver_task(env)
        api.receive()

        api.send(FLModel(params={"w": [2.0]}, params_type=ParamsType.FULL))

        assert api._closed is True
        assert env.stopped is True
        assert env.stop_calls == 1

    def test_send_before_receive_raises(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        try:
            with pytest.raises(RuntimeError, match="receive.*before sending"):
                api.send(FLModel(params={"w": [1.0]}))
        finally:
            api.shutdown()

    def test_send_tracks_actual_via_downloader_transaction_while_request_is_pending(
        self, bootstrap_path, env, monkeypatch
    ):
        waiter = MagicMock()
        waiter.done.return_value = False
        waiter.wait.return_value = SimpleNamespace(
            status=TransferProgressState.COMPLETED, reason="all_receivers_succeeded"
        )
        get_transfer_waiter = MagicMock(return_value=waiter)
        monkeypatch.setattr(cell_api.DownloadService, "get_transfer_waiter", get_transfer_waiter)

        def on_request(topic, target, request):
            if topic == Topic.HELLO:
                return _hello_accepted_reply()
            if topic == Topic.RESULT_READY:
                kwargs = env.request_kwargs[-1]
                cb = kwargs["fobs_ctx_props"][cell_api.RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY]
                cb(SimpleNamespace(tx_id="actual-via-tx"))
                assert kwargs["progress_wait_cb"]() is True
                return _result_accepted_reply()
            return make_cell_reply(CellReturnCode.OK)

        env.on_request = on_request
        api = _init_api(bootstrap_path, env)
        try:
            _deliver_task(env)
            api.receive()
            api.send(FLModel(params={"w": [2.0]}))
            waiter.done.assert_called()
            get_transfer_waiter.assert_called_once_with("actual-via-tx")
        finally:
            api.shutdown()

    def test_per_task_send_stays_alive_after_cj_acceptance_until_downstream_download_finishes(
        self, bootstrap_path, env, monkeypatch
    ):
        _set_launch_once(bootstrap_path, False)
        accepted = threading.Event()
        transfer_completed = threading.Event()
        waiter = MagicMock()
        waiter.done.side_effect = transfer_completed.is_set
        waiter.wait.side_effect = lambda timeout=None: (
            SimpleNamespace(status=TransferProgressState.COMPLETED, reason="all_receivers_succeeded")
            if transfer_completed.wait(timeout)
            else None
        )
        monkeypatch.setattr(cell_api.DownloadService, "get_transfer_waiter", lambda _tx_id: waiter)

        def on_request(topic, target, request):
            if topic == Topic.HELLO:
                return _hello_accepted_reply()
            if topic == Topic.RESULT_READY:
                kwargs = env.request_kwargs[-1]
                kwargs["fobs_ctx_props"][cell_api.RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY](
                    SimpleNamespace(tx_id="downstream-result-tx")
                )
                accepted.set()
                return _result_accepted_reply()
            return make_cell_reply(CellReturnCode.OK)

        env.on_request = on_request
        api = _init_api(bootstrap_path, env)
        errors = []
        try:
            _deliver_task(
                env,
                model=FLModel(params={"w": [1.0]}, optimizer_params={"momentum": [0.5]}, params_type=ParamsType.FULL),
            )
            received_model = api.receive()
            # FLModelUtils does not carry optimizer_params through Shareable; populate
            # task-local optimizer state directly so cache release covers it too.
            received_model.optimizer_params = {"momentum": [0.5]}
            sent_model = FLModel(
                params={"w": [2.0]}, optimizer_params={"momentum": [0.75]}, params_type=ParamsType.FULL
            )

            def send_result():
                try:
                    api.send(sent_model)
                except BaseException as e:
                    errors.append(e)

            sender = threading.Thread(target=send_result)
            sender.start()
            assert accepted.wait(0.5)
            time.sleep(0.05)
            assert sender.is_alive()
            assert env.stopped is False
            assert sent_model.params == {"w": [2.0]}
            assert sent_model.optimizer_params == {"momentum": [0.75]}
            assert received_model.params == {"w": [1.0]}
            assert received_model.optimizer_params == {"momentum": [0.5]}
            assert api._fl_model is received_model

            transfer_completed.set()
            sender.join(timeout=0.5)
            assert not sender.is_alive()
            assert errors == []
            assert env.stopped is True
            assert sent_model.params is None
            assert sent_model.optimizer_params is None
            assert received_model.params is None
            assert received_model.optimizer_params is None
            assert api._fl_model is None
            assert api._receive_called is False
            assert api._current_task is None
            assert api._result_receiver_ids is None
        finally:
            transfer_completed.set()
            api.shutdown()

    def test_launch_once_shutdown_waits_for_live_result_then_closes_cell(self, bootstrap_path, env, monkeypatch):
        accepted = threading.Event()
        transfer_completed = threading.Event()
        waiter = MagicMock()
        waiter.done.side_effect = transfer_completed.is_set
        waiter.wait.side_effect = lambda timeout=None: (
            SimpleNamespace(status=TransferProgressState.COMPLETED, reason="all_receivers_succeeded")
            if transfer_completed.wait(timeout)
            else None
        )
        monkeypatch.setattr(cell_api.DownloadService, "get_transfer_waiter", lambda _tx_id: waiter)

        def on_request(topic, target, request):
            if topic == Topic.HELLO:
                return _hello_accepted_reply()
            if topic == Topic.RESULT_READY:
                kwargs = env.request_kwargs[-1]
                kwargs["fobs_ctx_props"][cell_api.RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY](
                    SimpleNamespace(tx_id="shutdown-result-tx")
                )
                env.deliver(Topic.SHUTDOWN, CJ_FQCN, {MsgKey.SESSION_ID: SESSION_ID})
                accepted.set()
                return _result_accepted_reply()
            return make_cell_reply(CellReturnCode.OK)

        env.on_request = on_request
        api = _init_api(bootstrap_path, env)
        errors = []
        try:
            _deliver_task(env)
            api.receive()

            def send_result():
                try:
                    api.send(FLModel(params={"w": [2.0]}, params_type=ParamsType.FULL))
                except BaseException as e:
                    errors.append(e)

            sender = threading.Thread(target=send_result)
            sender.start()
            assert accepted.wait(0.5)
            time.sleep(0.05)
            assert sender.is_alive()
            assert env.stopped is False

            transfer_completed.set()
            sender.join(timeout=0.5)
            assert not sender.is_alive()
            assert errors == []
            assert env.stopped is True
        finally:
            transfer_completed.set()
            api.shutdown()

    def test_explicit_shutdown_defers_owned_f3_teardown_until_live_result_settles(
        self, bootstrap_path, env, monkeypatch
    ):
        accepted = threading.Event()
        transfer_completed = threading.Event()
        waiter = MagicMock()
        waiter.transaction_id = "explicit-shutdown-result-tx"
        waiter.done.side_effect = transfer_completed.is_set
        waiter.wait.side_effect = lambda timeout=None: (
            SimpleNamespace(status=TransferProgressState.COMPLETED, reason="all_receivers_succeeded")
            if transfer_completed.wait(timeout)
            else None
        )
        monkeypatch.setattr(cell_api.DownloadService, "get_transfer_waiter", lambda _tx_id: waiter)

        def on_request(topic, target, request):
            if topic == Topic.HELLO:
                return _hello_accepted_reply()
            if topic == Topic.RESULT_READY:
                kwargs = env.request_kwargs[-1]
                kwargs["fobs_ctx_props"][cell_api.RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY](
                    SimpleNamespace(tx_id="explicit-shutdown-result-tx")
                )
                accepted.set()
                return _result_accepted_reply()
            return make_cell_reply(CellReturnCode.OK)

        env.on_request = on_request
        api = _init_api(bootstrap_path, env)
        errors = []
        try:
            _deliver_task(env)
            api.receive()

            sender = threading.Thread(target=lambda: _send_and_capture_error(api, FLModel(params={"w": [2.0]}), errors))
            sender.start()
            assert accepted.wait(0.5)

            api.shutdown()

            assert sender.is_alive()
            assert env.stopped is False
            env.shutdown_f3_streaming.assert_not_called()
            assert waiter.done() is False

            transfer_completed.set()
            sender.join(timeout=0.5)

            assert not sender.is_alive()
            assert errors == []
            assert env.stopped is True
            assert env.stop_calls == 1
            env.shutdown_f3_streaming.assert_called_once_with()
        finally:
            transfer_completed.set()
            api.shutdown()

    def test_send_rejects_non_successful_terminal_result_transfer(self, bootstrap_path, env, monkeypatch):
        waiter = MagicMock()
        waiter.wait.return_value = SimpleNamespace(status=TransferProgressState.FAILED, reason="receiver_failed")
        monkeypatch.setattr(cell_api.DownloadService, "get_transfer_waiter", lambda _tx_id: waiter)

        def on_request(topic, target, request):
            if topic == Topic.HELLO:
                return _hello_accepted_reply()
            if topic == Topic.RESULT_READY:
                kwargs = env.request_kwargs[-1]
                kwargs["fobs_ctx_props"][cell_api.RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY](
                    SimpleNamespace(tx_id="failed-result-tx")
                )
                return _result_accepted_reply()
            return make_cell_reply(CellReturnCode.OK)

        env.on_request = on_request
        api = _init_api(bootstrap_path, env)
        try:
            _deliver_task(
                env,
                model=FLModel(params={"w": [1.0]}, optimizer_params={"momentum": [0.5]}, params_type=ParamsType.FULL),
                result_receiver_ids=("server.job",),
            )
            received_model = api.receive()
            received_model.optimizer_params = {"momentum": [0.5]}
            sent_model = FLModel(
                params={"w": [2.0]}, optimizer_params={"momentum": [0.75]}, params_type=ParamsType.FULL
            )

            with pytest.raises(TrainerSessionError, match="status=failed.*receiver_failed"):
                api.send(sent_model)

            assert env.stopped is False
            assert sent_model.params == {"w": [2.0]}
            assert sent_model.optimizer_params == {"momentum": [0.75]}
            assert received_model.params == {"w": [1.0]}
            assert received_model.optimizer_params == {"momentum": [0.5]}
            assert api._fl_model is received_model
            assert api._receive_called is False
            assert api._current_task is None
            assert api._result_receiver_ids is None
            with pytest.raises(RuntimeError, match='"receive" needs to be called'):
                api.send(sent_model)
        finally:
            api.shutdown()

    def test_send_succeeds_when_orderly_shutdown_follows_result_acceptance(self, bootstrap_path, env):
        def on_request(topic, target, request):
            if topic == Topic.HELLO:
                return _hello_accepted_reply()
            if topic == Topic.RESULT_READY:
                # A final task may complete the workflow immediately. CJ is then allowed
                # to issue SHUTDOWN before the trainer thread resumes from send_request().
                index = env.request_messages.index(request)
                result_cancel = env.request_kwargs[index]["abort_signal"]
                assert result_cancel is api._result_abort_signal
                shutdown_reply = env.deliver(Topic.SHUTDOWN, CJ_FQCN, {MsgKey.SESSION_ID: SESSION_ID})
                assert shutdown_reply.payload == {MsgKey.RESULT_SOURCE_LIVE: True}
                assert api._abort_signal.triggered is True
                assert result_cancel.triggered is False
                assert env.stopped is False, "SHUTDOWN must not close Cell while send is active"
                return _result_accepted_reply()
            return make_cell_reply(CellReturnCode.OK)

        env.on_request = on_request
        api = _init_api(bootstrap_path, env)
        try:
            _deliver_task(env)
            api.receive()

            api.send(FLModel(params={"w": [2.0]}))

            assert env.stopped is True
            assert api.is_running() is False
        finally:
            api.shutdown()

    def test_shutdown_after_send_reports_source_already_settled(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        try:
            _deliver_task(env)
            api.receive()
            api.send(FLModel(params={"w": [2.0]}))

            reply = env.deliver(Topic.SHUTDOWN, CJ_FQCN, {MsgKey.SESSION_ID: SESSION_ID})

            assert reply.payload == {MsgKey.RESULT_SOURCE_LIVE: False}
            assert env.stopped is False
        finally:
            api.shutdown()

    def test_send_raises_when_result_rejected(self, bootstrap_path, env):
        _set_launch_once(bootstrap_path, False)
        api = _init_api(bootstrap_path, env)
        try:
            task_id, _ = _deliver_task(
                env,
                model=FLModel(params={"w": [1.0]}, optimizer_params={"momentum": [0.5]}, params_type=ParamsType.FULL),
                result_receiver_ids=("server.job",),
            )
            received_model = api.receive()
            received_model.optimizer_params = {"momentum": [0.5]}
            sent_model = FLModel(
                params={"w": [2.0]}, optimizer_params={"momentum": [0.75]}, params_type=ParamsType.FULL
            )

            def reject(topic, target, request):
                if topic == Topic.RESULT_READY:
                    return make_cell_reply(
                        CellReturnCode.OK, body={MsgKey.REPLY_TOPIC: Topic.RESULT_REJECTED, MsgKey.REASON: "nope"}
                    )
                return _hello_accepted_reply()

            env.on_request = reject
            with pytest.raises(TrainerSessionError, match="rejected"):
                api.send(sent_model)
            assert api._closed is False, "only an accepted per-task result ends the session"
            assert env.stopped is False
            assert sent_model.params == {"w": [2.0]}
            assert sent_model.optimizer_params == {"momentum": [0.75]}
            assert received_model.params == {"w": [1.0]}
            assert received_model.optimizer_params == {"momentum": [0.5]}
            assert api._fl_model is received_model
            assert api._receive_called is True
            assert api._current_task[MsgKey.TASK_ID] == task_id
            assert api._result_receiver_ids == ("server.job",)
        finally:
            api.shutdown()

    def test_send_preserves_model_and_task_state_when_result_has_no_reply(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        try:
            task_id, _ = _deliver_task(env, result_receiver_ids=("server.job",))
            received_model = api.receive()
            sent_model = FLModel(
                params={"w": [2.0]}, optimizer_params={"momentum": [0.75]}, params_type=ParamsType.FULL
            )

            def no_result_reply(topic, target, request):
                if topic == Topic.RESULT_READY:
                    return None
                return _hello_accepted_reply()

            env.on_request = no_result_reply
            with pytest.raises(TrainerSessionError, match="no reply"):
                api.send(sent_model)

            assert sent_model.params == {"w": [2.0]}
            assert sent_model.optimizer_params == {"momentum": [0.75]}
            assert received_model.params == {"w": [1.0]}
            assert api._fl_model is received_model
            assert api._receive_called is True
            assert api._current_task[MsgKey.TASK_ID] == task_id
            assert api._result_receiver_ids == ("server.job",)
        finally:
            api.shutdown()

    def test_multi_round_loop(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        try:
            for _ in range(3):
                _deliver_task(env)
                model = api.receive()
                assert isinstance(model, FLModel)
                api.send(FLModel(params={"w": [9.0]}, params_type=ParamsType.FULL))
            results = [p for t, _, p in env.requests if t == Topic.RESULT_READY]
            assert len(results) == 3
        finally:
            api.shutdown()

    def test_diff_transfer_uses_received_model_state(self, bootstrap_path, env):
        torch = pytest.importorskip("torch")
        import numpy as np

        _set_formats(bootstrap_path, ExchangeFormat.PYTORCH, ExchangeFormat.NUMPY)
        api = _init_api(bootstrap_path, env)
        try:
            api._task_exchange[ConfigKey.TRANSFER_TYPE] = cell_api.TransferType.DIFF
            _deliver_task(
                env,
                model=FLModel(params={"w": np.asarray([1.0])}, params_type=ParamsType.FULL),
            )
            received = api.receive()
            assert isinstance(received.params["w"], torch.Tensor)
            api.send(
                FLModel(params={"w": received.params["w"] + 3}, params_type=ParamsType.FULL),
                clear_cache=False,
            )
            result_payload = [p for t, _, p in env.requests if t == Topic.RESULT_READY][0]
            result_model = FLModelUtils.from_shareable(result_payload[MsgKey.RESULT])
            assert result_model.params_type == ParamsType.DIFF
            assert isinstance(result_model.params["w"], np.ndarray)
            np.testing.assert_array_equal(result_model.params["w"], np.asarray([3.0]))
        finally:
            api.shutdown()


class TestSessionEnd:
    def test_shutdown_ends_the_loop_cleanly(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        try:
            reply = env.deliver(Topic.SHUTDOWN, CJ_FQCN, {MsgKey.SESSION_ID: SESSION_ID})
            assert reply.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.OK
            # SHUTDOWN stops admission and wakes the normal receive loop. Cell/F3 cleanup
            # happens when that loop observes the stop; an active result send instead
            # remains alive until its real DownloadService transaction settles.
            assert env.stopped is False
            assert api.receive() is None
            assert env.stopped is True
            assert env.stop_calls == 1
            env.shutdown_f3_streaming.assert_called_once_with()
            assert api.is_running() is False
        finally:
            api.shutdown()

    def test_is_running_stops_cell_once_after_orderly_shutdown(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        env.deliver(Topic.SHUTDOWN, CJ_FQCN, {MsgKey.SESSION_ID: SESSION_ID})

        assert api.is_running() is False
        assert env.stopped is True
        assert env.stop_calls == 1
        assert api._closed is True
        api.shutdown()
        assert env.stop_calls == 1

    def test_repeated_shutdown_retries_failed_streaming_cleanup(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        env.shutdown_f3_streaming.side_effect = [RuntimeError("cleanup failed"), None]

        api.shutdown()
        api.shutdown()

        assert env.stop_calls == 1
        assert env.shutdown_f3_streaming.call_count == 2

    def test_abort_raises_from_blocked_receive(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        try:
            # abort arrives while receive() is blocked waiting for a task
            def deliver_abort_soon():
                env.deliver(Topic.ABORT, CJ_FQCN, {MsgKey.SESSION_ID: SESSION_ID, MsgKey.REASON: "controller abort"})

            threading.Timer(0.2, deliver_abort_soon).start()
            with pytest.raises(TrainerSessionError, match="aborted"):
                api.receive(timeout=5.0)
            assert env.stopped is True
            # is_running() returns False on abort (loop exits) rather than raising
            assert api.is_running() is False
        finally:
            api.shutdown()

    def test_receive_timeout_returns_none(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        try:
            assert api.receive(timeout=0.1) is None  # no task delivered
        finally:
            api.shutdown()

    def test_shutdown_stops_cell(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        api.shutdown()
        assert env.stopped
        api.shutdown()  # idempotent


class TestHeartbeat:
    def test_heartbeat_thread_sends_and_stops_cleanly(self, bootstrap_path, env):
        env.heartbeat_interval = 0.01
        env.heartbeat_timeout = 0.2
        api = _init_api(bootstrap_path, env)
        thread = api._heartbeat_thread
        try:
            assert thread is not None and thread.is_alive()
            assert _wait_until(lambda: any(topic == Topic.HEARTBEAT for topic, _, _ in env.requests))
        finally:
            api.shutdown()
        assert not thread.is_alive()

    def test_hard_cj_loss_aborts_blocked_receive(self, bootstrap_path, env):
        def no_heartbeat_reply(topic, target, request):
            if topic == Topic.HELLO:
                return _hello_accepted_reply(heartbeat_interval=0.01, heartbeat_timeout=0.05)
            if topic == Topic.HEARTBEAT:
                return None
            return make_cell_reply(CellReturnCode.OK)

        env.on_request = no_heartbeat_reply
        api = _init_api(bootstrap_path, env)
        try:
            assert _wait_until(lambda: api._abort)
            assert "CJ heartbeat timed out" in api._abort_reason
            with pytest.raises(TrainerSessionError, match="CJ heartbeat timed out"):
                api.receive(timeout=0.1)
        finally:
            api.shutdown()

    def test_pending_inline_result_request_does_not_suppress_owner_loss(self, bootstrap_path, env):
        request_pending = threading.Event()

        def wedged_result_request(topic, target, request):
            if topic == Topic.HELLO:
                return _hello_accepted_reply(heartbeat_interval=0.01, heartbeat_timeout=0.05)
            if topic == Topic.HEARTBEAT:
                return None
            if topic == Topic.RESULT_READY:
                index = env.request_messages.index(request)
                cancel = env.request_kwargs[index]["abort_signal"]
                request_pending.set()
                while not cancel.triggered:
                    time.sleep(0.005)
                return None
            return make_cell_reply(CellReturnCode.OK)

        env.on_request = wedged_result_request
        api = _init_api(bootstrap_path, env)
        send_errors = []
        try:
            _deliver_task(env)
            api.receive()

            def send_result():
                try:
                    api.send(FLModel(params={"w": [2.0]}, params_type=ParamsType.FULL))
                except BaseException as e:
                    send_errors.append(e)

            sender = threading.Thread(target=send_result)
            sender.start()
            assert request_pending.wait(0.5)
            assert _wait_until(lambda: api._abort)
            sender.join(timeout=0.5)

            assert not sender.is_alive()
            assert send_errors
            assert "CJ heartbeat timed out" in api._abort_reason
        finally:
            api.shutdown()

    def test_live_result_transaction_suppresses_heartbeat_expiry(self, bootstrap_path, env, monkeypatch):
        transaction_created = threading.Event()
        transfer_completed = threading.Event()
        release_request = threading.Event()
        waiter = MagicMock()
        waiter.done.side_effect = transfer_completed.is_set
        waiter.wait.side_effect = lambda timeout=None: (
            SimpleNamespace(status=TransferProgressState.COMPLETED, reason="all_receivers_succeeded")
            if transfer_completed.is_set()
            else None
        )
        get_transfer_waiter = MagicMock(return_value=waiter)
        monkeypatch.setattr(cell_api.DownloadService, "get_transfer_waiter", get_transfer_waiter)

        def progressing_result_request(topic, target, request):
            if topic == Topic.HELLO:
                return _hello_accepted_reply(heartbeat_interval=0.01, heartbeat_timeout=0.05)
            if topic == Topic.HEARTBEAT:
                return None
            if topic == Topic.RESULT_READY:
                index = env.request_messages.index(request)
                tx_created = env.request_kwargs[index]["fobs_ctx_props"][cell_api.RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY]
                tx_created(SimpleNamespace(tx_id="live-result-tx"))
                transaction_created.set()
                release_request.wait(0.5)
                return _result_accepted_reply()
            return make_cell_reply(CellReturnCode.OK)

        env.on_request = progressing_result_request
        api = _init_api(bootstrap_path, env)
        send_errors = []
        try:
            _deliver_task(env)
            api.receive()

            def send_result():
                try:
                    api.send(FLModel(params={"w": [2.0]}, params_type=ParamsType.FULL))
                except BaseException as e:
                    send_errors.append(e)

            sender = threading.Thread(target=send_result)
            sender.start()
            assert transaction_created.wait(0.5)
            time.sleep(0.1)  # longer than the CJ heartbeat timeout
            assert api._abort is False

            transfer_completed.set()
            release_request.set()
            sender.join(timeout=0.5)
            assert not sender.is_alive()
            assert send_errors == []
            get_transfer_waiter.assert_called_once_with("live-result-tx")
        finally:
            release_request.set()
            api.shutdown()


class TestControlValidation:
    def test_task_ready_before_hello_is_rejected(self, bootstrap_path):
        api = CellClientAPI(bootstrap_file=bootstrap_path)
        shareable = FLModelUtils.to_shareable(FLModel(params={"w": [1.0]}))
        reply = api._handle_task_ready(
            new_cell_message(
                {MessageHeaderKey.ORIGIN: CJ_FQCN},
                {
                    MsgKey.TASK_ID: "t1",
                    MsgKey.TASK_NAME: "train",
                    MsgKey.MODEL: shareable,
                },
            )
        )

        assert reply.payload[MsgKey.REPLY_TOPIC] == Topic.TASK_FAILED
        assert reply.payload[MsgKey.REASON] == "no active trainer session"
        assert api._task_queue.empty()

    @pytest.mark.parametrize(
        "terminal_topic, terminal_fields",
        [
            (Topic.ABORT, {MsgKey.REASON: "controller abort"}),
            (Topic.SHUTDOWN, {}),
        ],
    )
    def test_task_ready_after_session_end_is_rejected(self, bootstrap_path, env, terminal_topic, terminal_fields):
        api = _init_api(bootstrap_path, env)
        try:
            terminal_payload = {MsgKey.SESSION_ID: SESSION_ID, **terminal_fields}
            env.deliver(terminal_topic, CJ_FQCN, terminal_payload)

            _, reply = _deliver_task(env)

            assert reply.payload[MsgKey.REPLY_TOPIC] == Topic.TASK_FAILED
            assert api._task_queue.empty()
        finally:
            api.shutdown()

    def test_get_config_preserves_legacy_shape_without_bootstrap_secret(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        try:
            config = api.get_config()
            assert config[ConfigKey.TASK_EXCHANGE][ConfigKey.TRAIN_TASK_NAME] == "train"
            assert config[FLMetaKey.JOB_ID] == "job-1"
            assert config[FLMetaKey.SITE_NAME] == "site-1"
            assert config[ConfigKey.MEMORY_GC_ROUNDS] == 3
            assert config[ConfigKey.CUDA_EMPTY_CACHE] is True
            assert BootstrapKey.LAUNCH_TOKEN not in config
        finally:
            api.shutdown()

    def test_task_ready_with_wrong_session_is_failed(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        try:
            reply = env.deliver(
                Topic.TASK_READY,
                CJ_FQCN,
                {MsgKey.SESSION_ID: "stale", MsgKey.TASK_ID: "t1", MsgKey.TASK_NAME: "train"},
            )
            assert reply.payload[MsgKey.REPLY_TOPIC] == Topic.TASK_FAILED
        finally:
            api.shutdown()

    def test_foreign_control_messages_are_rejected_without_mutating_session(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        try:
            task_reply = env.deliver(
                Topic.TASK_READY,
                "foreign.cell",
                {MsgKey.SESSION_ID: SESSION_ID, MsgKey.TASK_ID: "t1", MsgKey.TASK_NAME: "train"},
            )
            abort_reply = env.deliver(
                Topic.ABORT,
                "foreign.cell",
                {MsgKey.SESSION_ID: SESSION_ID, MsgKey.REASON: "forged abort"},
            )
            shutdown_reply = env.deliver(
                Topic.SHUTDOWN,
                "foreign.cell",
                {MsgKey.SESSION_ID: SESSION_ID},
            )

            assert task_reply.payload[MsgKey.REPLY_TOPIC] == Topic.TASK_FAILED
            assert abort_reply.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.INVALID_REQUEST
            assert shutdown_reply.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.INVALID_REQUEST
            assert api._task_queue.empty()
            assert api._abort is False
            assert api._stopped is False
        finally:
            api.shutdown()

    def test_log_sends_to_cj(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        try:
            api.log("accuracy", 0.9, AnalyticsDataType.SCALAR)
            logs = [f for f in env.fired if f[0] == Topic.LOG]
            assert logs and logs[0][2]["key"] == "accuracy" and logs[0][2]["value"] == 0.9
        finally:
            api.shutdown()

    def test_log_coerces_numpy_scalar_to_python_scalar(self, bootstrap_path, env):
        import numpy as np

        api = _init_api(bootstrap_path, env)
        try:
            api.log("weight_mean", np.float32(7.0), AnalyticsDataType.SCALAR)
            value = [f for f in env.fired if f[0] == Topic.LOG][0][2]["value"]
            # numpy scalar -> Python float so the CJ's analytics DXO validation accepts it
            assert type(value) is float and value == 7.0
            # arrays and plain scalars pass through unchanged
            arr = np.array([1.0, 2.0])
            assert cell_api._to_python_scalar(arr) is arr
            assert cell_api._to_python_scalar(5) == 5
        finally:
            api.shutdown()

    def test_system_info_and_ids(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        try:
            assert api.get_job_id() == "job-1"
            assert api.get_site_name() == "site-1"
            info = api.system_info()
            # SYS_ATTRS convention: lowercase job_id/site_name (FLMetaKey), as in_process uses
            assert info[FLMetaKey.JOB_ID] == "job-1" and info[FLMetaKey.SITE_NAME] == "site-1"
        finally:
            api.shutdown()
