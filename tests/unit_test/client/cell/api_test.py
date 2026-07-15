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

import threading
import time
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.apis.fl_constant import FLMetaKey
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.client.cell import api as cell_api
from nvflare.client.cell.api import CellClientAPI, TrainerSessionError
from nvflare.client.cell.bootstrap import (
    BOOTSTRAP_SCHEMA_VERSION,
    EXTERNAL_PROCESS_EXECUTION_MODE,
    BootstrapKey,
    read_bootstrap_config,
    write_bootstrap_config,
)
from nvflare.client.cell.defs import CHANNEL, PROTOCOL_VERSION, MsgKey, Topic
from nvflare.client.config import ConfigKey, ExchangeFormat
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply as make_cell_reply
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.f3.streaming.transfer_progress import TransferProgressState
from nvflare.fuel.utils.fobs import FOBSContextKey

CJ_FQCN = "site-1.job-1"
TRAINER_FQCN = "site-1.job-1.client_api_trainer_1"
SESSION_ID = "session-abc"


def _hello_accepted_reply(heartbeat_interval=0.05, heartbeat_timeout=0.0):
    return make_cell_reply(
        CellReturnCode.OK,
        body={
            MsgKey.REPLY_TOPIC: Topic.HELLO_ACCEPTED,
            MsgKey.SESSION_ID: SESSION_ID,
            MsgKey.JOB_ID: "job-1",
            MsgKey.SITE_NAME: "site-1",
            MsgKey.HEARTBEAT_INTERVAL: heartbeat_interval,
            MsgKey.HEARTBEAT_TIMEOUT: heartbeat_timeout,
        },
    )


def _result_accepted_reply(result_id):
    return make_cell_reply(
        CellReturnCode.OK, body={MsgKey.REPLY_TOPIC: Topic.RESULT_ACCEPTED, MsgKey.RESULT_ID: result_id}
    )


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
            return _hello_accepted_reply(self.heartbeat_interval, self.heartbeat_timeout)
        if topic == Topic.HEARTBEAT:
            return make_cell_reply(
                CellReturnCode.OK,
                body={MsgKey.REPLY_TOPIC: Topic.HEARTBEAT, MsgKey.SESSION_ID: SESSION_ID},
            )
        if topic == Topic.RESULT_READY:
            return _result_accepted_reply(request.payload[MsgKey.RESULT_ID])
        return make_cell_reply(CellReturnCode.OK)

    def fire_and_forget(self, channel, topic, targets, message, **kwargs):
        self.fired.append((topic, tuple(targets), message.payload))

    def deliver(self, topic, origin, payload):
        return self.cbs[topic](new_cell_message({MessageHeaderKey.ORIGIN: origin}, payload))


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
            # A bootstrap value is only an instruction from the parent; HELLO must report
            # the trainer code's compiled protocol constant instead of echoing this value.
            BootstrapKey.PROTOCOL_VERSION: PROTOCOL_VERSION + 99,
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


class TestInit:
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

    def test_non_control_rank_has_passive_api(self, bootstrap_path, env):
        api = CellClientAPI(bootstrap_file=bootstrap_path)
        api.init(rank="1")
        # rank != 0 opens no session (rank contract): no cell built, receive None, not running
        assert not env.started
        assert api.receive() is None
        assert api.is_running() is False


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
            assert payload[MsgKey.RESULT_ID]
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
        monkeypatch.setattr(cell_api.DownloadService, "get_transfer_waiter", lambda _tx_id: waiter)

        def on_request(topic, target, request):
            if topic == Topic.HELLO:
                return _hello_accepted_reply()
            if topic == Topic.RESULT_READY:
                kwargs = env.request_kwargs[-1]
                cb = kwargs["fobs_ctx_props"][cell_api.RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY]
                cb(SimpleNamespace(tx_id="actual-via-tx"))
                assert kwargs["progress_wait_cb"]() is True
                return _result_accepted_reply(request.payload[MsgKey.RESULT_ID])
            return make_cell_reply(CellReturnCode.OK)

        env.on_request = on_request
        api = _init_api(bootstrap_path, env)
        try:
            _deliver_task(env)
            api.receive()
            api.send(FLModel(params={"w": [2.0]}))
            waiter.done.assert_called()
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
                return _result_accepted_reply(request.payload[MsgKey.RESULT_ID])
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
                return _result_accepted_reply(request.payload[MsgKey.RESULT_ID])
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

    def test_send_rejects_non_successful_terminal_result_transfer(self, bootstrap_path, env, monkeypatch):
        _set_launch_once(bootstrap_path, False)
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
                return _result_accepted_reply(request.payload[MsgKey.RESULT_ID])
            return make_cell_reply(CellReturnCode.OK)

        env.on_request = on_request
        api = _init_api(bootstrap_path, env)
        try:
            _deliver_task(env)
            api.receive()

            with pytest.raises(TrainerSessionError, match="status=failed.*receiver_failed"):
                api.send(FLModel(params={"w": [2.0]}, params_type=ParamsType.FULL))

            assert env.stopped is True
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
                return _result_accepted_reply(request.payload[MsgKey.RESULT_ID])
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
            _deliver_task(env)
            api.receive()

            def reject(topic, target, request):
                if topic == Topic.RESULT_READY:
                    return make_cell_reply(
                        CellReturnCode.OK, body={MsgKey.REPLY_TOPIC: Topic.RESULT_REJECTED, MsgKey.REASON: "nope"}
                    )
                return _hello_accepted_reply()

            env.on_request = reject
            with pytest.raises(TrainerSessionError, match="rejected"):
                api.send(FLModel(params={"w": [2.0]}))
            assert api._closed is False, "only an accepted per-task result ends the session"
            assert env.stopped is False
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
            result_ids = [p[MsgKey.RESULT_ID] for t, _, p in env.requests if t == Topic.RESULT_READY]
            assert len(set(result_ids)) == 3
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
        monkeypatch.setattr(cell_api.DownloadService, "get_transfer_waiter", lambda _tx_id: waiter)

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
                return _result_accepted_reply(request.payload[MsgKey.RESULT_ID])
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
        finally:
            release_request.set()
            api.shutdown()


class TestControlValidation:
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
