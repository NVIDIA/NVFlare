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
HELLO handshake, receive() materializing a task the CJ delivers, send() publishing a result and
holding until the CJ certifies delivery, the batch-loop is_running() semantics, and ABORT/
SHUTDOWN session ends. The payload seam is faked here (its real-layer behavior is covered by
payload_transfer_integration_test.py)."""

import threading
import uuid
from unittest.mock import MagicMock

import pytest

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.client.cell import api as cell_api
from nvflare.client.cell.api import CellClientAPI, TrainerSessionError
from nvflare.client.cell.bootstrap import BootstrapKey, write_bootstrap_config
from nvflare.client.cell.defs import CHANNEL, PROTOCOL_VERSION, MsgKey, Topic
from nvflare.client.config import ConfigKey
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply as make_cell_reply
from nvflare.fuel.f3.cellnet.utils import new_cell_message

CJ_FQCN = "site-1.job-1"
TRAINER_FQCN = "site-1.job-1.client_api_trainer_1"
SESSION_ID = "session-abc"


def _hello_accepted_reply():
    return make_cell_reply(
        CellReturnCode.OK,
        body={
            MsgKey.REPLY_TOPIC: Topic.HELLO_ACCEPTED,
            MsgKey.SESSION_ID: SESSION_ID,
            MsgKey.JOB_ID: "job-1",
            MsgKey.SITE_NAME: "site-1",
        },
    )


def _result_accepted_reply(result_id):
    return make_cell_reply(
        CellReturnCode.OK, body={MsgKey.REPLY_TOPIC: Topic.RESULT_ACCEPTED, MsgKey.RESULT_ID: result_id}
    )


class FakeAttempt:
    """Stands in for payload_transfer.TaskPayloadAttempt (producer side of send())."""

    instances = []
    delivered = True

    def __init__(self, cell, obj, receiver_fqcn):
        FakeAttempt.instances.append(self)
        self.cell = cell
        self.obj = obj
        self.receiver_fqcn = receiver_fqcn
        self.ref_id = f"res-ref-{len(FakeAttempt.instances)}"
        self.tx_id = f"tx-{len(FakeAttempt.instances)}"
        self.terminated = False
        self.wait_calls = []

    @classmethod
    def reset(cls):
        cls.instances = []
        cls.delivered = True

    def wait(self, timeout=None, linger=None):
        self.wait_calls.append((timeout, linger))
        return FakeAttempt.delivered

    def terminate(self):
        self.terminated = True


class FakeCell:
    """The CJ cell as seen from the trainer: records the trainer's outbound requests/messages
    and lets a test deliver CJ->trainer control messages (TASK_READY/ABORT/SHUTDOWN)."""

    def __init__(self):
        self.fqcn = TRAINER_FQCN
        self.started = False
        self.stopped = False
        self.cbs = {}
        self.requests = []  # (topic, target, payload)
        self.fired = []  # (topic, targets, payload)
        self.on_request = None

    def get_fqcn(self):
        return self.fqcn

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def register_request_cb(self, channel, topic, cb):
        assert channel == CHANNEL
        self.cbs[topic] = cb

    def send_request(self, channel, topic, target, request, timeout=None, **kwargs):
        self.requests.append((topic, target, request.payload))
        if self.on_request is not None:
            return self.on_request(topic, target, request)
        if topic == Topic.HELLO:
            return _hello_accepted_reply()
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
            BootstrapKey.CONNECT_URL: "tcp://127.0.0.1:12345",
            BootstrapKey.CJ_FQCN: CJ_FQCN,
            BootstrapKey.TRAINER_FQCN: TRAINER_FQCN,
            BootstrapKey.LAUNCH_TOKEN: "the-token",
            BootstrapKey.PROTOCOL_VERSION: PROTOCOL_VERSION,
            BootstrapKey.JOB_ID: "job-1",
            BootstrapKey.SITE_NAME: "site-1",
            BootstrapKey.TASK_EXCHANGE: {
                ConfigKey.TRAIN_TASK_NAME: "train",
                ConfigKey.EVAL_TASK_NAME: "validate",
                ConfigKey.SUBMIT_MODEL_TASK_NAME: "submit_model",
            },
        },
    )
    return path


@pytest.fixture
def env(bootstrap_path, monkeypatch):
    cell = FakeCell()
    monkeypatch.setattr(cell_api, "Cell", MagicMock(return_value=cell))
    monkeypatch.setattr(cell_api, "payload_layer_available", lambda: True)
    monkeypatch.setattr(cell_api, "TaskPayloadAttempt", FakeAttempt)
    FakeAttempt.reset()
    # task/result payloads are faked at the seam; make fetch return a known FLModel shareable
    holder = {"task_shareable": FLModel(params={"w": [1.0]}, params_type=ParamsType.FULL)}

    def fake_fetch(cell_, from_fqcn, ref_ids, abort_signal=None):
        from nvflare.app_common.utils.fl_model_utils import FLModelUtils

        return [FLModelUtils.to_shareable(holder["task_shareable"])]

    monkeypatch.setattr(cell_api, "fetch_result_payload", fake_fetch)
    return cell


def _init_api(bootstrap_path, env, rank="0"):
    api = CellClientAPI(bootstrap_file=bootstrap_path)
    api.init(rank=rank)
    return api


def _deliver_task(env, task_name="train", task_id=None, with_payload=True):
    task_id = task_id or uuid.uuid4().hex
    payload = {
        MsgKey.SESSION_ID: SESSION_ID,
        MsgKey.TASK_ID: task_id,
        MsgKey.TASK_NAME: task_name,
        MsgKey.TRANSFER_ID: uuid.uuid4().hex,
        MsgKey.MODEL: {MsgKey.REF_IDS: ["task-ref-1"]} if with_payload else {},
    }
    reply = env.deliver(Topic.TASK_READY, CJ_FQCN, payload)
    return task_id, reply


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
    def test_receive_materializes_task_and_sends_payload_ready(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        try:
            task_id, accepted = _deliver_task(env, task_name="train")
            assert accepted.payload[MsgKey.REPLY_TOPIC] == Topic.TASK_ACCEPTED

            model = api.receive()
            assert isinstance(model, FLModel)
            assert model.params == {"w": [1.0]}
            assert api.get_task_name() == "train"
            assert api.is_train() is True and api.is_evaluate() is False
            # the trainer told the CJ the payload materialized
            assert [f for f in env.fired if f[0] == Topic.TASK_PAYLOAD_READY]
        finally:
            api.shutdown()

    def test_send_publishes_result_and_holds_until_delivered(self, bootstrap_path, env):
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
            assert payload[MsgKey.RESULT_ID] and payload[MsgKey.TRANSFER_ID]
            attempt = FakeAttempt.instances[0]
            assert payload[MsgKey.MANIFEST] == {MsgKey.REF_IDS: [attempt.ref_id]}
            # producer-liveness: held on the waiter, then terminated
            assert attempt.wait_calls and attempt.terminated
        finally:
            api.shutdown()

    def test_send_before_receive_raises(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        try:
            with pytest.raises(RuntimeError, match="receive.*before sending"):
                api.send(FLModel(params={"w": [1.0]}))
        finally:
            api.shutdown()

    def test_send_raises_when_not_certified_delivered(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        try:
            _deliver_task(env)
            api.receive()
            FakeAttempt.delivered = False  # the CJ never certifies the pull
            with pytest.raises(TrainerSessionError, match="not certified delivered"):
                api.send(FLModel(params={"w": [2.0]}))
            assert FakeAttempt.instances[0].terminated
        finally:
            api.shutdown()

    def test_send_raises_when_result_rejected(self, bootstrap_path, env):
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
            # each round used a fresh result attempt and transfer id
            transfer_ids = [p[MsgKey.TRANSFER_ID] for t, _, p in env.requests if t == Topic.RESULT_READY]
            assert len(set(transfer_ids)) == 3
        finally:
            api.shutdown()


class TestSessionEnd:
    def test_shutdown_ends_the_loop_cleanly(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        try:
            env.deliver(Topic.SHUTDOWN, CJ_FQCN, {MsgKey.SESSION_ID: SESSION_ID})
            assert api.receive() is None
            assert api.is_running() is False
        finally:
            api.shutdown()

    def test_abort_raises_from_blocked_receive(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        try:
            # abort arrives while receive() is blocked waiting for a task
            def deliver_abort_soon():
                env.deliver(Topic.ABORT, CJ_FQCN, {MsgKey.SESSION_ID: SESSION_ID, MsgKey.REASON: "controller abort"})

            threading.Timer(0.2, deliver_abort_soon).start()
            with pytest.raises(TrainerSessionError, match="aborted"):
                api.receive(timeout=5.0)
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

    def test_shutdown_sends_bye_and_stops_cell(self, bootstrap_path, env):
        api = _init_api(bootstrap_path, env)
        api.shutdown()
        assert [f for f in env.fired if f[0] == Topic.BYE]
        assert env.stopped
        api.shutdown()  # idempotent


class TestControlValidation:
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
