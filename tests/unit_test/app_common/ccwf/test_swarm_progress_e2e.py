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

import copy
import threading
import time
from unittest.mock import MagicMock

import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey, ReturnCode
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.ccwf.common import Constant
from nvflare.app_common.ccwf.swarm_client_ctl import SwarmClientController
from nvflare.app_common.executors import task_exchanger as task_exchanger_module
from nvflare.app_common.executors.task_exchanger import TaskExchanger
from nvflare.client.flare_agent import FlareAgent, _TaskContext
from nvflare.fuel.f3.cellnet import cell as cell_module
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.streaming import transfer_progress as transfer_progress_module
from nvflare.fuel.f3.streaming.download_service import TransactionDoneStatus
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.fobs.decomposers.via_downloader import (
    RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY,
    DownloadTransactionInfo,
    _tls,
    clear_download_initiated,
)
from nvflare.fuel.utils.pipe.cell_pipe import CellPipe
from nvflare.fuel.utils.pipe.pipe import Message, Pipe, Topic
from nvflare.fuel.utils.waiter_utils import WaiterRC


class _AbortSignal:
    triggered = False

    def trigger(self, reason=None):
        self.triggered = True


class _ReadyStreamFuture:
    def __init__(self, headers=None, payload=None):
        self.headers = headers or {}
        self._payload = payload
        self.error = False
        self.waiter = threading.Event()
        self.waiter.set()

    def get_progress(self):
        return 1

    def result(self):
        return self._payload


class _CellStackFake:
    """Fake only the network send_blob boundary while reusing Cell request logic."""

    def __init__(self):
        self.requests_dict = {}
        self.decode_pass_through_channels = set()
        self.sent_blobs = []
        self.logger = MagicMock()

    def get_fobs_context(self, props=None):
        ctx = {}
        if props:
            ctx.update(props)
        return ctx

    def send_blob(self, channel, topic, target, message, secure=False, optional=False):
        self.sent_blobs.append((channel, topic, target, message))
        return _ReadyStreamFuture()

    def send_request(self, *args, **kwargs):
        return cell_module.Cell._send_request(self, *args, **kwargs)

    def _encode_message(self, *args, **kwargs):
        return cell_module.Cell._encode_message(self, *args, **kwargs)

    def _send_one_request(self, *args, **kwargs):
        return cell_module.Cell._send_one_request(self, *args, **kwargs)

    def _get_result(self, *args, **kwargs):
        return cell_module.Cell._get_result(self, *args, **kwargs)

    def _future_wait(self, future, timeout, abort_signal):
        return True

    def __getattr__(self, name):
        raise AssertionError(f"Cell internal attribute '{name}' accessed; fake is incomplete")


def _make_cell_stack_pipe(cell):
    pipe = CellPipe.__new__(CellPipe)
    Pipe.__init__(pipe, Mode.ACTIVE)
    pipe.cell = cell
    pipe.channel = "cell_pipe.task"
    # Matches CellPipe's active peer FQCN convention:
    # FQCN.join([parent_fqcn, f"{site_name}_{token}_active"]).
    pipe.peer_fqcn = "site-2.site-2_job-1_active"
    pipe.hb_seq = 1
    pipe.pass_through_on_send = False
    pipe.logger = MagicMock()
    pipe.pipe_lock = threading.Lock()
    pipe.closed = False
    return pipe


def _make_swarm_controller(me="site-1", trainers=None, aggrs=None):
    ctl = SwarmClientController.__new__(SwarmClientController)
    ctl.logger = MagicMock()
    ctl.log_info = MagicMock()
    ctl.log_error = MagicMock()
    ctl.log_debug = MagicMock()
    ctl.log_warning = MagicMock()
    ctl.me = me
    ctl.trainers = trainers or ["site-1", "site-2"]
    ctl.aggrs = aggrs or ["site-1"]
    ctl.is_trainer = True
    ctl.is_aggr = me in ctl.aggrs
    ctl.learn_task_timeout = None
    ctl.learn_task_ack_timeout = 10
    ctl.request_to_submit_learn_result_task_name = "swarm_request_to_submit_learn_result"
    ctl.request_to_submit_result_max_wait = None
    ctl.request_to_submit_result_msg_timeout = 5.0
    ctl.request_to_submit_result_interval = 0.0
    ctl.report_learn_result_task_name = "swarm_report_learn_result"
    ctl.last_aggr_round_done = -1
    ctl.learn_task_abort_timeout = 10.0
    ctl.memory_gc_rounds = 0
    ctl.cuda_empty_cache = False
    ctl._aggr_round_count = 0
    ctl.gatherer = None
    ctl.gatherer_waiter = MagicMock()
    ctl.shareable_generator = MagicMock()
    ctl.aggregator = MagicMock()
    ctl.update_status = MagicMock()
    ctl.fire_event = MagicMock()
    ctl.record_last_result = MagicMock()
    ctl._distribute_final_results = MagicMock()
    ctl.set_learn_task = MagicMock(return_value=True)

    def _config(key, *default):
        mapping = {
            Constant.TRAIN_CLIENTS: ctl.trainers,
            Constant.AGGR_CLIENTS: ctl.aggrs,
            Constant.CLIENTS: sorted(set(ctl.trainers + ctl.aggrs)),
            Constant.START_ROUND: 0,
            AppConstants.NUM_ROUNDS: 1,
        }
        return mapping.get(key, default[0] if default else None)

    ctl.get_config_prop = MagicMock(side_effect=_config)
    return ctl


def _make_fl_ctx(job_id="job-1", site_name="site-1"):
    fl_ctx = MagicMock()
    fl_ctx.get_job_id.return_value = job_id
    fl_ctx.get_identity_name.return_value = site_name
    fl_ctx.get_prop.return_value = MagicMock()
    engine = MagicMock()
    engine.get_client_from_name.return_value = None
    fl_ctx.get_engine.return_value = engine
    return fl_ctx


def _patch_task_exchanger_logs(monkeypatch):
    logs = []
    monkeypatch.setattr(TaskExchanger, "log_info", lambda self, fl_ctx, msg: logs.append(("info", msg)))
    monkeypatch.setattr(TaskExchanger, "log_debug", lambda self, fl_ctx, msg: logs.append(("debug", msg)))
    monkeypatch.setattr(TaskExchanger, "log_warning", lambda self, fl_ctx, msg: logs.append(("warning", msg)))
    monkeypatch.setattr(TaskExchanger, "log_error", lambda self, fl_ctx, msg: logs.append(("error", msg)))
    return logs


def _make_model_shareable():
    dxo = DXO(
        data_kind=DataKind.WEIGHTS,
        data={
            "layer.weight": [1.0, 2.0],
            "layer.bias": [0.0],
        },
    )
    return dxo.to_shareable()


def _progress_message(task_id, job_id, transfer_id, sequence, bytes_done, receiver_id=None, state="active"):
    data = {
        "job_id": job_id,
        "task_id": task_id,
        "transfer_id": transfer_id,
        "direction": task_exchanger_module.DIRECTION_TASK_PAYLOAD_DOWNLOAD,
        "sequence": sequence,
        "bytes_done": bytes_done,
        "state": state,
    }
    if receiver_id is not None:
        data["receiver_id"] = receiver_id
    return Message.new_request(Topic.STREAM_PROGRESS, data)


def test_swarm_scatter_peer_parent_progress_reaches_cellpipe_task_send(monkeypatch):
    """Swarm parent scatter drives peer CJ task send and progress-aware wait in one simulated path.

    E2E counterpart to the unit-level
    ``test_swarm_task_payload_progress_from_peer_parent_suppresses_resend`` in
    ``task_exchanger_stream_progress_test.py``: that test pins the wait-policy
    decision logic directly, while this one exercises the full
    ``_scatter -> send_learn_task -> TaskExchanger._send_task_to_peer ->
    Cell._send_one_request`` chain across the swarm controller boundary.
    """

    _patch_task_exchanger_logs(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(task_exchanger_module.time, "time", lambda: now[0])
    monkeypatch.setattr(transfer_progress_module.time, "time", lambda: now[0])
    monkeypatch.setattr("nvflare.app_common.ccwf.swarm_client_ctl.random.choice", lambda candidates: "site-1")

    parent_ctl = _make_swarm_controller(me="site-1", trainers=["site-1", "site-2"], aggrs=["site-1"])
    parent_fl_ctx = _make_fl_ctx(job_id="job-1", site_name="site-1")
    task_data = _make_model_shareable()
    remote_observations = {}

    def _send_learn_task(targets, request, fl_ctx):
        assert targets == ["site-2"]
        assert request.get_header(Constant.AGGREGATOR) == "site-1"

        executor = TaskExchanger(
            pipe_id="pipe",
            peer_read_timeout=1.0,
            streaming_idle_timeout=10.0,
            resend_interval=0.01,
            max_resends=0,
        )
        cell = _CellStackFake()
        executor.pipe = _make_cell_stack_pipe(cell)
        handler = executor._create_pipe_handler()
        task_id = "swarm-round-0-site-2"
        transfer_id = "site-1-global-model-ref"
        decisions = []
        original_should_continue = executor._should_continue_task_send_waiting

        def _record_should_continue(*args, **kwargs):
            decision = original_should_continue(*args, **kwargs)
            decisions.append(decision)
            return decision

        monkeypatch.setattr(executor, "_should_continue_task_send_waiting", _record_should_continue)

        def _conditional_wait(event, timeout, abort_signal, **kwargs):
            if len(decisions) < 2:
                now[0] += timeout + 0.1
                handler.msg_cb(
                    _progress_message(
                        task_id=task_id,
                        job_id="job-1",
                        transfer_id=transfer_id,
                        sequence=len(decisions) + 1,
                        bytes_done=(len(decisions) + 1) * 1024 * 1024,
                        receiver_id="site-2.job-1",
                    )
                )
                return WaiterRC.TIMEOUT

            waiter = next(iter(cell.requests_dict.values()))
            waiter.receiving_future = _ReadyStreamFuture(
                headers={MessageHeaderKey.RETURN_CODE: CellReturnCode.OK},
                payload=None,
            )
            waiter.in_receiving.set()
            return WaiterRC.IS_SET

        monkeypatch.setattr(cell_module, "conditional_wait", _conditional_wait)

        remote_request = copy.deepcopy(request)
        remote_request.set_header(FLContextKey.TASK_ID, task_id)
        req = Message.new_request("swarm_learn", remote_request, msg_id=task_id)
        sent = executor._send_task_to_peer(req, _make_fl_ctx(job_id="job-1", site_name="site-2"), _AbortSignal())
        remote_observations.update(
            {
                "sent": sent,
                "send_count": len(cell.sent_blobs),
                "decisions": decisions,
                "cache_released": not hasattr(req, "_cached_cell_msg"),
            }
        )
        return sent

    parent_ctl.send_learn_task = _send_learn_task

    assert parent_ctl._scatter(task_data, for_round=0, fl_ctx=parent_fl_ctx) is True
    assert parent_ctl.set_learn_task.called
    assert remote_observations == {
        "sent": True,
        "send_count": 1,
        "decisions": [True, True],
        "cache_released": True,
    }


@pytest.fixture(autouse=True)
def _no_os_exit(monkeypatch):
    monkeypatch.setattr("nvflare.client.flare_agent.os._exit", lambda code: None)


def _make_agent_for_result_upload():
    pipe = MagicMock(spec=CellPipe)
    pipe.pass_through_on_send = True
    pipe.closed = False
    pipe.cell = MagicMock()
    pipe.cell.get_fobs_context.return_value = {}

    agent = FlareAgent.__new__(FlareAgent)
    agent.logger = MagicMock()
    agent.pipe = pipe
    agent.submit_result_timeout = 30.0
    agent._download_complete_timeout = 0.001
    agent._streaming_idle_timeout = 0.5
    agent._result_upload_poll_interval = 0.001
    agent._launch_once = False
    agent.asked_to_stop = False
    agent.pipe_handler = MagicMock()
    agent.pipe_handler.asked_to_stop = False
    result_shareable = Shareable()
    result_shareable.set_header(FLMetaKey.JOB_ID, "job-1")
    agent.task_result_to_shareable = MagicMock(return_value=result_shareable)
    return agent


def test_swarm_do_learn_task_stamps_receiver_and_result_upload_tracks_peer_receiver():
    """Trainer side do_learn_task stamps the swarm aggregator receiver and FlareAgent waits on that receiver."""

    clear_download_initiated()
    trainer_ctl = _make_swarm_controller(me="site-2", trainers=["site-1", "site-2"], aggrs=["site-1"])
    trainer_fl_ctx = _make_fl_ctx(job_id="job-1", site_name="site-2")
    aggr_client = MagicMock()
    aggr_client.get_fqcn.return_value = "relay.site-1"
    trainer_fl_ctx.get_engine.return_value.get_client_from_name.return_value = aggr_client
    permission_reply = make_reply(ReturnCode.OK)
    trainer_fl_ctx.get_engine.return_value.send_aux_request.return_value = {"site-1": permission_reply}
    trainer_ctl.broadcast_and_wait = MagicMock(return_value={"site-1": make_reply(ReturnCode.OK)})
    progress_receivers = []

    def _execute_learn_task(task_data, fl_ctx, abort_signal):
        receiver_ids = task_data.get_header(fobs.FOBSContextKey.RECEIVER_IDS)
        assert receiver_ids == ["relay.site-1.job-1"]

        agent = _make_agent_for_result_upload()
        transaction = DownloadTransactionInfo("tx-swarm", (("ref-swarm", "relay.site-1.job-1"),), time.time())
        callbacks = {}

        def _send(reply, timeout):
            ctx = agent.pipe.cell.update_fobs_context.call_args.args[0]
            callbacks["progress"] = ctx[fobs.FOBSContextKey.STREAM_PROGRESS_CB]
            callbacks["complete"] = ctx[fobs.FOBSContextKey.DOWNLOAD_COMPLETE_CB]
            ctx[RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY](transaction)
            # send_to_peer is mocked in this E2E boundary test, so the real FOBS encode path does not run here.
            # via_downloader_test.py covers encode-time population of this thread-local state.
            _tls.download_initiated = True
            _tls.download_transactions = [transaction]

            callbacks["progress"](
                tx_id="tx-swarm",
                transfer_id="ref-swarm",
                receiver_id="relay.site-1.job-1",
                sequence=1,
                bytes_done=1024,
                state="active",
            )
            progress_receivers.append("relay.site-1.job-1")
            callbacks["progress"](
                tx_id="tx-swarm",
                transfer_id="ref-swarm",
                receiver_id="relay.site-1.job-1",
                sequence=2,
                bytes_done=2048,
                state="completed",
            )
            callbacks["complete"]("tx-swarm", TransactionDoneStatus.FINISHED, [])
            return True

        agent.pipe_handler.send_to_peer.side_effect = _send
        ok = agent._do_submit_result(
            _TaskContext(
                "tid-swarm",
                "swarm_learn",
                "msg-swarm",
                result_receiver_ids=tuple(receiver_ids),
            ),
            None,
            "OK",
        )
        assert ok is True
        result = Shareable()
        result.set_return_code(ReturnCode.OK)
        return result

    trainer_ctl.execute_learn_task = _execute_learn_task

    task_data = _make_model_shareable()
    task_data.set_header(AppConstants.CURRENT_ROUND, 0)
    task_data.set_header(Constant.AGGREGATOR, "site-1")

    trainer_ctl.do_learn_task("swarm_learn", task_data, trainer_fl_ctx, _AbortSignal())

    assert progress_receivers == ["relay.site-1.job-1"]
    trainer_fl_ctx.get_engine.return_value.send_aux_request.assert_called_once()
    trainer_ctl.broadcast_and_wait.assert_called_once()
    assert trainer_ctl.update_status.call_args.kwargs["action"] == "finished_learn_task"


def test_swarm_result_upload_receiver_id_falls_back_when_client_fqcn_unknown():
    trainer_ctl = _make_swarm_controller(me="site-2", trainers=["site-1", "site-2"], aggrs=["site-1"])
    trainer_fl_ctx = _make_fl_ctx(job_id="job-1", site_name="site-2")
    task_data = Shareable()

    trainer_ctl._stamp_result_upload_receiver_ids(task_data, "site-1", trainer_fl_ctx)

    assert task_data.get_header(fobs.FOBSContextKey.RECEIVER_IDS) is None
    trainer_ctl.log_warning.assert_called_once()
