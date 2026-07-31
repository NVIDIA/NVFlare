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

"""Two-site/two-round Swarm execution-mode matrix.

Each mode pair runs twice:

- round 0: site-1 aggregates
- round 1: site-2 aggregates

The test exercises learner-input ref ownership, local-result handling, remote
result PASS_THROUGH, and terminal aggregation-CJ offload policy. Transport and
disk writing primitives are covered separately by the Cell/FOBS tests.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.ccwf.common import Constant
from nvflare.app_common.ccwf.swarm_client_ctl import SwarmClientController
from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor, ExecutionMode
from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef

_KIND = "test.swarm_matrix.kind"
_STORAGE = "test.swarm_matrix.storage"
_LEARN_INPUT = "learn_input"
_LOCAL_RESULT = "local_result"
_REMOTE_RESULT = "remote_result"
_DISK = "disk"
_MEMORY = "memory"


class _AbortSignal:
    triggered = False


class _MatrixGatherer:
    def __init__(self, for_round, **_kwargs):
        self.for_round = for_round
        self.submissions = {}

    def gather(self, client_name, request, _fl_ctx):
        self.submissions[client_name] = request
        return make_reply(ReturnCode.OK)


def _make_lazy_shareable(kind: str, current_round: int) -> Shareable:
    data = {
        "layer.weight": LazyDownloadRef(
            fqcn=f"source.round-{current_round}",
            ref_id=f"ref-{kind}-{current_round}",
            item_id="T0",
        ),
        "layer.bias": LazyDownloadRef(
            fqcn=f"source.round-{current_round}",
            ref_id=f"ref-{kind}-{current_round}",
            item_id="T1",
        ),
    }
    result = DXO(data_kind=DataKind.WEIGHTS, data=data).to_shareable()
    result.set_header(AppConstants.CURRENT_ROUND, current_round)
    result.set_header(_KIND, kind)
    return result


def _make_memory_shareable(kind: str, current_round: int) -> Shareable:
    data = {
        "layer.weight": np.full((2, 2), current_round + 1, dtype=np.float32),
        "layer.bias": np.full((2,), current_round + 1, dtype=np.float32),
    }
    result = DXO(data_kind=DataKind.WEIGHTS, data=data).to_shareable()
    result.set_header(AppConstants.CURRENT_ROUND, current_round)
    result.set_header(_KIND, kind)
    result.set_header(_STORAGE, _MEMORY)
    return result


def _make_executor(mode: str) -> ClientAPIExecutor:
    if mode == ExecutionMode.EXTERNAL_PROCESS:
        return ClientAPIExecutor(execution_mode=mode, command="python train.py")
    return ClientAPIExecutor(execution_mode=mode, task_script_path="train.py")


def _make_context(site_name: str, engine) -> FLContext:
    fl_ctx = FLContext()
    fl_ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=False)
    fl_ctx.set_prop(ReservedKey.IDENTITY_NAME, site_name, private=True, sticky=False)
    fl_ctx.set_prop(ReservedKey.RUN_NUM, "matrix-job", private=True, sticky=False)
    return fl_ctx


def _make_peer_context(site_name: str) -> FLContext:
    peer_ctx = FLContext()
    peer_ctx.set_prop(ReservedKey.IDENTITY_NAME, site_name, private=True, sticky=False)
    return peer_ctx


def _make_controller(site_name: str, mode: str):
    ctl = SwarmClientController.__new__(SwarmClientController)
    ctl.logger = MagicMock()
    ctl.log_info = MagicMock()
    ctl.log_error = MagicMock()
    ctl.log_debug = MagicMock()
    ctl.log_warning = MagicMock()
    ctl.me = site_name
    ctl.trainers = ["site-1", "site-2"]
    ctl.aggrs = ["site-1", "site-2"]
    ctl.is_trainer = True
    ctl.is_aggr = True
    ctl.enable_tensor_disk_offload = True
    ctl.learn_executor = _make_executor(mode)
    ctl.learn_task_timeout = 60
    ctl.learn_task_ack_timeout = 10
    ctl.request_to_submit_learn_result_task_name = "swarm_request_to_submit_learn_result"
    ctl.request_to_submit_result_max_wait = None
    ctl.request_to_submit_result_msg_timeout = 5.0
    ctl.request_to_submit_result_interval = 0.0
    ctl.report_learn_result_task_name = "swarm_report_learn_result"
    ctl.last_aggr_round_done = -1
    ctl.learn_task_abort_timeout = 10.0
    ctl.min_responses_required = 2
    ctl.wait_time_after_min_resps_received = 0.0
    ctl.max_concurrent_submissions = 1
    ctl.memory_gc_rounds = 0
    ctl.cuda_empty_cache = False
    ctl._aggr_round_count = 0
    ctl.gatherer = None
    ctl.gatherer_waiter = MagicMock()
    ctl.metric_comparator = None
    ctl.shareable_generator = MagicMock()
    ctl.shareable_generator.shareable_to_learnable.return_value = {}
    ctl.aggregator = MagicMock()
    ctl.update_status = MagicMock()
    ctl.fire_event = MagicMock()
    ctl._stamp_result_upload_receiver_ids = MagicMock()
    ctl.resolve_records = []

    def _config(key, default=None):
        values = {
            Constant.CLIENTS: ["site-1", "site-2"],
            Constant.TRAIN_CLIENTS: ["site-1", "site-2"],
            Constant.AGGR_CLIENTS: ["site-1", "site-2"],
            Constant.PRIVATE_P2P: False,
        }
        return values.get(key, default)

    ctl.get_config_prop = MagicMock(side_effect=_config)

    def _resolve(payload, _fl_ctx, enable_tensor_disk_offload=None):
        use_disk = ctl.enable_tensor_disk_offload if enable_tensor_disk_offload is None else enable_tensor_disk_offload
        kind = payload.get_header(_KIND)
        was_lazy = ctl._has_lazy_refs(payload)
        ctl.resolve_records.append(
            {
                "kind": kind,
                "round": payload.get_header(AppConstants.CURRENT_ROUND),
                "use_disk": use_disk,
                "was_lazy": was_lazy,
            }
        )
        if not was_lazy:
            return payload

        resolved = _make_memory_shareable(kind, payload.get_header(AppConstants.CURRENT_ROUND))
        if use_disk:
            resolved.set_header(_STORAGE, _DISK)
        return resolved

    ctl._resolve_lazy_refs = _resolve
    return ctl


@pytest.mark.parametrize(
    ("site_1_mode", "site_2_mode"),
    [
        (ExecutionMode.IN_PROCESS, ExecutionMode.EXTERNAL_PROCESS),
        (ExecutionMode.IN_PROCESS, ExecutionMode.IN_PROCESS),
        (ExecutionMode.EXTERNAL_PROCESS, ExecutionMode.EXTERNAL_PROCESS),
        (ExecutionMode.EXTERNAL_PROCESS, ExecutionMode.IN_PROCESS),
    ],
    ids=[
        "site1-in_site2-ex",
        "site1-in_site2-in",
        "site1-ex_site2-ex",
        "site1-ex_site2-in",
    ],
)
def test_two_round_aggregation_switch_covers_execution_mode_matrix(site_1_mode, site_2_mode):
    modes = {"site-1": site_1_mode, "site-2": site_2_mode}
    controllers = {site: _make_controller(site, mode) for site, mode in modes.items()}
    contexts = {}
    trainer_inputs = {}
    remote_pass_through = {}

    for site, ctl in controllers.items():
        engine = MagicMock()
        engine.new_context.side_effect = lambda site=site: _make_peer_context(site)

        def _grant(targets, **_kwargs):
            return {targets[0]: make_reply(ReturnCode.OK)}

        engine.send_aux_request.side_effect = _grant
        contexts[site] = _make_context(site, engine)

        def _execute(task_data, _fl_ctx, _abort_signal, site=site, ctl=ctl):
            current_round = task_data.get_header(AppConstants.CURRENT_ROUND)
            trainer_inputs[(current_round, site)] = ctl._has_lazy_refs(task_data)
            if modes[site] == ExecutionMode.EXTERNAL_PROCESS:
                return _make_lazy_shareable(_LOCAL_RESULT, current_round)
            return _make_memory_shareable(_LOCAL_RESULT, current_round)

        ctl.execute_learn_task = _execute

    def _make_broadcast(sender: str):
        def _broadcast(task, targets, **_kwargs):
            aggregator = targets[0]
            current_round = task.data.get_header(AppConstants.CURRENT_ROUND)
            remote_pass_through[(current_round, sender)] = task.data.get_header(
                ReservedHeaderKey.PASS_THROUGH,
                False,
            )

            # Model Adapter.call(PASS_THROUGH=True): both an external trainer ref
            # and an in-process tensor stream arrive as LazyDownloadRef objects.
            wire_result = _make_lazy_shareable(_REMOTE_RESULT, current_round)
            wire_result.set_header(ReservedHeaderKey.PASS_THROUGH, True)
            aggr_fl_ctx = contexts[aggregator].clone()
            aggr_fl_ctx.set_peer_context(_make_peer_context(sender))
            reply = controllers[aggregator]._process_learn_result(
                wire_result,
                aggr_fl_ctx,
                _AbortSignal(),
            )
            return {aggregator: reply}

        return _broadcast

    for site, ctl in controllers.items():
        ctl.broadcast_and_wait = _make_broadcast(site)

    aggregators = {0: "site-1", 1: "site-2"}
    round_submissions = {}
    with patch("nvflare.app_common.ccwf.swarm_client_ctl.Gatherer", _MatrixGatherer):
        for current_round, aggregator in aggregators.items():
            for ctl in controllers.values():
                ctl.gatherer = None

            remote_site = "site-2" if aggregator == "site-1" else "site-1"
            for site in (aggregator, remote_site):
                task_data = _make_lazy_shareable(_LEARN_INPUT, current_round)
                task_data.set_header(Constant.AGGREGATOR, aggregator)
                controllers[site].do_learn_task(
                    "swarm_learn",
                    task_data,
                    contexts[site],
                    _AbortSignal(),
                )

            gatherer = controllers[aggregator].gatherer
            assert set(gatherer.submissions) == {"site-1", "site-2"}
            round_submissions[current_round] = gatherer.submissions

    for current_round, aggregator in aggregators.items():
        remote_site = "site-2" if aggregator == "site-1" else "site-1"

        # The aggregation controller needs an ordinary base model. The remote
        # external trainer is the only case allowed to retain the learn-task ref.
        for site in ("site-1", "site-2"):
            expected_lazy_input = modes[site] == ExecutionMode.EXTERNAL_PROCESS and site != aggregator
            assert trainer_inputs[(current_round, site)] is expected_lazy_input
            input_resolutions = [
                record
                for record in controllers[site].resolve_records
                if record["kind"] == _LEARN_INPUT and record["round"] == current_round
            ]
            if expected_lazy_input:
                assert input_resolutions == []
            else:
                assert len(input_resolutions) == 1
                assert input_resolutions[0]["use_disk"] is False
                assert input_resolutions[0]["was_lazy"] is True

        # Every cross-site result remains a ref until the aggregation CJ and is
        # therefore disk-backed there, independent of the sender mode.
        assert remote_pass_through[(current_round, remote_site)] is True
        remote_submission = round_submissions[current_round][remote_site]
        assert remote_submission.get_header(_STORAGE) == _DISK
        assert remote_submission.get_header(ReservedHeaderKey.PASS_THROUGH) is False
        remote_resolutions = [
            record
            for record in controllers[aggregator].resolve_records
            if record["kind"] == _REMOTE_RESULT and record["round"] == current_round
        ]
        assert len(remote_resolutions) == 1
        assert remote_resolutions[0]["use_disk"] is True
        assert remote_resolutions[0]["was_lazy"] is True

        # A local external trainer returns a transport ref, so the aggregation
        # CJ offloads it. A local in-process result already exists in CJ memory.
        expected_local_storage = _DISK if modes[aggregator] == ExecutionMode.EXTERNAL_PROCESS else _MEMORY
        assert round_submissions[current_round][aggregator].get_header(_STORAGE) == expected_local_storage
        local_resolutions = [
            record
            for record in controllers[aggregator].resolve_records
            if record["kind"] == _LOCAL_RESULT and record["round"] == current_round
        ]
        if modes[aggregator] == ExecutionMode.EXTERNAL_PROCESS:
            assert len(local_resolutions) == 1
            assert local_resolutions[0]["use_disk"] is True
            assert local_resolutions[0]["was_lazy"] is True
        else:
            assert local_resolutions == []
