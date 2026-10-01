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

from unittest.mock import MagicMock

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather


def _make_controller():
    controller = ScatterAndGather(
        min_clients=1,
        num_rounds=1,
        wait_time_after_min_received=0,
        snapshot_every_n_rounds=0,
        memory_gc_rounds=0,
    )
    controller._engine = MagicMock()
    controller._engine.get_clients.return_value = [Client("site-1", "token")]
    controller.aggregator = MagicMock()
    controller.shareable_gen = MagicMock()
    controller.shareable_gen.learnable_to_shareable.return_value = Shareable()
    controller.shareable_gen.shareable_to_learnable.return_value = make_model_learnable({}, {})
    controller.fire_event = MagicMock()
    controller.fire_event_with_data = MagicMock()
    controller.system_panic = MagicMock()
    return controller


def _make_result(round_num):
    result = Shareable()
    result.add_cookie(AppConstants.CONTRIBUTION_ROUND, round_num)
    return result


def test_control_flow_panics_before_aggregation_when_no_results_arrive():
    controller = _make_controller()
    controller.broadcast_and_wait = MagicMock()
    fl_ctx = FLContext()

    controller.control_flow(Signal(), fl_ctx)

    controller.system_panic.assert_called_once_with(
        reason="No successful client results were received in round 0.",
        fl_ctx=fl_ctx,
    )
    controller.aggregator.aggregate.assert_not_called()
    assert all(call.args[0] != AppEventType.BEFORE_AGGREGATION for call in controller.fire_event.call_args_list)
    assert controller._current_round == 0


def test_control_flow_aggregates_after_current_round_result():
    controller = _make_controller()
    controller.aggregator.accept.return_value = True
    controller.aggregator.aggregate.return_value = Shareable()
    fl_ctx = FLContext()

    def broadcast_and_accept(task, **kwargs):
        client_task = ClientTask(client=Client("site-1", "token"), task=task)
        client_task.result = _make_result(round_num=0)
        controller._process_train_result(client_task, fl_ctx)

    controller.broadcast_and_wait = MagicMock(side_effect=broadcast_and_accept)

    controller.control_flow(Signal(), fl_ctx)

    controller.system_panic.assert_not_called()
    controller.aggregator.aggregate.assert_called_once_with(fl_ctx)
    controller.fire_event_with_data.assert_any_call(
        AppEventType.AFTER_AGGREGATION,
        fl_ctx,
        AppConstants.AGGREGATION_RESULT,
        controller.aggregator.aggregate.return_value,
    )
    assert controller._current_round == 1


def test_stale_result_reaches_aggregator_without_counting_for_current_round():
    controller = _make_controller()
    controller._current_round = 1
    controller._round_result_counts[1] = 0
    controller.aggregator.accept.return_value = True
    result = _make_result(round_num=0)
    fl_ctx = FLContext()

    assert controller._accept_train_result("site-1", result, fl_ctx)

    controller.aggregator.accept.assert_called_once_with(result, fl_ctx)
    assert controller._round_result_counts[1] == 0
    assert controller._round_result_counts[0] == 1
