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

from unittest.mock import Mock

import numpy as np

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import TaskCompletionStatus
from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.impl.wf_comm_server import WFCommServer
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.model import ModelLearnableKey, make_model_learnable
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.aggregators.weighted_aggregation_helper import AggregationStatsKey
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather


def test_shape_rejection_keeps_broadcast_open_for_later_valid_result(monkeypatch):
    clients = [Client(f"site-{i}", f"token-{i}") for i in range(1, 4)]
    engine = Mock(spec=ServerEngineSpec)
    engine.get_clients.return_value = clients
    fl_ctx = FLContextManager(engine, "server", "job", {}, {}).new_context()
    engine.new_context.side_effect = lambda: fl_ctx
    aggregator = InTimeAccumulateWeightedAggregator()
    aggregator.handle_event(EventType.START_RUN, fl_ctx)
    persistor = Mock(spec=LearnablePersistor)
    persistor.load.return_value = make_model_learnable({"early": np.zeros(2), "w": np.zeros(2)}, {})
    components = {
        "aggregator": aggregator,
        "persistor": persistor,
        "shareable_generator": FullModelShareableGenerator(),
    }
    engine.get_component.side_effect = components.get
    controller = ScatterAndGather(
        min_clients=3, num_rounds=1, persistor_id="persistor", snapshot_every_n_rounds=0, memory_gc_rounds=0
    )
    comm = WFCommServer()
    comm._engine = engine
    controller.set_communicator(comm)
    controller.initialize(fl_ctx)
    panic = Mock()
    monkeypatch.setattr(controller, "system_panic", panic)
    accepted, statuses, tasks = [], [], []

    def broadcast(task, fl_ctx, min_responses, wait_time_after_min_received, **kwargs):
        tasks.append(task)
        comm.broadcast(task, fl_ctx, min_responses=min_responses, wait_time_after_min_received=0)
        assignments = [comm.process_task_request(client, fl_ctx) for client in clients]
        contributions = [
            ({"early": np.array([2.0, 4.0]), "w": np.array([1.0, 2.0])}, 1),
            ({"early": np.array([100.0, 200.0]), "w": np.array([10.0])}, 4),
            ({"early": np.array([6.0, 8.0]), "w": np.array([5.0, 6.0])}, 3),
        ]
        for client, (task_name, task_id, _), (data, steps) in zip(clients, assignments, contributions):
            result = DXO(DataKind.WEIGHT_DIFF, data, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: steps}).to_shareable()
            result.set_peer_props({ReservedKey.IDENTITY_NAME: client.name})
            result.add_cookie(AppConstants.CONTRIBUTION_ROUND, 0)
            comm.process_submission(client, task_name, task_id, result, fl_ctx)
            accepted.append(fl_ctx.get_prop(AppConstants.AGGREGATION_ACCEPTED))
            statuses.append(task.completion_status)
        comm.check_tasks()

    monkeypatch.setattr(controller, "broadcast_and_wait", broadcast)
    controller.control_flow(Signal(), fl_ctx)

    panic.assert_not_called()
    assert accepted == [True, False, True]
    assert statuses == [None, None, None]
    assert tasks[0].completion_status == TaskCompletionStatus.OK
    stats = fl_ctx.get_prop(AppConstants.AGGREGATION_STATS)
    assert stats[AggregationStatsKey.CONTRIBUTORS] == ["site-1", "site-3"]
    assert stats[AggregationStatsKey.ACCEPTED_CONTRIBUTIONS] == 2
    persistor.save.assert_called_once()
    saved_weights = persistor.save.call_args.args[0][ModelLearnableKey.WEIGHTS]
    np.testing.assert_allclose(saved_weights["early"], [5.0, 7.0])
    np.testing.assert_allclose(saved_weights["w"], [4.0, 5.0])
    assert controller._current_round == 1
