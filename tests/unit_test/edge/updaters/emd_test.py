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

import pytest

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.edge.aggregators.model_update_dxo_factory import ModelUpdateDXOAggrFactory
from nvflare.edge.mud import BaseState, ModelUpdate, StateUpdateReply, StateUpdateReport
from nvflare.edge.updaters.emd import EdgeModelUpdater, ModelAggrState


def _updater(history=None):
    updater = EdgeModelUpdater(ModelUpdateDXOAggrFactory(), history)
    updater.log_info = MagicMock()
    updater.log_debug = MagicMock()
    model = DXO(DataKind.WEIGHTS, {"w": [0.0]})
    updater.current_state = BaseState(1, model, 1, {"A": 1})
    return updater, FLContext()


def _reply(version, selection):
    return StateUpdateReply(version, DXO(DataKind.WEIGHTS, {"w": [0.0]}), version, selection).to_shareable()


@pytest.mark.parametrize("history,version", [(None, 2), (1, 3)])
def test_parent_reply_keeps_pending_updates_until_forwarded(history, version):
    updater, ctx = _updater(history)
    contribution = ModelUpdate(
        1,
        DXO(DataKind.WEIGHT_DIFF, {"dict": {"w": [10.0]}, "count": 1}).to_shareable(),
        {"A": 1.0},
    )
    updater.process_child_update(StateUpdateReport(1, 1, {1: contribution}, {}).to_shareable(), ctx)

    # A late result can arrive while the LCP waits for the parent's newer state.
    reply = _reply(version, {"B": version})
    updater.process_parent_update_reply(reply, ctx)
    assert 1 in updater.aggr_states

    report = StateUpdateReport.from_shareable(updater.prepare_update_for_parent(ctx))
    result = from_shareable(report.model_updates[1].update)
    assert report.model_updates[1].devices == {"A": 1.0}
    assert result.data["dict"]["w"][0] == 10.0
    assert result.data["count"] == 1

    # Once forwarded, the empty bucket can be removed by the next pruning pass.
    updater.process_parent_update_reply(reply, ctx)
    assert 1 not in updater.aggr_states


def test_parent_reply_keeps_current_empty_model_state():
    updater, ctx = _updater()
    updater.aggr_states[2] = ModelAggrState(updater.aggr_factory.get_aggregator(), 2)

    updater.process_parent_update_reply(_reply(2, {"A": 1}), ctx)

    assert 2 in updater.aggr_states


@pytest.mark.parametrize("history,version,retained", [(None, 2, True), (1, 3, False)])
def test_parent_reply_prunes_only_empty_retired_states(history, version, retained):
    updater, ctx = _updater(history)
    updater.aggr_states[1] = ModelAggrState(updater.aggr_factory.get_aggregator(), 1)

    updater.process_parent_update_reply(_reply(version, {"A": 1}), ctx)

    assert (1 in updater.aggr_states) is retained
