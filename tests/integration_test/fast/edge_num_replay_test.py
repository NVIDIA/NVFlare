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

"""Real numeric updater/assessor workflows, on CPU with no transport or workers."""

import pytest

from nvflare.apis.dxo import from_shareable
from nvflare.edge.mud import StateUpdateReply, StateUpdateReport
from tests.edge_num_test_utils import forward, make_workflow, numeric_update, report


@pytest.mark.parametrize("replay", [False, True])
def test_consecutive_reports_preserve_fresh_b_and_finish_model(replay):
    assessor, updater, ctx = make_workflow()
    a = numeric_update(1, ["A"], 10)
    updater.process_child_update(report(a), ctx)
    assert forward(updater, assessor, ctx)[0]
    selected, cookie = updater.current_state.is_device_selected("B", 0)
    assert selected

    if replay:
        updater.process_child_update(report(a), ctx)
    updater.process_child_update(report(numeric_update(1, ["B"], 1)), ctx)
    # B cannot be asked to retrain with the consumed selection cookie.
    assert not updater.current_state.is_device_selected("B", cookie)[0]
    accepted, sent = forward(updater, assessor, ctx)
    assert sent.model_updates[1].devices == {"B": 1.0}
    assert from_shareable(sent.model_updates[1].update).data == {"value": 1, "count": 1}
    assert accepted
    assert assessor.current_model_version == 2
    assert assessor.current_model.data["value"] == 5.5
    assert updater.current_state.model_version == 2
    assert StateUpdateReport.from_shareable(updater.prepare_update_for_parent(ctx)).model_updates is None


def test_replay_before_first_flush_is_not_aggregated_twice():
    assessor, updater, ctx = make_workflow()
    for device, value in [("A", 10), ("A", 10), ("B", 1)]:
        updater.process_child_update(report(numeric_update(1, [device], value)), ctx)
    accepted, sent = forward(updater, assessor, ctx)
    assert accepted
    assert from_shareable(sent.model_updates[1].update).data == {"value": 11, "count": 2}
    assert assessor.current_model.data["value"] == 5.5


@pytest.mark.parametrize("history", [None, 1])
def test_replay_after_updater_pruning_preserves_fresh_late_contribution(history):
    assessor, updater, ctx = make_workflow(history)
    a = numeric_update(1, ["A"], 10)
    updater.process_child_update(report(a), ctx)
    forward(updater, assessor, ctx)
    # The server retains V1, while this updater has no pending V1 devices and
    # prunes its aggregator (both inactive-selection and age-based cases).
    assessor._generate_new_model(ctx)
    assessor._generate_new_model(ctx)
    selection = {"B": 3} if history is None else {"B": 1}
    updater.process_parent_update_reply(StateUpdateReply(3, assessor.current_model, 3, selection).to_shareable(), ctx)
    assert 1 not in updater.aggr_states
    assert 1 in assessor.updates
    updater.process_child_update(report(a), ctx)
    updater.process_child_update(report(numeric_update(1, ["B"], 1)), ctx)
    accepted, sent = forward(updater, assessor, ctx)
    assert sent.model_updates[1].devices == {"B": 1.0}
    assert accepted
    assert assessor.updates[1].aggregator.value == 11
    assert assessor.updates[1].aggregator.count == 2


def test_disjoint_batches_and_same_device_on_multiple_versions():
    assessor, updater, ctx = make_workflow()
    updater.process_child_update(report(numeric_update(1, ["A", "B"], 30, 3)), ctx)
    assert forward(updater, assessor, ctx)[0]
    assert assessor.current_model.data["value"] == 10
    updater.process_child_update(report(numeric_update(1, ["C", "D"], 10, 2), numeric_update(2, ["A", "B"], 6, 2)), ctx)
    accepted, sent = forward(updater, assessor, ctx)
    assert accepted
    assert set(sent.model_updates) == {1, 2}
    assert from_shareable(sent.model_updates[1].update).data == {"value": 10, "count": 2}
    assert assessor.current_model_version == 3
    assert assessor.current_model.data["value"] == 7  # (40 / 5) / 2 + (6 / 2)


@pytest.mark.parametrize("kind,count,error", [("weights", 1, ValueError), ("number", "bad", TypeError)])
def test_updater_rejected_payload_allows_valid_retry(kind, count, error):
    assessor, updater, ctx = make_workflow()
    with pytest.raises(error):
        updater.process_child_update(report(numeric_update(1, ["A"], 99, count, kind=kind)), ctx)
    for device, value in [("A", 10), ("B", 1)]:
        updater.process_child_update(report(numeric_update(1, [device], value)), ctx)
    assert forward(updater, assessor, ctx)[0]
    assert assessor.current_model.data["value"] == 5.5


def test_updater_partial_overlap_fails_without_poisoning_fresh_retry():
    assessor, updater, ctx = make_workflow()
    updater.process_child_update(report(numeric_update(1, ["A"], 10)), ctx)
    forward(updater, assessor, ctx)
    with pytest.raises(ValueError, match="partial overlap"):
        updater.process_child_update(report(numeric_update(1, ["A", "B"], 11, 2)), ctx)
    updater.process_child_update(report(numeric_update(1, ["B"], 1)), ctx)
    assert forward(updater, assessor, ctx)[0]
    assert assessor.current_model.data["value"] == 5.5


@pytest.mark.parametrize("replay_first", [True, False])
def test_updater_reports_any_rejection_but_keeps_other_versions(replay_first):
    assessor, updater, ctx = make_workflow()
    old = numeric_update(1, ["A"], 10)
    updater.process_child_update(report(old), ctx)
    forward(updater, assessor, ctx)
    fresh = numeric_update(2, ["A"], 2)
    updates = (old, fresh) if replay_first else (fresh, old)
    accepted, _ = updater.process_child_update(report(*updates), ctx)
    assert not accepted
    sent = StateUpdateReport.from_shareable(updater.prepare_update_for_parent(ctx))
    assert set(sent.model_updates) == {2}
    assert from_shareable(sent.model_updates[2].update).data == {"value": 2, "count": 1}


def test_new_task_can_accept_same_device_and_version():
    assessor, updater, ctx = make_workflow()
    a = numeric_update(1, ["A"], 10)
    updater.process_child_update(report(a), ctx)
    forward(updater, assessor, ctx)
    updater.end_task(ctx)
    updater.start_task(assessor.start_task(ctx), ctx)
    accepted, _ = updater.process_child_update(report(a), ctx)
    assert accepted
    sent = StateUpdateReport.from_shareable(updater.prepare_update_for_parent(ctx))
    assert from_shareable(sent.model_updates[1].update).data == {"value": 10, "count": 1}
