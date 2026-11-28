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

import pytest

from nvflare.apis.fl_context import FLContext
from tests.edge_num_test_utils import make_assessor, numeric_update, report


def test_duplicate_report_does_not_change_average_or_release_selection():
    ctx = FLContext()
    assessor = make_assessor(ctx)
    assessor.process_child_update(report(numeric_update(1, ["A"], 10)), ctx)
    # A later selection must not be released by a replay of A's old contribution.
    assessor.current_selection["A"] = 2
    accepted, _ = assessor.process_child_update(report(numeric_update(1, ["A"], 10, timestamp=2.0)), ctx)
    selection_after_replay = assessor.current_selection.copy()
    assessor.process_child_update(report(numeric_update(1, ["B"], 1)), ctx)
    assert assessor.current_model_version == 2
    assert assessor.current_model.data["value"] == 5.5
    assert selection_after_replay["A"] == 2
    assert not accepted
    assert assessor.updates[1].devices == {"A": 1.0, "B": 1.0}


def test_disjoint_preaggregated_reports_keep_numeric_weights():
    ctx = FLContext()
    assessor = make_assessor(ctx, threshold=4)
    for devices, value, count in [(["A", "B"], 30, 3), (["C", "D"], 10, 2)]:
        accepted, _ = assessor.process_child_update(report(numeric_update(1, devices, value, count)), ctx)
        assert accepted
    assert assessor.current_model_version == 2
    assert assessor.current_model.data["value"] == 8


@pytest.mark.parametrize("replay_first", [True, False])
def test_retained_history_replay_does_not_hide_fresh_version(replay_first):
    ctx = FLContext()
    assessor = make_assessor(ctx, threshold=1)
    old = numeric_update(1, ["A"], 10)
    assessor.process_child_update(report(old), ctx)
    fresh = numeric_update(2, ["A"], 2)
    updates = (old, fresh) if replay_first else (fresh, old)
    accepted, _ = assessor.process_child_update(report(*updates), ctx)
    assert not accepted
    assert assessor.updates[1].aggregator.count == 1
    assert assessor.updates[2].devices == {"A": 1.0}
    assert assessor.current_model_version == 3
    assert assessor.current_model.data["value"] == 7


@pytest.mark.parametrize("kind,count,error", [("weights", 1, ValueError), ("number", "bad", TypeError)])
def test_rejected_payload_does_not_poison_device_or_numeric_state(kind, count, error):
    ctx = FLContext()
    assessor = make_assessor(ctx)
    with pytest.raises(error):
        assessor.process_child_update(report(numeric_update(1, ["A"], 99, count, kind=kind)), ctx)
    state = assessor.updates[1]
    assert state.devices == {}
    assert state.last_update_time is None
    assert (state.aggregator.value, state.aggregator.count) == (0, 0)
    assessor.process_child_update(report(numeric_update(1, ["A"], 10)), ctx)
    assessor.process_child_update(report(numeric_update(1, ["B"], 1)), ctx)
    assert assessor.current_model.data["value"] == 5.5


def test_partial_overlap_is_explicit_and_does_not_consume_fresh_device():
    ctx = FLContext()
    assessor = make_assessor(ctx)
    assessor.process_child_update(report(numeric_update(1, ["A"], 10)), ctx)
    with pytest.raises(ValueError, match="partial overlap"):
        assessor.process_child_update(report(numeric_update(1, ["A", "B"], 11, 2)), ctx)
    assert assessor.updates[1].devices == {"A": 1.0}
    assert assessor.current_selection == {"B": 1}
    assessor.process_child_update(report(numeric_update(1, ["B"], 1)), ctx)
    assert assessor.current_model.data["value"] == 5.5
