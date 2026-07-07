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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.dxo import DXO
from nvflare.apis.fl_context import FLContext
from nvflare.edge.aggregators.num_dxo import NumDXOAggregator
from nvflare.edge.assessor import Assessment
from nvflare.edge.assessors import async_num, num
from nvflare.edge.mud import BaseState, Device, ModelUpdate, StateUpdateReply, StateUpdateReport

ASSESSORS = [(num.NumAssessor, num._ModelState, num), (async_num.AsyncNumAssessor, async_num._ModelState, async_num)]


def _assessor(assessor_cls, **kwargs):
    params = {
        "num_updates_for_model": 1,
        "max_model_version": 3,
        "max_model_history": 2,
        "device_selection_size": 2,
        "min_hole_to_fill": 1,
        "device_reuse": True,
    }
    params.update(kwargs)
    assessor = assessor_cls(**params)
    assessor.log_debug = MagicMock()
    assessor.log_info = MagicMock()
    assessor.log_error = MagicMock()
    assessor.log_warning = MagicMock()
    return assessor


def _devices():
    return {
        "device-1": Device("device-1", "site-1", 1.0),
        "device-2": Device("device-2", "site-1", 2.0),
        "device-3": Device("device-3", "site-2", 3.0),
    }


@pytest.mark.parametrize("assessor_cls,state_cls,module", ASSESSORS)
def test_model_state_accepts_numeric_update(assessor_cls, state_cls, module):
    state = state_cls(NumDXOAggregator())
    update = ModelUpdate(1, DXO("number", {"value": 4.0, "count": 2}).to_shareable(), {"device-1": 1.0})

    assert state.accept(update, FLContext())
    assert state.devices == {"device-1": 1.0}
    assert state.aggregator.value == 4.0
    assert state.aggregator.count == 2
    assert state.last_update_time is not None


@pytest.mark.parametrize("assessor_cls,state_cls,module", ASSESSORS)
def test_start_task_returns_current_base_state(assessor_cls, state_cls, module):
    assessor = _assessor(assessor_cls)
    state = BaseState.from_shareable(assessor.start_task(FLContext()))
    assert state.model_version == 0
    assert state.device_selection_version == 0
    assert assessor.start_time is not None


@pytest.mark.parametrize("assessor_cls,state_cls,module", ASSESSORS)
def test_initial_device_report_generates_model_and_selection(assessor_cls, state_cls, module):
    assessor = _assessor(assessor_cls)
    report = StateUpdateReport(0, 0, None, _devices())

    accepted, reply_data = assessor.process_child_update(report.to_shareable(), FLContext())
    reply = StateUpdateReply.from_shareable(reply_data)

    assert accepted
    assert assessor.current_model_version == 1
    assert reply.model_version == 1
    assert reply.model.data == {"value": 0.0}
    assert len(reply.device_selection) == 2
    assert assessor.current_selection_version == 1
    assert 1 in assessor.updates


@pytest.mark.parametrize("assessor_cls,state_cls,module", ASSESSORS)
def test_model_update_is_accepted_and_generates_next_model(assessor_cls, state_cls, module):
    assessor = _assessor(assessor_cls, device_selection_size=1, device_reuse=False)
    assessor.process_child_update(StateUpdateReport(0, 0, None, _devices()).to_shareable(), FLContext())
    device_id = next(iter(assessor.current_selection))
    update = ModelUpdate(
        1,
        DXO("number", {"value": 8.0, "count": 2}).to_shareable(),
        {device_id: 5.0},
    )

    accepted, reply_data = assessor.process_child_update(
        StateUpdateReport(1, assessor.current_selection_version, {1: update}, None).to_shareable(), FLContext()
    )
    reply = StateUpdateReply.from_shareable(reply_data)

    assert accepted
    assert assessor.current_model_version == 2
    assert assessor.current_model.data["value"] == 4.0
    assert reply.model_version == 2
    assert device_id not in assessor.current_selection


@pytest.mark.parametrize("assessor_cls,state_cls,module", ASSESSORS)
def test_generate_model_weights_history_and_removes_old_versions(assessor_cls, state_cls, module):
    assessor = _assessor(assessor_cls, max_model_history=2)
    old = state_cls(NumDXOAggregator())
    old.aggregator.value = 6.0
    old.aggregator.count = 2
    current = state_cls(NumDXOAggregator())
    current.aggregator.value = 4.0
    current.aggregator.count = 2
    assessor.current_model_version = 2
    assessor.updates = {1: old, 2: current}

    assessor._generate_new_model(FLContext())

    assert assessor.current_model_version == 3
    assert assessor.current_model.data["value"] == 3.5
    assert 1 not in assessor.updates
    assert set(assessor.updates) == {2, 3}


@pytest.mark.parametrize("assessor_cls,state_cls,module", ASSESSORS)
def test_child_update_skips_bad_stale_and_unknown_versions(assessor_cls, state_cls, module):
    assessor = _assessor(assessor_cls)
    assessor.current_model_version = 5
    assessor.updates = {5: "bad state"}
    report = SimpleNamespace(
        available_devices={},
        model_updates={0: object(), 1: object(), 4: None, 3: object()},
        current_model_version=5,
    )

    with patch.object(module.StateUpdateReport, "from_shareable", return_value=report):
        accepted, reply = assessor.process_child_update(MagicMock(), FLContext())

    assert accepted
    assert StateUpdateReply.from_shareable(reply).model_version == 5
    assert assessor.log_error.call_count >= 2


@pytest.mark.parametrize("assessor_cls,state_cls,module", ASSESSORS)
def test_fill_selection_excludes_used_devices_without_reuse(assessor_cls, state_cls, module):
    assessor = _assessor(assessor_cls, device_selection_size=2, device_reuse=False)
    assessor.available_devices = _devices()
    # used at an older model version: without reuse it must stay excluded anyway
    assessor.used_devices = {"device-1": 5}
    assessor._fill_selection(FLContext())
    assert set(assessor.current_selection) == {"device-2", "device-3"}

    assessor.current_selection = {"device-2": assessor.current_selection["device-2"]}
    assessor._fill_selection(FLContext())
    assert set(assessor.current_selection) == {"device-2"}


def test_fill_selection_with_reuse_excludes_only_current_version_devices():
    assessor = _assessor(num.NumAssessor, device_selection_size=3, device_reuse=True)
    assessor.available_devices = _devices()
    assessor.used_devices = {"device-1": 0, "device-2": 5}
    assessor._fill_selection(FLContext())
    # device-1 was used for the current model version (0) and is excluded;
    # device-2 was used for a different version and becomes usable again
    assert set(assessor.current_selection) == {"device-2", "device-3"}


def test_async_fill_selection_excludes_used_devices_despite_reuse():
    assessor = _assessor(async_num.AsyncNumAssessor, device_selection_size=3, device_reuse=True)
    assessor.available_devices = _devices()
    assessor.used_devices = {"device-1": {"model_version": 5, "selection_version": 5}}
    assessor._fill_selection(FLContext())
    assert set(assessor.current_selection) == {"device-2", "device-3"}


@pytest.mark.parametrize("device_reuse", [True, False])
def test_async_update_frees_used_device_only_with_reuse(device_reuse):
    assessor = _assessor(
        async_num.AsyncNumAssessor, device_selection_size=1, min_hole_to_fill=5, device_reuse=device_reuse
    )
    assessor.process_child_update(StateUpdateReport(0, 0, None, _devices()).to_shareable(), FLContext())
    device_id = next(iter(assessor.current_selection))
    update = ModelUpdate(1, DXO("number", {"value": 8.0, "count": 2}).to_shareable(), {device_id: 5.0})

    assessor.process_child_update(
        StateUpdateReport(1, assessor.current_selection_version, {1: update}, None).to_shareable(), FLContext()
    )

    assert device_id not in assessor.current_selection
    assert (device_id in assessor.used_devices) is not device_reuse


@pytest.mark.parametrize("assessor_cls,state_cls,module", ASSESSORS)
def test_assess_stops_at_max_model_version(assessor_cls, state_cls, module):
    assessor = _assessor(assessor_cls, max_model_version=2)
    assert assessor.assess(FLContext()) == Assessment.CONTINUE
    assessor.current_model_version = 2
    assert assessor.assess(FLContext()) == Assessment.WORKFLOW_DONE
