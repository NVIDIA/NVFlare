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

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.edge.aggregators.model_update_dxo import ModelUpdateDXOAggregator
from nvflare.edge.assessors.buff_device_manager import BuffDeviceManager
from nvflare.edge.assessors.buff_model_manager import BuffModelManager, _ModelState
from nvflare.edge.assessors.model_update import ModelUpdateAssessor
from nvflare.edge.mud import Device, ModelUpdate, StateUpdateReport


def _manager(num_updates=2, history=None):
    manager = BuffModelManager(num_updates, max_model_history=history)
    for method in ("log_debug", "log_info", "log_warning", "log_error", "fire_event"):
        setattr(manager, method, MagicMock())
    ctx = FLContext()
    manager.initialize_model(DXO(DataKind.WEIGHTS, {"w": [0.0]}), ctx)
    manager.generate_new_model(ctx)
    return manager, ctx


def _update(version, devices, value):
    return ModelUpdate(
        version,
        DXO(DataKind.WEIGHT_DIFF, {"dict": {"w": [value]}, "count": len(devices)}).to_shareable(),
        {device: 1.0 for device in devices},
    )


def test_prune_keeps_current_model_without_assignments():
    manager, ctx = _manager()
    manager.prune_model_versions(set(), ctx)
    assert set(manager.updates) == {1}


@pytest.mark.parametrize("history", [None, 1])
def test_prune_keeps_pending_updates_until_consumed(history):
    manager, ctx = _manager(history=history)
    manager.process_updates({1: _update(1, ["A"], 10.0)}, ctx)
    # Exercise retention even when the version is outside the admission window.
    manager.current_model_version = 2
    manager.updates[2] = _ModelState(ModelUpdateDXOAggregator())

    manager.prune_model_versions(set(), ctx)

    assert set(manager.updates) == {1, 2}
    assert manager.num_updates_counter == 1
    manager.process_updates({2: _update(2, ["B"], 2.0)}, ctx)
    assert manager.current_model.data["w"] == [6.0]
    assert manager.num_updates_counter == 0

    manager.prune_model_versions(set(), ctx)
    assert set(manager.updates) == {3}


@pytest.mark.parametrize("history,active,expected", [(None, {1}, {1, 3}), (2, {1, 2}, {2, 3})])
def test_prune_removes_only_unused_or_expired_empty_versions(history, active, expected):
    manager, ctx = _manager(history=history)
    manager.current_model_version = 3
    manager.updates.update({2: _ModelState(ModelUpdateDXOAggregator()), 3: _ModelState(ModelUpdateDXOAggregator())})

    manager.prune_model_versions(active, ctx)

    assert set(manager.updates) == expected


def test_history_limit_rejects_new_stale_updates_without_counting_them():
    manager, ctx = _manager(num_updates=3, history=1)
    manager.process_updates({1: _update(1, ["A"], 10.0)}, ctx)
    manager.current_model_version = 2
    manager.updates[2] = _ModelState(ModelUpdateDXOAggregator())

    manager.process_updates({1: _update(1, ["B"], 100.0)}, ctx)
    manager.prune_model_versions(set(), ctx)

    assert manager.num_updates_counter == 1
    assert manager.updates[1].aggregator.count == 1
    assert manager.updates[1].aggregator.dict["w"][0] == 10.0


def test_buffer_threshold_counts_aggregated_contributions():
    manager, ctx = _manager(num_updates=3)
    manager.process_updates({1: _update(1, ["A", "B"], 6.0)}, ctx)
    manager.prune_model_versions(set(), ctx)
    assert manager.num_updates_counter == 2

    manager.process_updates({1: _update(1, ["C"], 3.0)}, ctx)

    assert manager.current_model_version == 2
    assert manager.current_model.data["w"] == [3.0]
    assert manager.num_updates_counter == 0


@pytest.mark.parametrize("strategy", ["balanced", "random"])
def test_late_update_survives_assessor_refill(strategy):
    manager, ctx = _manager()
    device_manager = BuffDeviceManager(3, min_hole_to_fill=3, device_sampling_strategy=strategy)
    assessor = ModelUpdateAssessor("persistor", "manager", "devices", max_model_version=5)
    assessor.model_manager = manager
    assessor.device_manager = device_manager
    for component in (device_manager, assessor):
        for method in ("log_debug", "log_info", "log_warning", "log_error"):
            setattr(component, method, MagicMock())
    devices = {device: Device(device, "lcp", 0.0) for device in ("A", "B", "C")}
    assessor.process_child_update(StateUpdateReport(0, 0, None, devices).to_shareable(), ctx)

    # A and B advance the model; C's V1 result arrives after V2 was generated.
    for device, version, value in [("A", 1, 1.0), ("B", 1, 1.0), ("C", 1, 10.0), ("A", 2, 2.0)]:
        assessor.process_child_update(
            StateUpdateReport(
                manager.current_model_version,
                device_manager.current_selection_version,
                {version: _update(version, [device], value)},
                None,
            ).to_shareable(),
            ctx,
        )

    assert manager.current_model_version == 3
    assert manager.current_model.data["w"] == [7.0]
    assert manager.num_updates_counter == 0
