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

import time
from unittest.mock import MagicMock

import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.edge.assessor import Assessment
from nvflare.edge.assessors.buff_device_manager import BuffDeviceManager
from nvflare.edge.assessors.buff_model_manager import BuffModelManager
from nvflare.edge.assessors.device_manager import DeviceManager
from nvflare.edge.assessors.model_update import ModelUpdateAssessor
from nvflare.edge.mud import Device, ModelUpdate, StateUpdateReply, StateUpdateReport


def _setup(
    num_updates_for_model: int = 3,
    device_sampling_strategy: str = "balanced",
    device_reuse: bool = True,
):
    """Build a ModelUpdateAssessor wired with a BuffModelManager and a BuffDeviceManager."""
    assessor = ModelUpdateAssessor(
        persistor_id="persistor",
        model_manager_id="model_manager",
        device_manager_id="device_manager",
        max_model_version=100,
    )
    model_manager = BuffModelManager(num_updates_for_model=num_updates_for_model)
    device_manager = BuffDeviceManager(
        device_selection_size=2,
        initial_min_client_num=1,
        min_hole_to_fill=1,
        device_reuse=device_reuse,
        device_sampling_strategy=device_sampling_strategy,
    )
    assessor.model_manager = model_manager
    assessor.device_manager = device_manager
    for comp in (assessor, model_manager, device_manager):
        comp.log_debug = MagicMock()
        comp.log_info = MagicMock()
        comp.log_error = MagicMock()
        comp.log_warning = MagicMock()
    fl_ctx = FLContext()
    model_manager.initialize_model(DXO(data_kind=DataKind.WEIGHTS, data={"w": [0.0]}), fl_ctx)
    return assessor, model_manager, device_manager, fl_ctx


def _devices(ids, client="site-1"):
    return {d: Device(d, client, float(i + 1)) for i, d in enumerate(ids)}


def _report(current_model_version, model_updates=None, devices=None):
    return StateUpdateReport(
        current_model_version=current_model_version,
        current_device_selection_version=0,
        model_updates=model_updates,
        available_devices=devices,
    ).to_shareable()


def _model_update(model_version, device_ids, value=1.0):
    """A WEIGHT_DIFF update contributed by the given devices for the given model version."""
    dxo = DXO(
        data_kind=DataKind.WEIGHT_DIFF,
        data={"dict": {"w": [float(value)]}, "count": len(device_ids)},
    )
    return ModelUpdate(model_version=model_version, update=dxo.to_shareable(), devices={d: 1.0 for d in device_ids})


@pytest.mark.parametrize("device_sampling_strategy", ["balanced", "random"])
def test_batched_reports_allow_partial_refill_and_do_not_stall(device_sampling_strategy):
    """End-to-end regression for the P1 liveness issue found in review of PR #5423.

    3 devices, device_selection_size=2, num_updates_for_model=3, device_reuse=True:
    A and B are selected for model version 1 and both report their updates in ONE batch.
    The model stays at V1 (2 of 3 buffered updates), both selection slots become empty, and
    the third device C is the only eligible one. Requiring 2 eligible devices for the 2 holes
    would leave C undispatched forever (the 3rd update never arrives -> indefinite stall).
    A partial refill must dispatch C, and C's update must advance the model to V2 and refill
    the selection for the new model version.
    """
    assessor, model_manager, device_manager, fl_ctx = _setup(device_sampling_strategy=device_sampling_strategy)

    # all three devices report availability: startup fill selects 2 of them for model version 1
    accepted, reply_shareable = assessor.process_child_update(_report(0, devices=_devices(["A", "B", "C"])), fl_ctx)
    assert accepted is True
    reply = StateUpdateReply.from_shareable(reply_shareable)
    assert reply.model_version == 1
    selected = set(reply.device_selection)
    assert len(selected) == 2
    third_device = ({"A", "B", "C"} - selected).pop()

    # both selected devices report in one batch: model stays at V1 (2 of 3 updates buffered),
    # both selection slots become empty and only the third device remains eligible
    accepted, reply_shareable = assessor.process_child_update(
        _report(1, model_updates={1: _model_update(1, selected)}), fl_ctx
    )
    assert accepted is True
    reply = StateUpdateReply.from_shareable(reply_shareable)
    assert reply.model_version == 1
    # the only eligible device must be dispatched (partial refill) instead of stalling
    assert set(reply.device_selection) == {third_device}

    # the third device reports: the 3rd buffered update advances the model to V2
    accepted, reply_shareable = assessor.process_child_update(
        _report(1, model_updates={1: _model_update(1, [third_device])}), fl_ctx
    )
    assert accepted is True
    reply = StateUpdateReply.from_shareable(reply_shareable)
    # the update of the partially-refilled device advanced the model
    assert reply.model_version == 2
    # the device manager's model version was re-synchronized after aggregation advanced it,
    # so all devices are eligible again and the selection is refilled for V2
    assert device_manager.current_model_version == 2
    assert len(reply.device_selection) == 2
    assert all(v == 2 for v in reply.device_selection.values())


def test_device_wait_timeout_still_stops_job_when_no_device_in_flight():
    """The legitimate device wait timeout must still stop the job when no device is in flight."""
    assessor, model_manager, device_manager, fl_ctx = _setup(num_updates_for_model=2)
    assessor.device_wait_timeout = -1

    # only one device joins and startup cannot fill the selection; no device is in flight, so the
    # wait timer is armed and assess() stops the job
    accepted, reply_shareable = assessor.process_child_update(_report(0, devices=_devices(["d1"])), fl_ctx)
    assert accepted is True
    assert assessor.device_wait_start_time is not None
    assert assessor.assess(fl_ctx) == Assessment.WORKFLOW_DONE


def test_device_wait_timeout_stops_job_when_pool_exhausted_with_reuse_disabled():
    """Regression for Greptile follow-up P1: an exhausted pool must still stop the job.

    With device_reuse=False, get_used_devices keeps already-finished devices, so a naive in-flight
    check based on used_devices would never let the wait timeout fire and the job would hang forever.
    The in-flight signal must be the current selection (unfinished work); once every device has
    reported and none can be re-dispatched, the configured wait timeout must stop the job.
    """
    assessor, model_manager, device_manager, fl_ctx = _setup(num_updates_for_model=2, device_reuse=False)
    # a positive timeout: with -1 the armed timer would fire on the very next report before the
    # in-flight device's update is processed, which is not what this scenario exercises
    assessor.device_wait_timeout = 10

    # both devices join and are selected for V1
    accepted, reply_shareable = assessor.process_child_update(_report(0, devices=_devices(["d1", "d2"])), fl_ctx)
    assert accepted is True
    reply = StateUpdateReply.from_shareable(reply_shareable)
    assert set(reply.device_selection) == {"d1", "d2"}

    # d1 reports while d2 is still in flight: a refill is attempted but no device is eligible,
    # so the wait timer is armed (original device_wait_timeout behavior)
    accepted, reply_shareable = assessor.process_child_update(
        _report(1, model_updates={1: _model_update(1, ["d1"])}), fl_ctx
    )
    assert accepted is True
    assert assessor.device_wait_start_time is not None

    # d2 reports -> pool exhausted (reuse disabled, neither device can be re-dispatched)
    accepted, reply_shareable = assessor.process_child_update(
        _report(1, model_updates={1: _model_update(1, ["d2"])}), fl_ctx
    )
    assert accepted is True

    # all devices finished and none can be re-dispatched; the current selection is empty, so the
    # wait timeout must stop the job instead of hanging forever. Simulate the wait having run past
    # its timeout (no sleeping in tests).
    assert device_manager.current_selection == {}
    assessor.device_wait_start_time = time.time() - 100
    assert assessor.assess(fl_ctx) == Assessment.WORKFLOW_DONE


class _CustomDeviceManager(DeviceManager):
    """A minimal custom device manager (not BuffDeviceManager) used to verify that the assessor's
    wait-timeout behavior is unchanged for third-party managers."""

    def __init__(self, device_selection_size: int = 2, device_reuse: bool = True):
        super().__init__()
        self.device_selection_size = device_selection_size
        self.device_reuse = device_reuse
        self.current_selection_version = 0
        self.current_model_version = 0
        self.device_client_map = {}

    def update_available_devices(self, devices, fl_ctx):
        self.available_devices.update(devices)
        for device_id in devices:
            self.device_client_map[device_id] = "client"

    def fill_selection(self, current_model_version, fl_ctx):
        self.current_model_version = current_model_version
        self.current_selection_version += 1
        for device_id in self.available_devices:
            if len(self.current_selection) >= self.device_selection_size:
                break
            if device_id not in self.current_selection:
                self.current_selection[device_id] = current_model_version
                self.used_devices[device_id] = current_model_version

    def remove_devices_from_selection(self, devices, fl_ctx):
        for device_id in devices:
            self.current_selection.pop(device_id, None)

    def remove_devices_from_used(self, devices, fl_ctx):
        for device_id in devices:
            self.used_devices.pop(device_id, None)

    def should_fill_selection(self, fl_ctx):
        return self.device_selection_size - len(self.current_selection) >= 1

    def has_enough_devices_and_clients(self, fl_ctx):
        usable = set(self.available_devices.keys()) - set(self.used_devices.keys()) - set(self.current_selection.keys())
        num_holes = self.device_selection_size - len(self.current_selection)
        return len(usable) >= num_holes

    def get_active_model_versions(self, fl_ctx):
        return set(self.current_selection.values())


def test_custom_device_manager_retains_finite_timeout_behavior():
    """Regression requested by the maintainer: a custom DeviceManager keeps main's finite-timeout behavior.

    Abandoned-assignment detection is explicitly out of scope for this PR, so the wait timeout keeps
    its original semantics: it is armed when a refill is attempted with too few eligible devices and
    it stops the job once it expires - for a custom device manager exactly as for BuffDeviceManager.
    """
    assessor, model_manager, device_manager, fl_ctx = _setup(num_updates_for_model=2)
    assessor.device_manager = _CustomDeviceManager(device_selection_size=2)
    assessor.device_wait_timeout = 10

    # only one device joins: the refill cannot be filled, so the wait timer is armed
    accepted, reply_shareable = assessor.process_child_update(_report(0, devices=_devices(["d1"])), fl_ctx)
    assert accepted is True
    assert assessor.device_wait_start_time is not None

    # once the wait timeout has elapsed the job is stopped - the custom manager is not exempted
    assessor.device_wait_start_time = time.time() - 100
    assert assessor.assess(fl_ctx) == Assessment.WORKFLOW_DONE
