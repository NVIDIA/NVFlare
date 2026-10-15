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
from nvflare.edge.assessor import Assessment
from nvflare.edge.assessors.buff_device_manager import BuffDeviceManager
from nvflare.edge.assessors.buff_model_manager import BuffModelManager
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


@pytest.mark.parametrize("device_sampling_strategy", ["balanced", "random"])
def test_job_not_stopped_while_device_still_training(device_sampling_strategy):
    """Regression for Greptile P1: the job must not be stopped while a device is still training.

    device_selection_size=2, num_updates_for_model=2, device_reuse=True, two devices.
    After startup both devices are selected for V1. The first device reports its update, which
    makes has_enough_devices_and_clients False because the reporter is now excluded and the second
    device is still in flight. The wait timer must NOT be armed (and assess() must not stop the
    job) while the second device is still training; only after it reports does the model advance
    and the selection refill for V2.
    """
    assessor, model_manager, device_manager, fl_ctx = _setup(
        num_updates_for_model=2, device_sampling_strategy=device_sampling_strategy
    )
    # negative timeout forces the timeout path immediately; the test asserts the in-flight guard
    # prevents it from firing while a device is training
    assessor.device_wait_timeout = -1

    accepted, reply_shareable = assessor.process_child_update(_report(0, devices=_devices(["d1", "d2"])), fl_ctx)
    assert accepted is True
    reply = StateUpdateReply.from_shareable(reply_shareable)
    assert reply.model_version == 1
    assert set(reply.device_selection) == {"d1", "d2"}

    # d1 reports its V1 update; d2 is still in flight (training)
    accepted, reply_shareable = assessor.process_child_update(
        _report(1, model_updates={1: _model_update(1, ["d1"])}), fl_ctx
    )
    assert accepted is True

    # the wait timer must NOT be armed while d2 is still in flight, and assess() must keep the
    # job running rather than stopping it before d2 reports its update
    assert assessor.device_wait_start_time is None
    assert assessor.assess(fl_ctx) != Assessment.WORKFLOW_DONE

    # d2 reports: the 2nd buffered update advances the model to V2, selection refills for V2
    accepted, reply_shareable = assessor.process_child_update(
        _report(1, model_updates={1: _model_update(1, ["d2"])}), fl_ctx
    )
    assert accepted is True
    reply = StateUpdateReply.from_shareable(reply_shareable)
    assert reply.model_version == 2
    assert set(reply.device_selection) == {"d1", "d2"}
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
    assessor.device_wait_timeout = -1

    # both devices join and are selected for V1
    accepted, reply_shareable = assessor.process_child_update(_report(0, devices=_devices(["d1", "d2"])), fl_ctx)
    assert accepted is True
    reply = StateUpdateReply.from_shareable(reply_shareable)
    assert set(reply.device_selection) == {"d1", "d2"}

    # d1 reports while d2 is still in flight: timer must NOT be armed (d2 is still in current selection)
    accepted, reply_shareable = assessor.process_child_update(
        _report(1, model_updates={1: _model_update(1, ["d1"])}), fl_ctx
    )
    assert accepted is True
    assert assessor.device_wait_start_time is None

    # d2 reports -> pool exhausted (reuse disabled, neither device can be re-dispatched)
    accepted, reply_shareable = assessor.process_child_update(
        _report(1, model_updates={1: _model_update(1, ["d2"])}), fl_ctx
    )
    assert accepted is True

    # all devices finished and none can be re-dispatched; the current selection is empty, so the
    # wait timeout must stop the job instead of hanging forever
    assert assessor.device_manager.get_inflight_devices(fl_ctx) == set()
    assert assessor.assess(fl_ctx) == Assessment.WORKFLOW_DONE
