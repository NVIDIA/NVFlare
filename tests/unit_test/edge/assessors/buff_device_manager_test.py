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

from nvflare.apis.fl_context import FLContext
from nvflare.edge.assessors.buff_device_manager import BuffDeviceManager
from nvflare.edge.assessors.device_manager import DeviceManager
from nvflare.edge.mud import Device


def _manager(**kwargs):
    params = {
        "device_selection_size": 2,
        "initial_min_client_num": 1,
        "min_hole_to_fill": 1,
        "device_reuse": True,
        "device_sampling_strategy": "balanced",
    }
    params.update(kwargs)
    mgr = BuffDeviceManager(**params)
    mgr.log_debug = MagicMock()
    mgr.log_info = MagicMock()
    mgr.log_error = MagicMock()
    mgr.log_warning = MagicMock()
    return mgr


def _devices(ids, client="site-1"):
    return {d: Device(d, client, float(i + 1)) for i, d in enumerate(ids)}


def _report_contribution(mgr, device_ids, fl_ctx):
    """Simulate a device reporting its update back for the model version it was selected for."""
    mgr.remove_devices_from_selection(set(device_ids), fl_ctx)
    if mgr.device_reuse:
        mgr.remove_devices_from_used(set(device_ids), fl_ctx)


def test_remove_devices_from_selection_records_contribution():
    mgr = _manager()
    fl_ctx = FLContext()
    mgr.available_devices = _devices(["d1", "d2"])
    mgr.update_available_devices(mgr.available_devices, fl_ctx)
    mgr.fill_selection(current_model_version=3, fl_ctx=fl_ctx)
    assert set(mgr.current_selection) == {"d1", "d2"}

    mgr.remove_devices_from_selection({"d1"}, fl_ctx)
    assert "d1" not in mgr.current_selection
    # the version the device was selected for is recorded as its last contribution
    assert mgr.device_last_contributed_version.get("d1") == 3
    # d2 is still in-flight and must not be recorded yet
    assert "d2" not in mgr.device_last_contributed_version


def test_fill_selection_excludes_devices_that_contributed_to_current_model_version():
    # Regression test for issue #5347: a device that already contributed to the current model
    # version must not be re-selected for that same version, otherwise it would get the same
    # selection id and BaseState.is_device_selected would reject it forever (RETRY stall).
    mgr = _manager(device_selection_size=2, min_hole_to_fill=1, device_reuse=True)
    fl_ctx = FLContext()
    mgr.available_devices = _devices(["d1", "d2"])
    mgr.update_available_devices(mgr.available_devices, fl_ctx)

    # first round: both devices selected for model version 1
    mgr.fill_selection(current_model_version=1, fl_ctx=fl_ctx)
    assert set(mgr.current_selection) == {"d1", "d2"}
    assert all(v == 1 for v in mgr.current_selection.values())

    # d1 reports back for model version 1
    _report_contribution(mgr, ["d1"], fl_ctx)

    # model version has NOT advanced yet (only 1 of N updates received).
    # a naive implementation would re-select d1 for the same version 1 -> indefinite RETRY stall.
    assert mgr.should_fill_selection(fl_ctx)
    mgr.fill_selection(current_model_version=1, fl_ctx=fl_ctx)

    # d1 must NOT be re-selected for the version it already contributed to
    assert "d1" not in mgr.current_selection
    # only d2 remains (still in-flight for version 1)
    assert set(mgr.current_selection) == {"d2"}
    assert mgr.device_last_contributed_version.get("d1") == 1


def test_fill_selection_reselects_after_model_advances():
    mgr = _manager(device_selection_size=2, min_hole_to_fill=1, device_reuse=True)
    fl_ctx = FLContext()
    mgr.available_devices = _devices(["d1", "d2"])
    mgr.update_available_devices(mgr.available_devices, fl_ctx)
    mgr.fill_selection(current_model_version=1, fl_ctx=fl_ctx)

    # both devices report for version 1
    _report_contribution(mgr, ["d1", "d2"], fl_ctx)
    assert mgr.current_selection == {}

    # model version advanced to 2: both devices become eligible again
    mgr.fill_selection(current_model_version=2, fl_ctx=fl_ctx)
    assert set(mgr.current_selection) == {"d1", "d2"}
    assert all(v == 2 for v in mgr.current_selection.values())


def test_fill_selection_excludes_inflight_devices():
    mgr = _manager(device_selection_size=2, min_hole_to_fill=1, device_reuse=True)
    fl_ctx = FLContext()
    mgr.available_devices = _devices(["d1", "d2", "d3"])
    mgr.update_available_devices(mgr.available_devices, fl_ctx)

    mgr.fill_selection(current_model_version=1, fl_ctx=fl_ctx)
    selected_before = set(mgr.current_selection)
    assert len(selected_before) == 2

    # report a device that was actually selected, so a real hole is created and refilled.
    # the sampler may have chosen any two of d1/d2/d3, so pick from the actual selection
    # (otherwise reporting an unselected device would skip the refill path entirely).
    reported = next(iter(selected_before))
    inflight = selected_before - {reported}
    _report_contribution(mgr, [reported], fl_ctx)
    mgr.fill_selection(current_model_version=1, fl_ctx=fl_ctx)

    # the in-flight device (the selected one other than the reported one) is preserved
    assert inflight <= set(mgr.current_selection)
    # the reported device is never re-selected, and exactly one new device fills the hole
    assert reported not in mgr.current_selection
    assert len(mgr.current_selection) == 2


def test_has_enough_devices_consistent_with_fill():
    # has_enough_devices_and_clients must use the same eligibility as fill_selection:
    # d1 is "available" again after reporting, but it is NOT eligible for the current
    # model version, so it must not count as enough to fill a hole.
    mgr = _manager(device_selection_size=2, min_hole_to_fill=1, device_reuse=True)
    fl_ctx = FLContext()
    mgr.available_devices = _devices(["d1", "d2"])
    mgr.update_available_devices(mgr.available_devices, fl_ctx)
    mgr.fill_selection(current_model_version=1, fl_ctx=fl_ctx)

    _report_contribution(mgr, ["d1"], fl_ctx)
    assert mgr.has_enough_devices_and_clients(fl_ctx) is False


def test_fedbuff_no_indefinite_stall_with_reuse():
    # End-to-end style regression for issue #5347 (Option B):
    # 2 devices, reuse=True, model version advances only after 2 updates.
    mgr = _manager(device_selection_size=2, min_hole_to_fill=1, device_reuse=True)
    fl_ctx = FLContext()
    mgr.available_devices = _devices(["d1", "d2"])
    mgr.update_available_devices(mgr.available_devices, fl_ctx)

    model_version = 1
    mgr.fill_selection(current_model_version=model_version, fl_ctx=fl_ctx)
    assert set(mgr.current_selection) == {"d1", "d2"}

    # d1 reports for v1; only 1 update so far -> model version stays 1
    _report_contribution(mgr, ["d1"], fl_ctx)
    assert mgr.should_fill_selection(fl_ctx)
    mgr.fill_selection(current_model_version=model_version, fl_ctx=fl_ctx)
    # d1 must not be stuck in a RETRY loop: it is NOT re-selected for the same version
    assert "d1" not in mgr.current_selection

    # d2 reports for v1 -> 2 updates -> model version advances to 2
    _report_contribution(mgr, ["d2"], fl_ctx)
    model_version = 2
    assert mgr.should_fill_selection(fl_ctx)
    mgr.fill_selection(current_model_version=model_version, fl_ctx=fl_ctx)
    # both devices eligible again for the new model version
    assert set(mgr.current_selection) == {"d1", "d2"}
    assert all(v == 2 for v in mgr.current_selection.values())


def test_has_enough_devices_requires_full_selection_at_startup():
    # At startup (no selection dispatched yet), the initial dispatch must be able to fill the
    # whole selection - a partial initial selection is not allowed.
    mgr = _manager(device_selection_size=2, min_hole_to_fill=1, device_reuse=True)
    fl_ctx = FLContext()
    mgr.available_devices = _devices(["d1"])
    mgr.update_available_devices(mgr.available_devices, fl_ctx)
    assert mgr.current_selection_version == 0
    assert mgr.has_enough_devices_and_clients(fl_ctx) is False


def test_has_enough_devices_allows_partial_refill_after_startup():
    # Follow-up to issue #5347: after startup, a refill must succeed with fewer eligible
    # devices than holes. Otherwise, when one report carries several buffered updates at once
    # and the remaining devices already contributed to the current model version, the only
    # eligible device is never dispatched and the job stalls forever.
    mgr = _manager(device_selection_size=2, min_hole_to_fill=1, device_reuse=True)
    fl_ctx = FLContext()
    mgr.available_devices = _devices(["d1", "d2", "d3"])
    mgr.update_available_devices(mgr.available_devices, fl_ctx)

    # startup: full selection for model version 1
    mgr.fill_selection(current_model_version=1, fl_ctx=fl_ctx)
    assert len(mgr.current_selection) == 2
    assert mgr.current_selection_version == 1
    inflight = set(mgr.current_selection)
    third_device = ({"d1", "d2", "d3"} - inflight).pop()

    # both in-flight devices report in one batch: model stays at v1 (updates still needed),
    # both slots become empty and only the third device is eligible for v1
    _report_contribution(mgr, inflight, fl_ctx)
    assert mgr.current_selection == {}
    num_holes = mgr.device_selection_size
    assert len(mgr._compute_usable_devices()) == 1

    # 1 eligible device for 2 holes is enough for a partial refill
    assert mgr.has_enough_devices_and_clients(fl_ctx) is True
    mgr.fill_selection(current_model_version=1, fl_ctx=fl_ctx)
    assert set(mgr.current_selection) == {third_device}
    assert all(v == 1 for v in mgr.current_selection.values())


class _MinimalDeviceManager(DeviceManager):
    """A custom device manager that does NOT override get_inflight_devices, exercising the
    default implementation inherited from DeviceManager."""

    def update_available_devices(self, devices, fl_ctx):
        pass

    def fill_selection(self, fl_ctx):
        pass

    def remove_devices_from_selection(self, devices, fl_ctx):
        pass

    def remove_devices_from_used(self, devices, fl_ctx):
        pass

    def should_fill_selection(self, fl_ctx):
        return True

    def has_enough_devices_and_clients(self, fl_ctx):
        return True

    def get_active_model_versions(self, fl_ctx):
        return set(self.current_selection.values())


def test_base_get_inflight_devices_default_for_custom_manager():
    # Regression for Greptile follow-up P1: a custom device manager that does not implement
    # get_inflight_devices must still work via the base default (derived from current_selection),
    # otherwise the wait-timeout logic raises AttributeError on a too-few-devices report.
    mgr = _MinimalDeviceManager()
    mgr.current_selection = {"d1": 1, "d2": 1}
    fl_ctx = FLContext()
    assert mgr.get_inflight_devices(fl_ctx) == {"d1", "d2"}
    mgr.current_selection = {}
    assert mgr.get_inflight_devices(fl_ctx) == set()


def test_is_any_inflight_device_recent_preserves_behavior_without_timestamps():
    # Regression for Greptile follow-up P1: a custom DeviceManager that does not record assignment
    # times must keep the previous behavior - its in-flight devices are treated as still training,
    # so the wait timeout cannot stop the job while a device is selected. Only managers that do
    # record timestamps get the assignment-expiry bound.
    mgr = _MinimalDeviceManager()
    mgr.current_selection = {"d1": 1}
    fl_ctx = FLContext()
    # no timestamps recorded -> previous behavior preserved (in-flight == still training)
    assert mgr.is_any_inflight_device_recent(fl_ctx, 10) is True
    # no in-flight devices -> nothing to protect
    mgr.current_selection = {}
    assert mgr.is_any_inflight_device_recent(fl_ctx, 10) is False


def test_tracks_assignment_time_default_and_override():
    fl_ctx = FLContext()
    # a custom DeviceManager does not record assignment times -> it keeps the previous behavior:
    # the wait timer is not counting down while a device is in flight
    assert _MinimalDeviceManager().tracks_assignment_time(fl_ctx) is False
    # BuffDeviceManager records them -> it gets the assignment-expiry bound instead
    buff_mgr = _manager(device_selection_size=2, min_hole_to_fill=1, device_reuse=True)
    assert buff_mgr.tracks_assignment_time(fl_ctx) is True
