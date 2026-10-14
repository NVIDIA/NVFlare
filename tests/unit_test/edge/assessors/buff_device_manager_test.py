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
from nvflare.edge.mud import BaseState, Device


def _manager():
    manager = BuffDeviceManager(device_selection_size=2)
    manager.log_debug = MagicMock()
    manager.log_info = MagicMock()
    manager.log_warning = MagicMock()
    manager.update_available_devices(
        {"A": Device("A", "site-1", 1.0), "B": Device("B", "site-1", 1.0)},
        FLContext(),
    )
    return manager


def _state(manager):
    selection = dict(manager.get_selection(FLContext()))
    return BaseState(0, None, manager.current_selection_version, selection)


def test_reselected_device_is_accepted_at_the_same_model_version():
    manager = _manager()
    manager.fill_selection(1, FLContext())
    selected, first_id = _state(manager).is_device_selected("A", 0)
    assert selected

    # A reports while the model is still version 1, so its slot is filled again, with A
    manager.remove_devices_from_selection({"A"}, FLContext())
    manager.remove_devices_from_used({"A"}, FLContext())
    manager.fill_selection(1, FLContext())

    selected, second_id = _state(manager).is_device_selected("A", first_id)
    assert selected
    assert second_id != first_id


def test_active_model_versions_still_follow_the_model_version():
    manager = _manager()
    manager.fill_selection(1, FLContext())
    manager.remove_devices_from_selection({"A"}, FLContext())
    manager.remove_devices_from_used({"A"}, FLContext())
    manager.fill_selection(2, FLContext())

    assert manager.get_active_model_versions(FLContext()) == {1, 2}
