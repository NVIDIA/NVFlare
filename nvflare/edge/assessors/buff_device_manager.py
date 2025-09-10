# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import random
from typing import Dict, Set

from nvflare.edge.assessors.device_manager import DeviceManager
from nvflare.fuel.utils.validation_utils import check_positive_int


class BuffDeviceManager(DeviceManager):
    def __init__(
        self,
        device_selection_size: int,
        min_hole_to_fill: int = 1,
        device_reuse: bool = True,
    ):
        """Initialize the BuffDeviceManager.
        BuffDeviceManager is responsible for managing the selection of devices for model training.
        It maintains a list of available devices, tracks the current selection, and refills the selection as needed.
        The device_selection_size determines how many "concurrent" devices can be selected for the training session.
        The min_hole_to_fill determines how many empty slots should be created before refilling.
            - An empty slot is created when any device reports its update back.
            - To fill a slot, a new device is selected from the available device pool.
        The device_reuse flag indicates whether devices can be reused across different model versions, if False, we will always select new devices when filling holes.
        Args:
            device_selection_size (int): Number of devices to select for each model update round.
            min_hole_to_fill (int): Minimum number of empty slots in device selection before refilling. Defaults to 1 - once received an update, immediately sample a new device and send the current task to it.
            device_reuse (bool): Whether to allow reusing devices across different model versions. Defaults to True.
        """
        super().__init__()
        check_positive_int("device_selection_size", device_selection_size)
        check_positive_int("min_hole_to_fill", min_hole_to_fill)

        self.device_selection_size = device_selection_size
        self.min_hole_to_fill = min_hole_to_fill
        self.device_reuse = device_reuse
        # also keep track of the current selection version and used devices
        self.current_selection_version = 0

    def update_available_devices(self, devices: Dict, fl_ctx) -> None:
        self.available_devices.update(devices)
        self.log_debug(
            fl_ctx,
            f"assessor got reported {len(devices)} available devices from child. "
            f"total num available devices: {len(self.available_devices)}",
        )

    def fill_selection(self, current_model_version: int, fl_ctx) -> None:
        num_holes = self.device_selection_size - len(self.current_selection)
        self.log_info(fl_ctx, f"filling {num_holes} holes in selection list")
        if num_holes > 0:
            self.current_selection_version += 1
            # remove all used devices from available devices
            usable_devices = set(self.available_devices.keys()) - set(self.used_devices.keys())

            if usable_devices:
                for _ in range(num_holes):
                    device_id = random.choice(list(usable_devices))
                    usable_devices.remove(device_id)
                    # current_selection keeps track of devices selected for a particular model version
                    self.current_selection[device_id] = current_model_version
                    self.used_devices[device_id] = {
                        "model_version": current_model_version,
                        "selection_version": self.current_selection_version,
                    }
                    if not usable_devices:
                        break
        self.log_info(
            fl_ctx,
            f"current selection with {len(self.current_selection)} items: V{self.current_selection_version}; {dict(sorted(self.current_selection.items()))}",
        )
        if len(self.current_selection) < self.device_selection_size:
            self.log_warning(
                fl_ctx,
                f"current selection has only {len(self.current_selection)} devices, which is less than the expected {self.device_selection_size} devices. Please check the configuration to make sure this is expected.",
            )

    def remove_devices_from_selection(self, devices: Set[str], fl_ctx) -> None:
        for device_id in devices:
            self.current_selection.pop(device_id, None)

    def remove_devices_from_used(self, devices: Set[str], fl_ctx) -> None:
        for device_id in devices:
            self.used_devices.pop(device_id, None)

    def has_enough_devices(self, fl_ctx) -> bool:
        num_holes = self.device_selection_size - len(self.current_selection)
        usable_devices = set(self.available_devices.keys()) - set(self.used_devices.keys())
        num_usable_devices = len(usable_devices)
        return num_usable_devices >= num_holes

    def should_fill_selection(self, fl_ctx) -> bool:
        num_holes = self.device_selection_size - len(self.current_selection)
        return num_holes >= self.min_hole_to_fill

    def get_active_model_versions(self, fl_ctx) -> Set[int]:
        return set(self.current_selection.values())
