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


class BuffDeviceManager(DeviceManager):
    def __init__(
        self,
        device_selection_size: int,
        min_hole_to_fill: int = 1,
        device_reuse: bool = True,
        const_selection: bool = False,
    ):
        """Initialize the BuffDeviceManager.
        BuffDeviceManager is responsible for managing the selection of devices for model training.
        It maintains a list of available devices, tracks the current selection, and refills the selection as needed.
        The device_selection_size determines how many "concurrent" devices can be selected for the training session.
        The min_hole_to_fill determines how many empty slots should be created before refilling.
            - An empty slot is created when any device reports its update back.
            - To fill a slot, a new device is selected from the available device pool.
        The device_reuse flag indicates whether devices can be reused across different model versions, if False, we will always select new devices when filling holes.
        The const_selection flag indicates whether the same devices should be selected across different model versions, if True, we will always select the same concurrent devices.
        Args:
            device_selection_size (int): Number of devices to select for each model update round.
            min_hole_to_fill (int): Minimum number of empty slots in device selection before refilling. Defaults to 1 - once received an update, immediately sample a new device and send the current task to it.
            device_reuse (bool): Whether to allow reusing devices across different model versions. Defaults to True.
            const_selection (bool): Whether to use constant device selection across rounds. Defaults to False.
        """
        super().__init__()
        self.device_selection_size = device_selection_size
        self.min_hole_to_fill = min_hole_to_fill
        self.device_reuse = device_reuse
        self.const_selection = const_selection
        # also keep track of the current selection version and used devices
        self.current_selection_version = 0
        self.used_devices = {}

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
            if self.const_selection:
                # if const_selection is True, we will always select the same devices
                usable_devices = set(self.available_devices.keys())
                self.log_info(fl_ctx, "constant selection enabled, use the same original set.")
            else:
                if not self.device_reuse:
                    # remove all used devices from available devices
                    usable_devices = set(self.available_devices.keys()) - set(self.used_devices.keys())
                else:
                    # remove only the devices that are associated with the current model version
                    usable_devices = set(self.available_devices.keys()) - set(
                        k for k, v in self.used_devices.items() if v == current_model_version
                    )

            if usable_devices:
                for _ in range(num_holes):
                    device_id = random.choice(list(usable_devices))
                    usable_devices.remove(device_id)
                    self.current_selection[device_id] = self.current_selection_version
                    self.used_devices[device_id] = current_model_version
                    if not usable_devices:
                        break
        self.log_info(
            fl_ctx,
            f"current selection: V{self.current_selection_version}; {dict(sorted(self.current_selection.items()))}",
        )

    def remove_devices_from_selection(self, devices: Set[str], fl_ctx) -> None:
        for device_id in devices:
            self.current_selection.pop(device_id, None)

    def has_enough_devices(self, fl_ctx) -> bool:
        return len(self.available_devices) >= self.device_selection_size

    def should_fill_selection(self, fl_ctx) -> bool:
        num_holes = self.device_selection_size - len(self.current_selection)
        return num_holes >= self.min_hole_to_fill
