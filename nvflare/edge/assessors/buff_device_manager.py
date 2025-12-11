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
from collections import Counter, defaultdict
from typing import Dict, Set

from nvflare.edge.assessors.device_manager import DeviceManager
from nvflare.edge.mud import PropKey
from nvflare.fuel.utils.validation_utils import check_positive_int


class BuffDeviceManager(DeviceManager):
    def __init__(
        self,
        device_selection_size: int,
        initial_min_client_num: int = 1,
        min_hole_to_fill: int = 1,
        device_reuse: bool = True,
        device_sampling_strategy: str = "balanced",
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
            initial_min_client_num (int): Minimum number of clients to have at the beginning. This can be useful for initial model dispatch.
            min_hole_to_fill (int): Minimum number of empty slots in device selection before refilling. Defaults to 1 - once received an update, immediately sample a new device and send the current task to it.
            device_reuse (bool): Whether to allow reusing devices across different model versions. Defaults to True.
            device_sampling_strategy (str): Strategy for sampling devices when filling selection. Defaults to "balanced".
                - "balanced": try to balance the usage of devices across clients.
                - "random": randomly select devices from the available pool.
        """
        super().__init__()
        check_positive_int("device_selection_size", device_selection_size)
        check_positive_int("min_hole_to_fill", min_hole_to_fill)
        check_positive_int("initial_min_client_num", initial_min_client_num)
        if device_sampling_strategy not in ("balanced", "random"):
            raise ValueError(
                f"device_sampling_strategy must be 'balanced' or 'random', got '{device_sampling_strategy}'"
            )
        self.device_selection_size = device_selection_size
        self.initial_min_client_num = initial_min_client_num
        self.min_hole_to_fill = min_hole_to_fill
        self.device_reuse = device_reuse
        self.device_sampling_strategy = device_sampling_strategy
        # also keep track of the current selection version and used devices
        self.current_selection_version = 0
        self.used_devices = {}
        # keep a map of device_id -> client_name
        self.device_client_map = {}

    def _balanced_device_sampling(self, usable_devices: Set[str], num_holes: int) -> Set[str]:
        """Sample devices while balancing across clients.

        Args:
            usable_devices: Set of device IDs that can be selected
            num_holes: Number of devices to sample

        Returns:
            Set of selected device IDs
        """
        if not usable_devices or num_holes <= 0:
            return set()

        # Count devices per client efficiently using Counter
        client_device_counts = Counter(
            self.device_client_map[device_id] for device_id in usable_devices if device_id in self.device_client_map
        )

        # Group devices by client using defaultdict for efficiency
        client_devices = defaultdict(list)
        for device_id in usable_devices:
            if device_id in self.device_client_map:
                client_devices[self.device_client_map[device_id]].append(device_id)

        if not client_device_counts:
            # Fallback to random sampling if no client mapping
            return set(random.sample(list(usable_devices), min(num_holes, len(usable_devices))))

        # Randomize client order for more balanced distribution
        clients_list = list(client_device_counts.items())
        random.shuffle(clients_list)

        selected_devices = set()
        remaining_holes = num_holes

        # First pass: assign minimum possible to each client
        min_per_client = remaining_holes // len(clients_list)
        extra_holes = remaining_holes % len(clients_list)

        for i, (client_name, device_count) in enumerate(clients_list):
            # Calculate how many devices this client should get
            if i < extra_holes:
                target_count = min_per_client + 1
            else:
                target_count = min_per_client

            # Don't exceed what the client has available
            actual_count = min(target_count, device_count)

            if actual_count > 0:
                # Randomly sample from this client's devices
                sampled = random.sample(client_devices[client_name], actual_count)
                selected_devices.update(sampled)
                remaining_holes -= actual_count

                # Remove selected devices from available pool
                client_devices[client_name] = [d for d in client_devices[client_name] if d not in sampled]

        # Second pass: if we still have holes and some clients have remaining devices,
        # distribute remaining holes as evenly as possible with random starting point
        if remaining_holes > 0:
            clients_with_devices = [(name, devices) for name, devices in client_devices.items() if devices]

            if clients_with_devices:
                # Shuffle clients to randomize the round-robin starting point
                random.shuffle(clients_with_devices)

                # Round-robin distribution of remaining holes
                client_idx = 0
                while remaining_holes > 0 and clients_with_devices:
                    client_name, devices = clients_with_devices[client_idx]

                    if devices:
                        # Take one device from this client
                        device_id = random.choice(devices)
                        selected_devices.add(device_id)
                        devices.remove(device_id)
                        remaining_holes -= 1

                        # Remove client if no more devices
                        if not devices:
                            clients_with_devices.pop(client_idx)
                            if clients_with_devices:
                                client_idx = client_idx % len(clients_with_devices)
                        else:
                            client_idx = (client_idx + 1) % len(clients_with_devices)
                    else:
                        clients_with_devices.pop(client_idx)
                        if clients_with_devices:
                            client_idx = client_idx % len(clients_with_devices)

        return selected_devices

    def update_available_devices(self, devices: Dict, fl_ctx) -> None:
        self.available_devices.update(devices)
        self.log_debug(
            fl_ctx,
            f"assessor got reported {len(devices)} available devices from child. "
            f"total num available devices: {len(self.available_devices)}",
        )
        # add new devices to device_client_map
        for device_id, device in devices.items():
            client_name = device.to_dict().get(PropKey.CLIENT_NAME)
            if client_name:
                self.device_client_map[device_id] = client_name

    def fill_selection(self, current_model_version: int, fl_ctx) -> None:
        num_holes = self.device_selection_size - len(self.current_selection)
        self.log_info(fl_ctx, f"filling {num_holes} holes in selection list")
        if num_holes > 0:
            self.current_selection_version += 1
            # remove all used devices from available devices
            usable_devices = set(self.available_devices.keys()) - set(self.used_devices.keys())

            if usable_devices:
                if self.device_sampling_strategy == "balanced":
                    # try to balance the usage of devices across clients
                    selected_devices = self._balanced_device_sampling(usable_devices, num_holes)
                    for device_id in selected_devices:
                        # current_selection keeps track of devices selected for a particular model version
                        self.current_selection[device_id] = current_model_version
                        self.used_devices[device_id] = {
                            "model_version": current_model_version,
                            "selection_version": self.current_selection_version,
                        }
                elif self.device_sampling_strategy == "random":
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
                else:
                    raise ValueError(f"Invalid device sampling strategy: {self.device_sampling_strategy}")
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

    def has_enough_devices_and_clients(self, fl_ctx) -> bool:
        num_holes = self.device_selection_size - len(self.current_selection)
        usable_devices = set(self.available_devices.keys()) - set(self.used_devices.keys())
        num_usable_devices = len(usable_devices)
        if num_usable_devices < num_holes:
            return False

        # Further check if we have enough clients
        unique_clients = set(self.device_client_map.values())
        return len(unique_clients) >= self.initial_min_client_num

    def should_fill_selection(self, fl_ctx) -> bool:
        num_holes = self.device_selection_size - len(self.current_selection)
        return num_holes >= self.min_hole_to_fill

    def get_active_model_versions(self, fl_ctx) -> Set[int]:
        return set(self.current_selection.values())
