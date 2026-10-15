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

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Set

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext


class DeviceManager(FLComponent, ABC):
    """Abstract base class for device managers in federated learning.

    This class defines the interface that all device managers must implement.
    Device managers are responsible for handling the selection of devices for model training.
    """

    def __init__(self):
        FLComponent.__init__(self)
        """Initialize the DeviceManager.
        DeviceManager keeps track of two dicts:
        - current_selection for devices of current task distribution: our default assumption is device_id => model_id
        - available_devices containing all devices that are available for selection
        - used_devices dict kept for record keeping, containing all devices that have participated
        """
        self.current_selection = {}
        self.available_devices = {}
        self.used_devices = {}
        self.selection_assignment_time = {}

    @abstractmethod
    def update_available_devices(self, devices: Dict, fl_ctx: FLContext) -> None:
        """Update the list of available devices.
        modify self.available_devices with devices input

        Args:
            devices (Dict): Dictionary of available devices to add
            fl_ctx: FLContext object

            Returns: none
        """
        pass

    @abstractmethod
    def fill_selection(self, fl_ctx: FLContext) -> None:
        """Fill the device selection sampled from available devices.
        update self.current_selection

        Args:
            current_model_version (int): Current version of the model
            fl_ctx: FLContext object

            Returns: none
        """
        pass

    @abstractmethod
    def remove_devices_from_selection(self, devices: Any, fl_ctx: FLContext) -> None:
        """Remove devices from the current selection.
        update self.current_selection

        Args:
            devices: Set of devices to remove
            fl_ctx: FLContext object

            Returns: none
        """
        pass

    @abstractmethod
    def remove_devices_from_used(self, devices: Any, fl_ctx: FLContext) -> None:
        """Remove devices from the used device set.
        update self.used_devices

        Args:
            devices: Set of devices to remove
            fl_ctx: FLContext object

            Returns: none
        """
        pass

    @abstractmethod
    def should_fill_selection(self, fl_ctx: FLContext) -> bool:
        """Determine if it is time to fill the device selection.

        Args:
            fl_ctx: FLContext object

        Returns:
            bool: True if it's time to fill the selection, False otherwise
        """
        pass

    @abstractmethod
    def has_enough_devices_and_clients(self, fl_ctx: FLContext) -> bool:
        """Check if there are enough devices and clients available to start task distribution.

        Args:
            fl_ctx: FLContext object

        Returns:
            bool: True if there are enough devices and clients to start task distribution, False otherwise
        """
        pass

    @abstractmethod
    def get_active_model_versions(self, fl_ctx: FLContext) -> Set[int]:
        """Get the active model versions that is associated with the current selection.

        Args:
            fl_ctx: FLContext object

        Returns:
            Set of active model versions
        """
        pass

    def get_selection(self, fl_ctx: FLContext) -> Any:
        """Get the current device selection.

        Args:
            fl_ctx: FLContext object

        Returns:
            Current device selection
        """
        return self.current_selection

    def get_inflight_devices(self, fl_ctx: FLContext) -> Set[str]:
        """Get the devices currently selected (dispatched) but not yet reported back.

        These are the devices still in flight / training. The wait-timeout logic uses this to
        distinguish "a device is still training" from "no devices are available to join". The
        default implementation derives it from ``current_selection``, which every DeviceManager
        maintains; subclasses (e.g. custom device managers) inherit this without overriding.
        """
        selection = self.current_selection
        if isinstance(selection, dict):
            return set(selection.keys())
        return set(selection)

    def tracks_assignment_time(self, fl_ctx: FLContext) -> bool:
        """Whether this manager records assignment timestamps for selected devices.

        Only managers returning True get the assignment-expiry bound on the wait-timeout exemption
        (so a device that never reports can still trip the timeout). Managers that return False keep
        the previous behavior: an in-flight device is always treated as still training, and the wait
        timer is not counting down while a device is in flight.
        """
        return False

    def get_usable_devices(self, fl_ctx: FLContext) -> Set[str]:
        """Devices eligible to be selected right now (default: available minus used).

        Subclasses with richer eligibility rules (e.g. contribution-based exclusion) should override
        this so that status logging and timeout diagnostics agree with scheduling.
        """
        return set(self.available_devices.keys()) - set(self.used_devices.keys())

    def is_any_inflight_device_recent(self, fl_ctx: FLContext, deadline: float) -> bool:
        """True if any currently-selected (in-flight) device was assigned within ``deadline`` seconds.

        The wait-timeout logic uses this to suppress the timeout only while an in-flight device is
        reasonably expected to report. A device that was selected but never reports (e.g.
        disconnected) eventually exceeds ``deadline`` and no longer suppresses the timeout, so the
        job can still be stopped instead of hanging forever (Copilot follow-up on the in-flight guard).

        Device managers that do not record assignment times (e.g. an existing custom ``DeviceManager``
        subclass) keep the previous behavior: their in-flight devices are always treated as still
        training, so the timeout cannot stop the job mid-training (Greptile follow-up: custom jobs
        stop during training).
        """
        inflight = self.get_inflight_devices(fl_ctx)
        if not inflight:
            return False
        tracked = [self.selection_assignment_time[d] for d in inflight if d in self.selection_assignment_time]
        if not tracked:
            # no assignment timestamps recorded for the in-flight devices - preserve the previous
            # behavior instead of assuming those devices are stuck.
            return True
        now = time.time()
        return any(now - t <= deadline for t in tracked)

    def get_available_devices(self, fl_ctx: FLContext) -> Set[str]:
        """Get the available devices.

        Args:
            fl_ctx: FLContext object

        Returns:
            Set of available devices
        """
        return self.available_devices

    def get_used_devices(self, fl_ctx: FLContext) -> Set[str]:
        """Get the used devices.

        Args:
            fl_ctx: FLContext object

        Returns:
            Set of used devices
        """
        return self.used_devices
