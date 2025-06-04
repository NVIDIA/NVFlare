import random
from typing import Dict, Set

from nvflare.apis.fl_component import FLComponent


class DeviceManager(FLComponent):
    def __init__(self, device_selection_size: int, min_hole_to_fill: int, device_reuse: bool, const_selection: bool):
        FLComponent.__init__(self)
        self.current_selection = {}
        self.current_selection_version = 0
        self.available_devices = {}
        self.used_devices = {}
        self.device_selection_size = device_selection_size
        self.min_hole_to_fill = min_hole_to_fill
        self.device_reuse = device_reuse
        self.const_selection = const_selection

    def update_available_devices(self, devices: Dict):
        self.available_devices.update(devices)

    def fill_selection(self, current_model_version: int, fl_ctx) -> None:
        num_holes = self.device_selection_size - len(self.current_selection)
        self.log_info(fl_ctx, f"filling {num_holes} holes in selection list")
        if num_holes > 0:
            self.current_selection_version += 1
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

    def repeat_selection(self, current_model_version: int, fl_ctx) -> None:
        num_holes = self.device_selection_size - len(self.current_selection)
        self.log_info(fl_ctx, f"filling {num_holes} holes in selection list")
        if num_holes > 0:
            self.current_selection_version += 1
            # fill the holes with the same devices
            usable_devices = set(self.used_devices.keys())
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

    def remove_devices_from_selection(self, devices: Set[str]) -> None:
        for device_id in devices:
            self.current_selection.pop(device_id, None)

    def get_selection(self) -> Dict:
        return self.current_selection
