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
from typing import Optional

from nvflare.apis.fl_context import FLContext
from nvflare.edge.simulation.simulated_device import DeviceFactory
from nvflare.edge.simulation.simulator import Simulator
from nvflare.edge.widgets.runner import SimulationRunner
from nvflare.fuel.utils.validation_utils import check_number_range, check_positive_int, check_positive_number, check_str


class DeviceRunner(SimulationRunner):

    def __init__(
        self,
        device_factory_id: str,
        num_active_devices=100,
        num_workers=10,
        num_devices=10000,
        cycle_duration: float = 30,
        device_reuse_rate: float = 0,
    ):
        """Constructor of DeviceRunner.
        A DeviceRunner is a component to be directly installed in CJs that simulates edge devices.
        No web nodes are needed for the simulated devices to communicate with Flare.

        Args:
            device_factory_id:
            num_active_devices:
            num_workers:
            num_devices:
            cycle_duration:
            device_reuse_rate:
        """
        SimulationRunner.__init__(self)

        check_str("device_factory_id", device_factory_id)
        check_positive_int("num_active_devices", num_active_devices)

        check_positive_int("num_devices", num_devices)
        check_number_range("num_devices", num_devices, num_active_devices, 1000000)

        check_positive_int("num_workers", num_workers)
        check_number_range("num_workers", num_workers, 1, num_active_devices)
        check_number_range("num_workers", num_workers, 1, 100)

        check_positive_number("cycle_duration", cycle_duration)
        check_number_range("device_reuse_rate", device_reuse_rate, 0.0, 1.0)

        self.device_factory_id = device_factory_id
        self.num_active_devices = num_active_devices
        self.num_devices = num_devices
        self.cycle_length = cycle_duration
        self.device_reuse_rate = device_reuse_rate
        self.num_workers = num_workers
        self.simulator = None

    def create_simulator(self, fl_ctx: FLContext) -> Optional[Simulator]:
        engine = fl_ctx.get_engine()
        factory = engine.get_component(self.device_factory_id)
        if not isinstance(factory, DeviceFactory):
            self.system_panic(
                f"component {self.device_factory_id} must be DeviceFactory but got {type(factory)}",
                fl_ctx,
            )
            return None

        return Simulator(
            device_factory=factory,
            num_active_devices=self.num_active_devices,
            num_devices=self.num_devices,
            num_workers=self.num_workers,
            cycle_duration=self.cycle_length,
            device_reuse_rate=self.device_reuse_rate,
        )
