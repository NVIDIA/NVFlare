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

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.edge.simulation.simulated_device import DeviceFactory
from nvflare.edge.simulation.simulator import Simulator
from nvflare.edge.widgets.runner import SimulationRunner
from nvflare.fuel.utils.validation_utils import check_number_range, check_positive_int, check_str


class DeviceRunner(SimulationRunner):

    def __init__(
        self,
        device_factory_id: str,
        num_workers=10,
        num_devices=10000,
    ):
        """Constructor of DeviceRunner.
        A DeviceRunner is a component to be directly installed in CJs that simulates edge devices.
        No web nodes are needed for the simulated devices to communicate with Flare.

        Args:
            device_factory_id:
            num_workers:
            num_devices:
        """
        SimulationRunner.__init__(self)

        check_str("device_factory_id", device_factory_id)

        check_positive_int("num_devices", num_devices)
        check_number_range("num_devices", num_devices, 10, 1000000)

        check_positive_int("num_workers", num_workers)
        check_number_range("num_workers", num_workers, 1, 100)

        self.device_factory_id = device_factory_id
        self.num_devices = num_devices
        self.num_workers = num_workers

    def create_simulator(self, fl_ctx: FLContext) -> Optional[Simulator]:
        engine = fl_ctx.get_engine()
        job_meta = fl_ctx.get_prop(FLContextKey.JOB_META)
        job_name = job_meta.get(JobMetaKey.JOB_NAME)

        self.log_debug(fl_ctx, f"got job name from meta: {job_name}")

        factory = engine.get_component(self.device_factory_id)
        if not isinstance(factory, DeviceFactory):
            self.system_panic(
                f"component {self.device_factory_id} must be DeviceFactory but got {type(factory)}",
                fl_ctx,
            )
            return None

        return Simulator(
            job_name=job_name,
            device_factory=factory,
            num_devices=self.num_devices,
            num_workers=self.num_workers,
        )
