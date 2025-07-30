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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.edge.simulation.device_task_processor import DeviceTaskProcessor
from nvflare.edge.simulation.devices.tp import TPODeviceFactory
from nvflare.edge.simulation.simulator import Simulator
from nvflare.edge.widgets.runner import SimulationRunner


class TPORunner(SimulationRunner):

    def __init__(
        self,
        task_processor_id: str,
        job_timeout: float = 60.0,
        num_devices: int = 1000,
        num_workers: int = 10,
    ):
        SimulationRunner.__init__(self)
        self.job_timeout = job_timeout
        self.num_devices = num_devices
        self.num_workers = num_workers
        self.task_processor_id = task_processor_id
        self.tpo = None
        self.register_event_handler(EventType.ABOUT_TO_START_RUN, self._tpo_about_to_start)

    def _tpo_about_to_start(self, event_type: str, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"TPO got event: {event_type}")
        engine = fl_ctx.get_engine()
        tpo = engine.get_component(self.task_processor_id)
        if not isinstance(tpo, DeviceTaskProcessor):
            raise ValueError(f"component {self.task_processor_id} should be DeviceTaskProcessor but got {type(tpo)}")
        self.tpo = tpo

    def create_simulator(self, fl_ctx: FLContext) -> Optional[Simulator]:
        self.log_info(fl_ctx, "TPO Create Simulator.")
        job_meta = fl_ctx.get_prop(FLContextKey.JOB_META)
        job_name = job_meta.get(JobMetaKey.JOB_NAME)

        return Simulator(
            job_name=job_name,
            get_job_timeout=self.job_timeout,
            device_factory=TPODeviceFactory(self.tpo),
            num_devices=self.num_devices,
            num_workers=self.num_workers,
        )
