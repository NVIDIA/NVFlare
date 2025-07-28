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
from nvflare.edge.simulation.config import ConfigParser
from nvflare.edge.simulation.devices.tp import TPDeviceFactory
from nvflare.edge.simulation.simulator import Simulator
from nvflare.edge.widgets.runner import SimulationRunner


class TPRunner(SimulationRunner):

    def __init__(self, config_file: str):
        SimulationRunner.__init__(self)
        self.config_file = config_file

    def create_simulator(self, fl_ctx: FLContext) -> Optional[Simulator]:
        parser = ConfigParser(self.config_file)
        return Simulator(
            job_name=parser.get_job_name(),
            get_job_timeout=parser.get_job_timeout,
            device_factory=TPDeviceFactory(parser),
            num_devices=parser.get_num_devices(),
            num_workers=parser.get_num_workers(),
        )
