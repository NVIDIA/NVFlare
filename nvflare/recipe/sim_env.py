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
import os.path
from typing import Optional, List

from pydantic import BaseModel, field_validator
from pydantic import PositiveInt

from nvflare.job_config.api import FedJob

from .spec import ExecEnv

WORKSPACE_ROOT = "/tmp/nvflare/simulation"

class SimEnv(ExecEnv, BaseModel):
    num_clients: Optional[PositiveInt] = None,
    clients: Optional[List[str]]= None
    num_threads: Optional[PositiveInt] = None,
    gpu_config: Optional[str] = None,
    log_config: Optional[str] = None,


    def deploy(self, job: FedJob):
        job.simulator_run(
            workspace=os.path.join(WORKSPACE_ROOT, job.name),
            n_clients=self.num_clients,
            clients= self.clients,
            threads=self.num_threads,
            gpu=self.gpu_config,
            log_config=self.log_config,
        )
