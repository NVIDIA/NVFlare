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

from nvflare.job_config.api import FedJob

from .spec import ExecEnv

WORKSPACE_ROOT = "/tmp/nvflare/simulation"


class SimulationExecEnv(ExecEnv):

    def __init__(
        self,
        workspace_name: str,
        num_clients: int = None,
        num_threads: int = None,
        gpu_config: str = None,
        log_config: str = None,
    ):
        self.workspace_name = workspace_name
        self.num_clients = num_clients
        self.num_threads = num_threads
        self.gpu_config = gpu_config
        self.log_config = log_config

    def deploy(self, job: FedJob):
        job.simulator_run(
            workspace=os.path.join(WORKSPACE_ROOT, self.workspace_name),
            n_clients=self.num_clients,
            threads=self.num_threads,
            gpu=self.gpu_config,
            log_config=self.log_config,
        )
