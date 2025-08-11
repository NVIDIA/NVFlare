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
from typing import List, Optional

from pydantic import BaseModel, model_validator

from nvflare.job_config.api import FedJob

from .spec import ExecEnv

WORKSPACE_ROOT = "/tmp/nvflare/simulation"


# Internal — not part of the public API
class _SimEnvValidator(BaseModel):
    num_clients: int  # num_clients is always an integer
    clients: Optional[List[str]] = None
    num_threads: int
    gpu_config: Optional[str] = None
    log_config: Optional[str] = None

    @model_validator(mode="after")
    def check_num_clients_consistency(self):
        if self.num_clients and self.clients and len(self.clients) != self.num_clients:
            raise ValueError(
                f"Inconsistent number of clients: num_clients={self.num_clients} "
                f"but clients list has {len(self.clients)} entries."
            )
        return self


class SimEnv(ExecEnv):
    def __init__(
        self,
        *,
        num_clients: int = 0,
        clients: Optional[List[str]] = None,
        num_threads: int = 0,
        gpu_config: str = None,
        log_config: str = None,
    ):
        v = _SimEnvValidator(
            num_clients=num_clients,
            clients=clients,
            num_threads=num_threads,
            gpu_config=gpu_config,
            log_config=log_config,
        )

        self.num_clients = v.num_clients
        self.num_threads = v.num_threads
        self.gpu_config = v.gpu_config
        self.log_config = v.log_config
        self.clients = v.clients

    def deploy(self, job: FedJob):
        job.simulator_run(
            workspace=os.path.join(WORKSPACE_ROOT, job.name),
            n_clients=self.num_clients,
            clients=self.clients,
            threads=self.num_threads,
            gpu=self.gpu_config,
            log_config=self.log_config,
        )
