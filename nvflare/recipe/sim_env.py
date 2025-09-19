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
from typing import Optional

from pydantic import BaseModel, model_validator

from nvflare.job_config.api import FedJob

from .spec import ExecEnv

WORKSPACE_ROOT = "/tmp/nvflare/simulation"


# Internal — not part of the public API
class _SimEnvValidator(BaseModel):
    num_clients: int
    clients: Optional[list[str]] = None
    num_threads: Optional[int] = None
    gpu_config: Optional[str] = None
    log_config: Optional[str] = None
    workspace_root: str = WORKSPACE_ROOT

    @model_validator(mode="after")
    def check_num_clients_consistency(self):
        # Check if both num_clients and clients are not specified (invalid)
        if self.num_clients == 0 and (self.clients is None or len(self.clients) == 0):
            raise ValueError(
                "Either 'num_clients' must be > 0 or 'clients' list must be provided. "
                "Cannot run simulation with no clients."
            )

        # Check if both are specified and inconsistent
        if self.num_clients > 0 and self.clients and len(self.clients) != self.num_clients:
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
        clients: Optional[list[str]] = None,
        num_threads: Optional[int] = None,
        gpu_config: str = None,
        log_config: str = None,
        workspace_root: str = WORKSPACE_ROOT,
        extra: dict = None,
    ):
        """Initialize simulation execution environment.

        Args:
            num_clients (int, optional): Number of simulated clients. Defaults to 0.
            clients (list[str], optional): List of client names. Defaults to None.
            num_threads (int, optional): Number of threads to run simulator. Defaults to None.
                If not provided, the number of threads will be set to the number of clients.
            gpu_config (str, optional): GPU configuration string. Defaults to None.
            log_config (str, optional): Log configuration string. Defaults to None.
            workspace_root (str, optional): Root directory for simulation workspace. Defaults to WORKSPACE_ROOT.
            extra: extra env config info
        """
        super().__init__(extra)

        v = _SimEnvValidator(
            num_clients=num_clients,
            clients=clients,
            num_threads=num_threads,
            gpu_config=gpu_config,
            log_config=log_config,
            workspace_root=workspace_root,
        )

        self.num_clients = v.num_clients
        self.num_threads = v.num_threads if v.num_threads is not None else v.num_clients
        self.gpu_config = v.gpu_config
        self.log_config = v.log_config
        self.clients = v.clients
        self.workspace_root = v.workspace_root

    def deploy(self, job: FedJob):
        job.simulator_run(
            workspace=os.path.join(self.workspace_root, job.name),
            n_clients=self.num_clients,
            clients=self.clients,
            threads=self.num_threads,
            gpu=self.gpu_config,
            log_config=self.log_config,
        )
        return job.name

    def get_job_status(self, job_id: str) -> Optional[str]:
        """Get job status - not supported in simulation environment."""
        print(
            f"Note, get_status returns None in SimEnv. The simulation logs can be found at {os.path.join(self.workspace_root, job_id)}"
        )
        return None

    def abort_job(self, job_id: str) -> None:
        """Abort job - not supported in simulation environment."""
        print("abort is not supported in a simulation environment, it will always run to completion.")

    def get_job_result(self, job_id: str, timeout: float = 0.0) -> Optional[str]:
        """Get job result workspace path."""
        if self.workspace_root is None:
            raise RuntimeError("Simulation workspace_root is None - SimEnv may not be properly initialized")
        return os.path.join(self.workspace_root, job_id)
