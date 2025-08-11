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

from abc import ABC, abstractmethod
from typing import List, Optional, Union

from pydantic import BaseModel, field_validator

from nvflare.job_config.api import FedJob
from nvflare.job_config.exec_env import ExecEnv, SimEnv
from nvflare.job_config.script_runner import FrameworkType


class JobRecipe(ABC, BaseModel):
    name: str
    framework: FrameworkType = FrameworkType.PYTORCH

    model_config = {
        # note can't validate FrameworkType as it is not Enum.
        # todo: change this later.
        "arbitrary_types_allowed": True
    }

    @field_validator("name")
    @classmethod
    def check_name(cls, v):
        if not v:
            raise ValueError("Name must not be empty")
        return v

    @abstractmethod
    def get_job(self, env: ExecEnv) -> FedJob:
        pass

    def execute(
        self,
        env: Optional[ExecEnv] = None,
        clients: Union[int, List[str]] = None,
        gpus: Union[int, List[int]] = None,
        workspace_dir: Optional[str] = None,
    ):

        if env is None:
            env = SimEnv(clients=clients, gpus=gpus, workspace_dir=workspace_dir)

        if isinstance(env, SimEnv):
            job: FedJob = self.get_job(env)
            # job.export_job("/tmp/nvflare/jobs/job_config")
            job.simulator_run(workspace=env.workspace_dir, gpu="0")

        pass
