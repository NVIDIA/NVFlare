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


from typing import Any, Callable, Optional

from pydantic import BaseModel

from nvflare import FedJob
from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.exec_env import ExecEnv
from nvflare.job_config.job_recipe import JobRecipe
from nvflare.job_config.script_runner import ScriptRunner


class FedAvgRecipe(JobRecipe, BaseModel):
    name: str = "fed_avg"
    min_clients: int
    num_rounds: int
    model: Any = (None,)
    client_script: str
    client_script_args: str = "--local_epochs 1"
    aggregate_fn: Optional[Callable] = None

    def model_post_init(self, __context):  # v2 pydantic
        if not self.client_script:
            raise ValueError("client script must provided")

    def get_job(self, env: ExecEnv) -> FedJob:
        # todo: fix this:  this is PT FedAvgJob only
        job = FedAvgJob(
            name=self.name,
            min_clients=self.min_clients,
            n_clients=len(env.client_names),
            num_rounds=self.num_rounds,
            initial_model=self.model,
        )
        # Add clients
        for client_name in env.client_names:
            executor = ScriptRunner(script=self.client_script, script_args=self.client_script_args)
            job.to(executor, client_name)

        return job
