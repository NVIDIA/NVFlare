# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import sys
from typing import Any, Callable, Optional

from pydantic import BaseModel

print(sys.path)

from nvflare import FedJob
from nvflare.app_common.workflows.cyclic import Cyclic
from nvflare.job_config.exec_env import ExecEnv
from nvflare.job_config.job_recipe import JobRecipe
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner


class CyclicRecipe(JobRecipe, BaseModel):
    name: str = "cyclic"
    min_clients: int
    num_rounds: int
    model: Optional[Any] = (None,)
    client_script: str
    client_script_args: str = ""
    framework: Optional[FrameworkType] = None
    load_model_fn: Optional[Callable] = None

    def post_model_init(self):
        if self.framework is None:
            self.framework = FrameworkType.PYTORCH

        if not self.model and self.load_model_fn:
            self.model = self.load_model_fn()

        if not self.mode:
            raise ValueError("model is None")

    def get_job(self, env: ExecEnv) -> FedJob:
        job = FedJob(name="cyclic")
        # Define the controller workflow and send to serv er
        controller = Cyclic(num_clients=len(env.client_names), num_rounds=self.num_rounds)
        job.to(controller, "server")

        # Define the initial global model and send to server
        job.to(self.model, "server")

        # Add clients
        for client_name in env.client_names:
            executor = ScriptRunner(
                script=self.client_script,
                script_args=self.client_script_args,
                framework=self.framework,
            )
            job.to(executor, client_name)

        return job
