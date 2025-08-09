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
from typing import Optional, List, Any

from pydantic import BaseModel, PositiveInt

from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.recipe.spec import Recipe


class FedAvgRecipe(BaseModel, Recipe):
    name: str = "fedavg"
    initial_model: Any = None
    clients: Optional[List[str]] = None
    num_clients: Optional[PositiveInt] = None
    min_clients: int = 0
    num_rounds: int = 2
    train_script: str
    train_args: str = ""
    # aggregate_fn: Optional[Callable] = None

    def model_post_init(self, __context):
        # Extra setup logic after validation & parsing
        if self.clients:
            if self.num_clients is None:
                self.num_clients = len(self.clients)
            elif len(self.clients) != self.min_clients:
                raise ValueError(" inconsistent number of clients")

        job = FedAvgJob(
            name=self.name,
            n_clients=0,  # for all clients
            min_clients=self.min_clients,
            num_rounds=self.num_rounds,
            initial_model=self.initial_model,
        )
        executor = ScriptRunner(script=self.train_script, script_args= self.train_args)
        if self.clients is None:
            job.to_clients(executor)
        else:
            for client in self.clients:
                job.to(executor, client)
        self.job = job
