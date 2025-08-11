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
from typing import Any, List, Optional

from pydantic import BaseModel, PositiveInt

from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.recipe.spec import Recipe


# Internal — not part of the public API
class _FedAvgValidator(BaseModel):
    name: str
    initial_model: Any
    clients: Optional[List[str]]
    num_clients: Optional[PositiveInt]
    min_clients: int
    num_rounds: int
    train_script: str
    train_args: str

    def model_post_init(self, __context):
        if self.clients and self.num_clients is None:
            self.num_clients = len(self.clients)
        elif self.clients and len(self.clients) != self.min_clients:
            raise ValueError("inconsistent number of clients")


class FedAvgRecipe(Recipe):
    def __init__(
        self,
        *,
        name: str = "fedavg",
        initial_model: Any = None,
        clients: Optional[List[str]] = None,
        num_clients: Optional[int] = None,
        min_clients: int = 0,
        num_rounds: int = 2,
        train_script: str,
        train_args: str = "",
        # aggregate_fn: Optional[Callable] = None
    ):
        # Validate inputs internally
        v = _FedAvgValidator(
            name=name,
            initial_model=initial_model,
            clients=clients,
            num_clients=num_clients,
            min_clients=min_clients,
            num_rounds=num_rounds,
            train_script=train_script,
            train_args=train_args,
        )

        self.name = v.name
        self.initial_model = v.initial_model
        self.clients = v.clients
        self.num_clients = v.num_clients
        self.min_clients = v.min_clients
        self.num_rounds = v.num_rounds
        self.initial_model = v.initial_model
        self.clients = v.clients
        self.train_script = v.train_script
        self.train_args = v.train_args

        job = FedAvgJob(
            name=self.name,
            n_clients=0,  # for all clients
            min_clients=self.min_clients,
            num_rounds=self.num_rounds,
            initial_model=self.initial_model,
        )
        executor = ScriptRunner(script=self.train_script, script_args=self.train_args)
        if self.clients is None:
            job.to_clients(executor)
        else:
            for client in self.clients:
                job.to(executor, client)

        Recipe.__init__(self, job)
