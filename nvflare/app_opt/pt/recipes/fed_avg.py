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
from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.recipe.spec import Recipe


class FedAvgRecipe(Recipe):

    def __init__(
        self,
        name: str,
        initial_model,
        num_rounds: int,
        min_clients: int,
        train_script: str,
        train_args: dict = None,
    ):
        job = FedAvgJob(
            name=name,
            n_clients=0,  # for all clients
            min_clients=min_clients,
            num_rounds=num_rounds,
            initial_model=initial_model,
        )
        if not train_args:
            train_args = {}

        executor = ScriptRunner(script=train_script, **train_args)
        job.to_clients(executor)
        Recipe.__init__(self, job)
