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
from nvflare.collab.examples import export_recipe
from nvflare.collab.examples.np.algos.client import NPTrainer
from nvflare.collab.examples.np.algos.strategies.avg_seq import NPFedAvgSequential
from nvflare.collab.examples.np.algos.widgets import MetricReceiver
from nvflare.collab.sys.recipe import CollabRecipe


def main():
    export_recipe("fox_fedavg_seq", _make_recipe)


def _make_recipe(job_name):
    recipe = CollabRecipe(
        job_name=job_name,
        server=NPFedAvgSequential(initial_model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], num_rounds=2),
        client=NPTrainer(delta=1.0),
        server_objects={"metric_receiver": MetricReceiver()},
    )
    recipe.set_server_prop("client_weight_config", {"red": 70, "blue": 100, "silver": 50})
    recipe.set_client_prop("client_delta", {"red": 1.0, "blue": 2.0, "silver": 3.0})
    return recipe


if __name__ == "__main__":
    main()
