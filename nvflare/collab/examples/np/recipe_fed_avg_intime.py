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
from nvflare.collab.examples.np.algos.filters import AddNoiseToModel, Print
from nvflare.collab.examples.np.algos.strategies.avg_intime import NPFedAvgInTime
from nvflare.collab.examples.np.algos.widgets import MetricReceiver
from nvflare.collab.sys.recipe import CollabRecipe


def main():
    export_recipe("fox_fedavg_intime", _make_recipe)


def _make_recipe(job_name):
    recipe = CollabRecipe(
        job_name=job_name,
        server=NPFedAvgInTime(initial_model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], num_rounds=2),
        server_objects={
            "metric_receiver": MetricReceiver(),
        },
        client=NPTrainer(delta=1.0),
    )

    print_filter = Print()
    recipe.add_server_outgoing_call_filters("*.train", [AddNoiseToModel()])
    recipe.add_server_incoming_result_filters("*.train", [print_filter])
    recipe.set_server_prop("default_timeout", 5.0)

    recipe.add_client_incoming_call_filters("*.train", [print_filter])
    recipe.add_client_outgoing_result_filters("*.train", [print_filter])
    recipe.set_client_prop("default_timeout", 8.0)
    return recipe


if __name__ == "__main__":
    main()
