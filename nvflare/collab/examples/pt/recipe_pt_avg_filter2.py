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
from nvflare.collab.examples.pt.filters2 import ModelFilter
from nvflare.collab.examples.pt.pt_avg_filter import PTFedAvg, PTTrainer
from nvflare.collab.sys.recipe import CollabRecipe


def main():
    export_recipe("collab_pt_fedavg_filter2", _make_recipe)


def _make_recipe(job_name):
    recipe = CollabRecipe(
        job_name=job_name,
        server=PTFedAvg(
            initial_model={
                "x": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "y": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "z": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            },
            num_rounds=2,
        ),
        client=PTTrainer(delta=1.0),
    )
    model_filter = ModelFilter("weights")
    recipe.add_server_outgoing_call_filters(
        pattern="*.train",
        filters=[model_filter],
    )
    recipe.add_server_incoming_result_filters(pattern="*.train", filters=[model_filter])

    recipe.add_client_incoming_call_filters(
        pattern="*.train",
        filters=[model_filter],
    )
    recipe.add_client_outgoing_result_filters(pattern="*.train", filters=[model_filter])
    return recipe


if __name__ == "__main__":
    main()
