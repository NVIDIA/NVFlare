# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Give each client its own training arguments using the recipe API."""

from src.network import create_model

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_final_global_evaluation, set_per_site_config


def create_recipe():
    recipe = FedAvgRecipe(
        name="chapter1-custom-clients",
        model=create_model(),
        min_clients=2,
        num_rounds=3,
        train_script="src/client.py",
    )
    set_per_site_config(
        recipe,
        {
            "site-1": {"train_args": ["--learning_rate", "0.1", "--batch_size", "32"]},
            "site-2": {"train_args": ["--learning_rate", "0.05", "--batch_size", "16"]},
        },
    )
    add_final_global_evaluation(recipe)
    return recipe


if __name__ == "__main__":
    run = create_recipe().execute(SimEnv(clients=["site-1", "site-2"]))
    print("Results:", run.get_result())
