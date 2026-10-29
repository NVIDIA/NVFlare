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

"""Compare exported standard and per-site job configurations."""

import argparse

from model import create_model

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_final_global_evaluation, set_per_site_config


def create_recipe(per_site=False):
    recipe = FedAvgRecipe(
        name="chapter1-structure-per-site" if per_site else "chapter1-structure",
        model=create_model(),
        min_clients=2,
        num_rounds=3,
        train_script="client.py",
    )
    if per_site:
        set_per_site_config(
            recipe,
            {
                "site-1": {"train_args": ["--learning_rate", "0.1", "--batch_size", "32"]},
                "site-2": {"train_args": ["--learning_rate", "0.05", "--batch_size", "16"]},
            },
        )
    add_final_global_evaluation(recipe)
    return recipe


def main():
    parser = argparse.ArgumentParser(epilog="Recipe options: --export --export-dir PATH --log_config progress")
    parser.add_argument("--per-site", action="store_true")
    args = parser.parse_args()
    env = (
        SimEnv(clients=["site-1", "site-2"], log_config="progress")
        if args.per_site
        else SimEnv(num_clients=2, log_config="progress")
    )
    run = create_recipe(args.per_site).execute(env)
    print("Results:", run.get_result())


if __name__ == "__main__":
    main()
