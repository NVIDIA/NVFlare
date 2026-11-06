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

"""Plug custom server aggregation into the Recipe API."""

import argparse

from aggregator import MedianAggregator
from model import create_model

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_final_global_evaluation

NUM_CLIENTS = 3


def create_recipe(num_rounds=5, stop_accuracy=80.0, patience=2, aggregation="median"):
    if aggregation not in ("median", "fedavg"):
        raise ValueError(f"Unknown aggregation: {aggregation}")
    recipe = FedAvgRecipe(
        name="chapter1-custom-server" if aggregation == "median" else "chapter1-custom-server-fedavg",
        model=create_model(),
        min_clients=NUM_CLIENTS,
        num_rounds=num_rounds,
        train_script="client.py",
        aggregator=MedianAggregator() if aggregation == "median" else None,
        key_metric="accuracy",
        key_metric_mode="max",
        stop_cond=f"accuracy >= {stop_accuracy}",
        patience=patience,
        best_model_filename="chapter1_best_model.pt",
    )
    add_final_global_evaluation(recipe)
    return recipe


def main():
    parser = argparse.ArgumentParser(epilog="Recipe options: --export --export-dir PATH --log_config progress")
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--stop_accuracy", type=float, default=80.0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--aggregation", choices=("median", "fedavg"), default="median")
    args = parser.parse_args()
    recipe = create_recipe(args.num_rounds, args.stop_accuracy, args.patience, args.aggregation)
    run = recipe.execute(SimEnv(num_clients=NUM_CLIENTS, log_config="progress"))
    print("Results:", run.get_result())


if __name__ == "__main__":
    main()
