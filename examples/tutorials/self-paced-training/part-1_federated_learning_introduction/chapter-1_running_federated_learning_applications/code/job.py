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

"""Run or export the Chapter 1 FedAvg recipe from this directory."""

import argparse
from pathlib import Path

from model import create_model

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking, add_final_global_evaluation


def create_recipe(num_rounds=3, tracking="none", client_script="client.py"):
    recipe = FedAvgRecipe(
        name=f"chapter1-{tracking}",
        model=create_model(),
        min_clients=2,
        num_rounds=num_rounds,
        train_script=client_script,
        train_args=["--tracking", tracking],
    )
    if client_script == "client.py":
        add_final_global_evaluation(recipe)
    if tracking == "tensorboard":
        add_experiment_tracking(recipe, "tensorboard")
    elif tracking == "mlflow":
        # An absolute SQLite URI works for both the receiver and the local UI.
        tracking_uri = "sqlite:///" + str(Path("mlflow.db").resolve())
        add_experiment_tracking(recipe, "mlflow", {"tracking_uri": tracking_uri})
    return recipe


def main():
    parser = argparse.ArgumentParser(epilog="Recipe options: --export --export-dir PATH --log_config progress")
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--tracking", choices=("none", "tensorboard", "mlflow"), default="none")
    parser.add_argument("--client-script", choices=("client.py", "client_train_only.py"), default="client.py")
    args = parser.parse_args()
    if args.client_script == "client_train_only.py" and args.tracking != "none":
        parser.error("the minimal client does not send tracking events; use client.py")
    recipe = create_recipe(args.num_rounds, args.tracking, args.client_script)
    run = recipe.execute(SimEnv(num_clients=2))
    print("Results:", run.get_result())


if __name__ == "__main__":
    main()
