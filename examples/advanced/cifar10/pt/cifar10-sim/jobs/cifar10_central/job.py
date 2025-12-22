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
"""
This code shows how to use NVIDIA FLARE Job Recipe to connect both Federated learning client and server algorithm
and run it under different environments
"""
import argparse
import os

from model import ModerateCNN

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-2, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of worker processes for data loading (0 = main process only)"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size for each client")
    parser.add_argument("--name", type=str, default=None, help="Custom name for the recipe (overrides default naming)")

    return parser.parse_args()


def main():
    args = define_parser()

    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    job_name = args.name if args.name else "cifar10_central"

    print(f"Running Centralized training with lr = {lr} and epochs = {epochs}")

    recipe = FedAvgRecipe(
        name=job_name,
        min_clients=1,
        num_rounds=1,
        initial_model=ModerateCNN(),
        train_script=os.path.join(os.path.dirname(__file__), "../../../src/client.py"),
        train_args=f"--central --lr {lr} --aggregation_epochs {epochs} --batch_size {batch_size} --num_workers {num_workers}",
    )
    add_experiment_tracking(recipe, tracking_type="tensorboard")

    env = SimEnv(num_clients=1)
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in :", run.get_result())
    print()


if __name__ == "__main__":
    main()
