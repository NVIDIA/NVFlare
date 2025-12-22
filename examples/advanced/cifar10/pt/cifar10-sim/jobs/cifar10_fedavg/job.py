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

from data.cifar10_data_split import split_and_save
from model import ModerateCNN

from nvflare.apis.dxo import DataKind
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=8, help="Number of federated learning clients to simulate")
    parser.add_argument(
        "--num_rounds", type=int, default=50, help="Number of federated learning rounds (global aggregation iterations)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of worker processes for data loading (0 = main process only)"
    )
    parser.add_argument("--lr", type=float, default=5e-2, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size for each client")
    parser.add_argument(
        "--aggregation_epochs", type=int, default=4, help="Number of local training epochs per client and FL round"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet distribution parameter (controls data heterogeneity: "
        "lower values create more heterogeneous distributions)",
    )
    parser.add_argument("--name", type=str, default=None, help="Custom name for the recipe (overrides default naming)")

    return parser.parse_args()


def main():
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds
    alpha = args.alpha
    num_workers = args.num_workers
    lr = args.lr
    batch_size = args.batch_size
    aggregation_epochs = args.aggregation_epochs
    job_name = args.name if args.name else f"cifar10_fedavg_alpha{alpha}"

    print(f"Running FedAvg ({num_rounds} rounds) with alpha = {alpha} and {n_clients} clients")

    if alpha > 0.0:
        print(f"Preparing CIFAR10 and doing data split with alpha = {alpha}")
        train_idx_root = split_and_save(
            num_sites=n_clients, alpha=alpha, split_dir_prefix="/tmp/cifar10_splits/cifar10_fedavg"
        )
    else:
        raise ValueError("Alpha must be greater than 0 for federated settings")

    recipe = FedAvgRecipe(
        name=job_name,
        min_clients=n_clients,
        num_rounds=num_rounds,
        initial_model=ModerateCNN(),
        train_script=os.path.join(os.path.dirname(__file__), "../../src/client.py"),
        train_args=f"--train_idx_root {train_idx_root} --num_workers {num_workers} --lr {lr} --batch_size {batch_size} --aggregation_epochs {aggregation_epochs}",
        aggregator_data_kind=DataKind.WEIGHT_DIFF,
    )
    add_experiment_tracking(recipe, tracking_type="tensorboard")

    env = SimEnv(num_clients=n_clients)
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in :", run.get_result())
    print()


if __name__ == "__main__":
    main()
