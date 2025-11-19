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
This code shows how to use NVIDIA FLARE Job Recipe to run federated learning with scikit-learn
linear models using FedAvg algorithm.

Per-Client Data Splits:
    The old prepare_job_config.sh approach is replaced by passing a dict for train_args.
    This allows different data ranges for each client, supporting non-IID scenarios.
"""

import argparse

from nvflare.app_opt.sklearn.recipes.fedavg import SklearnFedAvgRecipe
from nvflare.recipe import SimEnv


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=50, help="Number of training rounds")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/tmp/nvflare/dataset/HIGGS.csv",
        help="Path to HIGGS dataset CSV file",
    )
    parser.add_argument(
        "--split_method",
        type=str,
        default="uniform",
        choices=["uniform", "custom"],
        help="Data split method: 'uniform' (same args for all) or 'custom' (per-client ranges)",
    )

    return parser.parse_args()


def calculate_data_splits(n_clients: int, total_size: int = 11000000, valid_size: int = 1100000):
    """Calculate uniform data splits for clients.
    
    Args:
        n_clients: Number of clients
        total_size: Total dataset size (HIGGS has 11M rows)
        valid_size: Size of validation set (first N rows)
    
    Returns:
        dict mapping site names to (train_start, train_end, valid_start, valid_end)
    """
    train_size = total_size - valid_size
    train_per_client = train_size // n_clients
    
    splits = {}
    for i in range(n_clients):
        site_name = f"site-{i + 1}"
        train_start = valid_size + (i * train_per_client)
        train_end = valid_size + ((i + 1) * train_per_client) if i < n_clients - 1 else total_size
        splits[site_name] = {
            "train_start": train_start,
            "train_end": train_end,
            "valid_start": 0,
            "valid_end": valid_size,
        }
    
    return splits


def main():
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds
    data_path = args.data_path
    split_method = args.split_method

    print(f"Creating sklearn linear model recipe with {n_clients} clients for {num_rounds} rounds")
    print(f"Data path: {data_path}")
    print(f"Split method: {split_method}")

    # Configure train_args based on split method
    if split_method == "uniform":
        # Simple mode: all clients use same args and client.py defaults
        train_args = f"--data_path {data_path}"
        print("Using uniform split (client.py defaults for data ranges)")
    else:
        # Custom mode: per-client data ranges
        splits = calculate_data_splits(n_clients)
        train_args = {
            site_name: f"--data_path {data_path} --train_start {split['train_start']} "
            f"--train_end {split['train_end']} --valid_start {split['valid_start']} "
            f"--valid_end {split['valid_end']}"
            for site_name, split in splits.items()
        }
        print("Using custom per-client data splits:")
        for site_name, split in splits.items():
            print(f"  {site_name}: train [{split['train_start']}:{split['train_end']}], "
                  f"valid [{split['valid_start']}:{split['valid_end']}]")

    recipe = SklearnFedAvgRecipe(
        name="sklearn_linear",
        min_clients=n_clients,
        num_rounds=num_rounds,
        initial_params={
            "n_classes": 2,
            "learning_rate": "constant",
            "eta0": 1e-4,
            "loss": "log_loss",
            "penalty": "l2",
            "fit_intercept": 1,
        },
        train_script="client.py",
        train_args=train_args,
    )

    print("Executing recipe in simulation environment...")
    env = SimEnv(num_clients=n_clients, num_threads=n_clients)
    run = recipe.execute(env)

    print()
    print("=" * 60)
    print(f"Job Status: {run.get_status()}")
    print(f"Result Location: {run.get_result()}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()

