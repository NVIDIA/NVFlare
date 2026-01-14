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
This code shows how to use NVIDIA FLARE Job Recipe to run federated K-Means clustering
with scikit-learn using mini-batch aggregation.

Per-Client Data Splits:
    Data is automatically divided into non-overlapping ranges for each client.
    Customize the calculate_data_splits() function to implement different split strategies.
"""

import argparse

from nvflare.app_opt.sklearn import KMeansFedAvgRecipe
from nvflare.recipe import SimEnv


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=5, help="Number of training rounds")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters for K-Means")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/tmp/nvflare/dataset/sklearn_iris.csv",
        help="Path to iris dataset CSV file",
    )

    return parser.parse_args()


def calculate_data_splits(n_clients: int, total_size: int = 150, train_fraction: float = 0.8):
    """Calculate non-overlapping data splits for clients.

    Divides training data into equal chunks for each client. Validation data is shared.
    Users can modify this function to implement custom split strategies.

    Args:
        n_clients: Number of clients
        total_size: Total dataset size (Iris has 150 samples)
        train_fraction: Fraction of data for training (rest for validation)

    Returns:
        dict mapping site names to split configuration with train/valid ranges
    """
    train_size = int(total_size * train_fraction)
    valid_start = train_size
    train_per_client = train_size // n_clients

    splits = {}
    for i in range(n_clients):
        site_name = f"site-{i + 1}"
        train_start = i * train_per_client
        train_end = (i + 1) * train_per_client if i < n_clients - 1 else train_size
        splits[site_name] = {
            "train_start": train_start,
            "train_end": train_end,
            "valid_start": valid_start,
            "valid_end": total_size,
        }

    return splits


def main():
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds
    n_clusters = args.n_clusters
    data_path = args.data_path

    print(f"Creating K-Means clustering recipe with {n_clients} clients for {num_rounds} rounds")
    print(f"Number of clusters: {n_clusters}")
    print(f"Data path: {data_path}")

    # Calculate per-client data splits (non-overlapping ranges)
    splits = calculate_data_splits(n_clients)
    clients = [site_name for site_name in splits.keys()]
    per_site_config = {
        site_name: {
            "train_args": f"--data_path {data_path} --train_start {split['train_start']} "
            f"--train_end {split['train_end']} --valid_start {split['valid_start']} "
            f"--valid_end {split['valid_end']}"
        }
        for site_name, split in splits.items()
    }

    recipe = KMeansFedAvgRecipe(
        name="sklearn_kmeans",
        min_clients=n_clients,
        num_rounds=num_rounds,
        n_clusters=n_clusters,
        train_script="client.py",
        per_site_config=per_site_config,
    )

    print("Executing recipe in simulation environment...")
    env = SimEnv(clients=clients, num_threads=n_clients)
    run = recipe.execute(env)

    print()
    print("=" * 60)
    print(f"Job Status: {run.get_status()}")
    print(f"Result Location: {run.get_result()}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
