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
"""

import argparse

from nvflare.app_opt.sklearn.recipes.kmeans import KMeansFedAvgRecipe
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


def main():
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds
    n_clusters = args.n_clusters
    data_path = args.data_path

    print(f"Creating K-Means clustering recipe with {n_clients} clients for {num_rounds} rounds")
    print(f"Number of clusters: {n_clusters}")
    print(f"Data path: {data_path}")

    # Note: For simplicity, this recipe uses the same data range for all clients.
    # In a real scenario, you would want to split the data. The client script
    # accepts train_start, train_end, valid_start, valid_end arguments for this purpose.
    # Example: For client-specific data splits, you could use the utils/split_data.py
    # logic to generate per-client arguments.

    recipe = KMeansFedAvgRecipe(
        name="sklearn_kmeans",
        min_clients=n_clients,
        num_rounds=num_rounds,
        n_clusters=n_clusters,
        train_script="src/kmeans_fl.py",
        train_args=f"--data_path {data_path} --train_start 0 --train_end 50 "
        f"--valid_start 0 --valid_end 150",
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
    print("Note: For heterogeneous data splits across clients, you can use")
    print("the utils/split_data.py to generate per-client data ranges and")
    print("pass them as arguments to the train_script.")


if __name__ == "__main__":
    main()


