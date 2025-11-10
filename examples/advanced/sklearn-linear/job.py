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

    return parser.parse_args()


def main():
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds
    data_path = args.data_path

    print(f"Creating sklearn linear model recipe with {n_clients} clients for {num_rounds} rounds")
    print(f"Data path: {data_path}")

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
        train_args=f"--data_path {data_path}",
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

