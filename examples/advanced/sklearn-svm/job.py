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
This code shows how to use NVIDIA FLARE Job Recipe to run federated SVM
with scikit-learn using support vector aggregation.
"""

import argparse

from nvflare.app_opt.sklearn.recipes.svm import SVMFedAvgRecipe
from nvflare.recipe import SimEnv


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=3, help="Number of clients")
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        choices=["linear", "poly", "rbf", "sigmoid"],
        help="Kernel type for SVM",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="sklearn",
        choices=["sklearn", "cuml"],
        help="Backend library (sklearn or cuml)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/tmp/nvflare/dataset/cancer.csv",
        help="Path to breast cancer dataset CSV file",
    )

    return parser.parse_args()


def main():
    args = define_parser()

    n_clients = args.n_clients
    kernel = args.kernel
    backend = args.backend
    data_path = args.data_path

    print(f"Creating SVM recipe with {n_clients} clients")
    print(f"Kernel: {kernel}")
    print(f"Backend: {backend}")
    print(f"Data path: {data_path}")

    # Note: For simplicity, this recipe uses the same data range for all clients.
    # In a real scenario, you would want to split the data. The client script
    # accepts train_start, train_end, valid_start, valid_end arguments for this purpose.
    # Example: For client-specific data splits, you could use the utils/prepare_data.py
    # logic to generate per-client arguments.

    recipe = SVMFedAvgRecipe(
        name="sklearn_svm",
        min_clients=n_clients,
        kernel=kernel,
        train_script="client.py",
        train_args=f"--data_path {data_path} --backend {backend} "
        f"--train_start 0 --train_end 100 --valid_start 100 --valid_end 569",
        backend=backend,
    )

    print("Executing recipe in simulation environment...")
    print("Note: SVM training only requires 1 round (round 0 for training, round 1 for validation)")
    env = SimEnv(num_clients=n_clients, num_threads=n_clients)
    run = recipe.execute(env)

    print()
    print("=" * 60)
    print(f"Job Status: {run.get_status()}")
    print(f"Result Location: {run.get_result()}")
    print("=" * 60)
    print()
    print("Note: For heterogeneous data splits across clients, you can use")
    print("the utils/prepare_data.py to generate per-client data ranges and")
    print("pass them as arguments to the train_script.")


if __name__ == "__main__":
    main()

