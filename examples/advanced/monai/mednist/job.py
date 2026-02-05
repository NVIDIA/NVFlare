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

"""
MONAI MedNIST Classification with FedAvg

This example shows federated learning with MONAI using Client API and FedAvgRecipe.
"""

import argparse

from model import FLDenseNet121

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2, help="Number of simulated clients")
    parser.add_argument("--num_rounds", type=int, default=5, help="Number of FL rounds")
    parser.add_argument("--threads", type=int, default=2, help="Number of parallel threads")
    parser.add_argument("--workspace", type=str, default="fedavg_workspace", help="Workspace directory for simulation")
    parser.add_argument(
        "--tracking",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "mlflow", "both", "none"],
        help="Experiment tracking type",
    )
    args = parser.parse_args()

    # Create FedAvgRecipe
    recipe = FedAvgRecipe(
        name="mednist_fedavg",
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        initial_model=FLDenseNet121(
            spatial_dims=2, in_channels=1, out_channels=6
        ),  # We wrap the DenseNet121 into FLDenseNet121 to preserve the configuration when sending the model to the server.
        train_script="client.py",
    )

    # Add experiment tracking
    if args.tracking in ["tensorboard", "both"]:
        add_experiment_tracking(recipe, tracking_type="tensorboard")
    if args.tracking in ["mlflow", "both"]:
        add_experiment_tracking(recipe, tracking_type="mlflow")

    # Setup simulation environment
    env = SimEnv(num_clients=args.n_clients, num_threads=args.threads, workspace_root=args.workspace)

    # Execute the recipe
    run = recipe.execute(env)

    print()
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())
    print()


if __name__ == "__main__":
    main()
