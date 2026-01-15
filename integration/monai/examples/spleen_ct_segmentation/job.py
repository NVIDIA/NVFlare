# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
MONAI Spleen CT Segmentation with FedAvg

This example shows how to use NVIDIA FLARE with MONAI bundles for federated learning.
It uses the Client API and FedAvgRecipe for simplified configuration.
"""

import argparse
import os

import torch
from model import FLUNet

from nvflare.apis.dxo import DataKind
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle_root", type=str, default="bundles/spleen_ct_segmentation",
                        help="Path to MONAI bundle relative to job directory")
    parser.add_argument("--n_clients", type=int, default=2, help="Number of simulated clients")
    parser.add_argument("--num_rounds", type=int, default=100, help="Number of FL rounds")
    parser.add_argument("--local_epochs", type=int, default=10, help="Number of local training epochs per round")
    parser.add_argument("--threads", type=int, default=2, help="Number of parallel threads")
    parser.add_argument("--workspace", type=str, default="/tmp/nvflare/simulation",
                        help="Workspace directory for simulation")
    parser.add_argument("--send_weight_diff", action="store_true", help="Send weight differences instead of full weights")
    parser.add_argument("--tracking", type=str, default="tensorboard",
                        choices=["tensorboard", "mlflow", "both", "none"],
                        help="Experiment tracking type")
    args = parser.parse_args()

    # Set device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create train arguments for client.py
    train_args = f"--bundle_root {os.path.join(os.getcwd(), args.bundle_root)} --local_epochs {args.local_epochs}"
    if args.send_weight_diff:
        train_args += " --send_weight_diff"

    # Create FedAvgRecipe
    recipe = FedAvgRecipe(
        name="spleen_bundle_fedavg",
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        initial_model=FLUNet(spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=[
                16,
                32,
                64,
                128,
                256
            ],
            strides=[
                2,
                2,
                2,
                2
            ],
        num_res_units=2,
        norm="batch"),
        train_script="client.py",
        train_args=train_args,
        aggregator_data_kind=DataKind.WEIGHT_DIFF if args.send_weight_diff else DataKind.WEIGHTS,
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
