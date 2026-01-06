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

import argparse
from typing import Dict

from model import SAGE

from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import ProdEnv, SimEnv


def create_job(
    num_clients: int,
    num_rounds: int,
    epochs_per_round: int,
    data_path: str,
    output_path: str = "./output",
) -> FedAvgRecipe:
    """Create a financial transaction classification job using FedAvgRecipe.

    Args:
        num_clients: Number of clients
        num_rounds: Number of federated learning rounds
        epochs_per_round: Number of local epochs per round for each client
        data_path: Path to the Elliptic++ dataset
        output_path: Output path for results

    Returns:
        FedAvgRecipe configured for financial transaction classification
    """
    # Configure and instantiate finance-specific model
    model = SAGE(
        in_channels=165,
        hidden_channels=256,
        num_layers=3,
        num_classes=2,
    )

    # Create per-client arguments (client_id derived from site name in client script)
    train_args: Dict[str, str] = {}
    for i in range(1, num_clients + 1):
        site_name = f"site-{i}"
        train_args[site_name] = (
            f"--data_path {data_path} "
            f"--epochs {epochs_per_round} "
            f"--num_clients {num_clients} "
            f"--output_path {output_path}"
        )

    # Create FedAvgRecipe with initial_model to ensure persistor is added
    recipe = FedAvgRecipe(
        name="gnn_finance",
        initial_model=model,
        min_clients=num_clients,
        num_rounds=num_rounds,
        train_script="client.py",
        train_args=train_args,
    )

    # Deploy model.py file needed for finance task
    recipe.job.to("model.py", "server")
    for i in range(1, num_clients + 1):
        site_name = f"site-{i}"
        recipe.job.to("model.py", site_name)
        recipe.job.to("process_elliptic.py", site_name)

    # Add model selector for validation metric tracking
    recipe.job.to(IntimeModelSelector(key_metric="validation_auc"), "server", id="model_selector")

    return recipe


def main():
    parser = argparse.ArgumentParser(
        description="Financial transaction classification federated learning using FedAvgRecipe"
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=2,
        help="Number of clients (default: 2)",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=7,
        help="Number of federated learning rounds (default: 7)",
    )
    parser.add_argument(
        "--epochs_per_round",
        type=int,
        default=10,
        help="Number of local epochs per round for each client (default: 10)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/tmp/nvflare/datasets/elliptic_pp",
        help="Path to the Elliptic++ dataset (default: /tmp/nvflare/datasets/elliptic_pp)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output",
        help="Output path for results (default: ./output)",
    )
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default="/tmp/nvflare/gnn/finance_fl_workspace",
        help="Work directory for simulator runs (default: /tmp/nvflare/gnn/finance_fl_workspace)",
    )
    parser.add_argument(
        "--job_dir",
        type=str,
        default="/tmp/nvflare/jobs/gnn_finance",
        help="Directory for job export (default: /tmp/nvflare/jobs/gnn_finance)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of threads for FL simulation (default: number of clients)",
    )
    parser.add_argument(
        "--startup_kit_location",
        type=str,
        default=None,
        help="Startup kit location for production mode (default: None, runs in simulation mode)",
    )
    parser.add_argument(
        "--username",
        type=str,
        default="admin@nvidia.com",
        help="Username for production mode (default: admin@nvidia.com)",
    )
    args = parser.parse_args()

    print("Starting financial transaction classification federated learning job...")
    print("args:", args)

    num_threads = args.threads if args.threads else args.num_clients

    print(f"Number of clients: {args.num_clients}")
    print(f"Rounds: {args.num_rounds}")
    print(f"Epochs per round: {args.epochs_per_round}")
    print(f"Data path: {args.data_path}")

    # Create recipe
    recipe = create_job(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        epochs_per_round=args.epochs_per_round,
        data_path=args.data_path,
        output_path=args.output_path,
    )

    # Export job
    print(f"Exporting job to {args.job_dir}")
    recipe.job.export_job(args.job_dir)

    # Run recipe
    client_names = [f"site-{i}" for i in range(1, args.num_clients + 1)]

    if args.startup_kit_location:
        print("Running job in production mode...")
        print(f"Startup kit location: {args.startup_kit_location}")
        print(f"Username: {args.username}")
        env = ProdEnv(startup_kit_location=args.startup_kit_location, username=args.username)
    else:
        print("Running job in simulation mode...")
        print(f"Workspace directory: {args.workspace_dir}")
        print(f"Number of threads: {num_threads}")
        env = SimEnv(clients=client_names, num_threads=num_threads, workspace_root=args.workspace_dir)

    run = recipe.execute(env)
    print("Job Status is:", run.get_status())
    print("Job Result is:", run.get_result())


if __name__ == "__main__":
    main()
