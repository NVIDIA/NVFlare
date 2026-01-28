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

from model import Net

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe.utils import add_experiment_tracking


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--script", type=str, default="client.py")
    parser.add_argument("--launch_external_process", action="store_true", help="Launch as external process")
    parser.add_argument(
        "--streamed_to_clients",
        action="store_true",
        help="Enable client-side tracking (each client logs to its own WandB run)",
    )
    parser.add_argument(
        "--disable_server_tracking",
        action="store_true",
        help="Disable server-side tracking (by default, server-side tracking is enabled)",
    )
    parser.add_argument("--export_config", action="store_true", help="Export config without running")

    return parser.parse_args()


def main():
    args = define_parser()

    # Create FedAvg recipe
    recipe = FedAvgRecipe(
        name="fedavg_wandb",
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        initial_model=Net(),
        train_script=args.script,
        launch_external_process=args.launch_external_process,
    )

    # Configure WandB settings
    wandb_config = {
        "mode": "online",
        "wandb_args": {
            "project": "wandb-experiment",
            "name": "wandb",
            "notes": (
                "Federated Experiment tracking with W&B\n\n"
                "Example of using NVIDIA FLARE to train an image classifier using federated averaging (FedAvg) "
                "and PyTorch as the deep learning training framework. This example also highlights the FLARE "
                "streaming capability from the clients to the server and deliver to WandB.\n\n"
                "NOTE: This example uses the CIFAR-10 dataset and will load its data within the trainer code."
            ),
            "tags": ["baseline"],
            "job_type": "train-validate",
            "config": {"architecture": "CNN", "dataset_id": "CIFAR10", "optimizer": "SGD"},
        },
    }

    # Add experiment tracking
    # - server_side: Aggregates metrics from all clients to a single WandB run (default)
    # - client_side: Each client tracks its own local metrics to separate WandB runs
    add_experiment_tracking(
        recipe,
        "wandb",
        tracking_config=wandb_config,
        client_side=args.streamed_to_clients,
        server_side=not args.disable_server_tracking,
    )

    # Run or export
    if args.export_config:
        export_dir = "/tmp/nvflare/jobs/job_config"
        print(f"job exported to {export_dir}")
        recipe.export(export_dir)
    else:
        from nvflare.recipe import SimEnv

        env = SimEnv(num_clients=args.n_clients, workspace_root="/tmp/nvflare/jobs/workdir")
        run = recipe.execute(env)
        print()
        print("Result:", run.get_result())
        print("Status:", run.get_status())
        print()


if __name__ == "__main__":
    main()
