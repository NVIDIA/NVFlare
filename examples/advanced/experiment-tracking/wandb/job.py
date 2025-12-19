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

from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE
from nvflare.app_opt.pt.recipes import FedAvgRecipe
from nvflare.app_opt.tracking.wandb.wandb_receiver import WandBReceiver
from nvflare.recipe.utils import add_experiment_tracking


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--script", type=str, default="client.py")
    parser.add_argument("--launch_external_process", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--streamed_to_clients",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to stream to NVFlare client or not",
    )
    parser.add_argument(
        "--streamed_to_server",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to stream to NVFlare server or not",
    )
    parser.add_argument("--export_config", action=argparse.BooleanOptionalAction, default=False)

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
        analytics_receiver=False,  # We'll add WandB manually
    )

    # Configure WandB settings
    wandb_config = {
        "mode": "online",
        "wandb_args": {
            "project": "wandb-experiment",
            "name": "wandb",
            "notes": "Federated Experiment tracking with W&B \n Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) and [PyTorch](https://pytorch.org/) as the deep learning training framework. This example also highlights the Flare streaming capability from the clients to the server and deliver to WandB.\\n\\n> **_NOTE:_** \\n This example uses the *[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and will load its data within the trainer code.\n",
            "tags": ["baseline"],
            "job_type": "train-validate",
            "config": {"architecture": "CNN", "dataset_id": "CIFAR10", "optimizer": "SGD"},
        },
    }

    # Add server-side tracking if requested
    if args.streamed_to_server:
        add_experiment_tracking(recipe, "wandb", tracking_config=wandb_config)

    # Add client-side tracking if requested
    if args.streamed_to_clients:
        for i in range(args.n_clients):
            site_name = f"site-{i + 1}"

            # Client-side receivers listen to local events (not federated)
            client_config = wandb_config.copy()
            client_config["events"] = [ANALYTIC_EVENT_TYPE]

            receiver = WandBReceiver(**client_config)
            recipe.job.to(receiver, site_name, id="wandb_receiver")

    # Run or export
    if args.export_config:
        export_dir = "/tmp/nvflare/jobs/job_config"
        print(f"job exported to {export_dir}")
        recipe.export(export_dir)
    else:
        recipe.run("/tmp/nvflare/jobs/workdir", gpu="0")


if __name__ == "__main__":
    main()
