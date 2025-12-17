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
Job configuration for PyTorch federated learning.

Supports 4 training modes:
- pt:           Standard PyTorch (single GPU)
- pt_ddp:       PyTorch DDP (multi-GPU)
- lightning:    PyTorch Lightning (single GPU)
- lightning_ddp: PyTorch Lightning DDP (multi-GPU)
"""

import argparse

from lit_model import LitNet
from model import Net

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking

CLIENT_SCRIPTS = {
    "pt": "client.py",
    "pt_ddp": "client_ddp.py",
    "lightning": "client_lightning.py",
    "lightning_ddp": "client_lightning_ddp.py",
}

MODELS = {
    "pt": Net,
    "pt_ddp": Net,
    "lightning": LitNet,
    "lightning_ddp": LitNet,
}


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["pt", "pt_ddp", "lightning", "lightning_ddp"],
        default="pt",
        help="Training mode: pt, pt_ddp, lightning, lightning_ddp",
    )
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--use_tracking", action="store_true", help="Enable TensorBoard tracking")
    parser.add_argument("--export_config", action="store_true", help="Export job config only")
    return parser.parse_args()


def main():
    args = define_parser()

    # Select client script and model based on mode
    train_script = CLIENT_SCRIPTS[args.mode]
    initial_model = MODELS[args.mode]()

    # Build train_args based on mode
    train_args = ""
    if args.mode == "pt" and args.use_tracking:
        train_args = "--use_tracking"

    # DDP modes require external process launch
    launch_external = args.mode in ["pt_ddp", "lightning_ddp"]

    # Command for DDP
    if args.mode == "pt_ddp":
        command = "python3 -u -m torch.distributed.run --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=2 "
    else:
        command = "python3 -u"

    recipe = FedAvgRecipe(
        name=f"pt_{args.mode}",
        initial_model=initial_model,
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        train_script=train_script,
        train_args=train_args,
        launch_external_process=launch_external,
        command=command,
    )

    if args.use_tracking:
        add_experiment_tracking(recipe, tracking_type="tensorboard")

    if args.export_config:
        recipe.export("/tmp/nvflare/jobs/job_config")
        print("Job config exported to /tmp/nvflare/jobs/job_config")
    else:
        env = SimEnv(num_clients=args.n_clients)
        run = recipe.execute(env)
        print()
        print("Result:", run.get_result())
        print("Status:", run.get_status())
        print()


if __name__ == "__main__":
    main()
