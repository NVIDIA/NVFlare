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
import os

from data.cifar10_data_utils import cifar10_split
from model import ModerateTFNet

from nvflare.app_opt.tf.recipes import FedOptRecipe
from nvflare.recipe import SimEnv

SPLIT_DIR = "/tmp/cifar10_splits"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=8)
    parser.add_argument("--num_rounds", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--name", type=str, default="", help="Optional job name")

    # FedOpt server-side optimizer arguments
    parser.add_argument("--server_lr", type=float, default=1.0, help="Server-side learning rate")
    parser.add_argument("--server_momentum", type=float, default=0.6, help="Server-side momentum")
    parser.add_argument("--server_lr_decay_alpha", type=float, default=0.9, help="Server-side LR decay alpha")

    args = parser.parse_args()

    job_name = args.name if args.name else f"cifar10_tf_fedopt_alpha{args.alpha}"
    train_split_root = f"{SPLIT_DIR}/clients{args.n_clients}_alpha{args.alpha}"

    print(f"Running FedOpt ({args.num_rounds} rounds) with alpha = {args.alpha} and {args.n_clients} clients")

    # Prepare data splits
    if args.alpha > 0.0:
        print(f"Preparing CIFAR10 and doing data split with alpha = {args.alpha}")
        train_idx_paths = cifar10_split(num_sites=args.n_clients, alpha=args.alpha, split_dir=train_split_root)
        print(train_idx_paths)
    else:
        raise ValueError("Alpha must be greater than 0 for federated settings")

    # Create initial model
    initial_model = ModerateTFNet(input_shape=(None, 32, 32, 3))

    # Configure FedOpt optimizer arguments
    optimizer_args = {
        "path": "tensorflow.keras.optimizers.SGD",
        "args": {
            "learning_rate": args.server_lr,
            "momentum": args.server_momentum,
        },
        "config_type": "dict",
    }

    # Configure FedOpt learning rate scheduler arguments
    lr_scheduler_args = {
        "path": "tensorflow.keras.optimizers.schedules.CosineDecay",
        "args": {
            "initial_learning_rate": args.server_lr,
            "decay_steps": args.num_rounds,
            "alpha": args.server_lr_decay_alpha,
        },
        "config_type": "dict",
    }

    # Create FedOpt recipe
    # Model can be specified as class instance or dict config
    # Alternative: initial_model = {"path": "networks.tf_net.ModerateTFNet", "args": {...}}
    # For pre-trained weights: initial_ckpt="/server/path/to/pretrained.h5"
    recipe = FedOptRecipe(
        name=job_name,
        initial_model=initial_model,
        # initial_ckpt=initial_ckpt,  # Uncomment to use pre-trained weights
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        train_script=os.path.join(os.path.dirname(__file__), "client.py"),
        train_args=f"--batch_size {args.batch_size} --epochs {args.epochs} --train_idx_root {train_split_root}",
        optimizer_args=optimizer_args,
        lr_scheduler_args=lr_scheduler_args,
    )

    # Run using SimEnv
    env = SimEnv(num_clients=args.n_clients)
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in:", run.get_result())
    print()


if __name__ == "__main__":
    main()
