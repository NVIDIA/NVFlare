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

import argparse
import multiprocessing

from data.cifar10_data_utils import cifar10_split
from src.model import ModerateTFNet

from nvflare.app_opt.tf.recipes import FedAvgRecipe
from nvflare.job_config.api import FedJob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=8)
    parser.add_argument("--num_rounds", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--fedprox_mu", type=float, default=0.0)
    parser.add_argument("--workspace", type=str, default="/tmp")
    parser.add_argument("--gpu", type=str, default="0")

    args = parser.parse_args()
    multiprocessing.set_start_method("spawn")

    train_split_root = f"{args.workspace}/cifar10_splits/clients{args.n_clients}_alpha{args.alpha}"

    # Prepare data splits
    if args.alpha > 0.0:
        print(f"preparing CIFAR10 and doing alpha split with alpha = {args.alpha}")
        train_idx_paths = cifar10_split(num_sites=args.n_clients, alpha=args.alpha, split_dir=train_split_root)
        print(train_idx_paths)
    else:
        raise ValueError("Alpha must be greater than 0 for federated settings")

    # Create initial model
    initial_model = ModerateTFNet(input_shape=(None, 32, 32, 3))

    # Create FedAvg recipe (FedProx uses FedAvg on server side, proximal term is client-side)
    recipe = FedAvgRecipe(
        name=f"cifar10_tf_fedprox_alpha{args.alpha}_mu{args.fedprox_mu}",
        initial_model=initial_model,
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        train_script="cifar10_fedprox/client.py",
        train_args=f"--batch_size {args.batch_size} --epochs {args.epochs} --train_idx_root {train_split_root} --fedprox_mu {args.fedprox_mu}",
    )

    # Build the job
    job: FedJob = recipe.create_job()

    # Run the job using simulator
    job.simulator_run(f"{args.workspace}/nvflare/jobs/{job.name}", gpu=args.gpu)
