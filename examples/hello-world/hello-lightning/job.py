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

import torchvision.datasets as datasets
from model import LitNet

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe.sim_env import SimEnv

DATASET_ROOT = "/tmp/nvflare/data"


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=24)

    return parser.parse_args()


def download_data():
    datasets.CIFAR10(root=DATASET_ROOT, train=True, download=True)
    datasets.CIFAR10(root=DATASET_ROOT, train=False, download=True)


def main():
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds
    batch_size = args.batch_size

    recipe = FedAvgRecipe(
        min_clients=n_clients,
        num_rounds=num_rounds,
        # Model can be specified as class instance or dict config:
        model=LitNet(),
        # Alternative: model={"class_path": "model.LitNet", "args": {}},
        # For pre-trained weights: initial_ckpt="/server/path/to/pretrained.pt",
        train_script="client.py",
        train_args=f"--batch_size {batch_size}",
    )

    env = SimEnv(num_clients=n_clients, num_threads=n_clients)
    recipe.execute(env=env)


if __name__ == "__main__":
    download_data()
    main()
