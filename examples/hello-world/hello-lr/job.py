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

from nvflare.app_common.np.recipes.lr.fedavg import FedAvgLrRecipe
from nvflare.recipe import SimEnv

# from nvflare.recipe import PocEnv


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--data_root", type=str, default="/tmp/flare/dataset/heart_disease_data")

    return parser.parse_args()


def main():
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds
    data_root = args.data_root

    print("number of clients =", n_clients)
    recipe = FedAvgLrRecipe(
        min_clients=n_clients,
        num_rounds=num_rounds,
        damping_factor=0.8,
        num_features=13,  # Model is created internally based on num_features
        # For pre-trained weights: initial_ckpt="/server/path/to/lr_model.npy",
        train_script="client.py",
        train_args=f"--data_root {data_root}",
    )
    env = SimEnv(num_clients=n_clients, num_threads=n_clients)
    # env = PocEnv(num_clients=n_clients)
    run = recipe.execute(env)
    w = run.get_result()
    print("result location =", w)


if __name__ == "__main__":
    main()
