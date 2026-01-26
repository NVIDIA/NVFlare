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

from model import LitNet

from nvflare.app_opt.pt.recipes.fedeval import FedEvalRecipe
from nvflare.recipe.sim_env import SimEnv


PRETRAIN_MODEL_DIR = "/tmp/nvflare/pretrain_models"


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument(
        "--checkpoint", type=str, default=os.path.join(PRETRAIN_MODEL_DIR, "pretrained_model.pt"),
        help="Path to pre-trained model checkpoint"
    )

    return parser.parse_args()


def main():
    args = define_parser()

    n_clients = args.n_clients
    batch_size = args.batch_size
    checkpoint = args.checkpoint

    # Create recipe with checkpoint
    print(f"Using pre-trained checkpoint: {checkpoint}")

    recipe = FedEvalRecipe(
        min_clients=n_clients,
        initial_model=LitNet(checkpoint=os.path.abspath(checkpoint)),
        eval_script="client.py",
        eval_args=f"--batch_size {batch_size}",
    )

    env = SimEnv(num_clients=n_clients, num_threads=n_clients)
    recipe.execute(env=env)


if __name__ == "__main__":
    main()
