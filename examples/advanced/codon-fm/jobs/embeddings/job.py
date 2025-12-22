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

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=5)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data/checkpoints/NV-CodonFM-Encodon-80M-v1/NV-CodonFM-Encodon-80M-v1.safetensors",
    )
    parser.add_argument("--data_prefix", type=str, default="/data/federated_data")
    parser.add_argument("--data_type", type=str, default="train")
    return parser.parse_args()


def main():
    args = define_parser()

    n_clients = args.n_clients

    recipe = FedAvgRecipe(
        name="codon-fm-embeddings",
        min_clients=n_clients,
        num_rounds=1,
        initial_model=None,
        train_script="/data/jobs/embeddings/client.py",
        train_args=f"--checkpoint {args.checkpoint} --data_prefix {args.data_prefix} --data_type {args.data_type}",
    )

    env = SimEnv(num_clients=n_clients, workspace_root="/data/nvflare/simulation")
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in :", run.get_result())
    print()


if __name__ == "__main__":
    main()
