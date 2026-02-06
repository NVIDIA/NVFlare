# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
NVFlare job recipe for federated Qwen2.5-VL fine-tuning with PubMedVision.
Uses FedAvg with 3 clients; each client gets a site-specific data path.
"""
import argparse
import os

import torch
from model import Qwen2VLModelWrapper
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking


def define_parser():
    parser = argparse.ArgumentParser(description="Federated Qwen2.5-VL SFT (3 clients)")
    parser.add_argument("--n_clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=2, help="FL rounds")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root dir for site-1, site-2, site-3 data")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--train_script", type=str, default="client.py")
    parser.add_argument("--local_epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    return parser.parse_args()


def main():
    args = define_parser()
    n_clients = args.n_clients
    data_dir = os.path.abspath(args.data_dir)

    # Per-site train_args: each client gets its own data path
    per_site_config = {}
    for i in range(1, n_clients + 1):
        site_name = f"site-{i}"
        site_data_path = os.path.join(data_dir, site_name)
        per_site_config[site_name] = {
            "train_args": (
                f"--data_path {site_data_path} "
                f"--model_name_or_path {args.model_name_or_path} "
                f"--epochs {args.local_epochs} --batch_size {args.batch_size}"
            ),
        }

    # Initial model: Qwen2.5-VL (loaded on server for aggregation)
    initial_model = Qwen2VLModelWrapper(
        model_name_or_path=args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    recipe = FedAvgRecipe(
        name="qwen3-vl",
        min_clients=n_clients,
        num_rounds=args.num_rounds,
        initial_model=initial_model,
        train_script=args.train_script,
        train_args="",  # overridden by per_site_config
        per_site_config=per_site_config,
        key_metric="loss",
    )
    add_experiment_tracking(recipe, tracking_type="tensorboard")

    env = SimEnv(num_clients=n_clients)
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in:", run.get_result())
    print()


if __name__ == "__main__":
    main()
