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
This example demonstrates Cross-Site Evaluation using the Recipe API.

There are two modes:
1. Standalone CSE: Evaluate pre-trained models (run generate_pretrain_models.py first)
2. Training + CSE: Run FedAvg training followed by cross-site evaluation

This file shows the standalone CSE mode. For training + CSE, see job_train_and_cse.py.
"""

import argparse

from nvflare.app_common.np.recipes.cross_site_eval import NumpyCrossSiteEvalRecipe
from nvflare.recipe import SimEnv

SERVER_MODEL_DIR = "/tmp/nvflare/server_pretrain_models"
CLIENT_MODEL_DIR = "/tmp/nvflare/client_pretrain_models"


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument(
        "--mode",
        type=str,
        default="pretrained",
        choices=["pretrained"],
        help="Mode: 'pretrained' for standalone CSE with pre-trained models",
    )
    return parser.parse_args()


def main():
    args = define_parser()
    n_clients = args.n_clients

    # Cross-site evaluation with pre-trained models
    recipe = NumpyCrossSiteEvalRecipe(
        name="hello-numpy-cse",
        min_clients=n_clients,
        model_locator_config={
            "model_dir": SERVER_MODEL_DIR,
            "model_name": {"server_model_1": "server_1.npy", "server_model_2": "server_2.npy"},
        },
        client_model_dir=CLIENT_MODEL_DIR,
    )

    # Run in simulation environment
    env = SimEnv(num_clients=n_clients)
    run = recipe.execute(env)

    print()
    print("Cross-site evaluation complete!")
    print("Result can be found in:", run.get_result())
    print("Job Status is:", run.get_status())
    print()
    print("To view results:")
    print(f"  cat {run.get_result()}/cross_site_val/cross_val_results.json")
    print()


if __name__ == "__main__":
    main()
