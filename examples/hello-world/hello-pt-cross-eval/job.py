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
Standalone Cross-Site Evaluation Example with PyTorch using the Recipe API.

This example demonstrates evaluating pre-trained PyTorch models without any training.
It uses PyTorchCrossSiteEvalRecipe to load pre-trained models and evaluate them
across all client sites.

For Training + CSE, see the hello-pt example.
"""

import argparse

from model import SimpleNetwork

from nvflare.app_opt.pt.recipes import PyTorchCrossSiteEvalRecipe
from nvflare.recipe import SimEnv

SERVER_MODEL_DIR = "/tmp/nvflare/server_pretrain_models"


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2, help="Number of clients")
    return parser.parse_args()


def main():
    args = define_parser()

    print("\n=== Running Cross-Site Evaluation with Pre-trained Models ===\n")

    # Create cross-site evaluation recipe with pre-trained models
    recipe = PyTorchCrossSiteEvalRecipe(
        name="hello-pt-cse",
        min_clients=args.n_clients,
        model=SimpleNetwork(),
        model_dir=SERVER_MODEL_DIR,
        train_script="client.py",
        train_args="--epochs 1 --batch_size 4",
    )

    # Execute in simulation environment
    env = SimEnv(num_clients=args.n_clients)
    run = recipe.execute(env)

    print("\n" + "=" * 60)
    print("Cross-site evaluation complete!")
    print("=" * 60)
    print(f"Result location: {run.get_result()}")
    print(f"Job status: {run.get_status()}")
    print()
    print("To view results:")
    print(f"  cat {run.get_result()}/cross_site_val/cross_val_results.json")
    print()


if __name__ == "__main__":
    main()
