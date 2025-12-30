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
Cross-Site Evaluation Example with NumPy using the Recipe API.

This example demonstrates two modes:
1. Standalone CSE: Evaluate pre-trained models without training
2. Training + CSE: Run FedAvg training followed by cross-site evaluation

Both modes use the add_cross_site_evaluation() utility function for consistent behavior.
"""

import argparse

from nvflare import FedJob
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.np.np_model_locator import NPModelLocator
from nvflare.app_common.np.np_validator import NPValidator
from nvflare.app_common.np.recipes import NumpyFedAvgRecipe
from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
from nvflare.recipe import SimEnv
from nvflare.recipe.spec import Recipe
from nvflare.recipe.utils import add_cross_site_evaluation

SERVER_MODEL_DIR = "/tmp/nvflare/server_pretrain_models"
CLIENT_MODEL_DIR = "/tmp/nvflare/client_pretrain_models"


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2, help="Number of clients")
    parser.add_argument(
        "--mode",
        type=str,
        default="pretrained",
        choices=["pretrained", "training"],
        help="Mode: 'pretrained' for standalone CSE, 'training' for training+CSE",
    )
    parser.add_argument("--num_rounds", type=int, default=1, help="Number of training rounds (for training mode)")
    return parser.parse_args()


def run_pretrained_cse(n_clients: int):
    """Run standalone cross-site evaluation with pre-trained models.

    This mode demonstrates CSE without any training. It loads pre-trained models
    from specified directories and evaluates them across all client sites.
    """
    print("\n=== Running Cross-Site Evaluation with Pre-trained Models ===\n")
    print(f"Server models: {SERVER_MODEL_DIR}")
    print(f"Client models: {CLIENT_MODEL_DIR}\n")

    # Create a minimal FedJob for CSE-only workflow
    job = FedJob(name="hello-numpy-cse", min_clients=n_clients)

    # Configure model locator with pre-trained model locations
    model_locator_id = job.to_server(
        NPModelLocator(
            model_dir=SERVER_MODEL_DIR,
            model_name={"server_model_1": "server_1.npy", "server_model_2": "server_2.npy"},
        )
    )

    # Add cross-site evaluation controller
    eval_controller = CrossSiteModelEval(
        model_locator_id=model_locator_id,
        submit_model_timeout=600,
        validation_timeout=6000,
    )
    job.to_server(eval_controller)

    # Add validators to clients
    validator = NPValidator(
        validate_task_name=AppConstants.TASK_VALIDATION,
    )
    job.to_clients(validator, tasks=[AppConstants.TASK_VALIDATION])

    # Note: ValidationJsonGenerator is added by the job automatically
    # But we can't use add_cross_site_evaluation() here because we need
    # custom model locator configuration for pre-trained models

    # Create recipe and run
    recipe = Recipe(job)
    env = SimEnv(num_clients=n_clients)
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


def run_training_and_cse(n_clients: int, num_rounds: int):
    """Run FedAvg training followed by cross-site evaluation.

    This mode demonstrates the recommended pattern: create a standard FedAvg recipe,
    then add cross-site evaluation using the utility function.
    """
    print("\n=== Running Training + Cross-Site Evaluation ===\n")
    print(f"Configuration: {n_clients} clients, {num_rounds} training rounds\n")

    # Create standard FedAvg recipe
    recipe = NumpyFedAvgRecipe(
        name="hello-numpy-train-cse",
        min_clients=n_clients,
        num_rounds=num_rounds,
        train_script="client.py",
        train_args="",
    )

    # Add cross-site evaluation using utility function
    # This is the recommended approach from Branch A
    add_cross_site_evaluation(recipe, model_locator_type="numpy")

    # Run in simulation environment
    env = SimEnv(num_clients=n_clients)
    run = recipe.execute(env)

    print("\n" + "=" * 60)
    print("Training and cross-site evaluation complete!")
    print("=" * 60)
    print(f"Result location: {run.get_result()}")
    print(f"Job status: {run.get_status()}")
    print()
    print("To view training results:")
    print(f"  ls {run.get_result()}/")
    print()
    print("To view CSE results:")
    print(f"  cat {run.get_result()}/cross_site_val/cross_val_results.json")
    print()


def main():
    args = define_parser()

    if args.mode == "pretrained":
        run_pretrained_cse(args.n_clients)
    elif args.mode == "training":
        run_training_and_cse(args.n_clients, args.num_rounds)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
