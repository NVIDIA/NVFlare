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
This example demonstrates Cross-Site Evaluation using NVFlare.

Two modes are supported:
1. Standalone CSE: Evaluate pre-trained models (run generate_pretrain_models.py first)
   - Uses Recipe API (NumpyCrossSiteEvalRecipe)
2. Training + CSE: Run FedAvg training followed by cross-site evaluation
   - Uses FedJob API to chain both workflows

Usage:
    # Mode 1: Standalone CSE with pre-trained models
    python job.py --mode pretrained

    # Mode 2: Training + CSE
    python job.py --mode training
"""

import argparse

from nvflare import FedJob
from nvflare.apis.dxo import DataKind
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.np.np_formatter import NPFormatter
from nvflare.app_common.np.np_model_locator import NPModelLocator
from nvflare.app_common.np.np_model_persistor import NPModelPersistor
from nvflare.app_common.np.np_trainer import NPTrainer
from nvflare.app_common.np.np_validator import NPValidator
from nvflare.app_common.np.recipes.cross_site_eval import NumpyCrossSiteEvalRecipe
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.recipe import SimEnv

SERVER_MODEL_DIR = "/tmp/nvflare/server_pretrain_models"
CLIENT_MODEL_DIR = "/tmp/nvflare/client_pretrain_models"
WORKSPACE_DIR = "/tmp/nvflare/jobs/workdir"


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
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
    """Run standalone cross-site evaluation with pre-trained models using Recipe API."""
    print("\n=== Running Cross-Site Evaluation with Pre-trained Models ===\n")

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
    print("  Result location:", run.get_result())
    print("  Job status:", run.get_status())
    print()
    print("To view results:")
    print(f"  cat {run.get_result()}/cross_site_val/cross_val_results.json")
    print()


def run_training_and_cse(n_clients: int, num_rounds: int):
    """Run FedAvg training followed by cross-site evaluation using FedJob API."""
    print("\n=== Running Training + Cross-Site Evaluation ===\n")
    print(f"Configuration: {n_clients} clients, {num_rounds} training rounds\n")

    # Create FedJob with both training and CSE workflows
    job = FedJob(name="hello-numpy-train-cse", min_clients=n_clients)

    # Server components
    persistor_id = job.to_server(NPModelPersistor())
    aggregator_id = job.to_server(InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS))
    shareable_generator_id = job.to_server(FullModelShareableGenerator())
    model_locator_id = job.to_server(NPModelLocator())  # Will locate models from persistor
    formatter_id = job.to_server(NPFormatter())
    job.to_server(ValidationJsonGenerator())

    # Workflow 1: Training (ScatterAndGather)
    controller = ScatterAndGather(
        min_clients=n_clients,
        num_rounds=num_rounds,
        persistor_id=persistor_id,
        aggregator_id=aggregator_id,
        shareable_generator_id=shareable_generator_id,
    )
    job.to_server(controller)

    # Workflow 2: Cross-Site Evaluation (runs after training)
    controller = CrossSiteModelEval(
        model_locator_id=model_locator_id,
        formatter_id=formatter_id,
    )
    job.to_server(controller)

    # Client components
    trainer = NPTrainer(
        train_task_name=AppConstants.TASK_TRAIN,
        submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
    )
    job.to_clients(trainer, tasks=[AppConstants.TASK_TRAIN, AppConstants.TASK_SUBMIT_MODEL])

    validator = NPValidator(
        validate_task_name=AppConstants.TASK_VALIDATION,
    )
    job.to_clients(validator, tasks=[AppConstants.TASK_VALIDATION])

    # Run in simulator
    job.export_job("/tmp/nvflare/jobs")
    job.simulator_run(WORKSPACE_DIR, gpu="0", n_clients=n_clients)

    print()
    print("Training and cross-site evaluation complete!")
    print(f"  Result location: {WORKSPACE_DIR}/server/simulate_job")
    print()
    print("To view training results:")
    print(f"  ls {WORKSPACE_DIR}/server/simulate_job/")
    print()
    print("To view CSE results:")
    print(f"  cat {WORKSPACE_DIR}/server/simulate_job/cross_site_val/cross_val_results.json")
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
