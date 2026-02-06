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
This code shows how to use NVIDIA FLARE Job Recipe with FedAvgRecipe for AMPLIFY multi-task fine-tuning.
Each client trains a different downstream task while jointly fine-tuning the AMPLIFY trunk.
"""
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.filters import ExcludeParamsFilter
from src.model import AmplifyRegressor

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking

# Define the six antibody property tasks
TASKS = ["aggregation", "binding", "expression", "immunogenicity", "polyreactivity", "thermostability"]


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=600,
        help="Number of federated learning rounds (global aggregation iterations)",
    )
    parser.add_argument(
        "--local_epochs", type=int, default=1, help="Number of local training epochs per client and FL round"
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="chandar-lab/AMPLIFY_120M",
        choices=["chandar-lab/AMPLIFY_120M", "chandar-lab/AMPLIFY_350M"],
        help="AMPLIFY pretrained model to use",
    )
    parser.add_argument(
        "--layer_sizes",
        type=str,
        default="128,64,32",
        help="Comma-separated list of layer sizes for the regression MLP",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=f"{os.path.dirname(os.path.abspath(__file__))}/../FLAb/data_fl",
        help="Root directory for training and test data",
    )
    parser.add_argument("--exp_name", type=str, default="fedavg_multitask", help="Experiment name")
    parser.add_argument(
        "--sim_gpus",
        type=str,
        default="0",
        help="GPU indexes to simulate clients, e.g., '0,1,2,0,1,2' for 6 clients on 3 GPUs",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size for each client")
    parser.add_argument("--trunk_lr", type=float, default=1e-4, help="Learning rate for the AMPLIFY trunk")
    parser.add_argument("--regressor_lr", type=float, default=1e-2, help="Learning rate for the regression layers")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use per client (default: None, use all samples). Useful for quick testing.",
    )

    return parser.parse_args()


def main():
    args = define_parser()

    # Parse layer sizes from string to list of integers
    layer_sizes = [int(size) for size in args.layer_sizes.split(",")]

    # Build the initial model (one regressor per task, but each client will only use their own)
    model = AmplifyRegressor(pretrained_model_name_or_path=args.pretrained_model, layer_sizes=layer_sizes)

    print(f"Running FedAvg ({args.num_rounds} rounds) for multi-task AMPLIFY fine-tuning")
    print(f"Using {len(TASKS)} clients, one for each task: {TASKS}")
    print(f"Model: {args.pretrained_model}")
    print(f"Layer sizes: {layer_sizes}")

    # Create FedAvgRecipe - each client will be named after a task
    # The client.py will automatically load data based on site_name
    recipe = FedAvgRecipe(
        name=f"amplify_seqregression_{args.exp_name}",
        min_clients=len(TASKS),
        num_rounds=args.num_rounds,
        model=model,
        train_script="client.py",
        train_args=f"--data_root {args.data_root} --n_epochs {args.local_epochs} --pretrained_model {args.pretrained_model} --layer_sizes {args.layer_sizes} --batch_size {args.batch_size} --trunk_lr {args.trunk_lr} --regressor_lr {args.regressor_lr}"
        + (f" --max_samples {args.max_samples}" if args.max_samples else ""),
        # server_expected_format=ExchangeFormat.PYTORCH,
    )

    # Add filter to exclude regressor parameters (keep them private to each client)
    recipe.add_client_output_filter(ExcludeParamsFilter(exclude_vars="regressor"), tasks=["train", "validate"])

    # Add TensorBoard experiment tracking
    add_experiment_tracking(recipe, tracking_type="tensorboard")

    # Run FL simulation with task names as client names
    env = SimEnv(
        clients=TASKS,  # Explicitly set client names to task names
        workspace_root="/tmp/nvflare/AMPLIFY/multitask",
        gpu_config=args.sim_gpus,
    )
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in:", run.get_result())
    print()


if __name__ == "__main__":
    main()
