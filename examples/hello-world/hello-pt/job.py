# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Run the Hello PyTorch FedAvg job in an NVFLARE simulation."""

import argparse

from model import create_model
from prepare_data import DATASET_CHOICES, DATASET_PATH, DEFAULT_DATASET

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_final_global_evaluation

DEFAULT_NUM_CLIENTS = 2
DEFAULT_NUM_ROUNDS = 3
EXPORT_HELP = """NVFlare Recipe export options:
  --export                    Export the job instead of running it.
  --export-dir EXPORT_DIR     Parent directory for the exported job (default: ./fl_job).
"""


def define_parser() -> argparse.ArgumentParser:
    # Recipe consumes its system-level export flags before this parser runs, so
    # list them in the epilog to keep ``python job.py --help`` complete.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXPORT_HELP,
    )
    parser.add_argument("--n_clients", type=int, default=DEFAULT_NUM_CLIENTS)
    parser.add_argument("--num_rounds", type=int, default=DEFAULT_NUM_ROUNDS)

    # Keep the zero-argument quickstart deterministic and offline. CIFAR-10 is
    # still available, but selecting it explicitly permits a dataset download.
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument("--dataset", choices=DATASET_CHOICES, dest="dataset")
    dataset_group.add_argument(
        "--synthetic_data",
        action="store_const",
        const="synthetic",
        dest="dataset",
        help="Deprecated alias for --dataset synthetic.",
    )
    parser.set_defaults(dataset=DEFAULT_DATASET)
    parser.add_argument(
        "--data_root",
        default=DATASET_PATH,
        help="Client-local CIFAR-10 cache path. Ignored for the synthetic dataset.",
    )

    return parser


def create_recipe(args):
    train_args = ["--dataset", args.dataset]
    if args.dataset == "cifar10":
        train_args.extend(("--data_root", args.data_root))

    recipe = FedAvgRecipe(
        name="hello-pt",
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        # Model can be specified as class instance or dict config:
        model=create_model(),
        # Alternative: model={"class_path": "model.SimpleNetwork", "args": {}},
        # For pre-trained weights: initial_ckpt="/server/path/to/pretrained.pt",
        train_script="client.py",
        # Pass argv directly so a client-local path containing spaces, quotes,
        # or apostrophes keeps its exact argument boundary without shell parsing.
        train_args=train_args,
    )
    # Always verify the persisted final global model in the basic quickstart.
    add_final_global_evaluation(recipe)

    return recipe


def main(argv=None):
    args = define_parser().parse_args(argv)
    recipe = create_recipe(args)

    env = SimEnv(num_clients=args.n_clients)
    run = recipe.execute(env)
    result = run.get_result()
    print()
    # SimEnv runs synchronously. A normal return from execute/get_result means
    # the simulation completed, so keep the beginner-facing message direct.
    print("Simulation completed successfully.")
    print("Result can be found in :", result)
    print()
    return result


if __name__ == "__main__":
    main()
