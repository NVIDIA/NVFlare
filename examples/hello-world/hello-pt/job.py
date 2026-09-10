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
from pathlib import Path

from model import create_model
from prepare_data import add_dataset_arguments, validate_cifar10

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_final_global_evaluation
from nvflare.recipe.spec import _peek_recipe_args

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
    parser.add_argument(
        "--n_clients", type=int, default=DEFAULT_NUM_CLIENTS, help="Number of participating clients (default: 2)."
    )
    parser.add_argument(
        "--num_rounds", type=int, default=DEFAULT_NUM_ROUNDS, help="Number of federated training rounds (default: 3)."
    )

    add_dataset_arguments(parser)

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
    # Recipe consumes export flags at import time. An exported job's data may
    # exist only on remote clients; validate the local cache only for simulation.
    export_only, _ = _peek_recipe_args()
    if args.dataset == "cifar10" and not export_only:
        args.data_root = str(Path(args.data_root).expanduser().resolve())
        try:
            validate_cifar10(args.data_root)
        except FileNotFoundError as e:
            raise SystemExit(str(e)) from None
    recipe = create_recipe(args)

    env = SimEnv(num_clients=args.n_clients)
    run = recipe.execute(env)
    result = run.get_result()
    print()
    # SimEnv raises on execution failure; a returned result confirms completion.
    if result is None:
        raise RuntimeError("Simulation did not return a result.")
    print("Simulation completed successfully.")
    print("Result can be found in :", result)
    print()
    return result


if __name__ == "__main__":
    main()
