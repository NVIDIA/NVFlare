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

from model import MODEL_SEED, create_model
from prepare_data import add_dataset_arguments

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
    parser.add_argument(
        "--n_clients", type=int, default=DEFAULT_NUM_CLIENTS, help="Number of participating clients (default: 2)."
    )
    parser.add_argument(
        "--num_rounds", type=int, default=DEFAULT_NUM_ROUNDS, help="Number of federated training rounds (default: 3)."
    )

    add_dataset_arguments(parser)
    parser.add_argument("--seed", type=int, default=MODEL_SEED)
    parser.add_argument("--torch_threads", type=int, default=None, help="PyTorch CPU threads per client (default: 1).")
    parser.add_argument(
        "--resume_training",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Restore the PyTorch checkpoint between task workers (task mode defaults to enabled).",
    )
    parser.add_argument(
        "--execution_lifetime",
        choices=("job", "task"),
        default="job",
        help="Keep client compute for the job, or start a fresh CPU worker for each task.",
    )

    return parser


def create_recipe(args):
    seed = getattr(args, "seed", MODEL_SEED)
    train_args = ["--dataset", args.dataset]
    if seed != MODEL_SEED:
        train_args.extend(("--seed", str(seed)))
    if getattr(args, "resume_training", None) is not None:
        if getattr(args, "execution_lifetime", "job") != "task":
            raise ValueError("--resume_training requires --execution_lifetime task; job mode keeps state in memory")
        train_args.append("--resume_training" if args.resume_training else "--no-resume_training")
    if getattr(args, "torch_threads", None) is not None:
        train_args.extend(("--torch_threads", str(args.torch_threads)))
    if args.dataset == "cifar10":
        train_args.extend(("--data_root", args.data_root))

    recipe = FedAvgRecipe(
        name="hello-pt",
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        # Model can be specified as class instance or dict config:
        model=create_model(seed=seed),
        # Alternative: model={"class_path": "model.SimpleNetwork", "args": {}},
        # For pre-trained weights: initial_ckpt="/server/path/to/pretrained.pt",
        train_script="client_task.py" if getattr(args, "execution_lifetime", "job") == "task" else "client.py",
        # Pass argv directly so a client-local path containing spaces, quotes,
        # or apostrophes keeps its exact argument boundary without shell parsing.
        train_args=train_args,
        execution_lifetime=getattr(args, "execution_lifetime", "job"),
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
    # SimEnv raises on execution failure; a returned result confirms completion.
    if result is None:
        raise RuntimeError("Simulation did not return a result.")
    print("Simulation completed successfully.")
    print("Result can be found in :", result)
    print()
    return result


if __name__ == "__main__":
    main()
