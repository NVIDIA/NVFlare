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
"""Run the Hello PyTorch application across simulation, POC, or production."""

import argparse
import sys
from pathlib import Path

# Reuse the beginner example's real model, client, and data code. This example
# changes orchestration and execution environment, not the learning application.
HELLO_PT_DIR = Path(__file__).resolve().parents[2] / "hello-world" / "hello-pt"
sys.path.insert(0, str(HELLO_PT_DIR))

from model import create_model  # noqa: E402
from prepare_data import DATASET_CHOICES, DATASET_PATH, DEFAULT_DATASET  # noqa: E402

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import PocEnv, ProdEnv, SimEnv, add_experiment_tracking, add_final_global_evaluation
from nvflare.recipe.utils import add_cross_site_evaluation

DEFAULT_NUM_CLIENTS = 2
DEFAULT_NUM_ROUNDS = 3
SUCCESS_STATUS = "FINISHED:COMPLETED"
LEGACY_SUCCESS_STATUS = "FINISHED_OK"
SUCCESS_STATUSES = {SUCCESS_STATUS, LEGACY_SUCCESS_STATUS}
EXPORT_HELP = """NVFlare Recipe export options:
  --export                    Export the job instead of running it.
  --export-dir EXPORT_DIR     Parent directory for the exported job (default: ./fl_job).
"""


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continue Hello PyTorch through simulation, POC, or production.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXPORT_HELP,
    )
    parser.add_argument("--n_clients", type=int, default=DEFAULT_NUM_CLIENTS)
    parser.add_argument("--num_rounds", type=int, default=DEFAULT_NUM_ROUNDS)
    parser.add_argument(
        "--env",
        choices=("sim", "poc", "prod"),
        default="sim",
        help="Execution environment: local simulation, local POC processes, or a provisioned production system.",
    )
    parser.add_argument(
        "--startup-kit",
        default=None,
        help="Path to an admin startup kit. Required with --env prod.",
    )
    parser.add_argument(
        "--username",
        default=None,
        help="Production admin identity. Must match the startup kit; defaults to admin@nvidia.com.",
    )

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

    # Defaults for these optional overrides remain owned by the shared client.
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--evaluation",
        choices=("none", "final", "cross-site"),
        default="final",
        help="Post-training evaluation: none, final global model only, or all submitted client and server models.",
    )
    parser.add_argument(
        "--experiment_tracking",
        choices=("none", "tensorboard"),
        default="none",
        help="Optional server-side tracking receiver. TensorBoard requires the tensorboard package.",
    )
    parser.add_argument("--enable_log_streaming", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--launch_external_process",
        action="store_true",
        help="Run the shared client script in a separate subprocess instead of in-process.",
    )
    parser.add_argument(
        "--client_memory_gc_rounds",
        type=int,
        default=0,
        help="Release model parameters and run GC every N rounds. 0 disables this behavior.",
    )
    return parser


def parse_args(argv=None):
    parser = define_parser()
    args = parser.parse_args(argv)
    if args.env == "prod" and not args.startup_kit:
        parser.error("--startup-kit is required with --env prod")
    if args.env != "prod" and args.startup_kit:
        parser.error("--startup-kit can only be used with --env prod")
    if args.env != "prod" and args.username:
        parser.error("--username can only be used with --env prod")
    return args


def create_recipe(args):
    train_args = ["--dataset", args.dataset]
    if args.dataset == "cifar10":
        train_args.extend(("--data_root", args.data_root))
    for name in ("batch_size", "epochs", "learning_rate", "num_workers"):
        value = getattr(args, name)
        if value is not None:
            train_args.extend((f"--{name}", str(value)))

    recipe = FedAvgRecipe(
        name="hello-pt",
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        model=create_model(),
        train_script=str(HELLO_PT_DIR / "client.py"),
        # Preserve each client argument exactly across all execution environments.
        train_args=train_args,
        launch_external_process=args.launch_external_process,
        client_memory_gc_rounds=args.client_memory_gc_rounds,
    )

    if args.experiment_tracking != "none":
        add_experiment_tracking(recipe, tracking_type=args.experiment_tracking)

    if args.evaluation == "final":
        add_final_global_evaluation(recipe)
    elif args.evaluation == "cross-site":
        add_cross_site_evaluation(recipe)

    if args.enable_log_streaming:
        recipe.enable_log_streaming()
    return recipe


def create_environment(args):
    """Construct the selected environment without changing the application recipe."""
    if args.env == "sim":
        return SimEnv(num_clients=args.n_clients)
    if args.env == "poc":
        return PocEnv(num_clients=args.n_clients)
    if args.username:
        return ProdEnv(startup_kit_location=args.startup_kit, username=args.username)
    return ProdEnv(startup_kit_location=args.startup_kit)


def main(argv=None):
    args = parse_args(argv)
    recipe = create_recipe(args)
    env = create_environment(args)

    try:
        run = recipe.execute(env)
        # PocEnv downloads into its own workspace. Retain that workspace after
        # success so the result and service logs at the printed path still exist.
        result = run.get_result(clean_up=args.env != "poc")
        status = None if args.env == "sim" else run.get_status()
        if result is None:
            raise RuntimeError("Job monitoring did not return a result. Review the execution-environment logs.")
        if args.env != "sim" and status not in SUCCESS_STATUSES:
            raise RuntimeError(f"Job completed with unsuccessful status: {status}")
    except (Exception, KeyboardInterrupt):
        # Only delete a POC workspace that this invocation created or replaced.
        # A provisioning failure can otherwise leave an older retained result in
        # the shared path, and that prior workspace must be preserved.
        if args.env == "poc" and env.workspace_owned:
            env.stop(clean_up=True)
        raise

    print()
    if args.env == "sim":
        print("Simulation completed successfully.")
    else:
        print("Job Status is:", status)
    print("Result can be found in:", result)
    print()
    return result


if __name__ == "__main__":
    main()
