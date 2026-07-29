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

"""Run asynchronous CIFAR-10 training with ``CollabRecipe``.

From the ``examples/advanced/collab/pt_async_cifar10`` directory:

    python prepare_data.py
    python -m pt_async_cifar10
"""

import argparse
import logging
from pathlib import Path

from async_aggregator import Cifar10AsyncAggregator
from data import validate_prepared_data
from prepare_data import DEFAULT_DATA_ROOT
from trainer import Cifar10Trainer

from nvflare.collab import CollabRecipe, simple_logging
from nvflare.recipe import SimEnv

JOB_NAME = "collab_pt_async_cifar10"
EXAMPLE_DIR = Path(__file__).resolve().parent


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Asynchronous CIFAR-10 training with the Collab API")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Prepared data root from prepare_data.py")
    parser.add_argument("--num-clients", type=int, default=2, help="Number of physical simulator clients")
    parser.add_argument(
        "--clients-per-round",
        type=int,
        default=None,
        help="Physical clients sampled per round (default: all physical clients)",
    )
    parser.add_argument(
        "--min-response-clients",
        type=int,
        default=None,
        help="Successful responses required per round (default: all selected clients)",
    )
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--local-iters", type=int, default=25)
    parser.add_argument("--local-batch-size", type=int, default=32)
    parser.add_argument("--local-lr", type=float, default=0.0003)
    parser.add_argument("--server-lr", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=300)
    parser.add_argument("--call-timeout", type=float, default=3600.0)
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=0,
        help="Maximum in-flight client calls; 0 dispatches to every selected client",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--setup-seed", type=int, default=10)
    parser.add_argument("--run-seed", type=int, default=10)
    parser.add_argument("--workspace-root", default="/tmp/nvflare/collab")
    return parser


def make_recipe(args, manifest: dict) -> CollabRecipe:
    if args.num_clients < 1:
        raise ValueError("--num-clients must be >= 1")
    if args.num_clients > manifest["num_clients"]:
        raise ValueError(
            f"--num-clients ({args.num_clients}) exceeds the {manifest['num_clients']} prepared logical clients"
        )
    if args.num_rounds < 1 or args.local_iters < 1 or args.local_batch_size < 1:
        raise ValueError("--num-rounds, --local-iters, and --local-batch-size must be >= 1")
    if args.local_lr <= 0 or args.server_lr <= 0 or args.call_timeout <= 0:
        raise ValueError("--local-lr, --server-lr, and --call-timeout must be > 0")
    if args.max_parallel < 0:
        raise ValueError("--max-parallel must be >= 0")

    clients_per_round = args.num_clients if args.clients_per_round is None else args.clients_per_round
    if not 1 <= clients_per_round <= args.num_clients:
        raise ValueError("--clients-per-round must be between 1 and --num-clients")
    min_response_clients = clients_per_round if args.min_response_clients is None else args.min_response_clients
    if not 1 <= min_response_clients <= clients_per_round:
        raise ValueError("--min-response-clients must be between 1 and --clients-per-round")

    device = None if args.device == "auto" else args.device

    server = Cifar10AsyncAggregator(
        data_root=args.data_root,
        logical_num_clients=manifest["num_clients"],
        num_rounds=args.num_rounds,
        clients_per_round=clients_per_round,
        min_response_clients=min_response_clients,
        call_timeout=args.call_timeout,
        max_parallel=args.max_parallel,
        device=device,
        eval_interval=args.eval_interval,
        eval_batch_size=args.eval_batch_size,
        server_lr=args.server_lr,
        setup_seed=args.setup_seed,
        run_seed=args.run_seed,
    )
    client = Cifar10Trainer(
        data_root=args.data_root,
        local_batch_size=args.local_batch_size,
        local_iters=args.local_iters,
        local_lr=args.local_lr,
        device=device,
    )
    recipe = CollabRecipe(
        job_name=JOB_NAME,
        server=server,
        client=client,
        min_clients=args.num_clients,
        sync_task_timeout=600,
        max_call_threads_for_server=max(100, args.num_clients),
        max_call_threads_for_client=max(100, args.num_clients),
    )
    recipe.set_client_prop("data_root", args.data_root)
    recipe.add_server_file(str(EXAMPLE_DIR / "model.py"))
    recipe.add_client_file(str(EXAMPLE_DIR / "data.py"))
    recipe.add_client_file(str(EXAMPLE_DIR / "model.py"))
    return recipe


def main():
    args = define_parser().parse_args()
    args.data_root = str(Path(args.data_root).expanduser().resolve())
    manifest = validate_prepared_data(args.data_root)
    simple_logging(logging.INFO)
    recipe = make_recipe(args, manifest)
    clients_per_round = args.num_clients if args.clients_per_round is None else args.clients_per_round

    print("=" * 80)
    print("CIFAR-10 COLLAB ASYNC")
    print(f"  Prepared logical clients: {manifest['num_clients']}")
    print(f"  Physical simulator clients: {args.num_clients}")
    print(f"  Clients per round: {clients_per_round}")
    print(f"  Rounds: {args.num_rounds}")
    print(f"  Data root: {args.data_root}")
    print("=" * 80)

    run = recipe.execute(SimEnv(num_clients=args.num_clients, workspace_root=args.workspace_root))
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
