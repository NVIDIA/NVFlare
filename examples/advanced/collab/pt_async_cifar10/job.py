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

    ./prepare_data.sh
    python job.py
"""

import argparse
from pathlib import Path

from async_aggregator import Cifar10AsyncAggregator
from trainer import Cifar10Trainer

from nvflare.collab import CollabRecipe
from nvflare.recipe import SimEnv

JOB_NAME = "collab_pt_async_cifar10"
EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = "/tmp/cifar10"
DEFAULT_SPLIT_PREFIX = "/tmp/cifar10_splits/cifar10_fedavg"


def prepare_splits(num_clients: int, alpha: float, seed: int) -> str:
    """Use the splitter shared by the CIFAR-10 simulator examples."""
    import sys

    cifar10_src = EXAMPLE_DIR.parents[1] / "cifar10" / "pt" / "src"
    sys.path.insert(0, str(cifar10_src))
    from data.cifar10_data_split import split_and_save

    return split_and_save(
        split_dir_prefix=DEFAULT_SPLIT_PREFIX,
        num_sites=num_clients,
        alpha=alpha,
        seed=seed,
    )


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Asynchronous CIFAR-10 training with the Collab API")
    parser.add_argument("--num-clients", type=int, default=8, help="Number of physical simulator clients (default: 8)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet data heterogeneity parameter")
    parser.add_argument("--split-seed", type=int, default=0, help="CIFAR-10 partition seed")
    parser.add_argument(
        "--num-active-jobs",
        type=int,
        default=None,
        help="K: client-job slots kept active (default: all physical clients)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=4,
        help="B: returned updates aggregated into each new global model (default: 4)",
    )
    parser.add_argument(
        "--min-open-slots",
        type=int,
        default=1,
        help="O: open slots accumulated before dispatch; O=1 is FedBuff (default: 1)",
    )
    parser.add_argument("--num-rounds", type=int, default=100, help="Number of global aggregations (default: 100)")
    parser.add_argument("--local-batch-size", type=int, default=64)
    parser.add_argument("--local-lr", type=float, default=0.05)
    parser.add_argument("--aggregation-epochs", type=int, default=4)
    parser.add_argument("--total-client-rounds", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--server-lr", type=float, default=1.0)
    parser.add_argument("--eval-batch-size", type=int, default=300)
    parser.add_argument("--call-timeout", type=float, default=3600.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--setup-seed", type=int, default=10)
    parser.add_argument("--run-seed", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--workspace-root", default="/tmp/nvflare/collab")
    return parser


def make_recipe(args) -> CollabRecipe:
    if args.num_clients < 1:
        raise ValueError("--num-clients must be >= 1")
    if args.alpha <= 0:
        raise ValueError("--alpha must be > 0")
    if args.num_rounds < 1 or args.local_batch_size < 1 or args.aggregation_epochs < 1:
        raise ValueError("--num-rounds, --local-batch-size, and --aggregation-epochs must be >= 1")
    if args.total_client_rounds < 1 or args.num_workers < 0:
        raise ValueError("--total-client-rounds must be >= 1 and --num-workers must be >= 0")
    if args.local_lr <= 0 or args.server_lr <= 0 or args.call_timeout <= 0:
        raise ValueError("--local-lr, --server-lr, and --call-timeout must be > 0")
    num_active_jobs = args.num_clients if args.num_active_jobs is None else args.num_active_jobs
    if not 1 <= num_active_jobs <= args.num_clients:
        raise ValueError("--num-active-jobs must be between 1 and --num-clients")
    if args.buffer_size < 1:
        raise ValueError("--buffer-size must be >= 1")
    if not 1 <= args.min_open_slots <= num_active_jobs:
        raise ValueError("--min-open-slots must be between 1 and --num-active-jobs")

    device = None if args.device == "auto" else args.device

    server = Cifar10AsyncAggregator(
        data_root=args.data_root,
        num_rounds=args.num_rounds,
        num_active_jobs=num_active_jobs,
        buffer_size=args.buffer_size,
        min_open_slots=args.min_open_slots,
        call_timeout=args.call_timeout,
        device=device,
        eval_batch_size=args.eval_batch_size,
        server_lr=args.server_lr,
        setup_seed=args.setup_seed,
        run_seed=args.run_seed,
        checkpoint_interval=args.checkpoint_interval,
    )
    client = Cifar10Trainer(
        data_root=args.data_root,
        local_batch_size=args.local_batch_size,
        local_lr=args.local_lr,
        device=device,
        train_idx_root=args.train_idx_root,
        aggregation_epochs=args.aggregation_epochs,
        total_client_rounds=args.total_client_rounds,
        num_workers=args.num_workers,
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
    recipe.add_client_file(str(EXAMPLE_DIR / "model.py"))
    return recipe


def main():
    args = define_parser().parse_args()
    args.data_root = DEFAULT_DATA_ROOT
    if args.alpha <= 0:
        raise ValueError("--alpha must be > 0")
    args.train_idx_root = prepare_splits(args.num_clients, args.alpha, args.split_seed)
    recipe = make_recipe(args)
    num_active_jobs = args.num_clients if args.num_active_jobs is None else args.num_active_jobs

    if args.min_open_slots == args.buffer_size == num_active_jobs:
        schedule = "synchronous FedAvg-style"
    elif args.min_open_slots == 1:
        schedule = "FedBuff asynchronous"
    elif args.min_open_slots == args.buffer_size:
        schedule = "minimum-response synchronous"
    else:
        schedule = "custom buffered asynchronous"

    print("=" * 80)
    print("CIFAR-10 COLLAB FEDBUFF")
    print(f"  Physical simulator clients: {args.num_clients}")
    print(f"  Active jobs (K): {num_active_jobs}")
    print(f"  Update buffer (B): {args.buffer_size}")
    print(f"  Dispatch threshold (O): {args.min_open_slots}")
    print(f"  Schedule: {schedule}")
    print(f"  Global aggregations: {args.num_rounds}")
    print(f"  Data root: {args.data_root}")
    print("=" * 80)

    run = recipe.execute(SimEnv(num_clients=args.num_clients, workspace_root=args.workspace_root))
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
