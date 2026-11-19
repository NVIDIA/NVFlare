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

"""Run CIFAR-10 FedBuff with simulator-owned fixed or timed replay."""

import argparse
import math
from pathlib import Path

from pt_async_cifar10_reproducible.async_aggregator import Cifar10ModelLoader, Cifar10Observer
from pt_async_cifar10_reproducible.model import PTModelOps
from pt_async_cifar10_reproducible.seed_policy import SeedPolicy
from pt_async_cifar10_reproducible.trainer import Cifar10Trainer

from nvflare.collab import FedBuffRecipe
from nvflare.recipe import SimEnv

JOB_NAME = "collab_pt_async_cifar10_reproducible"
EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = "/tmp/cifar10"
DEFAULT_SPLIT_PREFIX = "/tmp/cifar10_splits/cifar10_fedavg"


def prepare_splits(num_logical_clients: int, alpha: float, seed: int) -> str:
    from pt_async_cifar10_reproducible.cifar10_data import split_and_save

    return split_and_save(DEFAULT_SPLIT_PREFIX, num_logical_clients, alpha, seed)


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-clients", type=int, default=2, help="Physical simulator workers")
    parser.add_argument("--num-logical-clients", type=int, default=8, help="CIFAR shards")
    parser.add_argument("--concurrency", type=int, default=None, help="Logical slots (default: all shards)")
    parser.add_argument("--buffer-size", type=int, default=4)
    parser.add_argument("--min-open-slots", type=int, default=1)
    parser.add_argument("--num-rounds", type=int, default=100, help="Global model publications")
    parser.add_argument("--plan-mode", choices=("fixed", "timed"), default="fixed")
    parser.add_argument("--horizon", type=float, help="Timed-mode virtual stop; defaults after all uploads")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--setup-seed", type=int, default=10)
    parser.add_argument("--run-seed", type=int, default=10)
    parser.add_argument("--local-batch-size", type=int, default=64)
    parser.add_argument("--local-lr", type=float, default=0.05)
    parser.add_argument("--aggregation-epochs", type=int, default=4)
    parser.add_argument("--total-client-rounds", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--server-lr", type=float, default=1.0)
    parser.add_argument("--eval-batch-size", type=int, default=300)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--call-timeout", type=float, default=3600.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--workspace-root", default="/tmp/nvflare/collab")
    return parser


def _validate_args(args) -> int:
    concurrency = args.num_logical_clients if args.concurrency is None else args.concurrency
    if args.num_clients < 1 or args.num_logical_clients < 1 or not 1 <= concurrency <= args.num_logical_clients:
        raise ValueError("physical and logical client counts must be positive; concurrency must fit logical clients")
    if (
        min(args.num_rounds, args.buffer_size, args.aggregation_epochs, args.local_batch_size, args.total_client_rounds)
        < 1
    ):
        raise ValueError("rounds, buffer size, batch size, and training epochs must be positive")
    if not 1 <= args.min_open_slots <= concurrency:
        raise ValueError("min_open_slots must be between one and concurrency")
    if args.num_workers < 0 or args.eval_batch_size < 1 or args.checkpoint_interval < 0:
        raise ValueError("worker, evaluation, or checkpoint setting is invalid")
    if not all(math.isfinite(x) and x > 0 for x in (args.alpha, args.local_lr, args.server_lr, args.call_timeout)):
        raise ValueError("alpha, learning rates, and call_timeout must be finite and positive")
    if args.horizon is not None and (args.plan_mode != "timed" or not math.isfinite(args.horizon) or args.horizon <= 0):
        raise ValueError("a positive finite horizon is supported only for timed plans")
    SeedPolicy(args.run_seed)
    return concurrency


def make_recipe(args) -> FedBuffRecipe:
    concurrency = _validate_args(args)
    recipe = FedBuffRecipe(
        name=JOB_NAME,
        client=Cifar10Trainer(
            data_root=DEFAULT_DATA_ROOT,
            local_batch_size=args.local_batch_size,
            local_lr=args.local_lr,
            device=args.device,
            train_idx_root=args.train_idx_root,
            aggregation_epochs=args.aggregation_epochs,
            total_client_rounds=args.total_client_rounds,
            num_workers=args.num_workers,
            num_logical_clients=args.num_logical_clients,
            run_seed=args.run_seed,
        ),
        model_loader=Cifar10ModelLoader(setup_seed=args.setup_seed, run_seed=args.run_seed),
        observer=Cifar10Observer(
            data_root=DEFAULT_DATA_ROOT,
            num_rounds=args.num_rounds,
            buffer_size=args.buffer_size,
            device=args.device,
            eval_batch_size=args.eval_batch_size,
            setup_seed=args.setup_seed,
            checkpoint_interval=args.checkpoint_interval,
        ),
        model_ops=PTModelOps(),
        num_rounds=args.num_rounds,
        concurrency=concurrency,
        buffer_size=args.buffer_size,
        min_open_slots=args.min_open_slots,
        server_lr=args.server_lr,
        weight_metric="num_steps",
        call_timeout=args.call_timeout,
        min_clients=concurrency,
        sync_task_timeout=600,
        max_call_threads_for_server=max(8, args.num_clients),
        max_call_threads_for_client=max(100, args.num_clients),
    )
    recipe.set_client_prop("data_root", DEFAULT_DATA_ROOT)
    # Bundle the complete example for both the exported job and catalog download.
    recipe.add_server_file(str(EXAMPLE_DIR))
    recipe.add_client_file(str(EXAMPLE_DIR))
    return recipe


def make_plan(args) -> dict:
    concurrency = _validate_args(args)
    total = args.num_rounds * args.buffer_size
    policy = SeedPolicy(args.run_seed)
    execution = {}
    if args.plan_mode == "fixed":
        order = list(range(total))
        if min(concurrency, total) >= 2:
            order[0], order[1] = order[1], order[0]
        # Give the next reservation the logical client just accepted.
        logical = [None] * total
        for accepted_position, assignment_id in enumerate(order):
            logical[assignment_id] = f"site-{accepted_position % args.num_logical_clients + 1}"
        execution["arrival_sequence"] = order
    else:
        logical = [f"site-{i % args.num_logical_clients + 1}" for i in range(total)]
        profile = {phase: {"distribution": "constant", "value": 1} for phase in ("download", "train", "upload")}
        execution["timing"] = {
            "starts": [4 * (i // concurrency) for i in range(total)],
            "seed": policy.seed("delays"),
            "profiles": {name: profile for name in sorted(set(logical))},
            "horizon": args.horizon if args.horizon is not None else 4 * math.ceil(total / concurrency),
        }
    execution["logical_sequence"] = logical
    execution["train_seeds"] = [
        policy.seed("train", assignment_id=i, logical_name=name) for i, name in enumerate(logical)
    ]
    return {"format_version": 1, "async_execution": execution}


def main():
    args = define_parser().parse_args()
    _validate_args(args)
    args.train_idx_root = prepare_splits(args.num_logical_clients, args.alpha, args.split_seed)
    run = make_recipe(args).execute(
        SimEnv(num_clients=args.num_clients, workspace_root=args.workspace_root, reprod=True, plan=make_plan(args))
    )
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
