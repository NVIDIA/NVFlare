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

"""Run matched NVIDIA FLARE CIFAR-10 experiments for adaptive aggregation.

All methods use the same Dirichlet training partition and initial server-model
seed for a given ``(n_clients, alpha, seed)``. After training, every method is
evaluated by the same post-training evaluator on the full CIFAR-10 test set and
on identical site-specific test partitions.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
PROJECT_SRC = PROJECT_DIR / "src"
CIFAR_PT_DIR = REPO_ROOT / "examples" / "advanced" / "cifar10" / "pt"
CIFAR_SRC = CIFAR_PT_DIR / "src"
CIFAR_SIM = CIFAR_PT_DIR / "cifar10-sim"
EVAL_DIR = Path(__file__).resolve().parent

for path in (str(PROJECT_SRC), str(CIFAR_SRC), str(EVAL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ["PYTHONPATH"] = os.pathsep.join([str(PROJECT_SRC), str(CIFAR_SRC), os.environ.get("PYTHONPATH", "")])

from adaptive_hetero.model_aggregator import AdaptiveHeterogeneityModelAggregator  # noqa: E402
from data.cifar10_data_split import split_and_save  # noqa: E402
from eval_split import create_eval_splits  # noqa: E402
from evaluate_result import evaluate_workspace  # noqa: E402
from model import ModerateCNN  # noqa: E402

from nvflare.apis.dxo import DataKind  # noqa: E402
from nvflare.app_opt.pt.recipes import FedAvgRecipe, FedCERecipe, FedOptRecipe, FedProxRecipe  # noqa: E402
from nvflare.app_opt.pt.recipes.scaffold import ScaffoldRecipe  # noqa: E402
from nvflare.client.config import TransferType  # noqa: E402
from nvflare.recipe import SimEnv  # noqa: E402

METHODS = ("fedavg", "fedopt", "fedprox", "scaffold", "fedce", "adaptive")


def _client_script(method: str) -> str:
    if method == "adaptive":
        return str(EVAL_DIR / "adaptive_client.py")
    if method == "fedce":
        return str(EVAL_DIR / "fedce_client.py")
    directory = {
        "fedavg": "cifar10_fedavg",
        "fedopt": "cifar10_fedopt",
        "fedprox": "cifar10_fedprox",
        "scaffold": "cifar10_scaffold",
    }[method]
    return str(CIFAR_SIM / directory / "client.py")


def _round_clients(n_clients: int, participation_rate: float) -> int:
    if not 0.0 < participation_rate <= 1.0:
        raise ValueError("participation_rate must be in (0, 1]")
    return max(2, min(n_clients, int(round(n_clients * participation_rate))))


def _common_train_args(args, train_idx_root: str) -> str:
    return (
        f"--train_idx_root {train_idx_root} --num_workers {args.num_workers} --lr {args.lr} "
        f"--batch_size {args.batch_size} --aggregation_epochs {args.aggregation_epochs}"
    )


def _local_metric_train_args(args, train_idx_root: str, eval_idx_root: str) -> str:
    return f"{_common_train_args(args, train_idx_root)} --eval_idx_root {eval_idx_root} --seed {args.seed}"


def _build_recipe(args, train_idx_root: str, eval_idx_root: str, round_clients: int):
    # This controls the server's initial model and the Dirichlet partition seed.
    # NVIDIA's stock CIFAR baseline client scripts do not expose a client RNG
    # argument, so local augmentation/minibatch stochasticity remains part of
    # run-to-run variance and is not claimed to be identically paired.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = ModerateCNN()
    common = _common_train_args(args, train_idx_root)
    name = (
        f"adaptive_hetero_cifar10_{args.method}_alpha{args.alpha:g}_seed{args.seed}_"
        f"p{args.participation_rate:g}"
    )

    if args.method == "fedavg":
        return FedAvgRecipe(
            name=name,
            min_clients=round_clients,
            num_rounds=args.num_rounds,
            model=model,
            train_script=_client_script(args.method),
            train_args=common,
            aggregator_data_kind=DataKind.WEIGHT_DIFF,
        )
    if args.method == "fedopt":
        if round_clients != args.n_clients:
            raise ValueError("FedOpt is included as a full-participation reference; use another method for partial runs")
        return FedOptRecipe(
            name=name,
            min_clients=args.n_clients,
            num_rounds=args.num_rounds,
            model=model,
            train_script=_client_script(args.method),
            train_args=common,
            optimizer_args={"path": "torch.optim.SGD", "args": {"lr": 1.0, "momentum": 0.6}},
            device="cpu" if args.gpu_config is None else "cuda:0",
        )
    if args.method == "fedprox":
        return FedProxRecipe(
            name=name,
            min_clients=round_clients,
            num_rounds=args.num_rounds,
            model=model,
            train_script=_client_script(args.method),
            train_args=common,
            aggregator_data_kind=DataKind.WEIGHT_DIFF,
            fedprox_mu=args.fedprox_mu,
        )
    if args.method == "scaffold":
        return ScaffoldRecipe(
            name=name,
            min_clients=round_clients,
            num_rounds=args.num_rounds,
            model=model,
            train_script=_client_script(args.method),
            train_args=common,
        )
    if args.method == "fedce":
        return FedCERecipe(
            name=name,
            min_clients=round_clients,
            num_rounds=args.num_rounds,
            model=model,
            train_script=_client_script(args.method),
            train_args=_local_metric_train_args(args, train_idx_root, eval_idx_root),
            fedce_mode=args.fedce_mode,
        )

    aggregator = AdaptiveHeterogeneityModelAggregator(
        sample_exponent=args.sample_exponent,
        representation_exponent=args.representation_exponent,
        metric_prior_strength=args.metric_prior_strength,
        max_blend_factor=args.max_blend_factor,
        activation_warmup_rounds=args.activation_warmup_rounds,
        activation_patience=args.activation_patience,
        require_stable_cohort=not args.allow_changing_cohort_evidence,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
    )
    return FedAvgRecipe(
        name=name,
        min_clients=round_clients,
        num_rounds=args.num_rounds,
        model=model,
        train_script=_client_script(args.method),
        train_args=_local_metric_train_args(args, train_idx_root, eval_idx_root),
        aggregator=aggregator,
        aggregator_data_kind=DataKind.WEIGHT_DIFF,
        params_transfer_type=TransferType.DIFF,
    )


def _append_jsonl(path: str, row: dict):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def main(args):
    if args.n_clients < 2:
        raise ValueError("CIFAR-10 evaluation requires at least two clients")
    if args.alpha <= 0.0:
        raise ValueError("alpha must be greater than zero")

    round_clients = _round_clients(args.n_clients, args.participation_rate)
    split_prefix = os.path.join(args.split_root, "shared_train")
    train_idx_root = split_and_save(
        split_dir_prefix=split_prefix,
        num_sites=args.n_clients,
        alpha=args.alpha,
        seed=args.seed,
    )
    eval_idx_root = os.path.join(
        args.split_root,
        f"shared_eval_{args.n_clients}sites_alpha{args.alpha:.2f}_seed{args.seed}",
    )
    create_eval_splits(train_idx_root, eval_idx_root, args.n_clients, args.seed)

    recipe = _build_recipe(args, train_idx_root, eval_idx_root, round_clients)
    workspace_root = os.path.abspath(args.workspace_root)
    env = SimEnv(
        num_clients=args.n_clients,
        num_threads=args.num_threads or args.n_clients,
        gpu_config=args.gpu_config,
        workspace_root=workspace_root,
    )
    run = recipe.execute(env)
    status = str(run.get_status())
    if "COMPLETED" not in status.upper():
        raise RuntimeError(f"CIFAR-10 evaluation did not complete successfully: {status}")

    job_workspace = os.path.join(workspace_root, recipe.name)
    evaluation = evaluate_workspace(
        workspace=job_workspace,
        eval_idx_root=eval_idx_root,
        n_clients=args.n_clients,
        batch_size=args.eval_batch_size,
        num_workers=args.eval_num_workers,
        device_name=args.eval_device,
    )
    record = {
        "method": args.method,
        "alpha": args.alpha,
        "seed": args.seed,
        "n_clients": args.n_clients,
        "round_clients": round_clients,
        "participation_rate": args.participation_rate,
        "num_rounds": args.num_rounds,
        "aggregation_epochs": args.aggregation_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_idx_root": train_idx_root,
        "eval_idx_root": eval_idx_root,
        "workspace": job_workspace,
        "job_name": recipe.name,
        "status": status,
        "result": str(run.get_result()),
        "pairing_scope": "dirichlet_split_and_server_initialization",
        **evaluation,
    }
    print("CIFAR10_EVAL_RESULT " + json.dumps(record, sort_keys=True), flush=True)
    if args.results_jsonl:
        _append_jsonl(args.results_jsonl, record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--n_clients", type=int, default=8)
    parser.add_argument("--participation_rate", type=float, default=1.0)
    parser.add_argument("--num_rounds", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_threads", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--aggregation_epochs", type=int, default=4)
    parser.add_argument("--fedprox_mu", type=float, default=0.01)
    parser.add_argument("--fedce_mode", choices=("plus", "times"), default="plus")
    parser.add_argument("--gpu_config", type=str, default=None)
    parser.add_argument("--workspace_root", default="/tmp/nvflare/adaptive_hetero_cifar10")
    parser.add_argument("--split_root", default="/tmp/cifar10_splits/adaptive_hetero_eval")
    parser.add_argument("--results_jsonl", default=None)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--eval_num_workers", type=int, default=0)
    parser.add_argument("--eval_device", default=None)

    parser.add_argument("--sample_exponent", type=float, default=0.65)
    parser.add_argument("--representation_exponent", type=float, default=0.70)
    parser.add_argument("--metric_prior_strength", type=float, default=100.0)
    parser.add_argument("--max_blend_factor", type=float, default=0.20)
    parser.add_argument("--activation_warmup_rounds", type=int, default=3)
    parser.add_argument("--activation_patience", type=int, default=2)
    parser.add_argument("--min_weight", type=float, default=0.0)
    parser.add_argument("--max_weight", type=float, default=1.0)
    parser.add_argument("--allow_changing_cohort_evidence", action="store_true")
    main(parser.parse_args())
