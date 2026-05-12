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
Merged NVFlare Auto-FL baseline.

Provenance:
- NVFlare execution patterns are adapted from hello-pt, the CIFAR-10 custom-aggregator example,
  and the CIFAR-10 scaffold client path.
- Repo-level orchestration is inspired by the public karpathy/autoresearch operating model,
  especially the use of program.md as the main agent control plane.
"""

import argparse
import os
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from tasks.shared.custom_aggregators import (
    FedAdamAggregator,
    FedAvgAggregator,
    FedAvgMAggregator,
    MedianAggregator,
    ScaffoldAggregator,
    WeightedAggregator,
)
from data.cifar10_data_split import split_and_save
from model import (
    DEFAULT_MAX_MODEL_PARAMS,
    DEFAULT_MODEL_ARCH,
    available_model_architectures,
    build_model,
    count_parameters,
)

from nvflare.apis.dxo import DataKind
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
from nvflare.app_opt.pt.file_model_locator import PTFileModelLocator
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking


def define_parser():
    parser = argparse.ArgumentParser(
        description="Merged NVFlare baseline: hello-pt style Client API + CIFAR10 custom aggregators"
    )
    parser.add_argument("--n_clients", type=int, default=8)
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--train_script", type=str, default="client.py")
    parser.add_argument("--key_metric", type=str, default="accuracy")
    parser.add_argument("--name", type=str, default=None)

    parser.add_argument("--launch_external_process", action="store_true")
    parser.add_argument("--client_memory_gc_rounds", type=int, default=0)
    parser.add_argument("--cross_site_eval", action="store_true")

    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model_arch",
        type=str,
        default=DEFAULT_MODEL_ARCH,
        choices=available_model_architectures(),
        help="Registered model architecture for the server seed model and every client.",
    )
    parser.add_argument(
        "--max_model_params",
        type=int,
        default=DEFAULT_MAX_MODEL_PARAMS,
        help="Maximum allowed model parameters for architecture-search campaigns. Use 0 to disable.",
    )
    parser.add_argument("--aggregation_epochs", type=int, default=4)
    parser.add_argument(
        "--local_train_steps",
        type=int,
        default=0,
        help="Exact optimizer steps per client per round. Use 0 for epoch-based training with --aggregation_epochs.",
    )
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1024,
        help="Batch size for validation/evaluation tasks. Does not change local training batch size.",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--no_deterministic_training",
        action="store_true",
        help="Disable deterministic PyTorch and DataLoader seeding in client training.",
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--no_lr_scheduler", action="store_true")
    parser.add_argument("--cosine_lr_eta_min_factor", type=float, default=0.01)
    parser.add_argument("--evaluate_local", action="store_true")
    parser.add_argument(
        "--eval_global_every_round",
        action="store_true",
        help="Ask clients to evaluate the received global model on every training round for telemetry.",
    )
    parser.add_argument(
        "--final_eval_clients",
        type=str,
        default="site-1",
        help="Comma-separated clients used for final global-model evaluation, or 'all'. Defaults to site-1.",
    )
    parser.add_argument(
        "--fedproxloss_mu",
        type=float,
        default=0.0,
        help="FedProx proximal-loss coefficient. 0 disables the proximal term.",
    )

    parser.add_argument(
        "--aggregator",
        type=str,
        default="weighted",
        choices=[
            "weighted",
            "fedavg",
            "fedavgm",
            "fedadam",
            "fedopt",
            "scaffold",
            "median",
            "default",
        ],
        help=(
            "weighted/fedavg=data-size weighted FedAvg, fedavgm/fedopt=server momentum "
            "over aggregated DIFFs, fedadam=server Adam over aggregated DIFFs, "
            "scaffold=SCAFFOLD with control-variate meta, median=robust median, "
            "default=built-in FedAvg"
        ),
    )
    parser.add_argument(
        "--server_lr",
        type=float,
        default=1.0,
        help="Server learning rate multiplier for FedOpt aggregators.",
    )
    parser.add_argument(
        "--server_momentum",
        type=float,
        default=0.6,
        help="Server momentum for fedavgm/fedopt aggregators.",
    )
    parser.add_argument(
        "--fedopt_beta1",
        type=float,
        default=0.9,
        help="First-moment coefficient for the fedadam aggregator.",
    )
    parser.add_argument(
        "--fedopt_beta2",
        type=float,
        default=0.99,
        help="Second-moment coefficient for the fedadam aggregator.",
    )
    parser.add_argument(
        "--fedopt_tau",
        type=float,
        default=1e-3,
        help="Numerical stabilizer for the fedadam aggregator.",
    )
    return parser.parse_args()


def parse_final_eval_clients(client_spec: str, n_clients: int):
    normalized = client_spec.strip()
    if normalized.lower() in {"all", "*"}:
        return None

    clients = []
    for item in normalized.split(","):
        client = item.strip()
        if not client:
            continue
        if client.isdigit():
            client = f"site-{client}"
        clients.append(client)

    if not clients:
        raise ValueError("final_eval_clients must name at least one client or be 'all'")

    known_clients = {f"site-{i}" for i in range(1, n_clients + 1)}
    unknown_clients = [client for client in clients if client not in known_clients]
    if unknown_clients:
        raise ValueError(
            "final_eval_clients contains clients outside this job: "
            f"{', '.join(unknown_clients)}; expected one of {', '.join(sorted(known_clients))} or 'all'"
        )

    return clients


def add_final_global_evaluation(recipe, participating_clients):
    comp_ids = getattr(recipe.job, "comp_ids", {})
    model_locator_id = comp_ids.get("locator_id", "")

    if not model_locator_id:
        persistor_id = comp_ids.get("persistor_id", "")
        if not persistor_id:
            raise ValueError("Final evaluation requires a PyTorch model persistor, but no persistor_id was found")
        model_locator_id = recipe.job.to_server(PTFileModelLocator(pt_persistor_id=persistor_id))

    recipe.job.to_server(ValidationJsonGenerator())
    recipe.job.to_server(
        CrossSiteModelEval(
            model_locator_id=model_locator_id,
            submit_model_task_name="",
            participating_clients=participating_clients,
        )
    )


def get_aggregator(args):
    kind = args.aggregator
    if kind == "weighted":
        print("Using WeightedAggregator")
        return WeightedAggregator()
    if kind == "fedavg":
        print("Using FedAvgAggregator")
        return FedAvgAggregator()
    if kind in {"fedavgm", "fedopt"}:
        print("Using FedAvgMAggregator " f"(server_lr={args.server_lr}, server_momentum={args.server_momentum})")
        return FedAvgMAggregator(
            server_lr=args.server_lr,
            server_momentum=args.server_momentum,
        )
    if kind == "fedadam":
        print(
            "Using FedAdamAggregator "
            f"(server_lr={args.server_lr}, beta1={args.fedopt_beta1}, "
            f"beta2={args.fedopt_beta2}, tau={args.fedopt_tau})"
        )
        return FedAdamAggregator(
            server_lr=args.server_lr,
            beta1=args.fedopt_beta1,
            beta2=args.fedopt_beta2,
            tau=args.fedopt_tau,
        )
    if kind == "scaffold":
        print("Using ScaffoldAggregator")
        return ScaffoldAggregator()
    if kind == "median":
        print("Using MedianAggregator")
        return MedianAggregator()
    if kind == "default":
        print("Using default FedAvg aggregator")
        return None
    raise ValueError(f"Unknown aggregator: {kind}")


def write_result_dir_sidecar(result_dir: str):
    sidecar_path = os.environ.get("AUTOFL_RESULT_DIR_FILE")
    if not sidecar_path:
        return

    path = Path(sidecar_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(f"{result_dir}\n", encoding="utf-8")
    os.replace(tmp_path, path)


def main():
    args = define_parser()

    if args.alpha <= 0.0:
        raise ValueError("alpha must be > 0 for federated CIFAR10 splits")
    if args.aggregation_epochs <= 0:
        raise ValueError("aggregation_epochs must be > 0")
    if args.local_train_steps < 0:
        raise ValueError("local_train_steps must be >= 0")

    if args.name:
        job_name = args.name
    else:
        arch_suffix = "" if args.model_arch == DEFAULT_MODEL_ARCH else f"_{args.model_arch}"
        job_name = f"autofl_cifar10{arch_suffix}_{args.aggregator}_alpha{args.alpha}_seed{args.seed}"

    seed_model = build_model(
        model_arch=args.model_arch,
        seed=args.seed,
        max_model_params=args.max_model_params,
    )
    print(
        f"Using model_arch={args.model_arch} "
        f"params={count_parameters(seed_model):,} max_model_params={args.max_model_params:,}"
    )

    train_idx_root = split_and_save(
        num_sites=args.n_clients,
        alpha=args.alpha,
        seed=args.seed,
        split_dir_prefix="/tmp/cifar10_splits/autofl_cifar10",
    )

    train_script = os.path.join(os.path.dirname(__file__), args.train_script)
    train_args = [
        "--train_idx_root",
        train_idx_root,
        "--num_workers",
        args.num_workers,
        "--seed",
        args.seed,
        "--model_arch",
        args.model_arch,
        "--max_model_params",
        args.max_model_params,
        "--lr",
        args.lr,
        "--batch_size",
        args.batch_size,
        "--eval_batch_size",
        args.eval_batch_size,
        "--aggregation_epochs",
        args.aggregation_epochs,
        "--local_train_steps",
        args.local_train_steps,
        "--momentum",
        args.momentum,
        "--weight_decay",
        args.weight_decay,
        "--cosine_lr_eta_min_factor",
        args.cosine_lr_eta_min_factor,
        "--fedproxloss_mu",
        args.fedproxloss_mu,
    ]
    if args.no_lr_scheduler:
        train_args.append("--no_lr_scheduler")
    if args.evaluate_local:
        train_args.append("--evaluate_local")
    if args.eval_global_every_round:
        train_args.append("--eval_global_every_round")
    if args.aggregator == "scaffold":
        train_args.append("--scaffold")
    if args.no_deterministic_training:
        train_args.append("--no_deterministic_training")
    train_args = " ".join(shlex.quote(str(item)) for item in train_args)

    recipe = FedAvgRecipe(
        name=job_name,
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        model=seed_model,
        train_script=train_script,
        train_args=train_args,
        aggregator=get_aggregator(args),
        key_metric=args.key_metric,
        aggregator_data_kind=DataKind.WEIGHT_DIFF,
        launch_external_process=args.launch_external_process,
        client_memory_gc_rounds=args.client_memory_gc_rounds,
    )

    add_experiment_tracking(recipe, tracking_type="tensorboard")
    if args.cross_site_eval:
        final_eval_clients = parse_final_eval_clients(args.final_eval_clients, args.n_clients)
        add_final_global_evaluation(recipe, participating_clients=final_eval_clients)

    env = SimEnv(num_clients=args.n_clients)
    run = recipe.execute(env)
    result_dir = str(run.get_result())
    write_result_dir_sidecar(result_dir)

    print()
    print("Job Status:", run.get_status())
    print("Results:", result_dir)
    print(f"tensorboard --logdir={result_dir}")
    print()


if __name__ == "__main__":
    main()
