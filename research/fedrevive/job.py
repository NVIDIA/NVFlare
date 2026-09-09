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

"""Run the FedRevive Figure 2 methods with the NVIDIA FLARE Collab API."""

import argparse
import os
from pathlib import Path

# SimEnv forks its server and clients from this process.  Configure native
# libraries before importing modules that import torch or numpy; otherwise each
# long-lived worker may inherit a host-wide BLAS/OpenMP pool.  Multiplying those
# threads and their stacks across simulator processes can drain host resources
# even though logical training is intended to be sequential per physical site.
for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

from client import FedReviveClient
from data import DELAY_SCHEDULES, load_manifest, prepare_cifar10
from fedrevive import FIGURE_2_METHOD_CONFIGS, ClassProportionSource, Method
from server import FedReviveServer

from nvflare.collab import CollabRecipe
from nvflare.recipe import SimEnv

PROJECT_DIR = Path(__file__).resolve().parent


def define_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=[method.value for method in Method],
        default=Method.FEDREVIVE.value,
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=2,
        help="Number of physical Collab worker sites",
    )
    parser.add_argument("--num-logical-clients", type=int, default=1000)
    parser.add_argument(
        "--num-active-jobs",
        type=int,
        default=None,
        help="K; defaults to the method preset",
    )
    parser.add_argument("--buffer-size", type=int, default=None, help="B; defaults to the method preset")
    parser.add_argument(
        "--min-open-slots",
        type=int,
        default=None,
        help="O; defaults to the method preset",
    )
    parser.add_argument("--server-lr", type=float, default=None)
    parser.add_argument(
        "--class-proportion-source",
        choices=[source.value for source in ClassProportionSource],
        default=ClassProportionSource.TRUE_HISTOGRAM.value,
        help="How FedRevive obtains each client's class proportions",
    )
    parser.add_argument(
        "--generation-interval",
        type=int,
        default=1,
        help="Generate synthetic data every N global model versions",
    )
    parser.add_argument(
        "--class-proportion-probe-count",
        type=int,
        default=64,
        help="Number of Gaussian inputs used for each class-proportion probe",
    )
    parser.add_argument("--max-time", type=float, default=200.0, help="Simulated-time budget")
    parser.add_argument("--max-model-versions", type=int, default=50000)
    parser.add_argument("--data-root", default="/tmp/cifar10")
    parser.add_argument("--prepared-data-root", default="/tmp/fedrevive/cifar10")
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument(
        "--delay-schedule",
        choices=DELAY_SCHEDULES,
        default="default",
        help="Prepared logical-client runtime schedule: Figure 2 default or Figure 3 shifted",
    )
    parser.add_argument("--client-data-size", type=int, default=350)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--local-batch-size", type=int, default=32)
    parser.add_argument("--local-iterations", type=int, default=25)
    parser.add_argument("--local-lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=300)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument(
        "--in-time",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accumulate accepted deltas in place; --no-in-time retains all B deltas (higher memory)",
    )
    parser.add_argument("--call-timeout", type=float, default=3600.0)
    parser.add_argument("--server-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--client-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--setup-seed",
        type=int,
        default=10,
        help="Model initialization seed; also used for profiles when --prepare-data is set",
    )
    parser.add_argument(
        "--run-seed",
        type=int,
        default=10,
        help="Logical participant-selection and delay-sampling seed; fixes the simulated arrival sequence",
    )
    parser.add_argument("--workspace-root", default="/tmp/nvflare/fedrevive")
    return parser


def _device(value: str):
    return None if value == "auto" else value


def validate_args(args):
    if args.num_clients < 1 or args.num_logical_clients < 1:
        raise ValueError("--num-clients and --num-logical-clients must be >= 1")
    if args.max_time <= 0 or args.local_iterations < 1 or args.local_batch_size < 1:
        raise ValueError("--max-time, --local-iterations, and --local-batch-size must be positive")
    if args.local_lr <= 0 or args.call_timeout <= 0 or args.num_workers < 0:
        raise ValueError("--local-lr and --call-timeout must be positive; --num-workers must be nonnegative")
    if args.generation_interval < 1 or args.class_proportion_probe_count < 1:
        raise ValueError("--generation-interval and --class-proportion-probe-count must be positive")
    if args.method != Method.FEDREVIVE.value and (
        args.class_proportion_source != ClassProportionSource.TRUE_HISTOGRAM.value
        or args.generation_interval != 1
        or args.class_proportion_probe_count != 64
    ):
        raise ValueError("class-proportion, probe-count, and generation options require --method fedrevive")


def make_recipe(args):
    method = Method(args.method)
    preset = FIGURE_2_METHOD_CONFIGS[method]
    active_jobs = preset.num_active_jobs if args.num_active_jobs is None else args.num_active_jobs
    buffer_size = preset.buffer_size if args.buffer_size is None else args.buffer_size
    open_slots = preset.min_open_slots if args.min_open_slots is None else args.min_open_slots
    if active_jobs < 1:
        raise ValueError("K must be positive")
    if buffer_size < 1 or not 1 <= open_slots <= active_jobs:
        raise ValueError("B must be positive and O must be between 1 and K")

    server = FedReviveServer(
        method=method,
        data_root=args.data_root,
        prepared_data_root=args.prepared_data_root,
        max_time=args.max_time,
        num_active_jobs=active_jobs,
        buffer_size=buffer_size,
        min_open_slots=open_slots,
        server_lr=args.server_lr,
        call_timeout=args.call_timeout,
        device=_device(args.server_device),
        eval_batch_size=args.eval_batch_size,
        eval_interval=args.eval_interval,
        in_time=args.in_time,
        setup_seed=args.setup_seed,
        run_seed=args.run_seed,
        max_model_versions=args.max_model_versions,
        class_proportion_source=args.class_proportion_source,
        generation_interval=args.generation_interval,
        class_proportion_probe_count=args.class_proportion_probe_count,
    )
    client = FedReviveClient(
        data_root=args.data_root,
        prepared_data_root=args.prepared_data_root,
        local_batch_size=args.local_batch_size,
        local_iterations=args.local_iterations,
        local_lr=args.local_lr,
        device=_device(args.client_device),
        num_workers=args.num_workers,
    )
    # SimEnv creates one process per physical site.  Physical sites are only a
    # bounded execution pool and are deliberately independent of logical K.
    # Bound Collab's call pools to that small pool instead of retaining the
    # recipe default of 100 threads in every process.  This prevents a K=100
    # experiment from multiplying thread stacks and queued RPC state while the
    # server-side event scheduler still reproduces all 100 logical jobs.
    server_call_threads = max(8, 2 * args.num_clients)
    client_call_threads = 4
    recipe = CollabRecipe(
        job_name=f"fedrevive_{method.value}",
        server=server,
        client=client,
        min_clients=args.num_clients,
        sync_task_timeout=600,
        max_call_threads_for_server=server_call_threads,
        max_call_threads_for_client=client_call_threads,
    )
    recipe.set_client_prop("data_root", args.data_root)
    recipe.set_client_prop("prepared_data_root", args.prepared_data_root)
    for filename in ("model.py", "data.py"):
        recipe.add_client_file(str(PROJECT_DIR / filename))
    for filename in ("model.py", "data.py", "fedrevive.py", "dfkd.py"):
        recipe.add_server_file(str(PROJECT_DIR / filename))
    return recipe, (active_jobs, buffer_size, open_slots)


def main():
    args = define_parser().parse_args()
    validate_args(args)
    if args.prepare_data:
        prepare_cifar10(
            output_root=args.prepared_data_root,
            download_root=args.data_root,
            num_logical_clients=args.num_logical_clients,
            client_data_size=args.client_data_size,
            alpha=args.alpha,
            setup_seed=args.setup_seed,
            delay_schedule=args.delay_schedule,
        )
    manifest = load_manifest(args.prepared_data_root)
    if int(manifest["num_logical_clients"]) != args.num_logical_clients:
        raise ValueError(
            f"Prepared data has {manifest['num_logical_clients']} logical clients, "
            f"but --num-logical-clients={args.num_logical_clients}"
        )
    prepared_schedule = manifest.get("delay_schedule", "default")
    if prepared_schedule != args.delay_schedule:
        raise ValueError(
            f"Prepared data uses delay schedule {prepared_schedule!r}, but --delay-schedule={args.delay_schedule!r}"
        )

    recipe, (active_jobs, buffer_size, open_slots) = make_recipe(args)
    print("=" * 80)
    print(f"FEDREVIVE COLLAB FIGURE 2: {args.method}")
    print(f"  Physical clients: {args.num_clients}")
    print(f"  Logical clients: {args.num_logical_clients}")
    print(f"  K/B/O: {active_jobs}/{buffer_size}/{open_slots}")
    print(f"  In-time accumulation: {args.in_time}")
    print(f"  Simulated-time budget: {args.max_time}")
    print(f"  Delay schedule: {args.delay_schedule}")
    print(f"  Class-proportion source: {args.class_proportion_source}")
    print(f"  Class-proportion probe count: {args.class_proportion_probe_count}")
    print(f"  Generation interval: {args.generation_interval}")
    print("=" * 80)
    run = recipe.execute(SimEnv(num_clients=args.num_clients, workspace_root=args.workspace_root))
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
