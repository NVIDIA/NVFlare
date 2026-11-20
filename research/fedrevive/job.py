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
from fedrevive import FIGURE_2_METHOD_CONFIGS, ClassProportionSource, Method, SynthesisLabelDistribution
from model import set_seed
from schedule_standard import make_paper_plan
from server import FedReviveModelLoader, FedReviveModelOps, FedReviveObserver
from update import FedAvgUpdate, FedReviveUpdate

from nvflare.apis.simulation_plan import ASYNC_EXECUTION, load_simulation_plan
from nvflare.collab import FedBuffRecipe
from nvflare.recipe import SimEnv

PROJECT_DIR = Path(__file__).resolve().parent


class FedReviveSimEnv(SimEnv):
    """Reserve a fresh experiment directory before simulator startup."""

    def deploy(self, job):
        workspace = Path(self.workspace_root) / job.name
        # Simulator startup briefly removes the job directory, so admission
        # has to be reserved beside it.
        reservation = workspace.with_name(f".{workspace.name}.running")
        try:
            reservation.mkdir(parents=True)
        except FileExistsError:
            raise FileExistsError(
                f"FedRevive workspace is reserved by {reservation}. Choose a fresh workspace root."
            ) from None
        try:
            if workspace.exists() or workspace.is_symlink():
                raise FileExistsError(
                    f"FedRevive workspace already exists: {workspace}. Choose a fresh workspace root."
                )
            return super().deploy(job)
        finally:
            reservation.rmdir()


def define_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arrival-mode",
        choices=("simulated", "actual", "replay"),
        default="simulated",
        help="Preplanned paper phases, live arrivals, or a supplied SimEnv plan",
    )
    parser.add_argument("--simulation-plan", help="SimEnv JSON plan required for --arrival-mode replay")
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
        "--concurrency",
        "--num-active-jobs",
        type=int,
        default=None,
        help="Maximum concurrent logical clients; defaults to the method preset",
    )
    parser.add_argument(
        "--buffer-size", type=int, default=None, help="Contributions per global model; defaults to the method preset"
    )
    parser.add_argument(
        "--min-open-slots",
        type=int,
        default=None,
        help="Free logical slots before refill; defaults to the method preset",
    )
    parser.add_argument("--server-lr", type=float, default=None)
    parser.add_argument(
        "--class-proportion-source",
        choices=[source.value for source in ClassProportionSource],
        default=None,
        help="How FedRevive obtains class proportions; defaults to estimated for FedRevive",
    )
    parser.add_argument(
        "--generation-interval",
        type=int,
        default=None,
        help="Generate synthetic data every N global model versions; defaults to 10 for FedRevive",
    )
    parser.add_argument(
        "--synthesis-label-distribution",
        choices=[distribution.value for distribution in SynthesisLabelDistribution],
        default=SynthesisLabelDistribution.UNIFORM.value,
        help="Draw synthesis targets uniformly or from the mean of the buffered teachers' class proportions",
    )
    parser.add_argument(
        "--class-proportion-probe-count",
        type=int,
        default=64,
        help="Number of Gaussian inputs used for each class-proportion probe",
    )
    parser.add_argument("--max-time", type=float, default=200.0, help="Simulated-time budget")
    parser.add_argument(
        "--num-rounds",
        "--max-model-versions",
        dest="max_model_versions",
        type=int,
        default=100000,
        help="Maximum global model publications",
    )
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
        help="Compatibility flag; shared async aggregation always accumulates in time",
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
        help="Seed for the simulated arrival sequence, client assignments, and server-side distillation",
    )
    parser.add_argument("--workspace-root", default="/tmp/nvflare/fedrevive")
    return parser


def _device(value: str):
    return None if value == "auto" else value


def validate_args(args):
    if (args.arrival_mode == "replay") != (args.simulation_plan is not None):
        raise ValueError("--arrival-mode replay requires --simulation-plan, which is only used in replay mode")
    if not args.in_time:
        raise ValueError("The shared async coordinator uses bounded in-time accumulation; remove --no-in-time")
    if args.num_clients < 1 or args.num_logical_clients < 1:
        raise ValueError("--num-clients and --num-logical-clients must be >= 1")
    if args.arrival_mode == "simulated" and not args.max_time > 0:
        raise ValueError("--max-time must be positive")
    if args.local_iterations < 1 or args.local_batch_size < 1:
        raise ValueError("--local-iterations and --local-batch-size must be positive")
    if args.local_lr <= 0 or args.call_timeout <= 0 or args.num_workers < 0:
        raise ValueError("--local-lr and --call-timeout must be positive; --num-workers must be nonnegative")
    if (args.generation_interval is not None and args.generation_interval < 1) or args.class_proportion_probe_count < 1:
        raise ValueError("--generation-interval and --class-proportion-probe-count must be positive")
    if args.method != Method.FEDREVIVE.value and (
        args.class_proportion_source not in (None, ClassProportionSource.TRUE_HISTOGRAM.value)
        or args.generation_interval not in (None, 1)
        or args.class_proportion_probe_count != 64
        or args.synthesis_label_distribution != SynthesisLabelDistribution.UNIFORM.value
    ):
        raise ValueError("class-proportion and synthesis options require --method fedrevive")


def _scheduling_options(args, preset):
    concurrency = preset.concurrency if args.concurrency is None else args.concurrency
    buffer_size = preset.buffer_size if args.buffer_size is None else args.buffer_size
    open_slots = preset.min_open_slots if args.min_open_slots is None else args.min_open_slots
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if buffer_size < 1 or not 1 <= open_slots <= concurrency:
        raise ValueError("buffer_size must be positive and min_open_slots must be between 1 and concurrency")
    return concurrency, buffer_size, open_slots


def make_recipe(args):
    validate_args(args)
    method = Method(args.method)
    preset = FIGURE_2_METHOD_CONFIGS[method]
    concurrency, buffer_size, open_slots = _scheduling_options(args, preset)
    class_proportion_source = args.class_proportion_source
    if class_proportion_source is None:
        class_proportion_source = (
            ClassProportionSource.ESTIMATED.value
            if method is Method.FEDREVIVE
            else ClassProportionSource.TRUE_HISTOGRAM.value
        )
    generation_interval = args.generation_interval
    if generation_interval is None:
        generation_interval = 10 if method is Method.FEDREVIVE else 1
    if args.arrival_mode == "actual" and (
        concurrency > args.num_clients or args.num_clients > args.num_logical_clients
    ):
        raise ValueError("Actual arrivals require concurrency <= physical clients <= prepared logical clients")
    server_lr = preset.server_lr if args.server_lr is None else args.server_lr
    if server_lr <= 0 or args.max_model_versions < 1:
        raise ValueError("server_lr and num_rounds must be positive")
    config = vars(args).copy()
    config.pop("simulation_plan")
    config.update(
        concurrency=concurrency,
        buffer_size=buffer_size,
        min_open_slots=open_slots,
        server_lr=server_lr,
        class_proportion_source=class_proportion_source,
        generation_interval=generation_interval,
    )
    observer = FedReviveObserver(
        data_root=args.data_root,
        run_config=config,
        device=_device(args.server_device),
        eval_batch_size=args.eval_batch_size,
        eval_interval=(1 if method is Method.FEDAVG else 10) if args.eval_interval is None else args.eval_interval,
    )
    transform = (
        FedReviveUpdate(
            args.prepared_data_root,
            device=_device(args.server_device),
            generation_interval=generation_interval,
            class_proportion_source=class_proportion_source,
            probe_count=args.class_proportion_probe_count,
            run_seed=args.run_seed,
            synthesis_label_distribution=args.synthesis_label_distribution,
        )
        if method is Method.FEDREVIVE
        else None
    )
    if method is Method.FEDAVG:
        transform = FedAvgUpdate(buffer_size)
    client = FedReviveClient(
        data_root=args.data_root,
        prepared_data_root=args.prepared_data_root,
        local_batch_size=args.local_batch_size,
        local_iterations=args.local_iterations,
        local_lr=args.local_lr,
        device=_device(args.client_device),
        num_workers=args.num_workers,
        run_seed=args.run_seed,
    )
    # SimEnv creates one process per physical site.  Physical sites are only a
    # bounded execution pool and are deliberately independent of logical concurrency.
    # Bound Collab's call pools to that small pool instead of retaining the
    # recipe default of 100 threads in every process.  This prevents a concurrency=100
    # experiment from multiplying thread stacks and queued RPC state while the
    # simulator still reproduces all 100 logical jobs.
    server_call_threads = max(8, 2 * args.num_clients)
    client_call_threads = 4
    recipe = FedBuffRecipe(
        name=f"fedrevive_{method.value}",
        model_loader=FedReviveModelLoader(setup_seed=args.setup_seed),
        model_ops=FedReviveModelOps(),
        observer=observer,
        num_rounds=args.max_model_versions,
        concurrency=concurrency,
        buffer_size=buffer_size,
        min_open_slots=open_slots,
        server_lr=server_lr,
        update_transform=transform,
        call_timeout=args.call_timeout,
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
    for filename in (
        "model.py",
        "data.py",
        "fedrevive.py",
        "dfkd.py",
        "update.py",
    ):
        recipe.add_server_file(str(PROJECT_DIR / filename))
    return recipe, (concurrency, buffer_size, open_slots)


def make_sim_env(args, manifest):
    options = dict(num_clients=args.num_clients, workspace_root=args.workspace_root)
    if args.arrival_mode == "actual":
        return FedReviveSimEnv(**options)
    profiles = manifest["runtime_profiles"]
    if args.arrival_mode == "simulated":
        concurrency, buffer_size, open_slots = _scheduling_options(args, FIGURE_2_METHOD_CONFIGS[Method(args.method)])
        plan = make_paper_plan(
            profiles,
            args.run_seed,
            args.max_time,
            num_rounds=args.max_model_versions,
            concurrency=concurrency,
            min_open_slots=open_slots,
            buffer_size=buffer_size,
        )
    else:
        plan = load_simulation_plan(args.simulation_plan)
        missing = set(plan[ASYNC_EXECUTION]["logical_sequence"]) - set(profiles)
        if missing:
            raise ValueError(f"Simulation plan names unprepared logical clients: {sorted(missing)}")
    return FedReviveSimEnv(reprod=True, plan=plan, **options)


def main():
    args = define_parser().parse_args()
    validate_args(args)
    set_seed(args.setup_seed)
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
    if args.arrival_mode == "simulated" and prepared_schedule != args.delay_schedule:
        raise ValueError(
            f"Prepared data uses delay schedule {prepared_schedule!r}, but --delay-schedule={args.delay_schedule!r}"
        )

    recipe, (concurrency, buffer_size, open_slots) = make_recipe(args)
    print("=" * 80)
    print(f"FEDREVIVE COLLAB FIGURE 2: {args.method}")
    print(f"  Physical clients: {args.num_clients}")
    print(f"  Logical clients: {args.num_logical_clients}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Buffer size: {buffer_size}")
    print(f"  Minimum open slots: {open_slots}")
    print(f"  In-time accumulation: {args.in_time}")
    if args.arrival_mode == "simulated":
        print(f"  Simulated-time budget: {args.max_time}")
        print(f"  Delay schedule: {args.delay_schedule}")
    print(f"  Global-version budget: {args.max_model_versions}")
    print(f"  Arrival mode: {args.arrival_mode}")
    print(f"  Class-proportion source: {recipe.server.observer.run_config['class_proportion_source']}")
    print(f"  Class-proportion probe count: {args.class_proportion_probe_count}")
    print(f"  Generation interval: {recipe.server.observer.run_config['generation_interval']}")
    print("=" * 80)
    run = recipe.execute(make_sim_env(args, manifest))
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
