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

import argparse
import os
import shlex
import sys
from typing import Iterable

FEDBPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(FEDBPT_DIR, os.pardir, os.pardir))
SRC_DIR = os.path.join(FEDBPT_DIR, "src")
TRAIN_SCRIPT = os.path.join(SRC_DIR, "fedbpt_train.py")
DEFAULT_WORKSPACE = "/tmp/nvflare/fedbpt"

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nvflare.client.config import ExchangeFormat, TransferType  # noqa: E402
from nvflare.fuel.utils.constants import FrameworkType  # noqa: E402
from nvflare.job_config.api import FedJob  # noqa: E402
from nvflare.job_config.script_runner import BaseScriptRunner  # noqa: E402


def define_parser():
    parser = argparse.ArgumentParser(description="Run or export the FedBPT NVFlare job.")

    parser.add_argument("--job_name", default="fedbpt", help="Name of the generated NVFlare job.")
    parser.add_argument("--num_clients", type=int, default=10, help="Number of FedBPT clients.")
    parser.add_argument("--min_clients", type=int, default=None, help="Minimum clients required to start the job.")
    parser.add_argument("--num_rounds", type=int, default=200, help="Number of global FedBPT rounds.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed used by the server and client scripts.")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="Simulation workspace root.")
    parser.add_argument("--threads", type=int, default=None, help="Number of simulator worker threads.")
    parser.add_argument("--gpu", default=None, help="Simulator GPU assignment string, for example '0,1,2,3'.")
    parser.add_argument("--log_config", default=None, help="Simulator log config mode or path.")
    parser.add_argument("--export", action="store_true", help="Export the job and exit without running simulation.")
    parser.add_argument("--export-dir", default="./jobs", help="Directory where the job folder is exported.")

    parser.add_argument("--frac", type=float, default=1.0, help="Fraction of clients used by GlobalES.")
    parser.add_argument("--sigma", type=float, default=1.0, help="Initial GlobalES sigma.")
    parser.add_argument("--intrinsic_dim", type=int, default=500, help="Intrinsic dimension for GlobalES and clients.")
    parser.add_argument("--bound", type=int, default=0, help="GlobalES bound, 0 disables bounds.")

    parser.add_argument("--task_name", default="sst2", help="Task name passed to fedbpt_train.py.")
    parser.add_argument("--n_prompt_tokens", type=int, default=50, help="Number of prompt tokens.")
    parser.add_argument("--k_shot", type=int, default=200, help="Few-shot samples per class.")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional client batch size override.")
    parser.add_argument("--device", default="cuda:0", help="Device passed to fedbpt_train.py.")
    parser.add_argument("--loss_type", default="ce", help="Loss type passed to fedbpt_train.py.")
    parser.add_argument("--cat_or_add", default="add", help="Prompt mode passed to fedbpt_train.py.")
    parser.add_argument("--local_iter", type=int, default=8, help="Local CMA iterations per round.")
    parser.add_argument("--num_users", type=int, default=None, help="Number of client data shards.")
    parser.add_argument("--iid", type=int, default=1, help="Whether to split client data IID.")
    parser.add_argument("--local_popsize", type=int, default=5, help="Local CMA population size.")
    parser.add_argument("--perturb", type=int, default=1, help="Whether to use perturbed data fitness.")
    parser.add_argument("--model_name", default="roberta-large", choices=["roberta-base", "roberta-large"])
    parser.add_argument("--eval_clients", default="site-1", help="Comma-separated clients that run global eval.")
    parser.add_argument("--llama_causal", type=int, default=1, help="FedBPT compatibility flag.")
    parser.add_argument(
        "--train_args",
        default="",
        help="Additional fedbpt_train.py arguments appended after generated arguments.",
    )
    return parser


def _quote_args(args: Iterable[object]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args if arg is not None)


def _make_train_args(args, extra_train_args: list[str]) -> str:
    num_users = args.num_users if args.num_users is not None else args.num_clients
    train_args = [
        "--task_name",
        args.task_name,
        "--n_prompt_tokens",
        args.n_prompt_tokens,
        "--intrinsic_dim",
        args.intrinsic_dim,
        "--k_shot",
        args.k_shot,
        "--device",
        args.device,
        "--seed",
        args.seed,
        "--loss_type",
        args.loss_type,
        "--cat_or_add",
        args.cat_or_add,
        "--local_iter",
        args.local_iter,
        "--num_users",
        num_users,
        "--iid",
        args.iid,
        "--local_popsize",
        args.local_popsize,
        "--perturb",
        args.perturb,
        "--model_name",
        args.model_name,
        "--eval_clients",
        args.eval_clients,
        "--llama_causal",
        args.llama_causal,
    ]
    if args.batch_size is not None:
        train_args.extend(["--batch_size", args.batch_size])
    if args.train_args:
        train_args.extend(shlex.split(args.train_args))
    train_args.extend(extra_train_args)
    return _quote_args(train_args)


def _load_fedbpt_components():
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)

    from decomposer_widget import RegisterDecomposer
    from global_es import GlobalES

    return GlobalES, RegisterDecomposer


def create_recipe(args, extra_train_args: list[str] | None = None):
    from nvflare.app_common.launchers.subprocess_launcher import SubprocessLauncher
    from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver
    from nvflare.recipe.spec import Recipe

    GlobalES, RegisterDecomposer = _load_fedbpt_components()

    extra_train_args = extra_train_args or []
    min_clients = args.min_clients if args.min_clients is not None else args.num_clients
    train_args = _make_train_args(args, extra_train_args)

    job = FedJob(name=args.job_name, min_clients=min_clients)
    job.to_server(
        GlobalES(
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
            frac=args.frac,
            sigma=args.sigma,
            intrinsic_dim=args.intrinsic_dim,
            seed=args.seed,
            bound=args.bound,
        ),
        id="global_es",
    )
    job.to_server(TBAnalyticsReceiver(events=["fed.analytix_log_stats"]), id="receiver")
    job.to_server(RegisterDecomposer(), id="register_decomposer")

    launcher = SubprocessLauncher(
        script=f"python3 -u custom/fedbpt_train.py {train_args}",
        launch_once=True,
        shutdown_timeout=0.0,
    )
    runner = BaseScriptRunner(
        script=TRAIN_SCRIPT,
        launch_external_process=True,
        framework=FrameworkType.NUMPY,
        server_expected_format=ExchangeFormat.NUMPY,
        params_transfer_type=TransferType.FULL,
        launcher=launcher,
    )
    job.to_clients(runner, tasks=["train"])
    job.to_clients(RegisterDecomposer(), id="register_decomposer")
    return Recipe(job)


def main():
    parser = define_parser()
    args, extra_train_args = parser.parse_known_args()

    recipe = create_recipe(args, extra_train_args)
    if args.export:
        recipe.export(args.export_dir)
        print(f"Job exported to: {os.path.join(args.export_dir, args.job_name)}")
        return

    from nvflare.recipe.sim_env import SimEnv

    env = SimEnv(
        num_clients=args.num_clients,
        num_threads=args.threads,
        gpu_config=args.gpu,
        log_config=args.log_config,
        workspace_root=args.workspace,
    )
    run = recipe.run(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in:", run.get_result(clean_up=False))
    print()


if __name__ == "__main__":
    main()
