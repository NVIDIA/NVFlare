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

import os
import shlex
import sys
from contextlib import contextmanager
from typing import Iterable, Optional

from nvflare.app_common.launchers.subprocess_launcher import SubprocessLauncher
from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.fuel.utils.constants import FrameworkType
from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import BaseScriptRunner
from nvflare.recipe.spec import ExecEnv, Recipe

FEDBPT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(FEDBPT_DIR, "src")
TRAIN_SCRIPT = os.path.join(SRC_DIR, "fedbpt_train.py")


@contextmanager
def _temporary_sys_path(path: str):
    added_path = path not in sys.path
    if added_path:
        sys.path.insert(0, path)
    try:
        yield
    finally:
        if added_path:
            sys.path.remove(path)


def _quote_args(args: Iterable[object]) -> str:
    args = list(args)
    if any(arg is None for arg in args):
        raise ValueError(f"None value encountered in train args list: {args}")
    return " ".join(shlex.quote(str(arg)) for arg in args)


class FedBPTRecipe(Recipe):
    """Configure the FedBPT GlobalES workflow and client training process."""

    def __init__(
        self,
        *,
        name: str = "fedbpt",
        num_clients: int = 10,
        min_clients: int | None = None,
        num_rounds: int = 200,
        seed: int = 1234,
        frac: float = 1.0,
        sigma: float = 1.0,
        intrinsic_dim: int = 500,
        bound: int = 0,
        task_name: str = "sst2",
        n_prompt_tokens: int = 50,
        k_shot: int = 200,
        batch_size: int | None = None,
        device: str = "cuda:0",
        loss_type: str = "ce",
        cat_or_add: str = "add",
        local_iter: int = 8,
        num_users: int | None = None,
        iid: int = 1,
        local_popsize: int = 5,
        perturb: int = 1,
        model_name: str = "roberta-large",
        eval_clients: str = "site-1",
        llama_causal: int = 1,
        train_args: str = "",
        extra_train_args: Iterable[str] | None = None,
    ):
        min_clients = num_clients if min_clients is None else min_clients
        num_users = num_clients if num_users is None else num_users
        client_args = [
            "--task_name",
            task_name,
            "--n_prompt_tokens",
            n_prompt_tokens,
            "--intrinsic_dim",
            intrinsic_dim,
            "--k_shot",
            k_shot,
            "--device",
            device,
            "--seed",
            seed,
            "--loss_type",
            loss_type,
            "--cat_or_add",
            cat_or_add,
            "--local_iter",
            local_iter,
            "--num_users",
            num_users,
            "--iid",
            iid,
            "--local_popsize",
            local_popsize,
            "--perturb",
            perturb,
            "--model_name",
            model_name,
            "--eval_clients",
            eval_clients,
            "--llama_causal",
            llama_causal,
        ]
        if batch_size is not None:
            client_args.extend(["--batch_size", batch_size])
        if train_args:
            client_args.extend(shlex.split(train_args))
        if extra_train_args:
            client_args.extend(extra_train_args)
        # Retain the exact launcher arguments so Recipe's pre-use secret scan covers
        # both train_args and extra_train_args after they have been combined.
        self.train_args = _quote_args(client_args)

        with _temporary_sys_path(SRC_DIR):
            from decomposer_widget import RegisterDecomposer
            from global_es import GlobalES

        job = FedJob(name=name, min_clients=min_clients)
        job.to_server(
            GlobalES(
                num_clients=num_clients,
                num_rounds=num_rounds,
                frac=frac,
                sigma=sigma,
                intrinsic_dim=intrinsic_dim,
                seed=seed,
                bound=bound,
            ),
            id="global_es",
        )
        job.to_server(TBAnalyticsReceiver(events=["fed.analytix_log_stats"]), id="receiver")
        job.to_server(RegisterDecomposer(), id="register_decomposer")

        launcher = SubprocessLauncher(
            script=f"python3 -u custom/fedbpt_train.py {self.train_args}",
            launch_once=True,
            shutdown_timeout=10.0,
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
        super().__init__(job)

    def export(
        self,
        job_dir: str,
        server_exec_params: Optional[dict] = None,
        client_exec_params: Optional[dict] = None,
        env: Optional[ExecEnv] = None,
    ):
        with _temporary_sys_path(SRC_DIR):
            return super().export(
                job_dir=job_dir,
                server_exec_params=server_exec_params,
                client_exec_params=client_exec_params,
                env=env,
            )

    def run(
        self,
        env: ExecEnv,
        server_exec_params: Optional[dict] = None,
        client_exec_params: Optional[dict] = None,
    ) -> "Run":
        with _temporary_sys_path(SRC_DIR):
            return super().run(
                env=env,
                server_exec_params=server_exec_params,
                client_exec_params=client_exec_params,
            )
