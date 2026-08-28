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

"""Opt-in compatibility transport for hosts without usable IPv4 loopback.

This module intentionally depends on NVFlare simulator internals. The normal
research workflow uses the public :class:`nvflare.recipe.SimEnv`; import this
module only when ``AGENTICFL_SIMULATOR_MODE=ipv6_unix`` is explicitly selected.
"""

from __future__ import annotations

import os
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from nvflare.job_config.api import FedJob
from nvflare.private.fed.app.deployer.simulator_deployer import SimulatorDeployer
from nvflare.private.fed.app.simulator.simulator_runner import SimulatorClientRunner, SimulatorRunner
from nvflare.recipe import SimEnv
from nvflare.recipe.utils import collect_non_local_scripts


def validate_private_simulator_contract() -> None:
    """Fail before deployment if the opt-in private hooks have changed."""

    required_hooks = (
        (SimulatorDeployer, "_create_simulator_server_config"),
        (SimulatorDeployer, "_create_simulator_client_config"),
        (SimulatorRunner, "run"),
        (SimEnv, "_ensure_default_component_policy"),
    )
    missing = [f"{owner.__name__}.{name}" for owner, name in required_hooks if not hasattr(owner, name)]
    if missing:
        raise RuntimeError(
            "AGENTICFL_SIMULATOR_MODE=ipv6_unix is incompatible with this NVFlare installation; "
            f"missing private hooks: {', '.join(missing)}"
        )


def use_ipv6_atcp(service: dict[str, Any]) -> None:
    """Rewrite one NVFlare service mapping without allocating a socket."""

    _host, port = str(service["target"]).rsplit(":", 1)
    service["scheme"] = "atcp"
    service["target"] = f"ip6-localhost:{port}"


class AgenticFLSimulatorDeployer(SimulatorDeployer):
    """Use NVFlare's asynchronous TCP driver over IPv6 loopback."""

    def _create_simulator_server_config(self, admin_storage: str, max_clients: int) -> dict[str, Any]:
        config = super()._create_simulator_server_config(admin_storage, max_clients)
        use_ipv6_atcp(config["service"])
        return config

    def _create_simulator_client_config(self, client_name: str, args: Any) -> tuple[dict[str, Any], Any]:
        config, build_ctx = super()._create_simulator_client_config(client_name, args)
        use_ipv6_atcp(config["servers"][0]["service"])
        return config, build_ctx


def simulator_worker_ipc_path(parent_pid: int, port_token: str | int) -> Path:
    """Return the shared Unix-domain address for one simulator task worker."""

    return Path("/tmp") / f"agenticfl-nvflare-{parent_pid}-{port_token}.sock"


class AgenticFLSimulatorClientRunner(SimulatorClientRunner):
    """Keep simulator-private task-worker IPC off IPv4 loopback."""

    def _create_connection(self, open_port: int, timeout: float = 60.0) -> Any:
        address = str(simulator_worker_ipc_path(os.getpid(), open_port))
        start = time.time()
        while True:
            try:
                return Client(address)
            except Exception as exc:
                if time.time() - start > timeout:
                    raise RuntimeError(
                        "Failed to create Unix-domain connection to the NVFlare "
                        f"simulator child process within {timeout} seconds"
                    ) from exc
                time.sleep(1.0)


class AgenticFLSimulatorRunner(SimulatorRunner):
    """Reuse SimulatorRunner with Unix-domain task-worker IPC."""

    def client_run(self, server_custom_folder: str, clients: list[Any], gpu: Any) -> None:
        client_runner = AgenticFLSimulatorClientRunner(
            server_custom_folder,
            self.args,
            clients,
            self.client_config,
            self.deploy_args,
            self.build_ctx,
        )
        client_runner.run(gpu)


class AgenticFLSimEnv(SimEnv):
    """SimEnv using IPv6 ATCP plus Unix-domain private worker handshakes."""

    def deploy(self, job: FedJob) -> str:
        non_local_scripts = collect_non_local_scripts(job)
        if non_local_scripts:
            raise ValueError(
                f"The following scripts do not exist locally: {non_local_scripts}. "
                "For SimEnv, all scripts must be present on the local machine."
            )

        job_id = job.name
        workspace = os.path.join(self.workspace_root, job_id)
        self._ensure_default_component_policy(workspace)
        original_cwd = os.getcwd()
        with TemporaryDirectory() as job_root:
            job.export_job(job_root)
            runner = AgenticFLSimulatorRunner(
                job_folder=os.path.join(job_root, job_id),
                workspace=workspace,
                clients=",".join(self.clients) if self.clients else None,
                n_clients=self.num_clients if self.clients is None else None,
                threads=self.num_threads,
                gpu=self.gpu_config,
                log_config=self.log_config,
            )
            runner.deployer = AgenticFLSimulatorDeployer()
            python_executable = sys.executable
            worker_wrapper = Path(job_root) / "agenticfl-python"
            worker_wrapper.write_text(
                f"#!{python_executable}\n"
                "import os, sys\n"
                "args = sys.argv[1:]\n"
                "if len(args) >= 2 and args[0] == '-m' and "
                "args[1] == 'nvflare.private.fed.app.simulator.simulator_worker':\n"
                "    args[1] = 'agenticfl.flare.simulator_worker'\n"
                f"os.execv({python_executable!r}, [{python_executable!r}, *args])\n",
                encoding="utf-8",
            )
            worker_wrapper.chmod(0o700)
            try:
                sys.executable = str(worker_wrapper)
                run_status = runner.run()
            finally:
                sys.executable = python_executable
                os.chdir(original_cwd)

        self.last_run_failed = run_status not in (None, 0)
        if self.last_run_failed:
            raise RuntimeError(
                f"Simulation failed with return code {run_status}. "
                f"Logs are in per-site subdirectories under {workspace}."
            )
        return job_id
