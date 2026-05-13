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

from typing import Optional

from pydantic import BaseModel, conint

from nvflare import FedJob
from nvflare.app_common.workflows.cmd_task_controller import CmdTaskController
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.fuel.utils.constants import FrameworkType
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.recipe.spec import Recipe


# Internal - not part of the public API
class _FedTaskValidator(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str
    task_name: str
    min_clients: conint(ge=1)
    num_clients: Optional[conint(ge=1)] = None
    min_responses: Optional[conint(ge=1)] = None
    timeout: conint(ge=0) = 0
    task_data: Optional[dict] = None
    task_meta: Optional[dict] = None
    task_script: str
    task_args: str = ""
    launch_external_process: bool = False
    command: str = "python3 -u"
    framework: FrameworkType = FrameworkType.RAW
    server_expected_format: ExchangeFormat = ExchangeFormat.RAW
    params_transfer_type: TransferType = TransferType.FULL
    launch_once: bool = True
    shutdown_timeout: float = 0.0
    client_memory_gc_rounds: int = 0
    cuda_empty_cache: bool = False


class FedTaskRecipe(Recipe):
    """A model-free recipe for running one federated task on participating clients.

    This recipe is intended for one-round workflows that do not have a global model
    lifecycle, such as embedding extraction, preprocessing, feature generation, local
    evaluation, or other client-side jobs coordinated by the server.

    Args:
        name: Name of the federated job. Defaults to "fed_task".
        task_name: Name of the task sent to clients. Defaults to "task".
        min_clients: Minimum number of clients required to start the job.
        num_clients: Number of clients to sample for the task. If None, all available clients are used.
        min_responses: Minimum number of task results to wait for. If None, waits for all selected clients.
        timeout: Task timeout in seconds. Defaults to 0, meaning no timeout.
        task_data: Optional params dict sent to each client as ``FLModel.params``.
        task_meta: Optional metadata dict sent to each client as ``FLModel.meta``.
        task_script: Path to the client script.
        task_args: Command line arguments passed to the client script.
        launch_external_process: Whether to launch the script in an external process.
        command: Command used when ``launch_external_process`` is True.
        framework: Framework used by ``ScriptRunner`` for parameter exchange. Defaults to RAW.
        server_expected_format: Server-side expected parameter format. Defaults to RAW.
        params_transfer_type: Parameter transfer type. Defaults to FULL.
        launch_once: Whether an external process is launched once for the whole job.
        shutdown_timeout: Seconds to wait before external process shutdown.
        client_memory_gc_rounds: Run client memory cleanup every N rounds. Set 0 to disable.
        cuda_empty_cache: Whether client memory cleanup also empties the CUDA cache.

    Example:
        >>> from nvflare.recipe import FedTaskRecipe, SimEnv
        >>>
        >>> recipe = FedTaskRecipe(
        ...     name="extract_embeddings",
        ...     task_name="embed",
        ...     min_clients=2,
        ...     task_script="client.py",
        ...     task_args="--data-root /data --out /tmp/embeddings",
        ... )
        >>> run = recipe.execute(SimEnv(num_clients=2))
    """

    def __init__(
        self,
        *,
        name: str = "fed_task",
        task_name: str = "task",
        min_clients: int,
        num_clients: Optional[int] = None,
        min_responses: Optional[int] = None,
        timeout: int = 0,
        task_data: Optional[dict] = None,
        task_meta: Optional[dict] = None,
        task_script: str,
        task_args: str = "",
        launch_external_process: bool = False,
        command: str = "python3 -u",
        framework: FrameworkType = FrameworkType.RAW,
        server_expected_format: ExchangeFormat = ExchangeFormat.RAW,
        params_transfer_type: TransferType = TransferType.FULL,
        launch_once: bool = True,
        shutdown_timeout: float = 0.0,
        client_memory_gc_rounds: int = 0,
        cuda_empty_cache: bool = False,
    ):
        v = _FedTaskValidator(
            name=name,
            task_name=task_name,
            min_clients=min_clients,
            num_clients=num_clients,
            min_responses=min_responses,
            timeout=timeout,
            task_data=task_data,
            task_meta=task_meta,
            task_script=task_script,
            task_args=task_args,
            launch_external_process=launch_external_process,
            command=command,
            framework=framework,
            server_expected_format=server_expected_format,
            params_transfer_type=params_transfer_type,
            launch_once=launch_once,
            shutdown_timeout=shutdown_timeout,
            client_memory_gc_rounds=client_memory_gc_rounds,
            cuda_empty_cache=cuda_empty_cache,
        )

        self.name = v.name
        self.task_name = v.task_name
        self.min_clients = v.min_clients
        self.num_clients = v.num_clients
        self.min_responses = v.min_responses
        self.timeout = v.timeout
        self.task_data = v.task_data
        self.task_meta = v.task_meta
        self.task_script = v.task_script
        self.task_args = v.task_args
        self.launch_external_process = v.launch_external_process
        self.command = v.command
        self.framework = v.framework
        self.server_expected_format = v.server_expected_format
        self.params_transfer_type = v.params_transfer_type
        self.launch_once = v.launch_once
        self.shutdown_timeout = v.shutdown_timeout
        self.client_memory_gc_rounds = v.client_memory_gc_rounds
        self.cuda_empty_cache = v.cuda_empty_cache

        job = FedJob(name=self.name, min_clients=self.min_clients)

        controller = CmdTaskController(
            task_name=self.task_name,
            task_data=self.task_data,
            task_meta=self.task_meta,
            num_clients=self.num_clients,
            min_responses=self.min_responses,
            timeout=self.timeout,
            persistor_id="",
        )
        job.to_server(controller)

        executor = ScriptRunner(
            script=self.task_script,
            script_args=self.task_args,
            launch_external_process=self.launch_external_process,
            command=self.command,
            framework=self.framework,
            server_expected_format=self.server_expected_format,
            params_transfer_type=self.params_transfer_type,
            launch_once=self.launch_once,
            shutdown_timeout=self.shutdown_timeout,
            memory_gc_rounds=self.client_memory_gc_rounds,
            cuda_empty_cache=self.cuda_empty_cache,
        )
        job.to_clients(executor, tasks=[self.task_name])

        super().__init__(job)
