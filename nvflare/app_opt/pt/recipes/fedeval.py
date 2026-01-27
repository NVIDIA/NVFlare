# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any, Optional

from nvflare.app_common.workflows.model_controller import ModelController
from nvflare.client.config import ExchangeFormat
from nvflare.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner
from nvflare.recipe.spec import Recipe


class EvalController(ModelController):
    def __init__(self, persistor_id: str, timeout: int):
        super().__init__(persistor_id=persistor_id)
        self.timeout = timeout

    def run(self):
        model = self.load_model()
        self.info("Sending model for evaluation")
        results = self.send_model_and_wait(targets=None, data=model, task_name="validate", timeout=self.timeout)
        self.info(f"Got {len(results)} results")
        for r in results:
            self.info(f"Metrics: {r.metrics}")


class FedEvalRecipe(Recipe):
    """A recipe for federated evaluation of a PyTorch model across multiple sites.

    This recipe sets up a federated evaluation workflow where a global model
    from the server is sent to multiple clients for evaluation. Each client evaluates
    the model on their local data and reports metrics back to the server.

    The recipe configures:
    - A federated job with an initial model to evaluate
    - EvalController for coordinating federated evaluation across clients
    - Script runners for client-side evaluation execution

    Args:
        name: Name of the federated evaluation job. Defaults to "eval".
        initial_model: Model structure to evaluate (nn.Module). Required.
            required to have a checkpoint attribute.
        min_clients: Minimum number of clients required to start evaluation.
        eval_script: Path to the evaluation script that will be executed on each client.
        eval_args: Command line arguments to pass to the evaluation script. Defaults to "".
        launch_external_process: Whether to launch the script in external process. Defaults to False.
        command: If launch_external_process=True, command to run script (prepended to script).
            Defaults to "python3 -u".
        server_expected_format: What format to exchange the parameters between server and client.
            Defaults to ExchangeFormat.NUMPY.
        validation_timeout: Timeout for evaluation task in seconds. Defaults to 6000.
        per_site_config: Per-site configuration for the evaluation job. Dictionary mapping
            site names to configuration dicts. Each config dict can contain optional overrides:
            eval_script, eval_args, launch_external_process, command, server_expected_format.
            If not provided, the same configuration will be used for all clients. Defaults to None.
        client_memory_gc_rounds: Run memory cleanup every N rounds on client. Defaults to 0 (disabled).
        torch_cuda_empty_cache: If True, call torch.cuda.empty_cache() during cleanup. Defaults to False.

    Example:
        Basic usage:

        ```python
        from nvflare.app_opt.pt.recipes.fedeval import FedEvalRecipe
        from model import LitNet

        recipe = FedEvalRecipe(
            name="eval_job",
            initial_model=LitNet(checkpoint="pretrained_model.pt"),
            min_clients=2,
            eval_script="client.py",
            eval_args="--batch_size 32",
        )
        ```
    """

    def __init__(
        self,
        *,
        name: str = "eval",
        initial_model: Any,
        min_clients: int,
        eval_script: str,
        eval_args: str = "",
        launch_external_process: bool = False,
        command: str = "python3 -u",
        server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY,
        validation_timeout: int = 6000,
        per_site_config: Optional[dict[str, dict]] = None,
        client_memory_gc_rounds: int = 0,
        torch_cuda_empty_cache: bool = False,
    ):
        self.name = name
        self.initial_model = initial_model
        self.min_clients = min_clients
        self.eval_script = eval_script
        self.eval_args = eval_args
        self.launch_external_process = launch_external_process
        self.command = command
        self.server_expected_format = server_expected_format
        self.validation_timeout = validation_timeout
        self.per_site_config = per_site_config
        self.client_memory_gc_rounds = client_memory_gc_rounds
        self.torch_cuda_empty_cache = torch_cuda_empty_cache
        self.source_checkpoint = initial_model.checkpoint
        if self.source_checkpoint is None:
            raise ValueError("initial_model must have a checkpoint attribute")

        # Create BaseFedJob
        job = BaseFedJob(
            name=self.name,
            min_clients=self.min_clients,
        )

        # Setup model and persistor using PTModel (handles serialization properly)
        import torch.nn as nn

        from nvflare.app_opt.pt import PTFileModelPersistor
        from nvflare.app_opt.pt.job_config.model import PTModel

        if not isinstance(self.initial_model, nn.Module):
            raise ValueError(f"initial_model must be nn.Module, got {type(self.initial_model)}")

        persistor = PTFileModelPersistor(model=self.initial_model, source_ckpt_file_full_name=self.source_checkpoint)

        # Use PTModel to add model and persistor properly
        pt_model = PTModel(model=self.initial_model, persistor=persistor)
        job.comp_ids.update(job.to_server(pt_model))
        persistor_id = job.comp_ids.get("persistor_id", "")

        # Simple controller
        controller = EvalController(persistor_id=persistor_id, timeout=self.validation_timeout)
        job.to_server(controller)

        # Add client executors
        if self.per_site_config is not None:
            for site_name, site_config in self.per_site_config.items():
                script = site_config.get("eval_script", self.eval_script)
                script_args = site_config.get("eval_args", self.eval_args)
                launch_external = site_config.get("launch_external_process", self.launch_external_process)
                cmd = site_config.get("command", self.command)
                expected_format = site_config.get("server_expected_format", self.server_expected_format)

                executor = ScriptRunner(
                    script=script,
                    script_args=script_args,
                    launch_external_process=launch_external,
                    command=cmd,
                    framework=FrameworkType.PYTORCH,
                    server_expected_format=expected_format,
                    memory_gc_rounds=self.client_memory_gc_rounds,
                    torch_cuda_empty_cache=self.torch_cuda_empty_cache,
                )
                job.to(executor, site_name)
        else:
            executor = ScriptRunner(
                script=self.eval_script,
                script_args=self.eval_args,
                launch_external_process=self.launch_external_process,
                command=self.command,
                framework=FrameworkType.PYTORCH,
                server_expected_format=self.server_expected_format,
                memory_gc_rounds=self.client_memory_gc_rounds,
                torch_cuda_empty_cache=self.torch_cuda_empty_cache,
            )
            job.to_clients(executor)

        Recipe.__init__(self, job)
