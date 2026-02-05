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

from typing import Any, Optional, Union

from pydantic import BaseModel

from nvflare.app_common.workflows.scaffold import Scaffold
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.app_opt.pt.job_config.model import PTModel
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.fuel.utils.constants import FrameworkType
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.recipe.spec import Recipe


# Internal — not part of the public API
class _ScaffoldValidator(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str
    initial_model: Any
    initial_ckpt: Optional[str] = None
    min_clients: int
    num_rounds: int
    train_script: str
    train_args: str
    launch_external_process: bool = False
    command: str = "python3 -u"
    server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY
    params_transfer_type: TransferType = TransferType.FULL
    server_memory_gc_rounds: int = 0


class ScaffoldRecipe(Recipe):
    """A recipe for implementing Scaffold in NVFlare.

    Implements the training algorithm proposed in
    Karimireddy et al. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
    (https://arxiv.org/abs/1910.06378). The client script is assumed to be using functions implemented in `PTScaffoldHelper` class.

    This recipe sets up a complete federated learning workflow with Scaffold controller.

    Args:
        name: Name of the federated learning job. Defaults to "scaffold".
        initial_model: Initial model to start federated training with. Can be:
            - nn.Module instance
            - Dict config: {"path": "module.ClassName", "args": {"param": value}}
            - None: no initial model
        initial_ckpt: Absolute path to a pre-trained checkpoint file. The file may not
            exist locally as it could be on the server. Used to load initial weights.
            Note: PyTorch requires initial_model when using initial_ckpt (for architecture).
        min_clients: Minimum number of clients required to start a training round. Defaults to 2.
        num_rounds: Number of federated training rounds to execute. Defaults to 2.
        train_script: Path to the training script that will be executed on each client. Defaults to "client.py".
        train_args: Command line arguments to pass to the training script. Defaults to "".
        server_memory_gc_rounds: Run memory cleanup (gc.collect + malloc_trim) every N rounds on server.
            Set to 0 to disable. Defaults to 0.
    Example:
        ```python
        recipe = ScaffoldRecipe(
            name="my_scaffold_job",
            initial_model=pretrained_model,
            min_clients=2,
            num_rounds=10,
            train_script="client.py",
            train_args="--epochs 5 --batch_size 32"
        )
        ```
    """

    def __init__(
        self,
        *,
        name: str = "scaffold",
        initial_model: Union[Any, dict[str, Any], None] = None,
        initial_ckpt: Optional[str] = None,
        min_clients: int,
        num_rounds: int = 2,
        train_script: str,
        train_args: str = "",
        launch_external_process: bool = False,
        command: str = "python3 -u",
        server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY,
        params_transfer_type: TransferType = TransferType.FULL,
        server_memory_gc_rounds: int = 0,
    ):
        # Validate inputs internally
        v = _ScaffoldValidator(
            name=name,
            initial_model=initial_model,
            initial_ckpt=initial_ckpt,
            min_clients=min_clients,
            num_rounds=num_rounds,
            train_script=train_script,
            train_args=train_args,
            launch_external_process=launch_external_process,
            command=command,
            server_expected_format=server_expected_format,
            params_transfer_type=params_transfer_type,
            server_memory_gc_rounds=server_memory_gc_rounds,
        )

        self.name = v.name
        self.initial_model = v.initial_model
        self.initial_ckpt = v.initial_ckpt

        # Validate inputs using shared utilities
        from nvflare.recipe.utils import validate_dict_model_config, validate_initial_ckpt

        validate_initial_ckpt(self.initial_ckpt)
        validate_dict_model_config(self.initial_model)

        self.min_clients = v.min_clients
        self.num_rounds = v.num_rounds
        self.train_script = v.train_script
        self.train_args = v.train_args
        self.launch_external_process = v.launch_external_process
        self.command = v.command
        self.server_expected_format: ExchangeFormat = v.server_expected_format
        self.params_transfer_type: TransferType = v.params_transfer_type
        self.server_memory_gc_rounds = v.server_memory_gc_rounds

        # Create BaseFedJob
        job = BaseFedJob(
            initial_model=None,  # We'll setup model below
            name=self.name,
            min_clients=self.min_clients,
        )

        # Setup model persistor using PTModel
        persistor_id = ""
        if self.initial_model is not None or self.initial_ckpt is not None:
            pt_model = PTModel(model=self.initial_model, initial_ckpt=self.initial_ckpt)
            result = job.to_server(pt_model, id="persistor")
            persistor_id = result["persistor_id"]

        # Define the controller and send to server
        controller = Scaffold(
            num_clients=self.min_clients,  # Scaffold controller requires the number of clients to be the same as the min_clients
            num_rounds=self.num_rounds,
            persistor_id=persistor_id,
            memory_gc_rounds=self.server_memory_gc_rounds,
        )
        # Send the controller to the server
        job.to_server(controller)

        # Add clients
        executor = ScriptRunner(
            script=self.train_script,
            script_args=self.train_args,
            launch_external_process=self.launch_external_process,
            command=self.command,
            framework=FrameworkType.PYTORCH,
            server_expected_format=self.server_expected_format,
            params_transfer_type=self.params_transfer_type,
        )
        job.to_clients(executor)

        Recipe.__init__(self, job)
