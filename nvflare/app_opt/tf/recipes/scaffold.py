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

from typing import Any

from pydantic import BaseModel

from nvflare.app_common.workflows.scaffold import Scaffold
from nvflare.app_opt.tf.job_config.base_fed_job import BaseFedJob
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner
from nvflare.recipe.spec import Recipe


class _ScaffoldValidator(BaseModel):
    # Allow custom types (e.g., Aggregator) in validation. Required by Pydantic v2.
    model_config = {"arbitrary_types_allowed": True}

    name: str = "scaffold"
    initial_model: Any = None
    min_clients: int
    num_rounds: int = 2
    train_script: str
    train_args: str = ""
    launch_external_process: bool = False
    command: str = "python3 -u"
    server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY
    params_transfer_type: TransferType = TransferType.FULL


class ScaffoldRecipe(Recipe):
    """A recipe for implementing SCAFFOLD in NVFlare with TensorFlow.

    Implements the training algorithm proposed in
    Karimireddy et al. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
    (https://arxiv.org/abs/1910.06378). The client script is assumed to be using functions
    implemented in `TFScaffoldHelper` class.

    This recipe sets up a complete federated learning workflow with SCAFFOLD controller.

    The recipe configures:
    - A federated job with initial TensorFlow model (optional)
    - SCAFFOLD controller for coordinating training rounds with control variates
    - Script runners for client-side training execution
    - Model selector for tracking best model

    Args:
        name: Name of the federated learning job. Defaults to "scaffold".
        initial_model: Initial TensorFlow model to start federated training with. If None,
            clients will start with their own local models.
        min_clients: Minimum number of clients required to start a training round.
        num_rounds: Number of federated training rounds to execute. Defaults to 2.
        train_script: Path to the training script that will be executed on each client.
        train_args: Command line arguments to pass to the training script. Defaults to "".
        launch_external_process: Whether to launch the script in external process. Defaults to False.
        command: If launch_external_process=True, command to run script (prepended to script).
            Defaults to "python3 -u".
        server_expected_format: What format to exchange the parameters between server and client.
            Defaults to ExchangeFormat.NUMPY.
        params_transfer_type: How to transfer the parameters between server and client.
            FULL means the whole model parameters are sent. DIFF means that only the difference is sent.
            Defaults to TransferType.FULL.

    Example:
        ```python
        from nvflare.app_opt.tf.recipes import ScaffoldRecipe
        from src.model import ModerateTFNet

        model = ModerateTFNet(input_shape=(None, 32, 32, 3))

        recipe = ScaffoldRecipe(
            name="my_scaffold_job",
            initial_model=model,
            min_clients=8,
            num_rounds=50,
            train_script="cifar10_scaffold/client.py",
            train_args="--batch_size 64 --epochs 4"
        )

        job = recipe.create_job()
        job.simulator_run("/tmp/nvflare/jobs/my_scaffold_job", gpu="0")
        ```

    Note:
        The client script must use `TFScaffoldHelper` from `nvflare.app_opt.tf.scaffold`
        to handle SCAFFOLD-specific operations including control variates.
    """

    def __init__(
        self,
        *,
        name: str = "scaffold",
        initial_model: Any = None,
        min_clients: int,
        num_rounds: int = 2,
        train_script: str,
        train_args: str = "",
        launch_external_process: bool = False,
        command: str = "python3 -u",
        server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY,
        params_transfer_type: TransferType = TransferType.FULL,
    ):
        # Validate inputs internally
        v = _ScaffoldValidator(
            name=name,
            initial_model=initial_model,
            min_clients=min_clients,
            num_rounds=num_rounds,
            train_script=train_script,
            train_args=train_args,
            launch_external_process=launch_external_process,
            command=command,
            server_expected_format=server_expected_format,
            params_transfer_type=params_transfer_type,
        )

        self.name = v.name
        self.initial_model = v.initial_model
        self.min_clients = v.min_clients
        self.num_rounds = v.num_rounds
        self.train_script = v.train_script
        self.train_args = v.train_args
        self.launch_external_process = v.launch_external_process
        self.command = v.command
        self.server_expected_format: ExchangeFormat = v.server_expected_format
        self.params_transfer_type: TransferType = v.params_transfer_type

        # Create BaseFedJob with initial model
        job = BaseFedJob(
            initial_model=self.initial_model,
            name=self.name,
            min_clients=self.min_clients,
        )

        # Add SCAFFOLD controller to server
        controller = Scaffold(
            num_clients=self.min_clients,
            num_rounds=self.num_rounds,
        )
        # Send the controller to the server
        job.to(controller, "server")

        # Add initial model to server if provided
        executor = ScriptRunner(
            script=self.train_script,
            script_args=self.train_args,
            launch_external_process=self.launch_external_process,
            command=self.command,
            framework=FrameworkType.TENSORFLOW,
            server_expected_format=self.server_expected_format,
            params_transfer_type=self.params_transfer_type,
        )
        job.to_clients(executor)

        Recipe.__init__(self, job)
