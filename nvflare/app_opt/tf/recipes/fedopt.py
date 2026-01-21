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

from nvflare.app_opt.tf.fedopt_ctl import FedOpt
from nvflare.app_opt.tf.job_config.base_fed_job import BaseFedJob
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner
from nvflare.recipe.spec import Recipe


class _FedOptValidator(BaseModel):
    # Allow custom types (e.g., Aggregator) in validation. Required by Pydantic v2.
    model_config = {"arbitrary_types_allowed": True}

    name: str = "fedopt"
    initial_model: Any = None
    min_clients: int
    num_rounds: int = 2
    train_script: str
    train_args: str = ""
    launch_external_process: bool = False
    command: str = "python3 -u"
    server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY
    params_transfer_type: TransferType = TransferType.FULL
    optimizer_args: dict = None
    lr_scheduler_args: dict = None


class FedOptRecipe(Recipe):
    """A recipe for implementing Federated Optimization (FedOpt) in NVFlare with TensorFlow.

    FedOpt is a federated learning algorithm that uses server-side optimization with momentum
    to improve convergence. The algorithm is proposed in Reddi et al. "Adaptive Federated
    Optimization." arXiv preprint arXiv:2003.00295 (2020).

    This recipe sets up a complete federated learning workflow with FedOpt controller that
    applies momentum-based updates on the server side.

    The recipe configures:
    - A federated job with initial TensorFlow model (optional)
    - FedOpt controller for coordinating training rounds with server-side optimization
    - Script runners for client-side training execution
    - Model selector for tracking best model

    Args:
        name: Name of the federated learning job. Defaults to "fedopt".
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
        optimizer_args: Dictionary of server-side optimizer arguments with keys 'path' and 'args'.
            Defaults to SGD with learning_rate=1.0 and momentum=0.6.
        lr_scheduler_args: Dictionary of server-side learning rate scheduler arguments with keys
            'path' and 'args'. Defaults to CosineDecay with initial_learning_rate=1.0 and alpha=0.9.

    Example:
        ```python
        from nvflare.app_opt.tf.recipes import FedOptRecipe
        from src.model import ModerateTFNet

        model = ModerateTFNet(input_shape=(None, 32, 32, 3))

        recipe = FedOptRecipe(
            name="my_fedopt_job",
            initial_model=model,
            min_clients=8,
            num_rounds=50,
            train_script="cifar10_fedopt/client.py",
            train_args="--batch_size 64 --epochs 4"
        )

        job = recipe.create_job()
        job.simulator_run("/tmp/nvflare/jobs/my_fedopt_job", gpu="0")
        ```

    Note:
        FedOpt applies server-side momentum to aggregated client updates, which can lead to
        faster convergence compared to standard FedAvg, especially in heterogeneous settings.
    """

    def __init__(
        self,
        *,
        name: str = "fedopt",
        initial_model: Any = None,
        min_clients: int,
        num_rounds: int = 2,
        train_script: str,
        train_args: str = "",
        launch_external_process: bool = False,
        command: str = "python3 -u",
        server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY,
        params_transfer_type: TransferType = TransferType.FULL,
        optimizer_args: dict = None,
        lr_scheduler_args: dict = None,
    ):
        # Validate inputs internally
        v = _FedOptValidator(
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
            optimizer_args=optimizer_args,
            lr_scheduler_args=lr_scheduler_args,
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
        self.optimizer_args = v.optimizer_args
        self.lr_scheduler_args = v.lr_scheduler_args

        # Create BaseFedJob with initial model
        job = BaseFedJob(
            initial_model=self.initial_model,
            name=self.name,
            min_clients=self.min_clients,
        )

        # Add FedOpt controller to server
        controller = FedOpt(
            num_clients=self.min_clients,
            num_rounds=self.num_rounds,
            optimizer_args=self.optimizer_args,
            lr_scheduler_args=self.lr_scheduler_args,
        )

        # Send the controller to the server
        job.to(controller, "server")

        # Add clients
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
