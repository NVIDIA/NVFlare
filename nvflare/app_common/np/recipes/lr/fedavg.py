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

from pydantic import BaseModel, PositiveInt

from nvflare import FedJob
from nvflare.app_common.workflows.lr.fedavg import FedAvgLR
from nvflare.app_common.workflows.lr.np_persistor import LRModelPersistor
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner
from nvflare.recipe.spec import Recipe


# Internal — not part of the public API
class _FedAvgValidator(BaseModel):
    name: str
    num_rounds: int
    damping_factor: float
    num_features: PositiveInt
    train_script: str
    train_args: str
    launch_external_process: bool = False
    command: str


class FedAvgLrRecipe(Recipe):
    """A recipe for implementing Federated Averaging (FedAvg) for Logistics Regression with Newton Raphson.
    FedAvg is a fundamental federated learning algorithm that aggregates model updates
    from multiple clients by computing a weighted average based on the amount of local
    training data. This recipe sets up a complete federated learning workflow with
    scatter-and-gather communication pattern.

    The recipe configures:
    - A federated job with initial model (optional)
    - Weighted aggregator for combining client model updates (or custom aggregator)
    - Script runners for client-side training execution

    Args:
        name: Name of the federated learning job. Defaults to "lr_fedavg".
        num_rounds: Number of federated training rounds to execute. Defaults to 2.
        damping_factor: default to 0.8
        train_script: Path to the training script that will be executed on each client.
        train_args: Command line arguments to pass to the training script.
        launch_external_process (bool): Whether to launch the script in external process. Defaults to False.
        command (str): If launch_external_process=True, command to run script (prepended to script). Defaults to "python3".

    Example:
        ```python
            recipe = FedAvgLrRecipe(num_rounds=num_rounds,
                            damping_factor=0.8,
                            num_features=13,
                            train_script="client.py",
                            train_args=f"--data_root {data_root}")
        ```
    """

    def __init__(
        self,
        *,
        name: str = "lr_fedavg",
        num_rounds: int = 2,
        damping_factor=0.8,
        num_features=13,
        train_script: str,
        train_args: str = "",
        launch_external_process=False,
        command: str = "python3 -u",
    ):
        # Validate inputs internally
        v = _FedAvgValidator(
            name=name,
            num_rounds=num_rounds,
            damping_factor=damping_factor,
            num_features=num_features,
            train_script=train_script,
            train_args=train_args,
            launch_external_process=launch_external_process,
            command=command,
        )

        self.name = v.name
        self.num_rounds = v.num_rounds
        self.damping_factor = v.damping_factor
        self.train_script = v.train_script
        self.train_args = v.train_args
        self.launch_external_process = v.launch_external_process
        self.command = v.command
        self.num_features = v.num_features

        # Create FedJob.
        job = FedJob(name=self.name)
        persistor_id = job.to_server(LRModelPersistor(n_features=13), id="lr_persistor")

        # Send custom controller to server
        controller = FedAvgLR(
            num_clients=0,
            damping_factor=self.damping_factor,
            n_features=num_features,
            num_rounds=self.num_rounds,
            persistor_id=persistor_id,
        )
        job.to(controller, "server")

        runner = ScriptRunner(
            script=self.train_script,
            script_args=self.train_args,
            launch_external_process=self.launch_external_process,
            command=self.command,
            framework=FrameworkType.RAW,
            server_expected_format=ExchangeFormat.RAW,
            params_transfer_type=TransferType.FULL,
        )

        job.to_clients(runner)
        Recipe.__init__(self, job)
