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

from typing import Optional

from pydantic import BaseModel, PositiveInt, field_validator

from nvflare import FedJob
from nvflare.app_common.workflows.lr.fedavg import FedAvgLR
from nvflare.app_common.workflows.lr.np_persistor import LRModelPersistor
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner
from nvflare.recipe.spec import Recipe
from nvflare.recipe.utils import validate_ckpt


# Internal — not part of the public API
class _FedAvgValidator(BaseModel):
    name: str
    min_clients: PositiveInt
    num_rounds: int
    damping_factor: float
    num_features: PositiveInt
    initial_ckpt: Optional[str] = None
    train_script: str
    train_args: str
    launch_external_process: bool = False
    command: str
    client_memory_gc_rounds: int = 0
    cuda_empty_cache: bool = False

    @field_validator("initial_ckpt")
    @classmethod
    def validate_initial_ckpt(cls, v):
        if v is not None:
            validate_ckpt(v)
        return v


class FedAvgLrRecipe(Recipe):
    """A recipe for implementing Federated Averaging (FedAvg) for Logistic Regression with Newton Raphson.

    FedAvg is a fundamental federated learning algorithm that aggregates model updates
    from multiple clients by computing a weighted average based on the amount of local
    training data. This recipe sets up a complete federated learning workflow using
    the FedAvgLR controller specifically designed for logistic regression.

    The recipe configures:
    - A federated job with logistic regression model
    - FedAvgLR controller for Newton-Raphson based aggregation
    - Script runners for client-side training execution

    Args:
        name: Name of the federated learning job. Defaults to "lr_fedavg".
        min_clients: Minimum number of clients required to start a training round.
        num_rounds: Number of federated training rounds to execute. Defaults to 2.
        damping_factor: default to 0.8
        num_features: Number of features for the logistic regression. Defaults to 13.
        initial_ckpt: Absolute path to a pre-trained checkpoint file (.npy).
            The file may not exist locally as it could be on the server.
            Used to resume training from previously saved weights.
        train_script: Path to the training script that will be executed on each client.
        train_args: Command line arguments to pass to the training script.
        launch_external_process (bool): Whether to launch the script in external process. Defaults to False.
        command (str): If launch_external_process=True, command to run script (prepended to script). Defaults to "python3".

    Example:
        ```python
            recipe = FedAvgLrRecipe(min_clients=2,
                            num_rounds=num_rounds,
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
        min_clients: int,
        num_rounds: int = 2,
        damping_factor=0.8,
        num_features=13,
        initial_ckpt: Optional[str] = None,
        train_script: str,
        train_args: str = "",
        launch_external_process=False,
        command: str = "python3 -u",
        client_memory_gc_rounds: int = 0,
        cuda_empty_cache: bool = False,
    ):
        # Validate inputs internally
        v = _FedAvgValidator(
            name=name,
            min_clients=min_clients,
            num_rounds=num_rounds,
            damping_factor=damping_factor,
            num_features=num_features,
            initial_ckpt=initial_ckpt,
            train_script=train_script,
            train_args=train_args,
            launch_external_process=launch_external_process,
            command=command,
            client_memory_gc_rounds=client_memory_gc_rounds,
            cuda_empty_cache=cuda_empty_cache,
        )

        self.name = v.name
        self.min_clients = v.min_clients
        self.num_rounds = v.num_rounds
        self.damping_factor = v.damping_factor
        self.initial_ckpt = v.initial_ckpt
        self.train_script = v.train_script
        self.train_args = v.train_args
        self.launch_external_process = v.launch_external_process
        self.command = v.command
        self.num_features = v.num_features
        self.client_memory_gc_rounds = v.client_memory_gc_rounds
        self.cuda_empty_cache = v.cuda_empty_cache

        # Create FedJob.
        job = FedJob(name=self.name, min_clients=self.min_clients)
        from nvflare.recipe.utils import prepare_initial_ckpt

        ckpt_path = prepare_initial_ckpt(self.initial_ckpt, job)
        persistor = LRModelPersistor(
            n_features=self.num_features,
            source_ckpt_file_full_name=ckpt_path,
        )
        persistor_id = job.to_server(persistor, id="lr_persistor")

        # Send custom controller to server
        controller = FedAvgLR(
            num_clients=self.min_clients,
            damping_factor=self.damping_factor,
            n_features=self.num_features,
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
            memory_gc_rounds=self.client_memory_gc_rounds,
            cuda_empty_cache=self.cuda_empty_cache,
        )

        job.to_clients(runner)
        Recipe.__init__(self, job)
