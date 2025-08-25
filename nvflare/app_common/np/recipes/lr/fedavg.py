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
from typing import Any, List, Optional

from pydantic import BaseModel, PositiveInt

from nvflare import FedJob
from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE
from nvflare.app_common.widgets.convert_to_fed_event import ConvertToFedEvent
from nvflare.app_common.workflows.lr.fedavg import FedAvgLR
from nvflare.app_common.workflows.lr.np_persistor import LRModelPersistor
from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner
from nvflare.recipe.spec import Recipe


# Internal — not part of the public API
class _FedAvgValidator(BaseModel):
    name: str
    initial_model: Any
    clients: Optional[List[str]]
    num_clients: Optional[PositiveInt]
    min_clients: int
    num_rounds: int
    damping_factor: float
    train_script: str
    train_args: str
    launch_external_process: bool = False
    command: str

    def model_post_init(self, __context):
        if self.clients and self.num_clients is None:
            self.num_clients = len(self.clients)
        elif self.clients and len(self.clients) != self.min_clients:
            raise ValueError("inconsistent number of clients")


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
        initial_model: Initial model to start federated training with. If None,
            clients will start with their own local models.
        clients: List of selected client names to participate in training. If None,
            all available clients will be used.
        num_clients: Number of sampled clients expected to participate. If clients is provided,
            this will be set automatically to len(clients).
        min_clients: Minimum number of clients required to start a training round.
            Defaults to 0 (no minimum).
        num_rounds: Number of federated training rounds to execute. Defaults to 2.
        damping_factor: default to 0.8
        train_script: Path to the training script that will be executed on each client.
        train_args: Command line arguments to pass to the training script.
        launch_external_process (bool): Whether to launch the script in external process. Defaults to False.
        command (str): If launch_external_process=True, command to run script (prepended to script). Defaults to "python3".

    Example:
        ```python
        recipe = FedAvgLrRecipe(
            name="lr_fedavg",
            initial_model=pretrained_model,
            num_clients=3,
            min_clients=2,
            num_rounds=10,
            train_script="client.py",
        )
        ```
    """

    def __init__(
        self,
        *,
        name: str = "lr_fedavg",
        initial_model: Any = None,
        clients: Optional[List[str]] = None,
        num_clients: Optional[int] = None,
        min_clients: int = 0,
        num_rounds: int = 2,
        damping_factor=0.8,
        train_script: str,
        train_args: str = "",
        launch_external_process=False,
        command: str = "python3 -u",
    ):
        # Validate inputs internally
        v = _FedAvgValidator(
            name=name,
            initial_model=initial_model,
            clients=clients,
            num_clients=num_clients,
            min_clients=min_clients,
            num_rounds=num_rounds,
            damping_factor=damping_factor,
            train_script=train_script,
            train_args=train_args,
            launch_external_process=launch_external_process,
            command=command,
        )

        self.name = v.name
        self.initial_model = v.initial_model
        self.clients = v.clients
        self.num_clients = v.num_clients
        self.min_clients = v.min_clients
        self.num_rounds = v.num_rounds
        self.damping_factor = v.damping_factor
        self.initial_model = v.initial_model
        self.clients = v.clients
        self.train_script = v.train_script
        self.train_args = v.train_args
        self.launch_external_process = v.launch_external_process
        self.command = v.command

        # Create FedJob.
        job = FedJob(name=self.name)
        persistor_id = job.to_server(LRModelPersistor(n_features=13), id="lr_persistor")

        # Send custom controller to server
        controller = FedAvgLR(
            n_clients=0,  # for all clients
            min_clients=self.min_clients,
            damping_factor=self.damping_factor,
            num_rounds=self.num_rounds,
            initial_model=self.initial_model,
            persistor_id=persistor_id,
        )
        job.to(controller, "server")

        # Send TBAnalyticsReceiver to server for tensorboard streaming.
        analytics_receiver = TBAnalyticsReceiver()
        job.to_server(
            id="receiver",
            obj=analytics_receiver,
        )
        convert_to_fed_event = ConvertToFedEvent(events_to_convert=[ANALYTIC_EVENT_TYPE])

        # Add clients
        if self.clients is None:
            clients = [f"site-{i + 1}" for i in range(self.num_clients)]
        else:
            clients = self.clients

        for client in clients:
            job.to(id="event_to_fed", obj=convert_to_fed_event, target=client)
            runner = ScriptRunner(
                script=self.train_script,
                script_args=self.train_args,
                launch_external_process=self.launch_external_process,
                command=self.command,
                framework=FrameworkType.RAW,
                server_expected_format=ExchangeFormat.RAW,
                params_transfer_type=TransferType.FULL,
            )
            job.to(runner, client)

        Recipe.__init__(self, job)
