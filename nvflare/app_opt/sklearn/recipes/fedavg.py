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

from typing import Dict, Optional, Union

from pydantic import BaseModel

from nvflare import FedJob
from nvflare.apis.dxo import DataKind
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator
from nvflare.app_common.shareablegenerators import FullModelShareableGenerator
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.sklearn.joblib_model_param_persistor import JoblibModelParamPersistor
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner
from nvflare.recipe.spec import Recipe


# Internal — not part of the public API
class _SklearnFedAvgValidator(BaseModel):
    # Allow custom types (e.g., Aggregator) in validation. Required by Pydantic v2.
    model_config = {"arbitrary_types_allowed": True}

    name: str
    min_clients: int
    num_rounds: int
    model_params: Optional[dict] = None
    train_script: str
    train_args: Union[str, Dict[str, str]]
    aggregator: Optional[Aggregator] = None
    aggregator_data_kind: DataKind = DataKind.WEIGHTS
    launch_external_process: bool = False
    command: str = "python3 -u"


class SklearnFedAvgRecipe(Recipe):
    """A recipe for implementing Federated Averaging (FedAvg) with Scikit-learn.

    This recipe sets up a complete federated learning workflow with scatter-and-gather
    communication pattern specifically designed for scikit-learn models.

    The recipe configures:
    - A federated job with initial parameters
    - Scatter-and-gather controller for coordinating training rounds
    - Weighted aggregator for combining client model updates (or custom aggregator)
    - Script runners for client-side training execution

    Args:
        name: Name of the federated learning job. Defaults to "sklearn_fedavg".
        min_clients: Minimum number of clients required to start a training round.
        num_rounds: Number of federated training rounds to execute. Defaults to 2.
        model_params: Model hyperparameters as a dictionary. For SGDClassifier, can include:
            n_classes, learning_rate, eta0, loss, penalty, fit_intercept, etc.
            Can also include initial weights if needed.
        train_script: Path to the training script that will be executed on each client.
        train_args: Command line arguments to pass to the training script. Can be:
            - str: Same arguments for all clients (uses job.to_clients)
            - dict[str, str]: Per-client arguments mapping site names to args (uses job.to per site)
        aggregator: Custom aggregator for combining client updates. If None,
            uses InTimeAccumulateWeightedAggregator with aggregator_data_kind.
        aggregator_data_kind: Data kind to use for the aggregator. Defaults to DataKind.WEIGHTS.
        launch_external_process: Whether to launch the script in external process. Defaults to False.
        command: If launch_external_process=True, command to run script (prepended to script).
            Defaults to "python3 -u".

    Example:
        ```python
        recipe = SklearnFedAvgRecipe(
            name="sklearn_linear",
            min_clients=5,
            num_rounds=50,
            model_params={
                "n_classes": 2,
                "learning_rate": "constant",
                "eta0": 1e-4,
                "loss": "log_loss",
                "penalty": "l2",
                "fit_intercept": 1,
            },
            train_script="client.py",
            train_args="--data_path /tmp/data/HIGGS.csv",
        )

        from nvflare.recipe import SimEnv
        env = SimEnv(num_clients=5)
        run = recipe.execute(env)
        print("Result:", run.get_result())
        ```

    Note:
        By default, this recipe implements the standard FedAvg algorithm where model updates
        are aggregated using weighted averaging based on the number of training
        samples provided by each client.

        If you want to use a custom aggregator, you can pass it in the aggregator parameter.
        The custom aggregator must be a subclass of the Aggregator class.
    """

    def __init__(
        self,
        *,
        name: str = "sklearn_fedavg",
        min_clients: int,
        num_rounds: int = 2,
        model_params: Optional[dict] = None,
        train_script: str,
        train_args: Union[str, Dict[str, str]] = "",
        aggregator: Optional[Aggregator] = None,
        aggregator_data_kind: DataKind = DataKind.WEIGHTS,
        launch_external_process: bool = False,
        command: str = "python3 -u",
    ):
        # Validate inputs internally
        v = _SklearnFedAvgValidator(
            name=name,
            min_clients=min_clients,
            num_rounds=num_rounds,
            model_params=model_params,
            train_script=train_script,
            train_args=train_args,
            aggregator=aggregator,
            aggregator_data_kind=aggregator_data_kind,
            launch_external_process=launch_external_process,
            command=command,
        )

        self.name = v.name
        self.min_clients = v.min_clients
        self.num_rounds = v.num_rounds
        self.model_params = v.model_params
        self.train_script = v.train_script
        self.train_args = v.train_args
        self.aggregator = v.aggregator
        self.aggregator_data_kind = v.aggregator_data_kind
        self.launch_external_process = v.launch_external_process
        self.command = v.command

        # Create FedJob
        job = FedJob(name=self.name, min_clients=self.min_clients)

        # Server components
        persistor = JoblibModelParamPersistor(initial_params=self.model_params or {})
        persistor_id = job.to_server(persistor, id="persistor")

        shareable_generator = FullModelShareableGenerator()
        shareable_generator_id = job.to_server(shareable_generator, id="shareable_generator")

        if self.aggregator is None:
            self.aggregator = InTimeAccumulateWeightedAggregator(expected_data_kind=self.aggregator_data_kind)
        else:
            if not isinstance(self.aggregator, Aggregator):
                raise ValueError(f"Invalid aggregator type: {type(self.aggregator)}. Expected type: {Aggregator}")
        aggregator_id = job.to_server(self.aggregator, id="aggregator")

        controller = ScatterAndGather(
            min_clients=self.min_clients,
            num_rounds=self.num_rounds,
            wait_time_after_min_received=0,
            aggregator_id=aggregator_id,
            persistor_id=persistor_id,
            shareable_generator_id=shareable_generator_id,
            train_task_name="train",
        )
        job.to_server(controller)

        # Client components
        if isinstance(self.train_args, dict):
            # Per-client configuration: add executor for each client with their specific args
            for site_name, site_args in self.train_args.items():
                executor = ScriptRunner(
                    script=self.train_script,
                    script_args=site_args,
                    launch_external_process=self.launch_external_process,
                    command=self.command,
                    framework=FrameworkType.RAW,
                    server_expected_format=ExchangeFormat.RAW,
                    params_transfer_type=TransferType.FULL,
                )
                job.to(executor, site_name)
        else:
            # Unified configuration: same args for all clients
            executor = ScriptRunner(
                script=self.train_script,
                script_args=self.train_args,
                launch_external_process=self.launch_external_process,
                command=self.command,
                framework=FrameworkType.RAW,
                server_expected_format=ExchangeFormat.RAW,
                params_transfer_type=TransferType.FULL,
            )
            job.to_clients(executor)

        Recipe.__init__(self, job)
