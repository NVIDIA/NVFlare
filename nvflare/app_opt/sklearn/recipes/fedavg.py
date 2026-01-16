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

from nvflare.apis.dxo import DataKind
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_opt.sklearn.joblib_model_param_persistor import JoblibModelParamPersistor
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.job_config.script_runner import FrameworkType
from nvflare.recipe.fedavg import FedAvgRecipe as UnifiedFedAvgRecipe


class SklearnFedAvgRecipe(UnifiedFedAvgRecipe):
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
        train_args: Command line arguments to pass to the training script.
        aggregator: Custom aggregator for combining client updates. If None,
            uses InTimeAccumulateWeightedAggregator with aggregator_data_kind.
        aggregator_data_kind: Data kind to use for the aggregator. Defaults to DataKind.WEIGHTS.
        launch_external_process: Whether to launch the script in external process. Defaults to False.
        command: If launch_external_process=True, command to run script (prepended to script).
            Defaults to "python3 -u".
        per_site_config: Per-site configuration for the federated learning job. Dictionary mapping
            site names to configuration dicts. If not provided, the same configuration will be used
            for all clients.
        launch_once: Whether the external process will be launched only once at the beginning
            or on each task. Only used if `launch_external_process` is True. Defaults to True.
        shutdown_timeout: If provided, will wait for this number of seconds before shutdown.
            Only used if `launch_external_process` is True. Defaults to 0.0.

    Example:
        Basic usage with same config for all clients:

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

        Per-site configuration:

        ```python
        from nvflare.app_opt.sklearn import SklearnFedAvgRecipe

        recipe = SklearnFedAvgRecipe(
            name="sklearn_linear",
            min_clients=3,
            num_rounds=50,
            model_params={"n_classes": 2, "learning_rate": "constant", "eta0": 1e-4},
            train_script="client.py",
            per_site_config={
                "site-1": {"train_args": "--data_path /tmp/data/site1.csv"},
                "site-2": {"train_args": "--data_path /tmp/data/site2.csv"},
                "site-3": {"train_args": "--data_path /tmp/data/site3.csv"},
            },
        )
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
        train_args: str = "",
        aggregator: Optional[Aggregator] = None,
        aggregator_data_kind: DataKind = DataKind.WEIGHTS,
        launch_external_process: bool = False,
        command: str = "python3 -u",
        per_site_config: Optional[dict[str, dict]] = None,
        launch_once: bool = True,
        shutdown_timeout: float = 0.0,
    ):
        # Create sklearn-specific persistor
        persistor = JoblibModelParamPersistor(initial_params=model_params or {})

        # Call the unified FedAvgRecipe with sklearn-specific settings
        super().__init__(
            name=name,
            min_clients=min_clients,
            num_rounds=num_rounds,
            train_script=train_script,
            train_args=train_args,
            aggregator=aggregator,
            aggregator_data_kind=aggregator_data_kind,
            launch_external_process=launch_external_process,
            command=command,
            framework=FrameworkType.RAW,  # sklearn uses RAW framework
            server_expected_format=ExchangeFormat.RAW,  # sklearn uses RAW exchange format
            params_transfer_type=TransferType.FULL,
            model_persistor=persistor,  # Pass sklearn-specific persistor
            per_site_config=per_site_config,
            launch_once=launch_once,
            shutdown_timeout=shutdown_timeout,
        )
