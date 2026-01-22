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

from typing import Any, Dict, Optional

from nvflare.apis.dxo import DataKind
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.job_config.script_runner import FrameworkType
from nvflare.recipe.fedavg import FedAvgRecipe as UnifiedFedAvgRecipe


class NumpyFedAvgRecipe(UnifiedFedAvgRecipe):
    """A recipe for implementing Federated Averaging (FedAvg) with NumPy in NVFlare.

    FedAvg is a fundamental federated learning algorithm that aggregates model updates
    from multiple clients by computing a weighted average based on the amount of local
    training data. This recipe sets up a complete federated learning workflow with
    memory-efficient InTime aggregation, specifically designed for NumPy-based models.

    The recipe configures:
    - A federated job with initial model (optional)
    - FedAvg controller with InTime aggregation for memory efficiency
    - Optional early stopping and model selection
    - Script runners for client-side training execution

    Args:
        name: Name of the federated learning job. Defaults to "fedavg".
        initial_model: Initial model (as list or numpy array) to start federated training with.
            Lists are preferred for JSON serialization compatibility. If None,
            clients will start with their own local models.
        min_clients: Minimum number of clients required to start a training round.
        num_rounds: Number of federated training rounds to execute. Defaults to 2.
        train_script: Path to the training script that will be executed on each client.
        train_args: Command line arguments to pass to the training script.
        aggregator: Custom aggregator (ModelAggregator) for combining client model updates.
            Must implement accept_model(), aggregate_model(), reset_stats() methods.
            If None, uses built-in memory-efficient weighted averaging.
        aggregator_data_kind: Data kind to use for the aggregator. Defaults to DataKind.WEIGHTS.
            Kept for backward compatibility.
        launch_external_process (bool): Whether to launch the script in external process. Defaults to False.
        command (str): If launch_external_process=True, command to run script (prepended to script).
            Defaults to "python3 -u".
        server_expected_format (str): What format to exchange the parameters between server and client.
        params_transfer_type (str): How to transfer the parameters. FULL means the whole model parameters are sent.
            DIFF means that only the difference is sent. Defaults to TransferType.FULL.
        per_site_config: Per-site configuration for the federated learning job.
        launch_once: Whether external process is launched once or per task. Defaults to True.
        shutdown_timeout: Seconds to wait before shutdown. Defaults to 0.0.
        key_metric: Metric used to determine if the model is globally best. Defaults to "accuracy".
        stop_cond: Early stopping condition based on metric. String literal in the format of
            '<key> <op> <value>' (e.g. "accuracy >= 80"). If None, early stopping is disabled.
        patience: Number of rounds with no improvement after which FL will be stopped.
        save_filename: Filename for saving the best model. Defaults to "FL_global_model.pt".
        exclude_vars: Regex pattern for variables to exclude from aggregation.
        aggregation_weights: Per-client aggregation weights dict. Defaults to equal weights.

    Example:
        ```python
        recipe = NumpyFedAvgRecipe(
            name="my_fedavg_job",
            initial_model=numpy_model,
            min_clients=2,
            num_rounds=10,
            train_script="client.py",
            train_args="--learning_rate 0.01",
            stop_cond="accuracy >= 95",
            patience=3
        )
        ```

    Note:
        This recipe uses InTime (streaming) aggregation for memory efficiency - each client
        result is aggregated immediately upon receipt rather than collecting all results first.
        Memory usage is constant regardless of the number of clients.

        By default, this recipe implements the standard FedAvg algorithm where model updates
        are aggregated using weighted averaging based on the number of training
        samples provided by each client.

        If you want to use a custom aggregator, you can pass it in the aggregator parameter.
        The custom aggregator must be a subclass of the ModelAggregator class.
    """

    def __init__(
        self,
        *,
        name: str = "fedavg",
        initial_model: Any = None,
        min_clients: int,
        num_rounds: int = 2,
        train_script: str,
        train_args: str = "",
        aggregator: Optional[Aggregator] = None,
        aggregator_data_kind: Optional[DataKind] = DataKind.WEIGHTS,
        launch_external_process: bool = False,
        command: str = "python3 -u",
        server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY,
        params_transfer_type: TransferType = TransferType.FULL,
        per_site_config: Optional[Dict[str, Dict]] = None,
        launch_once: bool = True,
        shutdown_timeout: float = 0.0,
        key_metric: str = "accuracy",
        # New FedAvg features
        stop_cond: Optional[str] = None,
        patience: Optional[int] = None,
        save_filename: str = "FL_global_model.pt",
        exclude_vars: Optional[str] = None,
        aggregation_weights: Optional[Dict[str, float]] = None,
    ):
        # Store initial_model for NumPy-specific setup
        self._np_initial_model = initial_model

        # Call the unified FedAvgRecipe with NumPy-specific settings
        super().__init__(
            name=name,
            initial_model=initial_model,
            min_clients=min_clients,
            num_rounds=num_rounds,
            train_script=train_script,
            train_args=train_args,
            aggregator=aggregator,
            aggregator_data_kind=aggregator_data_kind,
            launch_external_process=launch_external_process,
            command=command,
            framework=FrameworkType.NUMPY,
            server_expected_format=server_expected_format,
            params_transfer_type=params_transfer_type,
            model_persistor=None,  # We'll set up NPModelPersistor in _setup_model_and_persistor
            per_site_config=per_site_config,
            launch_once=launch_once,
            shutdown_timeout=shutdown_timeout,
            key_metric=key_metric,
            stop_cond=stop_cond,
            patience=patience,
            save_filename=save_filename,
            exclude_vars=exclude_vars,
            aggregation_weights=aggregation_weights,
        )

    def _setup_model_and_persistor(self, job) -> str:
        """Override to handle NumPy-specific model setup."""
        if self._np_initial_model is not None:
            from nvflare.app_common.np.np_model_persistor import NPModelPersistor

            persistor = NPModelPersistor(initial_model=self._np_initial_model)
            return job.to_server(persistor, id="persistor")
        return ""
