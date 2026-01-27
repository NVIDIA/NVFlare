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
from nvflare.app_common.abstract.model_locator import ModelLocator
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.job_config.script_runner import FrameworkType
from nvflare.recipe.fedavg import FedAvgRecipe as UnifiedFedAvgRecipe


class FedAvgRecipe(UnifiedFedAvgRecipe):
    """A recipe for implementing Federated Averaging (FedAvg) for PyTorch.

    FedAvg is a fundamental federated learning algorithm that aggregates model updates
    from multiple clients by computing a weighted average based on the amount of local
    training data. This recipe sets up a complete federated learning workflow with
    memory-efficient InTime aggregation.

    The recipe configures:
    - A federated job with initial model (optional)
    - FedAvg controller with InTime aggregation for memory efficiency
    - Optional early stopping and model selection
    - Script runners for client-side training execution

    Args:
        name: Name of the federated learning job. Defaults to "fedavg".
        initial_model: Initial model to start federated training with. If None,
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
        command (str): If launch_external_process=True, command to run script (prepended to script). Defaults to "python3 -u".
        server_expected_format (str): What format to exchange the parameters between server and client.
        params_transfer_type (str): How to transfer the parameters. FULL means the whole model parameters are sent.
            DIFF means that only the difference is sent. Defaults to TransferType.FULL.
        model_persistor: Custom model persistor. If None, PTFileModelPersistor will be used.
        model_locator: Custom model locator. If None, PTFileModelLocator will be used.
        per_site_config: Per-site configuration for the federated learning job. Dictionary mapping
            site names to configuration dicts. Each config dict can contain optional overrides:
            train_script, train_args, launch_external_process, command, framework,
            server_expected_format, params_transfer_type, launch_once, shutdown_timeout.
            If not provided, the same configuration will be used for all clients.
        launch_once: Controls the lifecycle of the external process. If True (default), the process
            is launched once at startup and persists throughout all rounds, handling multiple training
            requests. If False, a new process is launched and torn down for each individual request
            from the server (e.g., each train or validate request). Only used if `launch_external_process`
            is True. Defaults to True.
        shutdown_timeout: If provided, will wait for this number of seconds before shutdown.
            Only used if `launch_external_process` is True. Defaults to 0.0.
        key_metric: Metric used to determine if the model is globally best. If validation metrics are a dict,
            key_metric selects the metric used for global model selection by the IntimeModelSelector.
            Defaults to "accuracy".
        stop_cond: Early stopping condition based on metric. String literal in the format of
            '<key> <op> <value>' (e.g. "accuracy >= 80"). If None, early stopping is disabled.
        patience: Number of rounds with no improvement after which FL will be stopped.
        save_filename: Filename for saving the best model. Defaults to "FL_global_model.pt".
        exclude_vars: Regex pattern for variables to exclude from aggregation.
        aggregation_weights: Per-client aggregation weights dict. Defaults to equal weights.
        server_memory_gc_rounds: Run memory cleanup every N rounds on server. Defaults to 0 (disabled).
        client_memory_gc_rounds: Run memory cleanup every N rounds on client. Defaults to 0 (disabled).
        torch_cuda_empty_cache: If True, call torch.cuda.empty_cache() during cleanup. Defaults to False.

    Example:
        Basic usage with early stopping:

        ```python
        recipe = FedAvgRecipe(
            name="my_fedavg_job",
            initial_model=pretrained_model,
            min_clients=2,
            num_rounds=10,
            train_script="client.py",
            train_args="--epochs 5 --batch_size 32",
            stop_cond="accuracy >= 95",
            patience=3
        )
        ```

        Using launch_once=False to restart the external process for each request:

        ```python
        recipe = FedAvgRecipe(
            name="my_fedavg_job",
            initial_model=pretrained_model,
            min_clients=2,
            num_rounds=10,
            train_script="client.py",
            train_args="--epochs 5 --batch_size 32",
            launch_external_process=True,
            launch_once=False,  # Process restarts for each server request
            shutdown_timeout=10.0  # Wait 10 seconds before shutdown
        )
        ```

    Note:
        This recipe uses InTime (streaming) aggregation for memory efficiency - each client
        result is aggregated immediately upon receipt rather than collecting all results first.
        Memory usage is constant regardless of the number of clients.
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
        model_persistor: Optional[ModelPersistor] = None,
        model_locator: Optional[ModelLocator] = None,
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
        # Memory management
        server_memory_gc_rounds: int = 0,
        client_memory_gc_rounds: int = 0,
        torch_cuda_empty_cache: bool = False,
    ):
        # Store PyTorch-specific model_locator before calling parent
        self._pt_model_locator = model_locator

        # Call the unified FedAvgRecipe with PyTorch-specific settings
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
            framework=FrameworkType.PYTORCH,
            server_expected_format=server_expected_format,
            params_transfer_type=params_transfer_type,
            model_persistor=model_persistor,
            per_site_config=per_site_config,
            launch_once=launch_once,
            shutdown_timeout=shutdown_timeout,
            key_metric=key_metric,
            stop_cond=stop_cond,
            patience=patience,
            save_filename=save_filename,
            exclude_vars=exclude_vars,
            aggregation_weights=aggregation_weights,
            server_memory_gc_rounds=server_memory_gc_rounds,
            client_memory_gc_rounds=client_memory_gc_rounds,
            torch_cuda_empty_cache=torch_cuda_empty_cache,
        )

    def _setup_model_and_persistor(self, job) -> str:
        """Override to handle PyTorch-specific model setup."""
        if self.initial_model is not None:
            from nvflare.app_opt.pt.job_config.model import PTModel

            # Disable numpy conversion when using tensor format to keep PyTorch tensors
            allow_numpy_conversion = self.server_expected_format != ExchangeFormat.PYTORCH

            pt_model = PTModel(
                model=self.initial_model,
                persistor=self.model_persistor,
                locator=self._pt_model_locator,
                allow_numpy_conversion=allow_numpy_conversion,
            )
            job.comp_ids.update(job.to_server(pt_model))
            return job.comp_ids.get("persistor_id", "")
        return ""
