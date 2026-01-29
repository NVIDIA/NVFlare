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

from pydantic import BaseModel

from nvflare.apis.dxo import DataKind
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner
from nvflare.recipe.spec import Recipe


# Internal — not part of the public API
class _FedAvgValidator(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str
    initial_model: Any
    min_clients: int
    num_rounds: int
    train_script: str
    train_args: str
    # Legacy parameters for backward compatibility (not used by new FedAvg)
    aggregator: Optional[Aggregator] = None
    aggregator_data_kind: Optional[DataKind] = DataKind.WEIGHTS
    # Core parameters
    launch_external_process: bool
    command: str
    framework: FrameworkType
    server_expected_format: ExchangeFormat
    params_transfer_type: TransferType
    model_persistor: Optional[ModelPersistor] = None
    per_site_config: Optional[dict[str, dict]] = None
    launch_once: bool = True
    shutdown_timeout: float = 0.0
    key_metric: str = "accuracy"
    # New FedAvg features
    stop_cond: Optional[str] = None
    patience: Optional[int] = None
    save_filename: str = "FL_global_model.pt"
    exclude_vars: Optional[str] = None
    aggregation_weights: Optional[Dict[str, float]] = None
    # Memory management
    server_memory_gc_rounds: int = 0
    client_memory_gc_rounds: int = 0
    torch_cuda_empty_cache: bool = False


class FedAvgRecipe(Recipe):
    """Unified FedAvg recipe for PyTorch, TensorFlow, and Scikit-learn.

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
        initial_model: Initial model to start federated training with. Can be:
            - nn.Module for PyTorch (will call .state_dict())
            - tf.keras.Model for TensorFlow
            - dict for sklearn/RAW frameworks (model parameters)
            - None (no initial model)
        min_clients: Minimum number of clients required to start a training round.
        num_rounds: Number of federated training rounds to execute. Defaults to 2.
        train_script: Path to the training script that will be executed on each client.
        train_args: Command line arguments to pass to the training script.
        aggregator: Custom aggregator (ModelAggregator) for combining client model updates.
            Must implement accept_model(), aggregate_model(), reset_stats() methods.
            If None, uses built-in memory-efficient weighted averaging. Defaults to None.
        aggregator_data_kind: Data kind for aggregation (DataKind.WEIGHTS or DataKind.WEIGHT_DIFF).
            Kept for backward compatibility. Defaults to DataKind.WEIGHTS.
        launch_external_process: Whether to launch the script in external process. Defaults to False.
        command: If launch_external_process=True, command to run script (prepended to script).
            Defaults to "python3 -u".
        framework: The framework type. One of:
            - FrameworkType.PYTORCH (default)
            - FrameworkType.TENSORFLOW
            - FrameworkType.RAW (for custom frameworks, e.g., sklearn, XGBoost)
        server_expected_format: What format to exchange the parameters between server and client.
            Defaults to ExchangeFormat.NUMPY.
        params_transfer_type: How to transfer the parameters. FULL means the whole model parameters
            are sent. DIFF means that only the difference is sent. Defaults to TransferType.FULL.
        model_persistor: Custom model persistor for any framework. If None, uses simple
            file-based saving with save_filename.
        per_site_config: Per-site configuration for the federated learning job. Dictionary mapping
            site names to configuration dicts. Each config dict can contain optional overrides:
            - train_script (str): Training script path
            - train_args (str): Script arguments
            - launch_external_process (bool): Whether to launch external process
            - command (str): Command prefix for external process
            - framework (FrameworkType): Framework type
            - server_expected_format (ExchangeFormat): Exchange format
            - params_transfer_type (TransferType): Parameter transfer type
            - launch_once (bool): Whether to launch external process once or per task
            - shutdown_timeout (float): Shutdown timeout in seconds
            If not provided, the same configuration will be used for all clients.
        launch_once: Controls the lifecycle of the external process. If True (default), the process
            is launched once at startup and persists throughout all rounds, handling multiple training
            requests. If False, a new process is launched and torn down for each individual request
            from the server (e.g., each train or validate request). Only used if `launch_external_process`
            is True. Defaults to True.
        shutdown_timeout: If provided, will wait for this number of seconds before shutdown.
            Only used if `launch_external_process` is True. Defaults to 0.0.
        key_metric: Metric used to determine if the model is globally best. Defaults to "accuracy".
        stop_cond: Early stopping condition based on metric. String literal in the format of
            '<key> <op> <value>' (e.g. "accuracy >= 80"). If None, early stopping is disabled.
        patience: Number of rounds with no improvement after which FL will be stopped.
            Only applies if stop_cond is set. Defaults to None.
        save_filename: Filename for saving the best model. Defaults to "FL_global_model.pt".
        exclude_vars: Regex pattern for variables to exclude from aggregation.
        aggregation_weights: Per-client aggregation weights dict. Defaults to equal weights.
        server_memory_gc_rounds: Run memory cleanup (gc.collect + malloc_trim) every N rounds on server.
            Set to 0 to disable. Defaults to 0.
        client_memory_gc_rounds: Run memory cleanup every N rounds on client after sending model.
            Set to 0 to disable. Defaults to 0.
        torch_cuda_empty_cache: If True, call torch.cuda.empty_cache() during client memory cleanup.
            Only applicable to PyTorch GPU training. Defaults to False.

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
        # Legacy parameters for backward compatibility
        aggregator: Optional[Aggregator] = None,
        aggregator_data_kind: Optional[DataKind] = DataKind.WEIGHTS,
        # Core parameters
        launch_external_process: bool = False,
        command: str = "python3 -u",
        framework: FrameworkType = FrameworkType.PYTORCH,
        server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY,
        params_transfer_type: TransferType = TransferType.FULL,
        model_persistor: Optional[ModelPersistor] = None,
        per_site_config: Optional[dict[str, dict]] = None,
        launch_once: bool = True,
        shutdown_timeout: float = 0.0,
        key_metric: str = "accuracy",
        # New FedAvg features
        stop_cond: Optional[str] = None,
        patience: Optional[int] = None,
        save_filename: str = "FL_global_model.pt",
        exclude_vars: Optional[str] = None,
        aggregation_weights: Optional[Dict[str, float]] = None,
        server_memory_gc_rounds: int = 0,
        client_memory_gc_rounds: int = 0,
        torch_cuda_empty_cache: bool = False,
    ):
        # Validate inputs internally
        v = _FedAvgValidator(
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
            framework=framework,
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

        self.name = v.name
        self.initial_model = v.initial_model
        self.min_clients = v.min_clients
        self.num_rounds = v.num_rounds
        self.train_script = v.train_script
        self.train_args = v.train_args
        self.aggregator = v.aggregator
        self.aggregator_data_kind = v.aggregator_data_kind
        self.launch_external_process = v.launch_external_process
        self.command = v.command
        self.framework = v.framework
        self.server_expected_format = v.server_expected_format
        self.params_transfer_type = v.params_transfer_type
        self.model_persistor = v.model_persistor
        self.per_site_config = v.per_site_config
        self.launch_once = v.launch_once
        self.shutdown_timeout = v.shutdown_timeout
        self.key_metric = v.key_metric
        self.stop_cond = v.stop_cond
        self.patience = v.patience
        self.save_filename = v.save_filename
        self.exclude_vars = v.exclude_vars
        self.aggregation_weights = v.aggregation_weights
        self.server_memory_gc_rounds = v.server_memory_gc_rounds
        self.client_memory_gc_rounds = v.client_memory_gc_rounds
        self.torch_cuda_empty_cache = v.torch_cuda_empty_cache

        # Validate RAW framework requirements
        if self.framework == FrameworkType.RAW:
            if self.initial_model is None and self.model_persistor is None:
                raise ValueError(
                    "RAW framework requires either initial_model (dict or ModelPersistor) or model_persistor. "
                    "Consider using framework-specific wrappers (e.g., SklearnFedAvgRecipe) for convenience."
                )

        # Create BaseFedJob - all frameworks use it for consistency
        job = BaseFedJob(
            name=self.name,
            min_clients=self.min_clients,
            key_metric=self.key_metric,
        )

        # Setup framework-specific model components and persistor
        # Child classes (PT/TF wrappers) override this method for framework-specific logic
        persistor_id = self._setup_model_and_persistor(job)

        # Convert initial_model to dict if needed (e.g., PyTorch nn.Module)
        # Only pass to controller if no persistor is handling the model
        # (persistor already handles initial model via PTModel/TFModel)
        # Note: empty string "" means no persistor, so we need initial_model_params
        has_persistor = persistor_id != ""
        initial_model_params = None if has_persistor else self._get_initial_model_params()

        # Prepare aggregator for controller - must be ModelAggregator for FLModel-based aggregation
        model_aggregator = self._get_model_aggregator()

        # Add controller with InTime aggregation and all features
        controller = FedAvg(
            num_clients=self.min_clients,
            num_rounds=self.num_rounds,
            persistor_id=persistor_id,
            initial_model=initial_model_params,
            save_filename=self.save_filename,
            aggregator=model_aggregator,
            stop_cond=self.stop_cond,
            patience=self.patience,
            task_name="train",
            exclude_vars=self.exclude_vars,
            aggregation_weights=self.aggregation_weights,
            memory_gc_rounds=self.server_memory_gc_rounds,
        )
        job.to_server(controller)

        if self.per_site_config is not None:
            for site_name, site_config in self.per_site_config.items():
                # Use site-specific config or fall back to defaults
                script = (
                    site_config.get("train_script")
                    if site_config.get("train_script") is not None
                    else self.train_script
                )
                script_args = (
                    site_config.get("train_args") if site_config.get("train_args") is not None else self.train_args
                )
                launch_external = (
                    site_config.get("launch_external_process")
                    if site_config.get("launch_external_process") is not None
                    else self.launch_external_process
                )
                command = site_config.get("command") or self.command
                framework = site_config.get("framework") or self.framework
                expected_format = site_config.get("server_expected_format") or self.server_expected_format
                transfer_type = site_config.get("params_transfer_type") or self.params_transfer_type
                launch_once = (
                    site_config.get("launch_once") if site_config.get("launch_once") is not None else self.launch_once
                )
                shutdown_timeout = (
                    site_config.get("shutdown_timeout")
                    if site_config.get("shutdown_timeout") is not None
                    else self.shutdown_timeout
                )

                executor = ScriptRunner(
                    script=script,
                    script_args=script_args,
                    launch_external_process=launch_external,
                    command=command,
                    framework=framework,
                    server_expected_format=expected_format,
                    params_transfer_type=transfer_type,
                    launch_once=launch_once,
                    shutdown_timeout=shutdown_timeout,
                    memory_gc_rounds=self.client_memory_gc_rounds,
                    torch_cuda_empty_cache=self.torch_cuda_empty_cache,
                )
                job.to(executor, site_name)
        else:
            executor = ScriptRunner(
                script=self.train_script,
                script_args=self.train_args,
                launch_external_process=self.launch_external_process,
                command=self.command,
                framework=self.framework,
                server_expected_format=self.server_expected_format,
                params_transfer_type=self.params_transfer_type,
                launch_once=self.launch_once,
                shutdown_timeout=self.shutdown_timeout,
                memory_gc_rounds=self.client_memory_gc_rounds,
                torch_cuda_empty_cache=self.torch_cuda_empty_cache,
            )
            job.to_clients(executor)

        Recipe.__init__(self, job)

    def _get_initial_model_params(self) -> Optional[Dict]:
        """Convert initial_model to dict of params.

        Base implementation handles dict and None. Framework-specific subclasses
        should override this to handle their model types (e.g., nn.Module, tf.keras.Model).

        Returns:
            Optional[Dict]: model parameters as dict, or None
        """
        if self.initial_model is None:
            return None

        if isinstance(self.initial_model, dict):
            return self.initial_model

        # Unknown type - subclasses should override for framework-specific handling
        raise TypeError(
            f"initial_model must be a dict or None for the base recipe. "
            f"Got {type(self.initial_model).__name__}. "
            f"Use a framework-specific recipe (e.g., nvflare.app_opt.pt.recipes.FedAvgRecipe) "
            f"for nn.Module or other model types."
        )

    def _get_model_aggregator(self):
        """Get the ModelAggregator for the FedAvg controller.

        The FedAvg controller expects a ModelAggregator (works with FLModel).
        If no aggregator is provided, returns None (uses built-in weighted averaging).
        If a ModelAggregator is provided, returns it directly.

        Returns:
            ModelAggregator or None
        """
        if self.aggregator is None:
            return None

        # Import here to avoid circular imports
        from nvflare.app_common.aggregators.model_aggregator import ModelAggregator

        if isinstance(self.aggregator, ModelAggregator):
            return self.aggregator
        else:
            # It's a Shareable-based Aggregator - can't use directly with FedAvg
            # Log a warning and fall back to built-in aggregation
            import logging

            logging.getLogger(__name__).warning(
                f"Provided aggregator {type(self.aggregator).__name__} is not a ModelAggregator. "
                "Using built-in weighted averaging instead. For custom aggregation with FedAvg, "
                "please use a ModelAggregator subclass (e.g., from model_aggregator.py)."
            )
            return None

    def _setup_model_and_persistor(self, job: BaseFedJob) -> str:
        """Setup framework-specific model components and persistor.

        Returns:
            str: The persistor_id to be used by the controller.
        """
        if self.model_persistor is not None:
            return job.to_server(self.model_persistor, id="persistor")
        return ""
