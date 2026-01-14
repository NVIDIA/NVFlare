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

from typing import Any, Optional

from pydantic import BaseModel

from nvflare.apis.dxo import DataKind
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator
from nvflare.app_common.shareablegenerators import FullModelShareableGenerator
from nvflare.app_common.widgets.streaming import AnalyticsReceiver
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
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
    aggregator: Optional[Aggregator]
    aggregator_data_kind: Optional[DataKind]
    launch_external_process: bool
    command: str
    framework: FrameworkType
    server_expected_format: ExchangeFormat
    params_transfer_type: TransferType
    model_persistor: Optional[ModelPersistor]
    analytics_receiver: Any
    per_site_config: Optional[dict[str, dict]] = None
    launch_once: bool
    shutdown_timeout: float


class FedAvgRecipe(Recipe):
    """Unified FedAvg recipe for PyTorch, TensorFlow, and Scikit-learn.

    FedAvg is a fundamental federated learning algorithm that aggregates model updates
    from multiple clients by computing a weighted average based on the amount of local
    training data. This recipe sets up a complete federated learning workflow with
    scatter-and-gather communication pattern.

    The recipe configures:
    - A federated job with initial model (optional)
    - Scatter-and-gather controller for coordinating training rounds
    - Weighted aggregator for combining client model updates (or custom aggregator)
    - Script runners for client-side training execution

    Args:
        name: Name of the federated learning job. Defaults to "fedavg".
        initial_model: Initial model to start federated training with. Can be:
            - nn.Module for PyTorch
            - tf.keras.Model for TensorFlow
            - dict for sklearn/RAW frameworks (model parameters)
            - ModelPersistor for any framework (custom persistence logic)
            - None (no initial model)
        min_clients: Minimum number of clients required to start a training round.
        num_rounds: Number of federated training rounds to execute. Defaults to 2.
        train_script: Path to the training script that will be executed on each client.
        train_args: Command line arguments to pass to the training script.
        aggregator: Aggregator for combining client updates. If None,
            uses InTimeAccumulateWeightedAggregator with aggregator_data_kind.
        aggregator_data_kind: Data kind to use for the aggregator. Defaults to DataKind.WEIGHTS.
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
        model_persistor: Custom model persistor for any framework.
            - For PyTorch/TensorFlow: Optional (defaults will be used if not provided)
            - For RAW frameworks: Can be provided here OR passed as initial_model
            If None, framework-specific defaults will be used (PT/TF only).
        analytics_receiver: Component for receiving analytics data (e.g., TBAnalyticsReceiver for TensorBoard).
            If not provided, no experiment tracking will be enabled. Pass explicitly to enable tracking.
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
        launch_once: Whether the external process will be launched only once at the beginning
            or on each task. Only used if `launch_external_process` is True. Defaults to True.
        shutdown_timeout: If provided, will wait for this number of seconds before shutdown.
            Only used if `launch_external_process` is True. Defaults to 0.0.

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
        framework: FrameworkType = FrameworkType.PYTORCH,
        server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY,
        params_transfer_type: TransferType = TransferType.FULL,
        model_persistor: Optional[ModelPersistor] = None,
        analytics_receiver: Optional[AnalyticsReceiver] = None,
        per_site_config: Optional[dict[str, dict]] = None,
        launch_once: bool = True,
        shutdown_timeout: float = 0.0,
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
            analytics_receiver=analytics_receiver,
            per_site_config=per_site_config,
            launch_once=launch_once,
            shutdown_timeout=shutdown_timeout,
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
        self.analytics_receiver = v.analytics_receiver
        self.per_site_config = v.per_site_config
        self.launch_once = v.launch_once
        self.shutdown_timeout = v.shutdown_timeout
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
            analytics_receiver=self.analytics_receiver,
        )

        # Setup framework-specific model components and persistor
        # Child classes (PT/TF wrappers) override this method for framework-specific logic
        persistor_id = self._setup_model_and_persistor(job)

        # Setup aggregator
        if self.aggregator is None:
            self.aggregator = InTimeAccumulateWeightedAggregator(expected_data_kind=self.aggregator_data_kind)
        else:
            if not isinstance(self.aggregator, Aggregator):
                raise ValueError(f"Invalid aggregator type: {type(self.aggregator)}. Expected type: {Aggregator}")

        # Add shareable generator and aggregator
        shareable_generator = FullModelShareableGenerator()
        shareable_generator_id = job.to_server(shareable_generator, id="shareable_generator")
        aggregator_id = job.to_server(self.aggregator, id="aggregator")

        # Add controller
        controller = ScatterAndGather(
            min_clients=self.min_clients,
            num_rounds=self.num_rounds,
            wait_time_after_min_received=0,
            aggregator_id=aggregator_id,
            persistor_id=persistor_id,
            shareable_generator_id=shareable_generator_id,
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
            )
            job.to_clients(executor)

        Recipe.__init__(self, job)

    def _setup_model_and_persistor(self, job: BaseFedJob) -> str:
        """Setup framework-specific model components and persistor.

        Returns:
            str: The persistor_id to be used by the controller.
        """
        if self.model_persistor is not None:
            return job.to_server(self.model_persistor, id="persistor")
        return ""
