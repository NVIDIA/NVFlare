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

from typing import Any, Dict, Optional, Union

from pydantic import BaseModel

from nvflare.apis.dxo import DataKind
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator
from nvflare.app_common.shareablegenerators import FullModelShareableGenerator
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
    initial_params: Optional[dict]
    min_clients: int
    num_rounds: int
    train_script: str
    train_args: Union[str, Dict[str, str]]
    aggregator: Optional[Aggregator]
    aggregator_data_kind: Optional[DataKind]
    launch_external_process: bool
    command: str
    framework: FrameworkType
    server_expected_format: ExchangeFormat
    params_transfer_type: TransferType
    model_persistor: Optional[ModelPersistor]
    custom_persistor: Optional[ModelPersistor]


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
            - None for sklearn (use initial_params instead)
        initial_params: Initial model parameters (dict). Used for sklearn.
            If provided, initial_model should be None.
        min_clients: Minimum number of clients required to start a training round.
        num_rounds: Number of federated training rounds to execute. Defaults to 2.
        train_script: Path to the training script that will be executed on each client.
        train_args: Command line arguments to pass to the training script. Can be:
            - str: Same arguments for all clients (uses job.to_clients)
            - dict[str, str]: Per-client arguments mapping site names to args (uses job.to per site)
        aggregator: Aggregator for combining client updates. If None,
            uses InTimeAccumulateWeightedAggregator with aggregator_data_kind.
        aggregator_data_kind: Data kind to use for the aggregator. Defaults to DataKind.WEIGHTS.
        launch_external_process: Whether to launch the script in external process. Defaults to False.
        command: If launch_external_process=True, command to run script (prepended to script).
            Defaults to "python3 -u".
        framework: The framework type. One of:
            - FrameworkType.PYTORCH (default)
            - FrameworkType.TENSORFLOW
            - FrameworkType.RAW (for sklearn)
        server_expected_format: What format to exchange the parameters between server and client.
            Defaults to ExchangeFormat.NUMPY.
        params_transfer_type: How to transfer the parameters. FULL means the whole model parameters
            are sent. DIFF means that only the difference is sent. Defaults to TransferType.FULL.
        model_persistor: Custom model persistor. If None, framework-specific defaults will be used.
        custom_persistor: Custom persistor for RAW framework (sklearn). Required when framework=RAW.
            This allows framework-specific wrappers to provide their own persistor without the unified
            recipe depending on framework-specific components.

    Example (PyTorch):
        ```python
        import torch.nn as nn
        from nvflare.recipe.fedavg import FedAvgRecipe
        from nvflare.job_config.script_runner import FrameworkType

        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

        recipe = FedAvgRecipe(
            name="pt_fedavg",
            initial_model=model,
            min_clients=2,
            num_rounds=10,
            train_script="train.py",
            train_args="--epochs 5",
            framework=FrameworkType.PYTORCH,
        )
        ```

    Example (TensorFlow):
        ```python
        import tensorflow as tf
        from nvflare.recipe.fedavg import FedAvgRecipe
        from nvflare.job_config.script_runner import FrameworkType

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(5, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(2)
        ])

        recipe = FedAvgRecipe(
            name="tf_fedavg",
            initial_model=model,
            min_clients=2,
            num_rounds=10,
            train_script="train.py",
            train_args="--epochs 5",
            framework=FrameworkType.TENSORFLOW,
        )
        ```

    Example (Scikit-learn):
        ```python
        from nvflare.recipe.fedavg import FedAvgRecipe
        from nvflare.job_config.script_runner import FrameworkType
        from nvflare.client.config import ExchangeFormat

        recipe = FedAvgRecipe(
            name="sklearn_fedavg",
            initial_params={
                "n_classes": 2,
                "learning_rate": "constant",
                "eta0": 1e-4,
            },
            min_clients=5,
            num_rounds=50,
            train_script="train.py",
            train_args="--data_path /tmp/data.csv",
            framework=FrameworkType.RAW,
            server_expected_format=ExchangeFormat.RAW,
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
        name: str = "fedavg",
        initial_model: Any = None,
        initial_params: Optional[dict] = None,
        min_clients: int,
        num_rounds: int = 2,
        train_script: str,
        train_args: Union[str, Dict[str, str]] = "",
        aggregator: Optional[Aggregator] = None,
        aggregator_data_kind: Optional[DataKind] = DataKind.WEIGHTS,
        launch_external_process: bool = False,
        command: str = "python3 -u",
        framework: FrameworkType = FrameworkType.PYTORCH,
        server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY,
        params_transfer_type: TransferType = TransferType.FULL,
        model_persistor: Optional[ModelPersistor] = None,
        custom_persistor: Optional[ModelPersistor] = None,
    ):
        # Validate inputs internally
        v = _FedAvgValidator(
            name=name,
            initial_model=initial_model,
            initial_params=initial_params,
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
            custom_persistor=custom_persistor,
        )

        self.name = v.name
        self.initial_model = v.initial_model
        self.initial_params = v.initial_params
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
        self.custom_persistor = v.custom_persistor

        # Validate that only one of initial_model or initial_params is provided
        if self.initial_model is not None and self.initial_params is not None:
            raise ValueError(
                "Cannot provide both initial_model and initial_params. "
                "Use initial_model for PyTorch/TensorFlow, initial_params for sklearn."
            )

        # Validate RAW framework has custom_persistor
        if self.framework == FrameworkType.RAW and self.custom_persistor is None:
            raise ValueError(
                "custom_persistor is required when framework=FrameworkType.RAW. "
                "Use framework-specific wrappers (e.g., SklearnFedAvgRecipe) or provide a custom persistor."
            )

        # Create BaseFedJob - all frameworks use it for consistency
        # Provide default TBAnalyticsReceiver for PT/TF only
        analytics_receiver = None
        if self.framework in (FrameworkType.PYTORCH, FrameworkType.TENSORFLOW):
            from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver

            analytics_receiver = TBAnalyticsReceiver()

        job = BaseFedJob(
            initial_model=self.initial_model,
            initial_params=self.initial_params,
            name=self.name,
            min_clients=self.min_clients,
            model_persistor=self.model_persistor,
            framework=self.framework,
            analytics_receiver=analytics_receiver,
        )

        # Setup framework-specific model components and persistor
        persistor_id = ""
        if self.framework == FrameworkType.RAW:
            # Add sklearn-specific persistor
            persistor_id = job.to_server(self.custom_persistor, id="persistor")
        elif self.initial_model is not None:
            if self.framework == FrameworkType.PYTORCH:
                self._setup_pytorch_model(job, self.initial_model, self.model_persistor, model_locator=None)
            elif self.framework == FrameworkType.TENSORFLOW:
                self._setup_tensorflow_model(job, self.initial_model, self.model_persistor)
            persistor_id = job.comp_ids.get("persistor_id", "")

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
            train_task_name="train",
        )
        job.to_server(controller)

        # Add client executors
        if isinstance(self.train_args, dict):
            # Per-client configuration
            for site_name, site_args in self.train_args.items():
                executor = ScriptRunner(
                    script=self.train_script,
                    script_args=site_args,
                    launch_external_process=self.launch_external_process,
                    command=self.command,
                    framework=self.framework,
                    server_expected_format=self.server_expected_format,
                    params_transfer_type=self.params_transfer_type,
                )
                job.to(executor, site_name)
        else:
            # Unified configuration
            executor = ScriptRunner(
                script=self.train_script,
                script_args=self.train_args,
                launch_external_process=self.launch_external_process,
                command=self.command,
                framework=self.framework,
                server_expected_format=self.server_expected_format,
                params_transfer_type=self.params_transfer_type,
            )
            job.to_clients(executor)

        Recipe.__init__(self, job)

    def _setup_pytorch_model(
        self, job: BaseFedJob, model: Any, persistor: Optional[ModelPersistor], model_locator: Optional[Any] = None
    ):
        """Setup PyTorch model with persistor and locator."""
        from nvflare.app_opt.pt.job_config.model import PTModel

        pt_model = PTModel(model=model, persistor=persistor, locator=model_locator)
        job.comp_ids.update(job.to_server(pt_model))

    def _setup_tensorflow_model(self, job: BaseFedJob, model: Any, persistor: Optional[ModelPersistor]):
        """Setup TensorFlow model with persistor."""
        from nvflare.app_opt.tf.job_config.model import TFModel

        tf_model = TFModel(model=model, persistor=persistor)
        job.comp_ids["persistor_id"] = job.to_server(tf_model)
