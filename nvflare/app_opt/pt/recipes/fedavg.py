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

from nvflare.apis.dxo import DataKind
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.model_locator import ModelLocator
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.widgets.streaming import AnalyticsReceiver
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.job_config.script_runner import FrameworkType
from nvflare.recipe.fedavg import FedAvgRecipe as UnifiedFedAvgRecipe


class FedAvgRecipe(UnifiedFedAvgRecipe):
    """A recipe for implementing Federated Averaging (FedAvg) for PyTorch.

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
        initial_model: Initial model to start federated training with. If None,
            clients will start with their own local models.
        min_clients: Minimum number of clients required to start a training round.
        num_rounds: Number of federated training rounds to execute. Defaults to 2.
        train_script: Path to the training script that will be executed on each client.
        train_args: Command line arguments to pass to the training script.
        aggregator: Aggregator for combining client updates. If None,
            uses InTimeAccumulateWeightedAggregator with aggregator_data_kind.
        aggregator_data_kind: Data kind to use for the aggregator. Defaults to DataKind.WEIGHTS.
        launch_external_process (bool): Whether to launch the script in external process. Defaults to False.
        command (str): If launch_external_process=True, command to run script (prepended to script). Defaults to "python3".
        server_expected_format (str): What format to exchange the parameters between server and client.
        params_transfer_type (str): How to transfer the parameters. FULL means the whole model parameters are sent.
            DIFF means that only the difference is sent. Defaults to TransferType.FULL.
        model_persistor: Custom model persistor. If None, PTFileModelPersistor will be used.
        model_locator: Custom model locator. If None, PTFileModelLocator will be used.
        analytics_receiver: Component for receiving analytics data (e.g., TBAnalyticsReceiver for TensorBoard).
            If not provided, no experiment tracking will be enabled.
            To enable experiment tracking, either:
            - Pass an AnalyticsReceiver instance explicitly, OR
            - Use add_experiment_tracking() from nvflare.recipe.utils after recipe creation
        per_site_config: Per-site configuration for the federated learning job. Dictionary mapping
            site names to configuration dicts. Each config dict can contain optional overrides:
            train_script, train_args, launch_external_process, command, framework,
            server_expected_format, params_transfer_type.
            If not provided, the same configuration will be used for all clients.
    Example:
        Basic usage without experiment tracking:

        ```python
        recipe = FedAvgRecipe(
            name="my_fedavg_job",
            initial_model=pretrained_model,
            min_clients=2,
            num_rounds=10,
            train_script="client.py",
            train_args="--epochs 5 --batch_size 32"
        )
        ```

        Enable TensorBoard experiment tracking (Option 1 - pass explicitly):

        ```python
        from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver

        recipe = FedAvgRecipe(
            name="my_fedavg_job",
            initial_model=pretrained_model,
            min_clients=2,
            num_rounds=10,
            train_script="client.py",
            train_args="--epochs 5 --batch_size 32",
            analytics_receiver=TBAnalyticsReceiver()
        )
        ```

        Enable experiment tracking (Option 2 - add after creation):

        ```python
        from nvflare.recipe.utils import add_experiment_tracking

        recipe = FedAvgRecipe(...)  # Create recipe first
        add_experiment_tracking(recipe, "tensorboard")  # Add tracking later
        # Also supports: "mlflow", "wandb"
        ```

    Note:
        By default, this recipe implements the standard FedAvg algorithm where model updates
        are aggregated using weighted averaging based on the number of training
        samples provided by each client.

        Experiment tracking is opt-in. No tracking components are configured by default,
        avoiding unnecessary dependencies.

        If you want to use a custom aggregator, you can pass it in the aggregator parameter.
        The custom aggregator must be a subclass of the Aggregator or ModelAggregator class.
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
        analytics_receiver: Optional[AnalyticsReceiver] = None,
        per_site_config: Optional[dict[str, dict]] = None,
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
            analytics_receiver=analytics_receiver,
            per_site_config=per_site_config,
        )

    def _setup_model_and_persistor(self, job) -> str:
        """Override to handle PyTorch-specific model setup."""
        if self.initial_model is not None:
            from nvflare.app_opt.pt.job_config.model import PTModel

            pt_model = PTModel(model=self.initial_model, persistor=self.model_persistor, locator=self._pt_model_locator)
            job.comp_ids.update(job.to_server(pt_model))
            return job.comp_ids.get("persistor_id", "")
        return ""
