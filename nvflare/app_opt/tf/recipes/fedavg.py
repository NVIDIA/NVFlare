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

from typing import Any, Optional, Union

from nvflare.apis.dxo import DataKind
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.fuel.utils.constants import FrameworkType
from nvflare.recipe.fedavg import FedAvgRecipe as UnifiedFedAvgRecipe


class FedAvgRecipe(UnifiedFedAvgRecipe):
    """A recipe for implementing Federated Averaging (FedAvg) for TensorFlow.

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
        model: Initial model to start federated training with. Can be:
            - tf.keras.Model instance
            - Dict config: {"class_path": "module.ClassName", "args": {"param": value}}
            - None: no initial model
        initial_ckpt: Absolute path to a pre-trained checkpoint file (.h5, .keras, or SavedModel dir).
            The file may not exist locally as it could be on the server.
            Note: TensorFlow can load full models from .h5/SavedModel without model.
        min_clients: Minimum number of clients required to start a training round.
        num_rounds: Number of federated training rounds to execute. Defaults to 2.
        train_script: Path to the training script that will be executed on each client.
        train_args: Command line arguments to pass to the training script.
        aggregator: Aggregator for combining client updates. If None,
            uses InTimeAccumulateWeightedAggregator with aggregator_data_kind.
        aggregator_data_kind: Data kind to use for the aggregator. Defaults to DataKind.WEIGHTS.
        launch_external_process (bool): Whether to launch the script in external process. Defaults to False.
        command (str): If launch_external_process=True, command to run script (prepended to script). Defaults to "python3".
        framework (str): The framework to use for the training script. Defaults to FrameworkType.TENSORFLOW.
        server_expected_format (str): What format to exchange the parameters between server and client.
        params_transfer_type (str): How to transfer the parameters. FULL means the whole model parameters are sent.
            DIFF means that only the difference is sent. Defaults to TransferType.FULL.
        model_persistor: Custom model persistor. If None, TFModelPersistor will be used.
        per_site_config: Per-site configuration for the federated learning job. Dictionary mapping
            site names to configuration dicts. Each config dict can contain optional overrides:
            train_script, train_args, launch_external_process, command, framework,
            server_expected_format, params_transfer_type, launch_once, shutdown_timeout.
            If not provided, the same configuration will be used for all clients.
        launch_once: Whether the external process will be launched only once at the beginning
            or on each task. Only used if `launch_external_process` is True. Defaults to True.
        shutdown_timeout: If provided, will wait for this number of seconds before shutdown.
            Only used if `launch_external_process` is True. Defaults to 0.0.
        key_metric: Metric used to determine if the model is globally best. If validation metrics are a dict,
            key_metric selects the metric used for global model selection by the IntimeModelSelector.
            Defaults to "accuracy".

    Example:
        Basic usage without experiment tracking:

        ```python
        recipe = FedAvgRecipe(
            name="my_fedavg_job",
            model=pretrained_model,
            min_clients=2,
            num_rounds=10,
            train_script="client.py",
            train_args="--epochs 5 --batch_size 32"
        )
        ```

    Note:
        By default, this recipe implements the standard FedAvg algorithm where model updates
        are aggregated using weighted averaging based on the number of training
        samples provided by each client.

        If you want to use a custom aggregator, you can pass it in the aggregator parameter.
        The custom aggregator must be a subclass of the Aggregator or ModelAggregator class.
    """

    def __init__(
        self,
        *,
        name: str = "fedavg",
        model: Union[Any, dict[str, Any], None] = None,
        initial_ckpt: Optional[str] = None,
        min_clients: int,
        num_rounds: int = 2,
        train_script: str,
        train_args: str = "",
        aggregator: Optional[Aggregator] = None,
        aggregator_data_kind: Optional[DataKind] = DataKind.WEIGHTS,
        launch_external_process: bool = False,
        command: str = "python3 -u",
        framework: FrameworkType = FrameworkType.TENSORFLOW,
        server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY,
        params_transfer_type: TransferType = TransferType.FULL,
        model_persistor: Optional[ModelPersistor] = None,
        per_site_config: Optional[dict[str, dict]] = None,
        launch_once: bool = True,
        shutdown_timeout: float = 0.0,
        key_metric: str = "accuracy",
        server_memory_gc_rounds: int = 0,
    ):
        # Call the unified FedAvgRecipe with TensorFlow-specific settings
        super().__init__(
            name=name,
            model=model,
            initial_ckpt=initial_ckpt,
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
            server_memory_gc_rounds=server_memory_gc_rounds,
        )

    def _setup_model_and_persistor(self, job) -> str:
        """Override to handle TensorFlow-specific model setup."""
        if self.model is not None or self.initial_ckpt is not None:
            from nvflare.app_opt.tf.job_config.model import TFModel
            from nvflare.recipe.utils import prepare_initial_ckpt

            ckpt_path = prepare_initial_ckpt(self.initial_ckpt, job)
            tf_model = TFModel(
                model=self.model,
                initial_ckpt=ckpt_path,
                persistor=self.model_persistor,
            )
            job.comp_ids["persistor_id"] = job.to_server(tf_model)
            return job.comp_ids.get("persistor_id", "")
        return ""
