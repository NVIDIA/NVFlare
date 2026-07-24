# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.app_common.abstract.model_locator import ModelLocator
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.utils.fedprox_utils import validate_fedprox_mu
from nvflare.client.config import ExchangeFormat, TransferType

from .fedavg import FedAvgRecipe


class FedProxRecipe(FedAvgRecipe):
    """A PyTorch FedProx recipe built on FedAvg aggregation.

    The recipe sends ``fedprox_mu`` with every training model. A patched PyTorch Lightning
    trainer consumes this metadata automatically. Raw PyTorch clients must read the metadata,
    snapshot the received global model, and apply :class:`nvflare.app_opt.pt.PTFedProxLoss`.
    Clients that ignore the metadata are not FedProx-compatible.

    All aggregation, persistence, transfer, and memory-management options are inherited from
    :class:`FedAvgRecipe`.

    Args:
        name: Name of the federated learning job. Defaults to "fedprox".
        model: Initial PyTorch model, model configuration dictionary, or None.
        initial_ckpt: Absolute path to a pre-trained checkpoint.
        min_clients: Minimum number of clients required to start a training round.
        num_rounds: Number of federated training rounds. Defaults to 2.
        train_script: Client training script path.
        train_args: Command-line arguments passed to the client training script.
        aggregator: Optional custom model aggregator.
        aggregator_data_kind: Data kind accepted by the aggregator.
        launch_external_process: Whether to launch the client script externally.
        command: Command prepended to the script for external launches.
        server_expected_format: Parameter format expected by the server.
        params_transfer_type: Full-model or model-difference transfer.
        model_persistor: Optional custom model persistor.
        model_locator: Optional custom model locator.
        per_site_config: Deprecated per-site constructor configuration.
        launch_once: Whether an external client process is launched once.
        shutdown_timeout: Seconds to wait for client shutdown.
        key_metric: Metric used for best-model selection.
        stop_cond: Optional early-stopping condition.
        patience: Optional early-stopping patience.
        best_model_filename: Optional best-model filename.
        save_filename: Deprecated alias for ``best_model_filename``.
        exclude_vars: Optional regex for variables excluded from aggregation.
        aggregation_weights: Optional per-client aggregation weights.
        server_memory_gc_rounds: Server garbage-collection interval.
        enable_tensor_disk_offload: Enable server tensor disk offload.
        client_memory_gc_rounds: Client garbage-collection interval.
        cuda_empty_cache: Whether clients empty the CUDA cache during cleanup.
        fedprox_mu: Finite positive proximal coefficient. Defaults to 0.01.
    """

    def __init__(
        self,
        *,
        name: str = "fedprox",
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
        server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY,
        params_transfer_type: TransferType = TransferType.FULL,
        model_persistor: Optional[ModelPersistor] = None,
        model_locator: Optional[ModelLocator] = None,
        per_site_config: Optional[dict[str, dict]] = None,
        launch_once: bool = True,
        shutdown_timeout: float = 0.0,
        key_metric: str = "accuracy",
        stop_cond: Optional[str] = None,
        patience: Optional[int] = None,
        best_model_filename: Optional[str] = None,
        save_filename: Optional[str] = None,
        exclude_vars: Optional[str] = None,
        aggregation_weights: Optional[dict[str, float]] = None,
        server_memory_gc_rounds: int = 0,
        enable_tensor_disk_offload: bool = False,
        client_memory_gc_rounds: int = 0,
        cuda_empty_cache: bool = False,
        fedprox_mu: float = 0.01,
    ):
        # FedAvgRecipe builds the controller through _get_controller_kwargs(), so this must precede super().__init__.
        self.fedprox_mu = validate_fedprox_mu(fedprox_mu)
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
            server_expected_format=server_expected_format,
            params_transfer_type=params_transfer_type,
            model_persistor=model_persistor,
            model_locator=model_locator,
            per_site_config=per_site_config,
            launch_once=launch_once,
            shutdown_timeout=shutdown_timeout,
            key_metric=key_metric,
            stop_cond=stop_cond,
            patience=patience,
            best_model_filename=best_model_filename,
            save_filename=save_filename,
            exclude_vars=exclude_vars,
            aggregation_weights=aggregation_weights,
            server_memory_gc_rounds=server_memory_gc_rounds,
            enable_tensor_disk_offload=enable_tensor_disk_offload,
            client_memory_gc_rounds=client_memory_gc_rounds,
            cuda_empty_cache=cuda_empty_cache,
        )

    def _get_controller_kwargs(self) -> dict[str, Any]:
        kwargs = super()._get_controller_kwargs()
        kwargs["fedprox_mu"] = self.fedprox_mu
        return kwargs
