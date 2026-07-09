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

from typing import Any, Dict, List, Optional, Union

from nvflare.app_opt.pt.fedce import FedCEModelAggregator
from nvflare.client.config import ExchangeFormat, TransferType

from .fedavg import FedAvgRecipe


class FedCERecipe(FedAvgRecipe):
    """PyTorch recipe for federated training via contribution estimation (FedCE).

    FedCE uses FedAvg's single-global-model lifecycle, but replaces ordinary
    sample-count aggregation with contribution-aware aggregation. Client scripts
    must return weight differences and set ``fedce_minus_val`` in the
    outgoing ``FLModel.meta``. Starting in round 1, the received global model
    metadata contains ``fedce_coef`` for constructing the local
    leave-one-out model.

    This recipe intentionally requires PyTorch exchange format and DIFF transfer.
    It is not composable with FedSM.
    """

    def __init__(
        self,
        *,
        name: str = "fedce",
        model: Union[Any, Dict[str, Any], None] = None,
        initial_ckpt: Optional[str] = None,
        min_clients: int,
        num_rounds: int = 2,
        train_script: str,
        train_args: str = "",
        fedce_mode: str = "plus",
        trainable_param_names: Optional[List[str]] = None,
        launch_external_process: bool = False,
        command: str = "python3 -u",
        server_expected_format: ExchangeFormat = ExchangeFormat.PYTORCH,
        per_site_config: Optional[Dict[str, Dict]] = None,
        launch_once: bool = True,
        shutdown_timeout: float = 0.0,
        key_metric: str = "accuracy",
        stop_cond: Optional[str] = None,
        patience: Optional[int] = None,
        best_model_filename: Optional[str] = None,
        server_memory_gc_rounds: int = 0,
        enable_tensor_disk_offload: bool = False,
        client_memory_gc_rounds: int = 0,
        cuda_empty_cache: bool = False,
    ):
        if server_expected_format != ExchangeFormat.PYTORCH:
            raise ValueError("FedCERecipe requires server_expected_format=ExchangeFormat.PYTORCH")

        if trainable_param_names is None:
            named_parameters = getattr(model, "named_parameters", None)
            if not callable(named_parameters):
                raise ValueError(
                    "trainable_param_names is required when model does not expose named_parameters(), "
                    "including dict-config models"
                )
            trainable_param_names = [name for name, parameter in named_parameters() if parameter.requires_grad]
        elif (
            not isinstance(trainable_param_names, list)
            or not trainable_param_names
            or any(not isinstance(name, str) or not name for name in trainable_param_names)
            or len(trainable_param_names) != len(set(trainable_param_names))
        ):
            raise ValueError("trainable_param_names must be a non-empty list of unique parameter names")
        if not trainable_param_names:
            raise ValueError("FedCE requires at least one trainable parameter")

        self.fedce_mode = fedce_mode
        self.trainable_param_names = list(trainable_param_names)
        self.fedce_aggregator = FedCEModelAggregator(
            mode=fedce_mode,
            trainable_param_names=self.trainable_param_names,
        )
        super().__init__(
            name=name,
            model=model,
            initial_ckpt=initial_ckpt,
            min_clients=min_clients,
            num_rounds=num_rounds,
            train_script=train_script,
            train_args=train_args,
            aggregator=self.fedce_aggregator,
            launch_external_process=launch_external_process,
            command=command,
            server_expected_format=server_expected_format,
            params_transfer_type=TransferType.DIFF,
            per_site_config=per_site_config,
            launch_once=launch_once,
            shutdown_timeout=shutdown_timeout,
            key_metric=key_metric,
            stop_cond=stop_cond,
            patience=patience,
            best_model_filename=best_model_filename,
            server_memory_gc_rounds=server_memory_gc_rounds,
            enable_tensor_disk_offload=enable_tensor_disk_offload,
            client_memory_gc_rounds=client_memory_gc_rounds,
            cuda_empty_cache=cuda_empty_cache,
        )
