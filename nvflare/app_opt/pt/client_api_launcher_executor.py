# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.client_api_launcher_executor import ClientAPILauncherExecutor
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.app_opt.pt.numpy_params_converter import NumpyToPTParamsConverter, PTToNumpyParamsConverter
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.client.constants import CLIENT_API_CONFIG
from nvflare.fuel.utils import fobs


class PTClientAPILauncherExecutor(ClientAPILauncherExecutor):
    def __init__(
        self,
        pipe_id: str,
        launcher_id: Optional[str] = None,
        launch_timeout: Optional[float] = None,
        task_wait_timeout: Optional[float] = None,
        last_result_transfer_timeout: float = 300.0,
        external_pre_init_timeout: float = 300.0,
        peer_read_timeout: Optional[float] = 300.0,
        monitor_interval: float = 0.01,
        read_interval: float = 0.5,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 300.0,
        workers: int = 4,
        train_with_evaluation: bool = False,
        train_task_name: str = AppConstants.TASK_TRAIN,
        evaluate_task_name: str = AppConstants.TASK_VALIDATION,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
        from_nvflare_converter_id: Optional[str] = None,
        to_nvflare_converter_id: Optional[str] = None,
        server_expected_format: str = ExchangeFormat.NUMPY,
        params_exchange_format: str = ExchangeFormat.PYTORCH,
        params_transfer_type: str = TransferType.FULL,
        config_file_name: str = CLIENT_API_CONFIG,
    ) -> None:
        ClientAPILauncherExecutor.__init__(
            self,
            pipe_id=pipe_id,
            launcher_id=launcher_id,
            launch_timeout=launch_timeout,
            task_wait_timeout=task_wait_timeout,
            last_result_transfer_timeout=last_result_transfer_timeout,
            external_pre_init_timeout=external_pre_init_timeout,
            peer_read_timeout=peer_read_timeout,
            monitor_interval=monitor_interval,
            read_interval=read_interval,
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
            workers=workers,
            train_with_evaluation=train_with_evaluation,
            train_task_name=train_task_name,
            evaluate_task_name=evaluate_task_name,
            submit_model_task_name=submit_model_task_name,
            from_nvflare_converter_id=from_nvflare_converter_id,
            to_nvflare_converter_id=to_nvflare_converter_id,
            server_expected_format=server_expected_format,
            params_exchange_format=params_exchange_format,
            params_transfer_type=params_transfer_type,
            config_file_name=config_file_name,
        )

    def initialize(self, fl_ctx: FLContext) -> None:
        fobs.register(TensorDecomposer)
        super().initialize(fl_ctx)

        if (
            self._server_expected_format == ExchangeFormat.NUMPY
            and self._params_exchange_format == ExchangeFormat.PYTORCH
        ):
            if self._from_nvflare_converter is None:
                self._from_nvflare_converter = NumpyToPTParamsConverter(
                    [AppConstants.TASK_TRAIN, AppConstants.TASK_VALIDATION]
                )

            if self._to_nvflare_converter is None:
                self._to_nvflare_converter = PTToNumpyParamsConverter(
                    [AppConstants.TASK_TRAIN, AppConstants.TASK_SUBMIT_MODEL]
                )
