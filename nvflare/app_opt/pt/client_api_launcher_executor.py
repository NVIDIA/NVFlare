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

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.client_api_launcher_executor import ClientAPILauncherExecutor
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.app_opt.pt.numpy_params_converter import NumpyToPTParamsConverter, PTToNumpyParamsConverter
from nvflare.app_opt.pt.tensor_params_converter import PTReceiveParamsConverter, PTSendParamsConverter
from nvflare.client.config import ExchangeFormat
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.log_utils import get_obj_logger


class PTClientAPILauncherExecutor(ClientAPILauncherExecutor):
    def initialize(self, fl_ctx: FLContext) -> None:
        fobs.register(TensorDecomposer)
        super().initialize(fl_ctx)
        self.logger = get_obj_logger(self)
        if self._from_nvflare_converter is None:
            # if not specified, assign defaults
            if self._params_exchange_format == ExchangeFormat.NUMPY:
                self.logger.debug("Numpy from_nvflare_converter initialized")
                self._from_nvflare_converter = NumpyToPTParamsConverter(
                    [AppConstants.TASK_TRAIN, AppConstants.TASK_VALIDATION]
                )
            elif self._params_exchange_format == ExchangeFormat.PYTORCH:
                self.logger.debug("Pytorch from_nvflare_converter initialized")
                self._from_nvflare_converter = PTReceiveParamsConverter(
                    [AppConstants.TASK_TRAIN, AppConstants.TASK_VALIDATION]
                )
            else:
                self._from_nvflare_converter = None

        if self._to_nvflare_converter is None:
            # if not specified, assign defaults
            if self._params_exchange_format == ExchangeFormat.NUMPY:
                self.logger.debug("Numpy to_nvflare_converter initialized")
                self._to_nvflare_converter = PTToNumpyParamsConverter(
                    [AppConstants.TASK_TRAIN, AppConstants.TASK_SUBMIT_MODEL]
                )
            elif self._params_exchange_format == ExchangeFormat.PYTORCH:
                self.logger.debug("Pytorch to_nvflare_converter initialized")
                self._to_nvflare_converter = PTSendParamsConverter(
                    [AppConstants.TASK_TRAIN, AppConstants.TASK_SUBMIT_MODEL]
                )
            else:
                self._to_nvflare_converter = None
