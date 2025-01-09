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
from nvflare.app_opt.tf.params_converter import KerasModelToNumpyParamsConverter, NumpyToKerasModelParamsConverter
from nvflare.client.config import ExchangeFormat


class TFClientAPILauncherExecutor(ClientAPILauncherExecutor):
    def initialize(self, fl_ctx: FLContext) -> None:
        self._params_exchange_format = ExchangeFormat.NUMPY
        super().initialize(fl_ctx)
        if self._from_nvflare_converter is None:
            self._from_nvflare_converter = NumpyToKerasModelParamsConverter(
                [AppConstants.TASK_TRAIN, AppConstants.TASK_VALIDATION]
            )
        if self._to_nvflare_converter is None:
            self._to_nvflare_converter = KerasModelToNumpyParamsConverter(
                [AppConstants.TASK_TRAIN, AppConstants.TASK_SUBMIT_MODEL]
            )
