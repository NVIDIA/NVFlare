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
from nvflare.app_common.executors.client_api_launcher_executor import ClientAPILauncherExecutor
from nvflare.app_common.model_exchange.constants import ModelExchangeFormat
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.app_opt.pt.params_converter import NumpyToPTParamsConverter, PTToNumpyParamsConverter
from nvflare.fuel.utils import fobs


class PTClientAPILauncherExecutor(ClientAPILauncherExecutor):
    def initialize(self, fl_ctx: FLContext) -> None:
        fobs.register(TensorDecomposer)
        self._params_exchange_format = ModelExchangeFormat.PYTORCH
        super().initialize(fl_ctx)
        if self._from_nvflare_converter is None:
            self._from_nvflare_converter = NumpyToPTParamsConverter()
        if self._to_nvflare_converter is None:
            self._to_nvflare_converter = PTToNumpyParamsConverter()
