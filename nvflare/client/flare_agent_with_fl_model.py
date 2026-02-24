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

from typing import Any, Optional

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.abstract.params_converter import ParamsConverter
from nvflare.app_common.utils.fl_model_utils import FLModelUtils

from .flare_agent import FlareAgent


class _ConverterContext:
    """Minimal duck-type stub for FLContext used by ParamsConverters.

    ParamsConverter.process() requires a context object with get_prop/set_prop.
    In the subprocess there is no FLContext, so this lightweight stub is used instead.
    """

    def __init__(self):
        self._props = {}

    def get_prop(self, key: str, default=None):
        return self._props.get(key, default)

    def set_prop(self, key: str, value: Any, private: Optional[bool] = None, sticky: Optional[bool] = None):
        _ = private
        _ = sticky
        self._props[key] = value


class FlareAgentWithFLModel(FlareAgent):
    def __init__(
        self,
        *args,
        from_nvflare_converter: Optional[ParamsConverter] = None,
        to_nvflare_converter: Optional[ParamsConverter] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.from_nvflare_converter = from_nvflare_converter
        self.to_nvflare_converter = to_nvflare_converter
        self._converter_ctx = _ConverterContext()

    def shareable_to_task_data(self, shareable: Shareable) -> FLModel:
        if self.from_nvflare_converter is not None:
            task_name = shareable.get_header(FLContextKey.TASK_NAME, "")
            shareable = self.from_nvflare_converter.process(task_name, shareable, self._converter_ctx)
        model = FLModelUtils.from_shareable(shareable)
        return model

    def task_result_to_shareable(self, result: FLModel, rc) -> Shareable:
        shareable = FLModelUtils.to_shareable(result)
        if self.to_nvflare_converter is not None:
            task_name = self.current_task.task_name if self.current_task else ""
            shareable = self.to_nvflare_converter.process(task_name, shareable, self._converter_ctx)
        shareable.set_return_code(rc)
        return shareable
