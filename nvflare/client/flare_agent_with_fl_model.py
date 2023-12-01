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


from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.utils.fl_model_utils import FLModelUtils

from .flare_agent import FlareAgent


class FlareAgentWithFLModel(FlareAgent):
    def shareable_to_task_data(self, shareable: Shareable) -> FLModel:
        model = FLModelUtils.from_shareable(shareable)
        return model

    def task_result_to_shareable(self, result: FLModel, rc) -> Shareable:
        shareable = FLModelUtils.to_shareable(result)
        shareable.set_return_code(rc)
        return shareable
