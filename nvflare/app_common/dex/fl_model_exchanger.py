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

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.dex.dxo_exchanger import DXOExchanger
from nvflare.app_common.utils.fl_model_utils import FLModelUtils


class FLModelExchanger(DXOExchanger):
    def get_model(self, data_id: str, timeout: Optional[float] = None) -> FLModel:
        dxo = self.get(data_id, timeout)
        return FLModelUtils.from_dxo(dxo)

    def put_model(self, data_id: str, model: FLModel):
        dxo = FLModelUtils.to_dxo(model)
        self.put(data_id, dxo)
