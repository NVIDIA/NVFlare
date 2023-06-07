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
from typing import Dict, List

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.fuel.utils.fobs import fobs


class FLModelConverter:
    @staticmethod
    def fl_model_to_shareable(fl_model: FLModel) -> Shareable:
        o = fobs.serialize(fl_model)
        data = {"model": o}
        dxo = DXO(DataKind.MODEL, data=data, meta={"fob_serialized": True})
        return dxo.to_shareable()

    @staticmethod
    def fl_model_list_to_shareable(fl_models: List[FLModel]) -> Shareable:
        dxo_list = []
        for fl_model in fl_models:
            dxo_list.append(FLModelConverter.fl_model_to_shareable(fl_model))

        data = {"model_list": dxo_list}
        dxo = DXO(data_kind=DataKind.COLLECTION, data=data, meta={})
        return dxo.to_shareable()

    @staticmethod
    def fl_model_dict_to_shareable(fl_models: Dict[str, FLModel]) -> Shareable:
        data = {"model_dict": fl_models}
        dxo = DXO(data_kind=DataKind.COLLECTION, data=data, meta={})
        return dxo.to_shareable()

    def shareable_to_model(self, shareable: Shareable) -> FLModel:
        pass
