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

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.fl_model import FLModel, FLModelConst, TransferType
from nvflare.app_common.app_constant import AppConstants

MODEL_ATTRS = [
    FLModelConst.MODEL,
    FLModelConst.METRICS,
    FLModelConst.TRANSFER_TYPE,
    FLModelConst.OPTIMIZER,
    FLModelConst.METRICS,
    FLModelConst.CONFIGS,
    FLModelConst.CLIENT_WEIGHTS,
    FLModelConst.ROUND,
    FLModelConst.TOTAL_ROUNDS,
    FLModelConst.META,
]


transfer_type_to_data_kind = {
    TransferType.MODEL.value: DataKind.WEIGHTS,
    TransferType.MODEL_DIFF.value: DataKind.WEIGHT_DIFF,
    TransferType.METRICS_ONLY.value: DataKind.METRICS,
}
data_kind_to_transfer_type = {v: k for k, v in transfer_type_to_data_kind.items()}


class FLModelUtils:
    @staticmethod
    def to_shareable(fl_model: FLModel) -> Shareable:
        """From FLModel to NVFlare side shareable.

        This is a temporary solution to converts FLModel to the shareable of existing style,
        so that we can reuse the existing components we have.

        In the future, we should be using the to_dxo, from_dxo directly.
        And all the components should be changed to accept the standard DXO.
        """
        if fl_model.transfer_type.value not in transfer_type_to_data_kind:
            raise ValueError(f"Invalid TransferType: ({fl_model.transfer_type.value}).")

        data_kind = transfer_type_to_data_kind[fl_model.transfer_type.value]

        dxo = DXO(data_kind, data=fl_model.model, meta={})
        shareable = dxo.to_shareable()
        if fl_model.round:
            shareable.set_header(AppConstants.CURRENT_ROUND, fl_model.round)
        if fl_model.total_rounds:
            shareable.set_header(AppConstants.NUM_ROUNDS, fl_model.total_rounds)
        if fl_model.meta:
            if AppConstants.VALIDATE_TYPE in fl_model.meta:
                shareable.set_header(AppConstants.VALIDATE_TYPE, fl_model.meta[AppConstants.VALIDATE_TYPE])
        return shareable

    @staticmethod
    def from_shareable(shareable: Shareable) -> FLModel:
        """From NVFlare side shareable to FLModel.

        This is a temporary solution to converts the shareable of existing style to FLModel,
        so that we can reuse the existing components we have.

        In the future, we should be using the to_dxo, from_dxo directly.
        And all the components should be changed to accept the standard DXO.
        """
        kwargs = {}
        dxo = from_shareable(shareable)
        if dxo.data_kind not in data_kind_to_transfer_type:
            raise ValueError(f"Invalid shareable with dxo that has data kind: {dxo.data_kind}")

        kwargs[FLModelConst.TRANSFER_TYPE] = TransferType(data_kind_to_transfer_type[dxo.data_kind])

        current_round = shareable.get_header(AppConstants.CURRENT_ROUND, None)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS, None)
        validate_type = shareable.get_header(AppConstants.VALIDATE_TYPE, None)

        kwargs[FLModelConst.MODEL] = dxo.data
        kwargs[FLModelConst.ROUND] = current_round
        kwargs[FLModelConst.TOTAL_ROUNDS] = total_rounds
        kwargs[FLModelConst.META] = {AppConstants.VALIDATE_TYPE: validate_type}

        result = FLModel(**kwargs)
        return result

    @staticmethod
    def to_dxo(fl_model: FLModel) -> DXO:
        """Converts FLModel to a DXO."""
        dxo_dict = {}
        for attr in MODEL_ATTRS:
            value = getattr(fl_model, attr, None)
            if value is not None:
                dxo_dict[attr] = value
        result = DXO(data_kind=DataKind.FL_MODEL, data=dxo_dict)
        return result

    @staticmethod
    def from_dxo(dxo: DXO) -> FLModel:
        """Converts DXO to FLModel."""
        if dxo.data_kind != DataKind.FL_MODEL:
            raise ValueError(f"Invalid dxo with data_kind: {dxo.data_kind}")

        if not isinstance(dxo.data, dict):
            raise ValueError(f"Invalid dxo with data of type: {type(dxo.data)}")

        kwargs = {}
        for attr in MODEL_ATTRS:
            value = dxo.data.get(attr, None)
            if value is not None:
                kwargs[attr] = value
        return FLModel(**kwargs)
