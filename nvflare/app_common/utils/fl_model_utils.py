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

from abc import ABC, abstractmethod
from typing import Dict, Optional

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.dxo import MetaKey as DXOMetaKey
from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.fl_model import FLModel, FLModelConst, MetaKey, ParamsType
from nvflare.app_common.app_constant import AppConstants

MODEL_ATTRS = [
    FLModelConst.PARAMS_TYPE,
    FLModelConst.PARAMS,
    FLModelConst.METRICS,
    FLModelConst.OPTIMIZER_PARAMS,
    FLModelConst.CURRENT_ROUND,
    FLModelConst.TOTAL_ROUNDS,
    FLModelConst.META,
]


params_type_to_data_kind = {
    ParamsType.FULL.value: DataKind.WEIGHTS,
    ParamsType.DIFF.value: DataKind.WEIGHT_DIFF,
}
data_kind_to_params_type = {v: k for k, v in params_type_to_data_kind.items()}


class ParamsConverter(ABC):
    """This class converts params from one format to the other."""

    @abstractmethod
    def convert(self, params: Dict) -> Dict:
        pass


class FLModelUtils:
    @staticmethod
    def to_shareable(fl_model: FLModel, params_converter: Optional[ParamsConverter] = None) -> Shareable:
        """From FLModel to NVFlare side shareable.

        This is a temporary solution to converts FLModel to the shareable of existing style,
        so that we can reuse the existing components we have.

        In the future, we should be using the to_dxo, from_dxo directly.
        And all the components should be changed to accept the standard DXO.
        """
        if fl_model.params is None and fl_model.metrics is None:
            raise ValueError("FLModel without params and metrics is NOT supported.")
        elif fl_model.params is not None:
            data_kind = params_type_to_data_kind.get(fl_model.params_type)
            if data_kind is None:
                raise ValueError(f"Invalid ModelType: ({fl_model.params_type}).")
            if params_converter is not None:
                fl_model.params = params_converter.convert(fl_model.params)

            if fl_model.metrics is None:
                dxo = DXO(data_kind, data=fl_model.params, meta={})
            else:
                # if both params and metrics are presented, will be treated as initial evaluation on the global model
                dxo = DXO(data_kind, data=fl_model.params, meta={DXOMetaKey.INITIAL_METRICS: fl_model.metrics})
        else:
            dxo = DXO(DataKind.METRICS, data=fl_model.metrics, meta={})

        shareable = dxo.to_shareable()
        if fl_model.current_round is not None:
            shareable.set_header(AppConstants.CURRENT_ROUND, fl_model.current_round)

        meta = fl_model.meta if fl_model.meta is not None else {}
        meta[MetaKey.CURRENT_ROUND] = fl_model.current_round
        meta[MetaKey.TOTAL_ROUNDS] = fl_model.total_rounds

        dxo.meta.update(meta)
        if MetaKey.VALIDATE_TYPE in meta:
            shareable.set_header(AppConstants.VALIDATE_TYPE, meta[MetaKey.VALIDATE_TYPE])
        if MetaKey.TOTAL_ROUNDS in meta:
            shareable.set_header(AppConstants.NUM_ROUNDS, meta[MetaKey.TOTAL_ROUNDS])
        return shareable

    @staticmethod
    def from_shareable(
        shareable: Shareable, params_converter: Optional[ParamsConverter] = None, fl_ctx: Optional[FLContext] = None
    ) -> FLModel:
        """From NVFlare side shareable to FLModel.

        This is a temporary solution to converts the shareable of existing style to FLModel,
        so that we can reuse the existing components we have.

        In the future, we should be using the to_dxo, from_dxo directly.
        And all the components should be changed to accept the standard DXO.
        """
        kwargs = {}
        dxo = from_shareable(shareable)

        if dxo.data_kind == DataKind.METRICS:
            kwargs[FLModelConst.METRICS] = dxo.data
        else:
            params_type = data_kind_to_params_type.get(dxo.data_kind)
            if params_type is None:
                raise ValueError(f"Invalid shareable with dxo that has data kind: {dxo.data_kind}")
            kwargs[FLModelConst.PARAMS_TYPE] = ParamsType(params_type)
            if params_converter:
                dxo.data = params_converter.convert(dxo.data)
            kwargs[FLModelConst.PARAMS] = dxo.data

        current_round = shareable.get_header(AppConstants.CURRENT_ROUND, None)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS, None)
        validate_type = shareable.get_header(AppConstants.VALIDATE_TYPE, None)

        kwargs[FLModelConst.CURRENT_ROUND] = current_round
        kwargs[FLModelConst.META] = dxo.meta
        if validate_type is not None:
            kwargs[FLModelConst.META][MetaKey.VALIDATE_TYPE] = validate_type
        if total_rounds is not None:
            kwargs[FLModelConst.TOTAL_ROUNDS] = total_rounds

        if fl_ctx is not None:
            kwargs[FLModelConst.META][MetaKey.JOB_ID] = fl_ctx.get_job_id()
            kwargs[FLModelConst.META][MetaKey.SITE_NAME] = fl_ctx.get_identity_name()

        result = FLModel(**kwargs)
        return result

    @staticmethod
    def to_dxo(fl_model: FLModel) -> DXO:
        """Converts FLModel to a DXO."""
        attr_dict = {}
        for attr in MODEL_ATTRS:
            value = getattr(fl_model, attr, None)
            if value is not None:
                attr_dict[attr] = value
        result = DXO(data_kind=DataKind.FL_MODEL, data=attr_dict)
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
