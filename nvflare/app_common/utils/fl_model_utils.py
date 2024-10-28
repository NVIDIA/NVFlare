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

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.fl_model import FLModel, FLModelConst, MetaKey, ParamsType
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, make_model_learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.fuel.utils.validation_utils import check_object_type

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


class FLModelUtils:
    @staticmethod
    def to_shareable(fl_model: FLModel) -> Shareable:
        """From FLModel to NVFlare side shareable.

        This is a temporary solution to converts FLModel to the shareable of existing style,
        so that we can reuse the existing components we have.

        In the future, we should be using the to_dxo, from_dxo directly.
        And all the components should be changed to accept the standard DXO.
        """
        if fl_model.params is None and fl_model.metrics is None:
            raise ValueError("FLModel without params and metrics is NOT supported.")
        elif fl_model.params is not None:
            if fl_model.params_type is None:
                fl_model.params_type = ParamsType.FULL

            data_kind = params_type_to_data_kind.get(fl_model.params_type.value)
            if data_kind is None:
                raise ValueError(f"Invalid ParamsType: ({fl_model.params_type}).")

            if fl_model.metrics is None:
                dxo = DXO(data_kind, data=fl_model.params, meta={})
            else:
                # if both params and metrics are presented, will be treated as evaluation on the global model
                dxo = DXO(data_kind, data=fl_model.params, meta={MetaKey.INITIAL_METRICS: fl_model.metrics})
        else:
            dxo = DXO(DataKind.METRICS, data=fl_model.metrics, meta={})

        meta = fl_model.meta if fl_model.meta is not None else {}
        dxo.meta.update(meta)

        shareable = dxo.to_shareable()
        if fl_model.start_round is not None:
            shareable.set_header(AppConstants.START_ROUND, fl_model.start_round)
        if fl_model.current_round is not None:
            shareable.set_header(AppConstants.CURRENT_ROUND, fl_model.current_round)
        if fl_model.total_rounds is not None:
            shareable.set_header(AppConstants.NUM_ROUNDS, fl_model.total_rounds)

        if MetaKey.VALIDATE_TYPE in meta:
            shareable.set_header(AppConstants.VALIDATE_TYPE, meta[MetaKey.VALIDATE_TYPE])
        return shareable

    @staticmethod
    def from_shareable(shareable: Shareable, fl_ctx: Optional[FLContext] = None) -> FLModel:
        """From NVFlare side shareable to FLModel.

        This is a temporary solution to converts the shareable of existing style to FLModel,
        so that we can reuse the existing components we have.

        In the future, we should be using the to_dxo, from_dxo directly.
        And all the components should be changed to accept the standard DXO.
        """
        metrics = None
        params_type = None
        params = None
        meta = {}

        submit_model_name = shareable.get_header(AppConstants.SUBMIT_MODEL_NAME)
        if submit_model_name:
            # this only happens in cross-site eval right now
            meta[MetaKey.SUBMIT_MODEL_NAME] = submit_model_name
        else:
            dxo = from_shareable(shareable)
            meta = dict(dxo.meta)
            if dxo.data_kind == DataKind.METRICS:
                metrics = dxo.data
            else:
                params_type = data_kind_to_params_type.get(dxo.data_kind)
                params = dxo.data
                if params_type is None:
                    if params is None:
                        raise ValueError(f"Invalid shareable with dxo that has data kind: {dxo.data_kind}")
                    else:
                        params_type = ParamsType.FULL

                params_type = ParamsType(params_type)

                if MetaKey.INITIAL_METRICS in meta:
                    metrics = meta[MetaKey.INITIAL_METRICS]

        start_round = shareable.get_header(AppConstants.START_ROUND, None)
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND, None)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS, None)
        validate_type = shareable.get_header(AppConstants.VALIDATE_TYPE, None)
        if validate_type is not None:
            meta[MetaKey.VALIDATE_TYPE] = validate_type

        job_id = shareable.get_header(MetaKey.JOB_ID, None)
        if job_id is not None:
            meta[MetaKey.JOB_ID] = job_id

        site_name = shareable.get_header(MetaKey.SITE_NAME, None)
        if site_name is not None:
            meta[MetaKey.SITE_NAME] = site_name

        result = FLModel(
            params_type=params_type,
            params=params,
            metrics=metrics,
            start_round=start_round,
            current_round=current_round,
            total_rounds=total_rounds,
            meta=meta,
        )
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

        params = dxo.data.get(FLModelConst.PARAMS, None)
        params_type = dxo.data.get(FLModelConst.PARAMS_TYPE, None)
        metrics = dxo.data.get(FLModelConst.METRICS, None)
        optimizer_params = dxo.data.get(FLModelConst.OPTIMIZER_PARAMS, None)
        start_round = dxo.data.get(FLModelConst.START_ROUND, None)
        current_round = dxo.data.get(FLModelConst.CURRENT_ROUND, None)
        total_rounds = dxo.data.get(FLModelConst.TOTAL_ROUNDS, None)
        meta = dxo.data.get(FLModelConst.META, None)

        return FLModel(
            params=params,
            params_type=params_type,
            metrics=metrics,
            optimizer_params=optimizer_params,
            start_round=start_round,
            current_round=current_round,
            total_rounds=total_rounds,
            meta=meta,
        )

    @staticmethod
    def to_model_learnable(fl_model: FLModel) -> ModelLearnable:
        return make_model_learnable(weights=fl_model.params, meta_props=fl_model.meta)

    @staticmethod
    def from_model_learnable(model_learnable: ModelLearnable) -> FLModel:
        return FLModel(
            params_type=ParamsType.FULL,
            params=model_learnable[ModelLearnableKey.WEIGHTS],
            meta=model_learnable[ModelLearnableKey.META],
        )

    @staticmethod
    def get_meta_prop(model: FLModel, key: str, default=None):
        check_object_type("model", model, FLModel)
        if not model.meta:
            return default
        else:
            return model.meta.get(key, default)

    @staticmethod
    def set_meta_prop(model: FLModel, key: str, value: Any):
        check_object_type("model", model, FLModel)
        model.meta[key] = value

    @staticmethod
    def get_configs(model: FLModel) -> Optional[dict]:
        return FLModelUtils.get_meta_prop(model, MetaKey.CONFIGS)

    @staticmethod
    def update_model(model: FLModel, model_update: FLModel, replace_meta: bool = True) -> FLModel:
        if model.params_type != ParamsType.FULL:
            raise RuntimeError(f"params_type {model.params_type} of `model` not supported! Expected `ParamsType.FULL`.")

        if replace_meta:
            model.meta = model_update.meta
        else:
            model.meta.update(model_update.meta)

        model.metrics = model_update.metrics

        if model_update.params_type == ParamsType.FULL:
            model.params = model_update.params
        elif model_update.params_type == ParamsType.DIFF:
            for v_name, v_value in model_update.params.items():
                model.params[v_name] = model.params[v_name] + v_value
        else:
            raise RuntimeError(f"params_type {model_update.params_type} of `model_update` not supported!")
        return model
