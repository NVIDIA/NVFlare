# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.dxo import DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, model_learnable_to_dxo
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants


class FullModelShareableGeneratorFedSM(ShareableGenerator):
    def learnable_to_shareable(self, model_learnable: ModelLearnable, fl_ctx: FLContext) -> Shareable:
        """Convert ModelLearnable to Shareable.

        Args:
            model_learnable (ModelLearnable): model to be converted
            fl_ctx (FLContext): FL context

        Returns:
            Shareable: a shareable containing a DXO object.
        """
        dxo = model_learnable_to_dxo(model_learnable)
        return dxo.to_shareable()

    def shareable_to_learnable(self, shareable: Shareable, client_ids: list, fl_ctx: FLContext) -> ModelLearnable:
        """Convert Shareable to ModelLearnable.

        Supporting TYPE == TYPE_WEIGHT_DIFF or TYPE_WEIGHTS

        Args:
            shareable (Shareable): Shareable that contains a DXO object
            client_ids: client id list for getting the personalized models
            fl_ctx (FLContext): FL context

        Returns:
            A ModelLearnable object

        Raises:
            TypeError: if shareable is not of type shareable
            ValueError: if data_kind is not `DataKind.WEIGHTS` and is not `DataKind.WEIGHT_DIFF`
        """
        if not isinstance(shareable, Shareable):
            raise TypeError("shareable must be Shareable, but got {}.".format(type(shareable)))

        # base_model is contains several models with ids "select_model", "global_model", and client_ids
        base_model = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
        if not base_model:
            self.system_panic(reason="No FedSM base model!", fl_ctx=fl_ctx)
            return base_model
        # dxo are organized as ['global_weights', 'select_weights', 'person_weights']
        # where 'person_weights' is a dxo collection containing dxo for each client_id
        dxo = from_shareable(shareable)

        # selector model is single model, directly provides weights
        dxo_select = dxo.data.get("select_weights")
        if not dxo_select:
             self.log_error(fl_ctx, "Aggregated selector model weights are missing!")
             return
        base_select_model = base_model["select_model"]
        if not base_select_model:
            self.system_panic(reason="No base model for selector!", fl_ctx=fl_ctx)
            return base_select_model
        weights = base_select_model[ModelLearnableKey.WEIGHTS]
        if dxo_select.data_kind == DataKind.WEIGHT_DIFF:
            if dxo_select.data is not None:
                model_diff = dxo_select.data
                for v_name, v_value in model_diff.items():
                    weights[v_name] = weights[v_name] + v_value
        elif dxo_select.data_kind == DataKind.WEIGHTS:
            weights = dxo_select.data
            if not weights:
                self.log_info(fl_ctx, "No model weights found. Model will not be updated.")
            else:
                base_select_model[ModelLearnableKey.WEIGHTS] = weights
        else:
            raise ValueError(
                "data_kind should be either DataKind.WEIGHTS or DataKind.WEIGHT_DIFF, but got {}".format(dxo.data_kind)
            )
        base_select_model[ModelLearnableKey.META] = dxo_select.get_meta_props()
        base_model["select_model"] = base_select_model

        # global model is single model, directly provides weights
        dxo_global = dxo.data.get("global_weights")
        if not dxo_global:
             self.log_error(fl_ctx, "Aggregated global model weights are missing!")
             return
        base_global_model = base_model["global_model"]
        if not base_global_model:
            self.system_panic(reason="No base global model!", fl_ctx=fl_ctx)
            return base_global_model
        weights = base_global_model[ModelLearnableKey.WEIGHTS]
        if dxo_global.data_kind == DataKind.WEIGHT_DIFF:
            if dxo_global.data is not None:
                model_diff = dxo_global.data
                for v_name, v_value in model_diff.items():
                    weights[v_name] = weights[v_name] + v_value
        elif dxo_global.data_kind == DataKind.WEIGHTS:
            weights = dxo_global.data
            if not weights:
                self.log_info(fl_ctx, "No model weights found. Model will not be updated.")
            else:
                base_global_model[ModelLearnableKey.WEIGHTS] = weights
        else:
            raise ValueError(
                "data_kind should be either DataKind.WEIGHTS or DataKind.WEIGHT_DIFF, but got {}".format(dxo.data_kind)
            )
        base_global_model[ModelLearnableKey.META] = dxo_global.get_meta_props()
        base_model["global_model"] = base_global_model

        # personalized weights is a set of models, provides each weights under client_ids
        dxo_person = dxo.data.get("person_weights")
        for client_id in client_ids:
            dxo_person_single = dxo_person.data.get(client_id)
            if not dxo_person_single:
                self.log_error(fl_ctx, "Aggregated personalized model weights for {} are missing!".format(client_id))
                return
            base_model_person = base_model[client_id]
            if not base_model_person:
                self.system_panic(reason="No base personalized model for {}!".format(client_id), fl_ctx=fl_ctx)
                return base_model_person
            weights = base_model_person[ModelLearnableKey.WEIGHTS]
            if dxo_person.data_kind == DataKind.WEIGHT_DIFF:
                if dxo_person_single is not None:
                    model_diff = dxo_person_single
                    # add corresponding weight_diff from aggregator record
                    for v_name in model_diff.keys():
                        weights[v_name] = weights[v_name] + model_diff[v_name]
            elif dxo_person.data_kind == DataKind.WEIGHTS:
                weights = dxo_person_single
                if not weights:
                    self.log_info(fl_ctx, "No model weights found. Model will not be updated.")
                else:
                    base_model_person[ModelLearnableKey.WEIGHTS] = weights
            else:
                raise ValueError(
                    "data_kind should be either DataKind.WEIGHTS or DataKind.WEIGHT_DIFF, but got {}".format(
                        dxo.data_kind)
                )
            base_model_person[ModelLearnableKey.META] = dxo_person.get_meta_props()
            base_model[client_id] = base_model_person

        return base_model
