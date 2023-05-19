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

from nvflare.apis.dxo import DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.model import (
    ModelLearnable,
    ModelLearnableKey,
    make_model_learnable,
    model_learnable_to_dxo,
)
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants


class FullModelShareableFedSMGenerator(ShareableGenerator):
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

    def update_single_model(self, dxo_single_model, base_model_set, model_id, fl_ctx: FLContext):
        if not dxo_single_model:
            self.log_error(fl_ctx, f"Aggregated model weights for {model_id} are missing!")
            return
        # get base_model from the base_model_set
        base_model = base_model_set[model_id]
        if not base_model:
            self.system_panic(
                reason=f"No base personalized model for {model_id}!",
                fl_ctx=fl_ctx,
            )
            return base_model
        weights = base_model[ModelLearnableKey.WEIGHTS]
        # update with aggregated dxo
        if dxo_single_model.data_kind == DataKind.WEIGHT_DIFF:
            # add aggregated weight_diff from aggregator record to the base model weights
            if dxo_single_model is not None:
                model_diff = dxo_single_model.data
                for v_name in model_diff.keys():
                    weights[v_name] = weights[v_name] + model_diff[v_name]
        elif dxo_single_model.data_kind == DataKind.WEIGHTS:
            # update weights directly
            weights_new = dxo_single_model.data
            if not weights_new:
                self.log_info(
                    fl_ctx,
                    f"No model weights for {model_id} found. Model will not be updated.",
                )
            else:
                base_model[ModelLearnableKey.WEIGHTS] = weights_new
        else:
            raise ValueError(
                f"data_kind should be either DataKind.WEIGHTS or DataKind.WEIGHT_DIFF, but got {dxo_single_model.data_kind}"
            )
        # set meta and set base_model_set
        base_model[ModelLearnableKey.META] = dxo_single_model.get_meta_props()
        base_model_set[model_id] = base_model

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
            raise TypeError(f"shareable must be Shareable, but got {type(shareable)}.")

        # base_model_set is a "flattened set", containing all models with ids
        # "select_weights", "select_exp_avg", "select_exp_avg_sq", "global_weights", and client_ids
        base_model_set = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)["weights"]
        meta = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)["meta"]
        if not base_model_set:
            self.system_panic(reason="No FedSM base model set!", fl_ctx=fl_ctx)
            return base_model_set

        # dxo from aggregator is hierarchically organized as ["select_weights", "global_weights", "person_weights"]
        # "global_weights" is a dxo for global model
        # "person_weights" is a dxo collection containing dxo for each client_id
        # "select_weights" is a dxo collection containing dxo for ["select_weights", "exp_avg", "exp_avg_sq"]
        dxo = from_shareable(shareable)

        dxo_global = dxo.data.get("global_weights")
        self.update_single_model(dxo_global, base_model_set, "global_weights", fl_ctx)

        dxo_person = dxo.data.get("person_weights")
        for model_id in client_ids:
            dxo_single = dxo_person.get(model_id)
            self.update_single_model(dxo_single, base_model_set, model_id, fl_ctx)

        dxo_select = dxo.data.get("select_weights")
        for model_id in ["select_weights", "select_exp_avg", "select_exp_avg_sq"]:
            dxo_single = dxo_select.get(model_id)
            self.update_single_model(dxo_single, base_model_set, model_id, fl_ctx)

        model_set = make_model_learnable(base_model_set, meta)

        return model_set
