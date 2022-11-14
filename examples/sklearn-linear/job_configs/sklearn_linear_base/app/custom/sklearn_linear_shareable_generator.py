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

import json

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, model_learnable_to_dxo
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants


class LinearModelShareableGenerator(ShareableGenerator):
    def __init__(self):
        super().__init__()
        self.shareable = None

    def learnable_to_shareable(self, model_learnable: ModelLearnable, fl_ctx: FLContext) -> Shareable:
        """Convert ModelLearnable to Shareable.

        Args:
            model_learnable (ModelLearnable): model to be converted
            fl_ctx (FLContext): FL context

        Returns:
            Shareable: a shareable containing a DXO object.
        """

        if not self.shareable:
            # initialization or recovering from previous training
            model = model_learnable[ModelLearnableKey.WEIGHTS]
            if model:
                dxo = DXO(data_kind=DataKind.SKL_LINEAR_MODEL, data={"model_data": model})
            else:
                # initial run, starting from empty model
                dxo = model_learnable_to_dxo(model_learnable)
            return dxo.to_shareable()
        else:
            # return shareable saved from previous call to shareable_to_learnable
            return self.shareable

    def shareable_to_learnable(self, shareable: Shareable, fl_ctx: FLContext) -> ModelLearnable:
        """Convert Shareable to ModelLearnable.

        Args:
            shareable (Shareable): Shareable that contains a DXO object
            fl_ctx (FLContext): FL context

        Returns:
            A ModelLearnable object

        Raises:
            TypeError: if shareable is not of type shareable
            ValueError: if data_kind is not `DataKind.XGB_MODEL`
        """
        if not isinstance(shareable, Shareable):
            raise TypeError("shareable must be Shareable, but got {}.".format(type(shareable)))

        base_model = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
        if not base_model:
            self.system_panic(reason="No global base model!", fl_ctx=fl_ctx)
            return base_model

        dxo = from_shareable(shareable)

        if dxo.data_kind == DataKind.SKL_LINEAR_MODEL:
            model_update = dxo.data
            if not model_update:
                self.log_info(fl_ctx, "No model update found. Model will not be updated.")
            else:
                model_data_dict = model_update.get("model_params")
                base_model[ModelLearnableKey.WEIGHTS] = model_data_dict
            self.shareable = dxo.to_shareable()
        else:
            raise ValueError("data_kind should be either DataKind.SKL_LINEAR_MODEL, but got {}".format(dxo.data_kind))
        return base_model
