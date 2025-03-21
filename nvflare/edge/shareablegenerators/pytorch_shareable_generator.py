# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants
from nvflare.edge.constants import MsgKey
from nvflare.edge.model_protocol import ModelBufferType, ModelEncoding, ModelExchangeFormat, ModelNativeFormat


class PytorchShareableGenerator(ShareableGenerator):
    def learnable_to_shareable(self, model_learnable: ModelLearnable, fl_ctx: FLContext) -> Shareable:
        """Convert ModelLearnable to Shareable.

        Args:
            model_learnable (ModelLearnable): model to be converted
            fl_ctx (FLContext): FL context

        Returns:
            Shareable: a shareable containing a DXO object.
        """
        # Compose shareable
        task_data = Shareable()
        model_weights = model_learnable[ModelLearnableKey.WEIGHTS]
        # convert to list
        model_weights = {k: v.tolist() for k, v in model_weights.items()}
        task_data[MsgKey.PAYLOAD] = {
            ModelExchangeFormat.MODEL_BUFFER: model_weights,
            ModelExchangeFormat.MODEL_BUFFER_TYPE: ModelBufferType.PYTORCH,
            ModelExchangeFormat.MODEL_BUFFER_NATIVE_FORMAT: ModelNativeFormat.STRING,
            ModelExchangeFormat.MODEL_BUFFER_ENCODING: ModelEncoding.NONE,
        }
        return task_data

    def shareable_to_learnable(self, shareable: Shareable, fl_ctx: FLContext) -> ModelLearnable:
        """Convert Shareable to ModelLearnable.

        Supporting TYPE == TYPE_WEIGHT_DIFF or TYPE_WEIGHTS

        Args:
            shareable (Shareable): Shareable that contains a DXO object
            fl_ctx (FLContext): FL context

        Returns:
            A ModelLearnable object

        Raises:
            TypeError: if shareable is not of type shareable
            ValueError: if data_kind is not `DataKind.WEIGHTS` and is not `DataKind.WEIGHT_DIFF`
        """
        if not isinstance(shareable, Shareable):
            raise TypeError("shareable must be Shareable, but got {}.".format(type(shareable)))

        base_model = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
        result = shareable.get(MsgKey.RESULT)
        weight_to_add = result.get(MsgKey.WEIGHTS)
        mode = result.get(MsgKey.MODE)
        divide_factor = shareable.get(MsgKey.NUM_DEVICES)

        if mode == "diff":
            if not base_model:
                self.system_panic(reason="No global base model found for processing WEIGHT_DIFF!", fl_ctx=fl_ctx)
                return base_model
            weights = base_model[ModelLearnableKey.WEIGHTS]
            # apply updates
            for v_name, v_value in weight_to_add.items():
                weights[v_name] = weights[v_name] + np.array(v_value) / divide_factor
        elif mode == "weight":
            if not base_model:
                base_model = ModelLearnable()
            weights = base_model[ModelLearnableKey.WEIGHTS]
            # apply updates
            for v_name, v_value in weight_to_add.items():
                weights[v_name] = np.array(v_value) / divide_factor
        else:
            raise ValueError(f"data_kind should be either diff or weight, but got {mode}")

        return base_model
