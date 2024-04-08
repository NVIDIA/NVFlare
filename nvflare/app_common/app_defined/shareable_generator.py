# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Any

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, make_model_learnable
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants

from .component_base import ComponentBase


class AppDefinedShareableGenerator(ShareableGenerator, ComponentBase, ABC):
    def __init__(self):
        ShareableGenerator.__init__(self)
        ComponentBase.__init__(self)
        self.current_round = None

    @abstractmethod
    def model_to_trainable(self, model_obj: Any) -> (Any, dict):
        """Convert the model weights and meta to a format that can be sent to clients to do training

        Args:
            model_obj: model object

        Returns: a tuple of (weights, meta)

        The returned weights and meta will be for training and serializable
        """
        pass

    @abstractmethod
    def update_model(self, model_obj: Any, training_result: Any, meta: dict) -> Any:
        """Update model with training result and meta

        Args:
            model_obj: base model object to be updated
            training_result: training result to be applied to the model object
            meta: trained meta

        Returns: the updated model object

        """
        pass

    def learnable_to_shareable(self, learnable: Learnable, fl_ctx: FLContext) -> Shareable:
        self.fl_ctx = fl_ctx
        self.current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        self.debug(f"{learnable=}")
        base_model_obj = learnable.get(ModelLearnableKey.WEIGHTS)
        trainable_weights, trainable_meta = self.model_to_trainable(base_model_obj)
        self.debug(f"trainable weights: {trainable_weights}")
        dxo = DXO(
            data_kind=DataKind.APP_DEFINED,
            data=trainable_weights,
            meta=trainable_meta,
        )
        self.debug(f"learnable_to_shareable: {dxo.data}")
        return dxo.to_shareable()

    def shareable_to_learnable(self, shareable: Shareable, fl_ctx: FLContext) -> Learnable:
        self.fl_ctx = fl_ctx
        self.current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        base_model_learnable = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)

        if not base_model_learnable:
            self.system_panic(reason="No global base model!", fl_ctx=fl_ctx)
            return base_model_learnable

        if not isinstance(base_model_learnable, ModelLearnable):
            raise ValueError(f"expect global model to be ModelLearnable but got {type(base_model_learnable)}")
        base_model_obj = base_model_learnable.get(ModelLearnableKey.WEIGHTS)

        dxo = from_shareable(shareable)
        training_result = dxo.data
        trained_meta = dxo.meta
        model_obj = self.update_model(model_obj=base_model_obj, training_result=training_result, meta=trained_meta)
        return make_model_learnable(model_obj, {})
