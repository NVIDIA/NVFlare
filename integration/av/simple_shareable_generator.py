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
from typing import Any

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants

from .utils import unwrap_dict, wrap_with_dict


class SimpleShareableGenerator(ShareableGenerator, ABC):
    def __init__(self):
        ShareableGenerator.__init__(self)
        self.fl_ctx = None

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
    def apply_weights_to_model(self, model_obj: Any, weights: Any, meta: dict) -> Any:
        """Apply trained weights and meta to the base model

        Args:
            model_obj: base model object that weights will be applied to
            weights: trained weights
            meta: trained meta

        Returns: the updated model object

        """
        pass

    def learnable_to_shareable(self, learnable: Learnable, fl_ctx: FLContext) -> Shareable:
        self.fl_ctx = fl_ctx
        trainable_weights, trainable_meta = self.model_to_trainable(unwrap_dict(learnable))
        print(f"trainable weights: {trainable_weights}")
        dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data=wrap_with_dict(trainable_weights),
            meta=trainable_meta,
        )
        print(f"learnable_to_shareable: {dxo.data}")
        return dxo.to_shareable()

    def shareable_to_learnable(self, shareable: Shareable, fl_ctx: FLContext) -> Learnable:
        self.fl_ctx = fl_ctx
        base_model_learnable = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
        if not base_model_learnable:
            self.system_panic(reason="No global base model!", fl_ctx=fl_ctx)
            return base_model_learnable

        dxo = from_shareable(shareable)
        trained_weights = unwrap_dict(dxo.data)
        trained_meta = dxo.meta
        model_obj = self.apply_weights_to_model(
            model_obj=unwrap_dict(base_model_learnable), weights=trained_weights, meta=trained_meta
        )
        return Learnable(wrap_with_dict(model_obj))
