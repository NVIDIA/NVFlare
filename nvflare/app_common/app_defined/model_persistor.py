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

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor

from .component_base import ComponentBase


class AppDefinedModelPersistor(ModelPersistor, ComponentBase, ABC):
    def __init__(self):
        ModelPersistor.__init__(self)
        ComponentBase.__init__(self)

    @abstractmethod
    def read_model(self) -> Any:
        """Load model object.

        Returns: a model object
        """
        pass

    @abstractmethod
    def write_model(self, model_obj: Any):
        """Save the model object

        Args:
            model_obj: the model object to be saved

        Returns: None

        """
        pass

    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        self.fl_ctx = fl_ctx
        model = self.read_model()
        return make_model_learnable(weights=model, meta_props={})

    def save_model(self, learnable: ModelLearnable, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx
        self.write_model(learnable.get(ModelLearnableKey.WEIGHTS))
