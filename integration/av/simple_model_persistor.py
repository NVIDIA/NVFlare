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

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor

from .utils import unwrap_dict, wrap_with_dict


class SimpleModelPersistor(LearnablePersistor, ABC):
    def __init__(self):
        LearnablePersistor.__init__(self)
        self.fl_ctx = None

    @abstractmethod
    def load_model(self) -> Any:
        """Load model object.

        Returns: a model object
        """
        pass

    @abstractmethod
    def save_model(self, model_obj: Any):
        """Save the model object

        Args:
            model_obj: the model object to be saved

        Returns: None

        """
        pass

    def load(self, fl_ctx: FLContext) -> Learnable:
        self.fl_ctx = fl_ctx
        model = self.load_model()
        return Learnable(wrap_with_dict(model))

    def save(self, learnable: Learnable, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx
        self.save_model(unwrap_dict(learnable))
