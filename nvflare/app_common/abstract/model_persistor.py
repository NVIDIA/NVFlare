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

from abc import ABC, abstractmethod

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.model_desc import ModelDescriptor

from .learnable_persistor import LearnablePersistor
from .model import ModelLearnable


class ModelPersistor(LearnablePersistor, ABC):
    def load(self, fl_ctx: FLContext):
        return self.load_model(fl_ctx)

    def save(self, learnable: ModelLearnable, fl_ctx: FLContext):
        self.save_model(learnable, fl_ctx)

    @abstractmethod
    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        """Initialize and load the model.

        Args:
            fl_ctx: FLContext

        Returns:
            Model object

        """
        pass

    @abstractmethod
    def save_model(self, model: ModelLearnable, fl_ctx: FLContext):
        """Persist the model object.

        Args:
            model: Model object to be saved
            fl_ctx: FLContext

        """
        pass

    def get_model_inventory(self, fl_ctx: FLContext) -> {str: ModelDescriptor}:
        """Get the model inventory of the ModelPersister.

        Args:
            fl_ctx: FLContext

        Returns: { model_kind: ModelDescriptor }

        """
        pass

    def get_model(self, model_file, fl_ctx: FLContext) -> object:
        pass
