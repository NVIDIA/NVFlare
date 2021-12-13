# Copyright (c) 2021, NVIDIA CORPORATION.
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
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.model import ModelLearnable


class ModelPersistor(LearnablePersistor, ABC):
    def __init__(self):
        """Abstract class for ModelPersistor to save or load models."""
        super().__init__()

    def load(self, fl_ctx: FLContext):
        return self.load_model(fl_ctx)

    def save(self, learnable: ModelLearnable, fl_ctx: FLContext):
        self.save_model(learnable, fl_ctx)

    @abstractmethod
    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        """Initialize and load the model.

        Args:
            fl_ctx (FLContext): FLContext used to pass data.

        Returns:
            ModelLearnable object.

        """
        pass

    @abstractmethod
    def save_model(self, model: ModelLearnable, fl_ctx: FLContext):
        """Persist the model object.

        Args:
            model (ModelLearnable): Model object to be saved.
            fl_ctx (FLContext): fl context used to pass around data

        Returns:
            None

        """
        pass
