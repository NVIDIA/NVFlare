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

import os

import tensorflow as tf

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_opt.tf.utils import get_flat_weights, load_flat_weights


class TFModelPersistor(ModelPersistor):
    def __init__(self, model: tf.keras.Model, save_name="tf_model.ckpt"):
        super().__init__()
        self.save_name = save_name
        self.model = model

    def _initialize(self, fl_ctx: FLContext):
        workspace = fl_ctx.get_engine().get_workspace()
        app_root = workspace.get_app_dir(fl_ctx.get_job_id())
        self._model_save_path = os.path.join(app_root, self.save_name)

    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        """Initializes and loads the Model.

        Args:
            fl_ctx: FLContext

        Returns:
            ModelLearnable object
        """

        if os.path.exists(self._model_save_path):
            self.logger.info("Loading server model and weights")
            self.model.load_weights(self._model_save_path)

        # get flat model parameters
        result = get_flat_weights(self.model)

        model_learnable = make_model_learnable(result, dict())
        return model_learnable

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.START_RUN:
            self._initialize(fl_ctx)

    def save_model(self, model_learnable: ModelLearnable, fl_ctx: FLContext):
        """Saves model.

        Args:
            model_learnable: ModelLearnable object
            fl_ctx: FLContext
        """
        load_flat_weights(self.model, model_learnable[ModelLearnableKey.WEIGHTS])
        self.model.save_weights(self._model_save_path)
