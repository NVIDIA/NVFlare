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

import os

import numpy as np

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.app_constant import AppConstants

from .constants import NPConstants


class NPModelPersistor(ModelPersistor):
    def __init__(self, model_dir="models", model_name="server.npy"):
        super().__init__()

        self.model_dir = model_dir
        self.model_name = model_name

        # This is default model that will be used if not local model is provided.
        self.default_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        # Get start round from FLContext. If start_round > 0, we will try loading model from disk.
        start_round = fl_ctx.get_prop(AppConstants.START_ROUND, 0)
        engine = fl_ctx.get_engine()
        run_number = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
        run_dir = engine.get_workspace().get_run_dir(run_number)
        model_path = os.path.join(run_dir, self.model_dir, self.model_name)

        # Create a new numpy model
        if start_round > 0:
            try:
                data = np.load(model_path)
            except Exception as e:
                self.log_exception(
                    fl_ctx, f"Unable to load model from {model_path}. Using default data instead.", fire_event=False
                )
                data = self.default_data.copy()
        else:
            data = self.default_data.copy()

        # Generate model dictionary and create model_learnable.
        weights = {NPConstants.NUMPY_KEY: data}
        model_learnable = make_model_learnable(weights, {})

        self.logger.info(f"Loaded initial model: {model_learnable[ModelLearnableKey.WEIGHTS]}")
        return model_learnable

    def save_model(self, model: ModelLearnable, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        run_number = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
        run_dir = engine.get_workspace().get_run_dir(run_number)
        model_path = os.path.join(run_dir, self.model_dir)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model_save_path = os.path.join(model_path, self.model_name)
        if model_save_path:
            with open(model_save_path, "wb") as f:
                np.save(f, model[ModelLearnableKey.WEIGHTS][NPConstants.NUMPY_KEY])
            self.log_info(fl_ctx, f"Saved numpy model to: {model_save_path}")
