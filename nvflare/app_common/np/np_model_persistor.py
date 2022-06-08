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

from .constants import NPConstants


def _get_run_dir(fl_ctx: FLContext):
    engine = fl_ctx.get_engine()
    if engine is None:
        raise RuntimeError("engine is missing in fl_ctx.")
    job_id = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
    if job_id is None:
        raise RuntimeError("job_id is missing in fl_ctx.")
    run_dir = engine.get_workspace().get_run_dir(job_id)
    return run_dir


class NPModelPersistor(ModelPersistor):
    def __init__(self, model_dir="models", model_name="server.npy"):
        super().__init__()

        self.model_dir = model_dir
        self.model_name = model_name

        # This is default model that will be used if not local model is provided.
        self.default_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        run_dir = _get_run_dir(fl_ctx)
        model_path = os.path.join(run_dir, self.model_dir, self.model_name)
        try:
            # try loading previous model
            data = np.load(model_path)
        except Exception as e:
            self.log_exception(
                fl_ctx,
                f"Unable to load model from {model_path}: {e}. Using default data instead.",
                fire_event=False,
            )
            data = self.default_data.copy()

        model_learnable = make_model_learnable(weights={NPConstants.NUMPY_KEY: data}, meta_props={})

        self.log_info(fl_ctx, f"Loaded initial model: {model_learnable[ModelLearnableKey.WEIGHTS]}")
        return model_learnable

    def save_model(self, model_learnable: ModelLearnable, fl_ctx: FLContext):
        run_dir = _get_run_dir(fl_ctx)
        model_root_dir = os.path.join(run_dir, self.model_dir)
        if not os.path.exists(model_root_dir):
            os.makedirs(model_root_dir)

        model_path = os.path.join(model_root_dir, self.model_name)
        np.save(model_path, model_learnable[ModelLearnableKey.WEIGHTS][NPConstants.NUMPY_KEY])
        self.log_info(fl_ctx, f"Saved numpy model to: {model_path}")
