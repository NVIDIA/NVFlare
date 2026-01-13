# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Optional

import numpy as np

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.security.logging import secure_format_exception

from .constants import NPConstants


def _get_run_dir(fl_ctx: FLContext):
    workspace = fl_ctx.get_workspace()
    job_id = fl_ctx.get_job_id()
    if job_id is None:
        raise RuntimeError("job_id is missing in fl_ctx.")
    return workspace.get_run_dir(job_id)


class NPModelPersistor(ModelPersistor):
    def __init__(self, model_dir="models", model_name="server.npy", initial_model: Optional[list] = None):
        """Model persistor for numpy arrays.

        Note:
            This persistor first tries to load a previously saved numpy array from
            ``<run_dir>/<model_dir>/<model_name>``.

            If the file cannot be loaded (e.g. does not exist on the first run),
            it falls back to an "initial model" provided via ``initial_model``.
            If ``initial_model`` is not provided, a small built-in default array
            (``[[1, 2, 3], [4, 5, 6], [7, 8, 9]]``) is used.

        Args:
            model_dir (str, optional): model directory. Defaults to "models".
            model_name (str, optional): model name. Defaults to "server.npy".
            initial_model (list, optional): fallback initial model as a (JSON-serializable) list.
                This is only used when a previously saved model cannot be loaded.
                It will be converted to numpy array when ``load_model`` is called.
                Defaults to None.
        """
        super().__init__()

        self.model_dir = model_dir
        self.model_name = model_name
        # Keep as list for JSON serialization during job config generation.
        # Conversion to numpy happens in load_model().
        self.initial_model = initial_model

    def _get_initial_model_as_numpy(self) -> np.ndarray:
        """Return the fallback initial model as a numpy array.

        This is used by ``load_model`` when the model file cannot be loaded.
        """
        if self.initial_model is None:
            return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        else:
            return np.array(self.initial_model, dtype=np.float32)

    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        run_dir = _get_run_dir(fl_ctx)
        model_path = os.path.join(run_dir, self.model_dir, self.model_name)
        try:
            # try loading previous model
            data = np.load(model_path)
        except Exception as e:
            self.log_info(
                fl_ctx,
                f"Unable to load model from {model_path}: {secure_format_exception(e)}. Using default data instead.",
                fire_event=False,
            )
            data = self._get_initial_model_as_numpy().copy()

        model_learnable = make_model_learnable(weights={NPConstants.NUMPY_KEY: data}, meta_props={})

        self.log_info(fl_ctx, f"Loaded initial model: {model_learnable[ModelLearnableKey.WEIGHTS]}")
        return model_learnable

    def save_model(self, model_learnable: ModelLearnable, fl_ctx: FLContext):
        workspace = fl_ctx.get_workspace()
        job_id = fl_ctx.get_job_id()
        model_root_dir = os.path.join(workspace.get_result_root(job_id), self.model_dir)
        if not os.path.exists(model_root_dir):
            os.makedirs(model_root_dir)

        model_path = os.path.join(model_root_dir, self.model_name)
        np.save(model_path, model_learnable[ModelLearnableKey.WEIGHTS][NPConstants.NUMPY_KEY])
        self.log_info(fl_ctx, f"Saved numpy model to: {model_path}")
