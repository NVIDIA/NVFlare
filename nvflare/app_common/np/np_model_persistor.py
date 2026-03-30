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

from .constants import NPConstants
from .utils import load_numpy_model


def _get_run_dir(fl_ctx: FLContext):
    workspace = fl_ctx.get_workspace()
    job_id = fl_ctx.get_job_id()
    if job_id is None:
        raise RuntimeError("job_id is missing in fl_ctx.")
    return workspace.get_run_dir(job_id)


class NPModelPersistor(ModelPersistor):
    def __init__(
        self,
        model_dir="models",
        model_name="server.npy",
        model: Optional[list] = None,
        source_ckpt_file_full_name: Optional[str] = None,
    ):
        """Model persistor for numpy arrays.

        Note:
            This persistor loads model data in the following priority:
            1. source_ckpt_file_full_name (if provided and exists)
            2. Previously saved numpy array from ``<run_dir>/<model_dir>/<model_name>``
            3. model (if provided)
            4. Default array ``[[1, 2, 3], [4, 5, 6], [7, 8, 9]]``

        Args:
            model_dir (str, optional): model directory. Defaults to "models".
            model_name (str, optional): model name. Defaults to "server.npy".
            model (list, optional): fallback initial model as a (JSON-serializable) list.
                This is only used when a previously saved model cannot be loaded.
                It will be converted to numpy array when ``load_model`` is called.
                Defaults to None.
            source_ckpt_file_full_name (str, optional): Full path to source checkpoint file.
                This path may not exist locally (server-side path). If provided and exists
                at runtime, it takes priority over other loading methods.
        """
        super().__init__()

        self.model_dir = model_dir
        self.model_name = model_name
        # Keep as list for JSON serialization during job config generation.
        # Conversion to numpy happens in load_model().
        self.model = model
        self.source_ckpt_file_full_name = source_ckpt_file_full_name
        # Note: We don't validate existence here because the checkpoint path may be
        # a server-side path that doesn't exist on the job submission machine.

    def _get_initial_model_as_numpy(self) -> np.ndarray:
        """Return the fallback initial model as a numpy array.

        This is used by ``load_model`` when the model file cannot be loaded.
        """
        if self.model is None:
            return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        else:
            return np.array(self.model, dtype=np.float32)

    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        run_dir = _get_run_dir(fl_ctx)
        model_path = os.path.join(run_dir, self.model_dir, self.model_name)

        data = load_numpy_model(
            fl_ctx=fl_ctx,
            logger=self,
            source_ckpt_file_full_name=self.source_ckpt_file_full_name,
            model_file_path=model_path,
            get_fallback_data=lambda: self._get_initial_model_as_numpy().copy(),
        )

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
