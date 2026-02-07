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
from typing import Any, Dict, Optional

from joblib import dump, load

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.app_constant import AppConstants

# Key in initial_params that means "load model from this path"
MODEL_PATH_KEY = "model_path"


def validate_model_path(path: Optional[str]) -> None:
    """Require model_path to be absolute if provided.

    All sklearn recipes use this so construction fails fast instead of at runtime
    when the persistor's load_model() runs. Call from recipe __init__ or validators.
    """
    if path is not None and not os.path.isabs(path):
        raise ValueError(
            f"model_path must be an absolute path, got: {path!r}. "
            "Use absolute paths like '/workspace/model.joblib' for server-side model files."
        )


class JoblibModelParamPersistor(ModelPersistor):
    def __init__(
        self,
        initial_params: Optional[Dict[str, Any]] = None,
        save_name: str = "model_param.joblib",
        model_path: Optional[str] = None,
    ):
        """Persist global model parameters from a dict to a joblib file.

        Note that this contains the necessary information to build
        a certain model but may not be directly loadable.

        Unlike PTFileModelPersistor, this persistor does NOT instantiate model classes.
        It only stores and transmits parameter values (e.g., hyperparameters, weights).
        The sklearn model class is instantiated on the client side using these params.

        Args:
            initial_params: Initial parameters dict (e.g., {"n_clusters": 3, "kernel": "rbf"}).
                Hyperparameters/config only; do not put model_path here—use the model_path
                argument instead. Used as fallback when model_path is None and no saved
                model exists in save_path.
            save_name: Filename for saving model params. Defaults to "model_param.joblib".
            model_path: Optional absolute path to a saved model file (.joblib, .pkl).
                If provided, the model is loaded from this path at runtime (file must exist).
                Defaults to None. For backward compatibility, initial_params may still
                contain key "model_path" and will be used if model_path is None.
        """
        super().__init__()
        self.initial_params = initial_params or {}
        self.save_name = save_name
        self.model_path = model_path

    def _initialize(self, fl_ctx: FLContext):
        # get save path from FLContext
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        self.log_dir = app_root
        self.save_path = os.path.join(self.log_dir, self.save_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        fl_ctx.sync_sticky()

    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        """Initialize and load the Model.

        Args:
            fl_ctx: FLContext

        Returns:
            ModelLearnable object
        """
        model = None
        initial_params = self.initial_params if isinstance(self.initial_params, dict) else {}

        # Priority 1: Load from explicit model_path argument, or from initial_params (backward compat)
        path = self.model_path or initial_params.get(MODEL_PATH_KEY)
        if path:
            if not os.path.isabs(str(path)):
                raise ValueError(f"model_path must be a non-empty absolute path, got: {path!r}")
            if not os.path.exists(path):
                raise ValueError(f"Model file not found: {path}. Check that the file exists at runtime.")
            self.logger.info(f"Loading model from {path}")
            model = load(path)

        # Priority 2: Load from previously saved model
        if model is None and os.path.exists(self.save_path):
            self.logger.info("Loading server model")
            model = load(self.save_path)

        # Priority 3: Use initial_params as config (exclude model_path so clients don't see it)
        if model is None:
            if initial_params:
                config = {k: v for k, v in initial_params.items() if k != MODEL_PATH_KEY}
                if config:
                    self.logger.info(f"Initialization, sending global settings: {config}")
                    model = config
            if model is None:
                raise ValueError(
                    "No model parameters available. Provide initial_params (hyperparameters) "
                    "and/or model_path (absolute path to saved .joblib/.pkl), "
                    "or ensure a previously saved model exists."
                )

        model_learnable = make_model_learnable(weights=model, meta_props=dict())

        return model_learnable

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.START_RUN:
            self._initialize(fl_ctx)

    def save_model(self, model_learnable: ModelLearnable, fl_ctx: FLContext):
        """Persists the Model object.

        Args:
            model_learnable: ModelLearnable object
            fl_ctx: FLContext
        """
        if model_learnable:
            current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
            num_rounds = fl_ctx.get_prop(AppConstants.NUM_ROUNDS)
            # Save model only on the last round, or always if NUM_ROUNDS is not set
            if num_rounds is None or current_round == num_rounds - 1:
                self.logger.info(f"Saving received model to {os.path.abspath(self.save_path)}")
                # save 'weights' which contains model parameters
                model = model_learnable[ModelLearnableKey.WEIGHTS]
                dump(model, self.save_path, compress=1)
