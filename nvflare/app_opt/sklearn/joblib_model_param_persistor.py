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


class JoblibModelParamPersistor(ModelPersistor):
    def __init__(
        self,
        initial_params: Optional[Dict[str, Any]] = None,
        save_name: str = "model_param.joblib",
        source_ckpt_file_full_name: Optional[str] = None,
    ):
        """Persist global model parameters from a dict to a joblib file.

        Note that this contains the necessary information to build
        a certain model but may not be directly loadable.

        Unlike PTFileModelPersistor, this persistor does NOT instantiate model classes.
        It only stores and transmits parameter values (e.g., hyperparameters, weights).
        The sklearn model class is instantiated on the client side using these params.

        Args:
            initial_params: Initial parameters dict (e.g., {"n_clusters": 3, "kernel": "rbf"}).
                These are parameter VALUES, not a class path configuration.
                Used as fallback if no checkpoint or previously saved model is available.
            save_name: Filename for saving model params. Defaults to "model_param.joblib".
            source_ckpt_file_full_name: Full path to source checkpoint file.
                This path may not exist locally (server-side path). If provided
                and exists at runtime, it takes priority over initial_params.
        """
        super().__init__()
        self.initial_params = initial_params
        self.save_name = save_name
        self.source_ckpt_file_full_name = source_ckpt_file_full_name
        # Note: We don't validate existence here because the checkpoint path may be
        # a server-side path that doesn't exist on the job submission machine.

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

        # Priority 1: Load from source checkpoint if provided
        if self.source_ckpt_file_full_name:
            # If user explicitly specified a checkpoint, it MUST exist (fail fast to catch config errors)
            if not os.path.exists(self.source_ckpt_file_full_name):
                raise ValueError(
                    f"Source checkpoint not found: {self.source_ckpt_file_full_name}. "
                    "Check that the checkpoint exists at runtime."
                )
            self.logger.info(f"Loading model from source checkpoint: {self.source_ckpt_file_full_name}")
            model = load(self.source_ckpt_file_full_name)

        # Priority 2: Load from previously saved model
        if model is None and os.path.exists(self.save_path):
            self.logger.info("Loading server model")
            model = load(self.save_path)

        # Priority 3: Use initial params
        if model is None:
            if self.initial_params is not None:
                self.logger.info(f"Initialization, sending global settings: {self.initial_params}")
                model = self.initial_params
            else:
                raise ValueError(
                    "No model parameters available. Provide either source_ckpt_file_full_name, "
                    "a previously saved model, or initial_params."
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
