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

import json
import os

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.app_constant import AppConstants


class XGBModelPersistor(ModelPersistor):
    def __init__(self, save_name="xgboost_model.json", load_as_dict=True):
        super().__init__()
        self.save_name = save_name
        self.load_as_dict = load_as_dict

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

        if os.path.exists(self.save_path):
            self.logger.info("Loading server model")
            with open(self.save_path, "r") as json_file:
                model = json.load(json_file)
                if not self.load_as_dict:
                    model = bytearray(json.dumps(model), "utf-8")
        else:
            self.logger.info("Initializing server model as None")
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
            if fl_ctx.get_prop(AppConstants.CURRENT_ROUND) == fl_ctx.get_prop(AppConstants.NUM_ROUNDS) - 1:
                self.logger.info(f"Saving received model to {os.path.abspath(self.save_path)}")
                # save 'weights' which is actual model, loadable by xgboost library
                model = model_learnable[ModelLearnableKey.WEIGHTS]
                with open(self.save_path, "w") as f:
                    if isinstance(model, dict):
                        json.dump(model, f)
                    elif isinstance(model, bytes) or isinstance(model, bytearray) or isinstance(model, str):
                        # should already be json, but double check by loading and dumping at some extra cost
                        json.dump(json.loads(model), f)
                    else:
                        self.logger.error("unknown model format")
                        self.system_panic(reason="No global base model!", fl_ctx=fl_ctx)
