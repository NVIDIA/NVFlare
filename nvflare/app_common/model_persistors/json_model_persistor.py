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

import json
import os

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable, make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor


class JSONModelPersistor(ModelPersistor):
    def __init__(self, save_name="xgboost_model.json"):
        super().__init__()
        self.save_name = save_name

    def _initialize(self, fl_ctx: FLContext):
        # get save path from FLContext
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        self.log_dir = app_root
        self.save_path = os.path.join(self.log_dir, self.save_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        fl_ctx.sync_sticky()

    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        """
            initialize and load the Model.

        Args:
            fl_ctx: FLContext

        Returns:
            Model object
        """

        if os.path.exists(self.save_path):
            self.logger.info("Loading server model")
            with open(self.save_path, "rb") as json_file:
                model_learnable = json.load(json_file)
        else:
            self.logger.info("Initializing server model as None")
            var_dict = None
            model_learnable = make_model_learnable(var_dict, dict())
        return model_learnable

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.START_RUN:
            self._initialize(fl_ctx)

    def save_model(self, model_learnable: ModelLearnable, fl_ctx: FLContext):
        """
            persist the Model object

        Args:
            model: Model object
            fl_ctx: FLContext
        """
        self.logger.info("Saving received model")
        if model_learnable:
            with open(self.save_path, "w") as f:
                json.dump(model_learnable, f)
