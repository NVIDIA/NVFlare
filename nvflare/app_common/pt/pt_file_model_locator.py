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

from typing import List

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model_locator import ModelLocator
from nvflare.app_common.pt.pt_file_model_persistor import PTFileModelPersistor


class PTFileModelLocator(ModelLocator):
    def __init__(self, pt_persistor_id: str):
        """The ModelLocator's job is to find and locate the models inventory saved during training.

        Args:
            pt_persistor_id (str): ModelPersistor component ID
        """
        super().__init__()

        self.pt_persistor_id = pt_persistor_id

        self.model_inventory = {}

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)

    def initialize(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        self.model_persistor: PTFileModelPersistor = engine.get_component(self.pt_persistor_id)

    def get_model_names(self, fl_ctx: FLContext) -> List[str]:
        """Returns the list of model names that should be included from server in cross site validation.add()

        Args:
            fl_ctx (FLContext): FL Context object.

        Returns:
            List[str]: List of model names.
        """
        self.model_inventory: dict = self.model_persistor.get_model_inventory(fl_ctx)
        return list(self.model_inventory.keys())

    def locate_model(self, model_name, fl_ctx: FLContext) -> DXO:
        """Call to locate and load the model weights of "model_name"

        Args:
            model_name: name of the model
            fl_ctx: FLContext

        Returns: model_weight DXO

        """
        if model_name not in list(self.model_inventory.keys()):
            raise ValueError(f"model inventory does not contain: {model_name}")

        weights = self.model_persistor.get_model(model_name, fl_ctx)
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=weights, meta={})

        return dxo
