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
from nvflare.app_common.abstract.model_persistor import ModelPersistor


class PTModelLocator(ModelLocator):
    def __init__(self, pt_persistor_id: str):
        """The ModelLocator's job is to find the models to be included for cross site evaluation
        located on server. This PTModelLocator finds and extracts server global model  and best global model
        that is saved during training.

        Args:
            persistor_id (str): ModelPersistor component ID
        """
        super().__init__()

        self.pt_persistor_id = pt_persistor_id

        self.model_inventory = {}

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)

    def initialize(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        self.model_persistor: ModelPersistor = engine.get_component(self.pt_persistor_id)

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
        weights = self.model_inventory.get(model_name, {}).data
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=weights, meta={})

        return dxo
