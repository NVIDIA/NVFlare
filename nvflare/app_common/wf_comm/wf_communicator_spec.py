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

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from nvflare.apis.controller_spec import SendOrder
from nvflare.fuel.utils.class_utils import instantiate_class
from nvflare.fuel.utils.component_builder import ComponentBuilder
from nvflare.fuel.utils.fobs import fobs
from nvflare.fuel.utils.import_utils import optional_import


class WFCommunicatorSpec(ABC):
    def __init__(self):
        self.controller_config: Optional[Dict] = None

    @abstractmethod
    def broadcast_to_peers_and_wait(self, pay_load: Dict):
        pass

    @abstractmethod
    def broadcast_to_peers(self, pay_load: Dict):
        pass

    @abstractmethod
    def send_to_peers(self, pay_load: Dict, send_order: SendOrder = SendOrder.SEQUENTIAL):
        pass

    @abstractmethod
    def send_to_peers_and_wait(self, pay_load: Dict, send_order: SendOrder = SendOrder.SEQUENTIAL):
        pass

    @abstractmethod
    def relay_to_peers_and_wait(self, pay_load: Dict, send_order: SendOrder = SendOrder.SEQUENTIAL):
        pass

    @abstractmethod
    def relay_to_peers(self, pay_load: Dict, send_order: SendOrder = SendOrder.SEQUENTIAL):
        pass

    def set_controller_config(self, controller_config: Dict):
        if controller_config is None:
            raise ValueError("controller_config is None")

        if not isinstance(controller_config, dict):
            raise ValueError(f"controller_config should be Dict, found '{type(controller_config)}'")

        self.controller_config = controller_config

    def get_controller(self):
        controller = None
        if isinstance(self.controller_config, dict):
            controller = ComponentBuilder().build_component(self.controller_config)
            if controller is None:
                raise ValueError("wf_controller should provided, but get None")

        return controller

    def register_serializers(self, serializer_class_paths: List[str] = None):
        self.register_default_serializers()
        if serializer_class_paths:
            for class_path in serializer_class_paths:
                fobs.register(instantiate_class(class_path, {}))

    def register_default_serializers(self):
        torch, flag = optional_import("torch")
        if flag:
            from nvflare.app_opt.pt.decomposers import TensorDecomposer

            fobs.register(TensorDecomposer)
