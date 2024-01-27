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
from typing import Any, List

from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class ParamsConverter(ABC):
    def __init__(self, supported_tasks: List[str] = None):
        self.supported_tasks = supported_tasks

    def process(self, task_name: str, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        if not self.supported_tasks or task_name in self.supported_tasks:
            dxo = from_shareable(shareable)
            dxo.data = self.convert(dxo.data, fl_ctx)
            dxo.update_shareable(shareable)
        return shareable

    @abstractmethod
    def convert(self, params: Any, fl_ctx: FLContext) -> Any:
        pass
