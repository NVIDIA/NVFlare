# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from abc import ABC
from typing import Dict, List

from nvflare.apis.filter import Filter
from nvflare.apis.fl_component import FLComponent


class BaseAppConfig(ABC):
    def __init__(self) -> None:
        super().__init__()

        self.task_data_filters: [(List[str], Filter)] = []
        self.task_result_filters: [(List[str], Filter)] = []
        self.components: Dict[str, object] = {}

        self.handlers: [FLComponent] = []

    def add_component(self, cid: str, component):
        if cid in self.components.keys():
            raise RuntimeError(f"Component with ID:{cid} already exist.")

        self.components[cid] = component

        if isinstance(component, FLComponent):
            self.handlers.append(component)

    def add_task_data_filter(self, tasks: List[str], filter: Filter):
        self._add_task_filter(tasks, filter, self.task_data_filters)

    def add_task_result_filter(self, tasks: List[str], filter: Filter):
        self._add_task_filter(tasks, filter, self.task_result_filters)

    def _add_task_filter(self, tasks, filter, filters):
        if not isinstance(filter, Filter):
            raise RuntimeError(f"filter must be type of Filter, but got {filter.__class__}")
        for task in tasks:
            for fd in filters:
                if task in fd.tasks:
                    raise RuntimeError(f"Task {task} already defined in the task filters.")
        filters.append((tasks, filter))
