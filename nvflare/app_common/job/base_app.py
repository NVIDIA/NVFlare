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

from typing import Dict

from nvflare.apis.filter import Filter
from nvflare.apis.fl_component import FLComponent


class BaseApp(object):
    def __init__(self) -> None:
        super().__init__()

        self.task_data_filters: [dict[str, Filter]] = []
        self.task_result_filters: [dict[str, Filter]] = []
        self.handlers: [FLComponent] = []
        self.components: Dict[str, object] = {}

    def add_component(self, cid: str, component):
        if cid in self.components.keys():
            raise RuntimeError(f"Component with ID:{cid} already exist.")

        self.components[cid] = component

        if isinstance(component, FLComponent):
            self.handlers.append(component)

    def add_task_data_filter(self, tasks: [str], filter: Filter):
        if not isinstance(filter, Filter):
            raise RuntimeError(f"filter must be Filter, but got {filter.__class__}")

        self.task_data_filters[tasks] = filter

    def add_task_result_filter(self, tasks: [str], filter: Filter):
        if not isinstance(filter, Filter):
            raise RuntimeError(f"filter must be Filter, but got {filter.__class__}")

        self.task_result_filters[tasks] = filter

