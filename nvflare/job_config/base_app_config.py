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
import os.path
from abc import ABC
from typing import Dict, List

from nvflare.apis.filter import Filter
from nvflare.apis.fl_component import FLComponent


class BaseAppConfig(ABC):
    """BaseAppConfig holds the base essential component data for the ServerApp and ClientApp, including the
    task_data_filters, task_result_filters, system components and used external scripts.

    """

    def __init__(self) -> None:
        super().__init__()

        self.task_data_filters: [(List[str], Filter)] = []
        self.task_result_filters: [(List[str], Filter)] = []
        self.components: Dict[str, object] = {}
        self.ext_scripts = []
        self.ext_dirs = []

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

    def add_ext_script(self, ext_script: str):
        if not isinstance(ext_script, str):
            raise RuntimeError(f"ext_script must be type of str, but got {ext_script.__class__}")

        if not (os.path.isabs(ext_script) or os.path.exists(ext_script)):
            raise RuntimeError(f"Could not locate external script: {ext_script}")

        self.ext_scripts.append(ext_script)

    def add_ext_dir(self, ext_dir: str):
        if not (os.path.isdir(ext_dir) and os.path.exists(ext_dir)):
            raise RuntimeError(f"external resource dir: {ext_dir} does not exist")

        self.ext_dirs.append(ext_dir)

    def _add_task_filter(self, tasks, filter, filters):
        if not isinstance(filter, Filter):
            raise RuntimeError(f"filter must be type of Filter, but got {filter.__class__}")
        for task in tasks:
            for fd in filters:
                if task in fd.tasks:
                    raise RuntimeError(f"Task {task} already defined in the task filters.")
        filters.append((tasks, filter))
