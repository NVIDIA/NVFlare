# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from abc import ABC, abstractmethod


class Workspace(ABC):

    def __init__(self):
        self.resource_dirs = {}

    def add_resource_dir(self, name, resource_dir):
        if not os.path.isdir(resource_dir):
            raise ValueError(f"Resource dir {resource_dir} does not exist")
        self.resource_dirs[name] = resource_dir

    @abstractmethod
    def get_root_dir(self) -> str:
        pass

    @abstractmethod
    def get_work_dir(self) -> str:
        pass

    def get_resource_dir(self, name: str, create: bool = True) -> str:
        resource_dir = self.resource_dirs.get(name)
        if resource_dir:
            return resource_dir

        p = os.path.join(self.get_work_dir(), name)
        if not os.path.exists(p) and create:
            os.makedirs(p, exist_ok=True)
        return p

    @abstractmethod
    def get_experiment_dir(self) -> str:
        pass
