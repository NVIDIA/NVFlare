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

    @abstractmethod
    def get_root_dir(self) -> str:
        pass

    @abstractmethod
    def get_work_dir(self) -> str:
        pass

    def get_subdir(self, name: str, create: bool = True) -> str:
        p = f"{self.get_work_dir()}/{name}"
        if not os.path.exists(p) and create:
            os.makedirs(p, exist_ok=True)
        return p

    @abstractmethod
    def get_experiment_dir(self) -> str:
        pass
