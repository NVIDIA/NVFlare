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

import uuid
from typing import Dict, List, Optional

from nvflare.apis.fl_context import FLContext
from nvflare.apis.resource_manager_spec import ResourceManagerSpec


class ListResourceManager(ResourceManagerSpec):
    """Manage a list of resource units.

    For example:

        - require 2, resource is [0, 1, 2, 3, 4, 5] -> check if things in list is available => return [0,1]
        - require 3 => return [2, 3, 4]
        - free 2, require 3 => return [0, 1, 5]

    """

    def __init__(self, resources: Dict[str, List]):
        """Constructor

        Args:
            resources (dict): Specify the list of resources unit
        """
        super().__init__()
        if not isinstance(resources, dict):
            raise TypeError(f"resources should be of type dict, but got {type(resources)}.")
        self.resources = {}
        for k in resources:
            if not isinstance(resources[k], list):
                raise TypeError(f"item in resources should be of type list, but got {type(resources[k])}.")
            self.resources[k] = set(resources[k])

    def check_resources(self, resource_requirement: dict, fl_ctx: FLContext) -> (bool, Optional[str]):
        if not isinstance(resource_requirement, dict):
            raise TypeError(f"resource_requirement should be of type dict, but got {type(resource_requirement)}.")
        check_result = True
        for k in resource_requirement:
            if k in self.resources:
                if len(self.resources[k]) < resource_requirement[k]:
                    check_result = False
                    break
        return check_result, str(uuid.uuid4())

    def cancel_resources(self, resource_requirement: dict, token: str, fl_ctx: FLContext):
        return None

    def allocate_resources(self, resource_requirement: dict, token: str, fl_ctx: FLContext) -> dict:
        result = {}
        for k in resource_requirement:
            if k in self.resources:
                result[k] = []
                if len(self.resources[k]) < resource_requirement[k]:
                    raise RuntimeError("Not enough resources.")
                for i in range(resource_requirement[k]):
                    result[k].append(self.resources[k].pop())
        return result

    def free_resources(self, resources: dict, token: str, fl_ctx: FLContext):
        for k in resources:
            if k not in self.resources:
                raise RuntimeError(f"Key {k} is not in resource manager's resources.")
            self.resources[k].update(resources[k])
