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
from collections import deque
from threading import Lock
from typing import Dict, List, Optional

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.resource_manager_spec import ResourceManagerSpec


class ListResourceManager(ResourceManagerSpec, FLComponent):
    """Manage a list of resource units.

    For example:

        - require 2, current resources is [0, 1, 2, 3, 4, 5] => return [0,1]
          after allocation the current resources become [2, 3, 4, 5]
        - require 3, current resources [2, 3, 4, 5] => return [2, 3, 4]

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
            self.resources[k] = deque(list(dict.fromkeys(resources[k])))
        self.reserved_resources = {}
        self.lock = Lock()

    def check_resources(self, resource_requirement: dict, fl_ctx: FLContext) -> (bool, Optional[str]):
        if not isinstance(resource_requirement, dict):
            raise TypeError(f"resource_requirement should be of type dict, but got {type(resource_requirement)}.")

        check_result = True
        token = None
        with self.lock:
            for k in resource_requirement:
                if k in self.resources:
                    if len(self.resources[k]) < resource_requirement[k]:
                        check_result = False
                        self.log_debug(fl_ctx, f"Resource {k} is not enough.")
                        break
                else:
                    check_result = False
                    self.log_debug(fl_ctx, f"Missing {k} in resources.")
                    break

        # reserve resource only when check is True
        if check_result:
            token = str(uuid.uuid4())
            reserved_resources = {}
            with self.lock:
                for k in resource_requirement:
                    reserved_resource_units = []
                    for i in range(resource_requirement[k]):
                        reserved_resource_units.append(self.resources[k].popleft())
                    reserved_resources[k] = reserved_resource_units
                self.reserved_resources[token] = reserved_resources
        return check_result, token

    def cancel_resources(self, resource_requirement: dict, token: str, fl_ctx: FLContext):
        with self.lock:
            if token and token in self.reserved_resources:
                reserved_resources = self.reserved_resources.pop(token)
                for k in reserved_resources:
                    for i in reserved_resources[k]:
                        self.resources[k].append(i)
            else:
                self.log_debug(fl_ctx, f"Token {token} is not related to any reserved resources.")
        return None

    def allocate_resources(self, resource_requirement: dict, token: str, fl_ctx: FLContext) -> dict:
        result = {}
        with self.lock:
            if token and token in self.reserved_resources:
                result = self.reserved_resources[token]
            else:
                raise RuntimeError(f"allocate_resources: No reserved resources for token {token}.")
        return result

    def free_resources(self, resources: dict, token: str, fl_ctx: FLContext):
        with self.lock:
            if token and token in self.reserved_resources:
                reserved_resources = self.reserved_resources.pop(token)
                for k in reserved_resources:
                    for i in reserved_resources[k]:
                        self.resources[k].append(i)
            else:
                raise RuntimeError(f"free_resources: No reserved resources for token {token}.")
