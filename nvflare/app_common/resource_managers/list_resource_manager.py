# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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


from collections import deque
from typing import Dict, List

from nvflare.app_common.resource_managers.auto_clean_resource_manager import AutoCleanResourceManager


class ListResourceManager(AutoCleanResourceManager):
    """Manage a list of resource units.

    For example:

        - require 2, current resources is [0, 1, 2, 3, 4, 5] => return [0,1]
          after allocation the current resources become [2, 3, 4, 5]
        - require 3, current resources [2, 3, 4, 5] => return [2, 3, 4]

    """

    def __init__(self, resources: Dict[str, List], expiration_period: int = 30):
        """Constructor

        Args:
            resources (dict): Specify the list of resources unit
            expiration_period (int): Number of seconds to hold the resources reserved.
                If check_resources is called but after "expiration_period" no allocate resource is called,
                then the reserved resources will be released.
        """
        if not isinstance(resources, dict):
            raise TypeError(f"resources should be of type dict, but got {type(resources)}.")

        resource_queue = {}
        for k in resources:
            if not isinstance(resources[k], list):
                raise TypeError(f"item in resources should be of type list, but got {type(resources[k])}.")
            resource_queue[k] = deque(resources[k])
        super().__init__(resources=resource_queue, expiration_period=expiration_period)

    def _deallocate(self, resources: dict):
        for k, v in resources.items():
            for i in v:
                self.resources[k].appendleft(i)

    def _check_required_resource_available(self, resource_requirement: dict) -> bool:
        is_resource_enough = True
        for k in resource_requirement:
            if k in self.resources:
                if len(self.resources[k]) < resource_requirement[k]:
                    is_resource_enough = False
                    break
            else:
                is_resource_enough = False
                break
        return is_resource_enough

    def _reserve_resource(self, resource_requirement: dict) -> dict:
        reserved_resources = {}
        for k in resource_requirement:
            reserved_resource_units = []
            for i in range(resource_requirement[k]):
                reserved_resource_units.append(self.resources[k].popleft())
            reserved_resources[k] = reserved_resource_units
        return reserved_resources

    def _resource_to_dict(self) -> dict:
        return {
            "resources": {k: list(self.resources[k]) for k in self.resources},
            "reserved_resources": self.reserved_resources,
        }
