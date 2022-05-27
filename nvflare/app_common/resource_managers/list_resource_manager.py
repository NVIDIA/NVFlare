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

import time
import uuid
from collections import deque
from threading import Event, Lock, Thread
from typing import Dict, List, Optional

from nvflare.apis.event_type import EventType
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

    def __init__(self, resources: Dict[str, List], expiration_period: int = 30):
        """Constructor

        Args:
            resources (dict): Specify the list of resources unit
            expiration_period (int): Number of seconds to hold the resources reserved.
                If check_resources is called but after "expiration_period" no allocate resource is called,
                then the reserved resources will be released.
        """
        super().__init__()
        if not isinstance(resources, dict):
            raise TypeError(f"resources should be of type dict, but got {type(resources)}.")
        if not isinstance(expiration_period, int):
            raise TypeError(f"expiration_period should be of type int, but got {type(expiration_period)}.")
        if expiration_period < 0:
            raise ValueError("expiration_period should be greater than 0.")

        self.resources = {}
        for k in resources:
            if not isinstance(resources[k], list):
                raise TypeError(f"item in resources should be of type list, but got {type(resources[k])}.")
            self.resources[k] = deque(resources[k])

        self.expiration_period = expiration_period
        self.reserved_resources = {}
        self.lock = Lock()
        self.stop_event = Event()
        self.cleanup_thread = Thread(target=self._check_expired)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.SYSTEM_START:
            self.cleanup_thread.start()
        elif event_type == EventType.SYSTEM_END:
            self.stop_event.set()
            if self.cleanup_thread:
                self.cleanup_thread.join()
                self.cleanup_thread = None

    def _check_expired(self):
        while not self.stop_event.is_set():
            time.sleep(1)
            with self.lock:
                tokens_to_remove = []
                for k in self.reserved_resources:
                    r, t = self.reserved_resources[k]
                    t -= 1
                    if t == 0:
                        tokens_to_remove.append(k)
                    else:
                        self.reserved_resources[k] = r, t
                for token in tokens_to_remove:
                    reserved_resources, _ = self.reserved_resources.pop(token)
                    for k in reserved_resources:
                        for i in reserved_resources[k]:
                            self.resources[k].append(i)
                self.logger.debug(f"current resources: {self.resources}, reserved_resources {self.reserved_resources}.")

    def _check_all_required_resource_available(self, resource_requirement: dict, fl_ctx: FLContext) -> bool:
        check_result = True
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
        return check_result

    def check_resources(self, resource_requirement: dict, fl_ctx: FLContext) -> (bool, Optional[str]):
        if not isinstance(resource_requirement, dict):
            raise TypeError(f"resource_requirement should be of type dict, but got {type(resource_requirement)}.")

        check_result = self._check_all_required_resource_available(resource_requirement, fl_ctx)
        token = None

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
                self.reserved_resources[token] = (reserved_resources, self.expiration_period)
                self.log_debug(
                    fl_ctx, f"reserving resources: {reserved_resources} for requirements {resource_requirement}."
                )
                self.log_debug(
                    fl_ctx, f"current resources: {self.resources}, reserved_resources {self.reserved_resources}."
                )
        return check_result, token

    def cancel_resources(self, resource_requirement: dict, token: str, fl_ctx: FLContext):
        with self.lock:
            if token and token in self.reserved_resources:
                reserved_resources, _ = self.reserved_resources.pop(token)
                for k in reserved_resources:
                    for i in reserved_resources[k]:
                        self.resources[k].appendleft(i)
                self.log_debug(fl_ctx, f"cancelling resources: {reserved_resources}.")
                self.log_debug(
                    fl_ctx, f"current resources: {self.resources}, reserved_resources {self.reserved_resources}."
                )
            else:
                self.log_debug(fl_ctx, f"Token {token} is not related to any reserved resources.")
        return None

    def allocate_resources(self, resource_requirement: dict, token: str, fl_ctx: FLContext) -> dict:
        result = {}
        with self.lock:
            if token and token in self.reserved_resources:
                result, _ = self.reserved_resources.pop(token)
                self.log_debug(fl_ctx, f"allocating resources: {result} for requirements: {resource_requirement}.")
                self.log_debug(
                    fl_ctx, f"current resources: {self.resources}, reserved_resources {self.reserved_resources}."
                )
            else:
                raise RuntimeError(f"allocate_resources: No reserved resources for token {token}.")
        return result

    def free_resources(self, resources: dict, token: str, fl_ctx: FLContext):
        with self.lock:
            self.log_debug(fl_ctx, f"freeing resources: {resources}.")
            self.log_debug(
                fl_ctx, f"current resources: {self.resources}, reserved_resources {self.reserved_resources}."
            )
            for k in resources:
                for i in resources[k]:
                    self.resources[k].append(i)
