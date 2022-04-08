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

from abc import ABC, abstractmethod
from typing import Optional

from .fl_context import FLContext


class ResourceConsumerSpec(ABC):
    @abstractmethod
    def consume(self, resources: dict):
        pass


class ResourceManagerSpec(ABC):
    @abstractmethod
    def check_resources(self, resource_requirement: dict, fl_ctx: FLContext) -> (bool, Optional[str]):
        """Checks whether the specified resource requirement can be satisfied.

        This is to respond to the Scheduler's request to check resource availabilities

        Args:
            resource_requirement: a dict that specifies resource requirement
            fl_ctx: the FLContext

        Returns:
            A tuple of (check_result, token).

            check_result is a bool indicates whether there is enough resources;
            token (optional) is for resource reservation / cancellation for this check request.
        """
        pass

    @abstractmethod
    def cancel_resources(self, resource_requirement: dict, token: str, fl_ctx: FLContext):
        """Cancels resources reserved if any.

        This happens when the Scheduler check and Dispatcher decided to not dispatch the job to the client.

        Args:
            resource_requirement: a dict that specifies resource requirement
            token: a resource reservation token returned by check_resources
            fl_ctx: the FLContext

        Note:
            If check_resource didn't return a token, then don't need to call this method
        """
        pass

    @abstractmethod
    def allocate_resources(self, resource_requirement: dict, token: str, fl_ctx: FLContext) -> dict:
        """Allocate resources.

        This is called by the Dispatcher on the client after the Server deploys a job to the client.
        The Deployer calls this method to allocate/obtain resources for the requirement (committed when check_resource).

        Note:
            resource requirements and resources may be different things.

        Args:
            resource_requirement: a dict that specifies resource requirement
            token: a resource reservation token returned by check_resources
            fl_ctx: the FLContext

        Returns:
            A dict of allocated resources
        """
        pass

    @abstractmethod
    def free_resources(self, resources: dict, token: str, fl_ctx: FLContext):
        """Free the resources.

        Args:
            resources: resources to be freed
            token: a resource reservation token returned by check_resources
            fl_ctx: the FLContext
        """
        pass
