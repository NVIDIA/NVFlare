# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.resource_manager_spec import ResourceManagerSpec


class BEResourceManager(ResourceManagerSpec, FLComponent):
    """Best-effort resource manager that optimistically approves all resource requests.

    This implementation accepts every resource allocation request unconditionally,
    deferring actual resource availability checks to runtime. If the requested
    resources are not available when the job executes, the job will fail at that
    point rather than at scheduling time.

    Suitable for environments where resource pre-checking is not needed or where
    the underlying scheduler (e.g. Kubernetes) handles resource enforcement.
    """

    def __init__(self):
        """Initializes BEResourceManager."""
        super().__init__()

    def check_resources(self, resource_requirement: dict, fl_ctx: FLContext):
        """Checks resource requirements by always reporting sufficient resources.

        Args:
            resource_requirement: a dict specifying the requested resources.
            fl_ctx: the FLContext.

        Returns:
            A tuple of (True, token) where token is a newly generated UUID string.

        Raises:
            TypeError: if resource_requirement is not a dict.
        """
        if not isinstance(resource_requirement, dict):
            raise TypeError(f"resource_requirement should be of type dict, but got {type(resource_requirement)}.")

        token = str(uuid.uuid4())
        return True, token

    def cancel_resources(self, resource_requirement: dict, token: str, fl_ctx: FLContext):
        """Cancels a previously reserved resource allocation.

        This is a no-op since BEResourceManager does not actually reserve resources.

        Args:
            resource_requirement: a dict specifying the requested resources.
            token: the reservation token returned by check_resources.
            fl_ctx: the FLContext.
        """
        return None

    def allocate_resources(self, resource_requirement: dict, token: str, fl_ctx: FLContext) -> dict:
        """Allocates resources by returning an empty dict.

        Since this manager does not track resources, allocation is a no-op.

        Args:
            resource_requirement: a dict specifying the requested resources.
            token: the reservation token returned by check_resources.
            fl_ctx: the FLContext.

        Returns:
            An empty dict.
        """
        return {}

    def free_resources(self, resources: dict, token: str, fl_ctx: FLContext):
        """Frees previously allocated resources.

        This is a no-op since BEResourceManager does not track allocated resources.

        Args:
            resources: a dict of resources to free.
            token: the reservation token returned by check_resources.
            fl_ctx: the FLContext.
        """
        pass

    def report_resources(self, fl_ctx):
        """Reports current available resources.

        Args:
            fl_ctx: the FLContext.

        Returns:
            An empty dict, as this manager does not track resources.
        """
        return {}
