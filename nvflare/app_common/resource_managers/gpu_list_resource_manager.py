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

import os
from typing import Dict, List

from nvflare.apis.fl_context import FLContext

from .list_resource_manager import ListResourceManager

_CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"


class GPUListResourceManager(ListResourceManager):
    """Manage a list of GPUs."""

    def __init__(self, resources: Dict[str, List], expiration_period: int = 600, gpu_resource_key="gpu"):
        """Constructor

        Args:
            resources (dict): Specify the list of resources unit
            expiration_period (int): Number of seconds to hold the resources reserved.
                If check_resources is called but after "expiration_period" no allocate resource is called,
                then the reserved resources will be released.
        """
        if gpu_resource_key not in resources:
            raise ValueError("GPU resource is missing in resources.")
        super().__init__(resources=resources, expiration_period=expiration_period)
        self.original_cuda_visible_devices = None
        self.gpu_resource_key = gpu_resource_key

    def allocate_resources(self, resource_requirement: dict, token: str, fl_ctx: FLContext) -> dict:
        result = super().allocate_resources(resource_requirement, token, fl_ctx)
        self.original_cuda_visible_devices = os.environ.pop(_CUDA_VISIBLE_DEVICES, None)
        gpu_numbers = [str(x) for x in result[self.gpu_resource_key]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_numbers)
        return result

    def free_resources(self, resources: dict, token: str, fl_ctx: FLContext):
        if self.original_cuda_visible_devices:
            os.environ[_CUDA_VISIBLE_DEVICES] = self.original_cuda_visible_devices
        else:
            os.environ.pop(_CUDA_VISIBLE_DEVICES, None)
        super().free_resources(resources, token, fl_ctx)
