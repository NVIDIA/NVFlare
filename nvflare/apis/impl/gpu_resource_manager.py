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

from nvflare.apis.job_def import Job

try:
    import pynvml
except ImportError:
    pynvml = None

from typing import Dict, List, Optional

import psutil

from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.resource_manager import ResourceManager

from .scheduler_constants import GPUConstants


def _fetch_resources() -> Dict:
    infos = dict(psutil.virtual_memory()._asdict())
    if pynvml:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            gpu_info = {GPUConstants.GPU_COUNT: device_count}
            for index in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                gpu_info[f"{GPUConstants.GPU_DEVICE_PREFIX}{index}"] = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
            pynvml.nvmlShutdown()
            infos.update(gpu_info)
        except pynvml.nvml.NVMLError_LibraryNotFound:
            pass
    return infos


# TODO::: make a GPU resource manager more specific => if any we put in app_common


# TODO:: make this generic
# require 2, resource is [0, 1, 2, 3, 4, 5] -> check if things in list is available => return [0,1]
# require 3 => return [2, 3, 4]
# free 2, require 3 => return [0, 1, 5]
class ListResourceManager(ResourceManager):
    def __init__(self, resources: Dict[str, List]):
        super().__init__()
        self.committed_jobs: Optional[List[Job]] = None
        self.resources = resources

    def check_resources(self, resource_requirement: dict, fl_ctx: FLContext) -> (bool, Optional[str]):
        if not isinstance(resource_requirement, dict):
            raise TypeError("resource_requirement is a required argument.")
        info = _fetch_resources()
        check_result = True
        for k in resource_requirement:
            if k in info:
                if info[k] < resource_requirement[k]:
                    check_result = False
                    break
        return check_result, None

    def cancel_resources(self, resource_requirement: dict, token: str, fl_ctx: FLContext):
        return None

    def allocate_resources(self, resource_requirement: dict, token: str, fl_ctx: FLContext) -> dict:
        result = {}
        # TODO::
        #   if resource_requirements: {gpus: 2}
        #   return {"gpus": [gpu_0, gpu_1]}
        for k in resource_requirement:
            if k in self.resources:
                self.resources[k] -= resource_requirement[k]
                result[k] = resource_requirement[k]
        return result

    def free_resources(self, resources: dict, token: str, fl_ctx: FLContext):
        for k in resources:
            if k not in self.resources:
                raise RuntimeError(f"Key {k} is not in resource manager's resources.")
            self.resources[k] += resources[k]
