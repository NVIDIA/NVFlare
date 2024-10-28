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

import os

from nvflare.apis.resource_manager_spec import ResourceConsumerSpec
from nvflare.fuel.utils.gpu_utils import get_host_gpu_ids, get_host_gpu_memory_free


class GPUResourceConsumer(ResourceConsumerSpec):
    def __init__(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    def consume(self, resources: dict):
        host_gpus = get_host_gpu_ids()
        host_gpu_memory_free = get_host_gpu_memory_free(unit="MiB")
        for gpu_id, gpu_mem in resources.items():
            if gpu_id not in host_gpus:
                raise RuntimeError(f"GPU ID {gpu_id} does not exist")
            if gpu_mem * 1024.0 > host_gpu_memory_free[gpu_id]:
                raise RuntimeError("GPU free mem is not enough")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in resources.keys()])
