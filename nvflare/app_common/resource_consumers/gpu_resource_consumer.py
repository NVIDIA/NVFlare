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

from nvflare.apis.resource_manager_spec import ResourceConsumerSpec


class GPUResourceConsumer(ResourceConsumerSpec):
    def __init__(self, gpu_resource_key="gpu"):
        super().__init__()
        self.gpu_resource_key = gpu_resource_key
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    def consume(self, resources: dict):
        gpu_numbers = []
        if self.gpu_resource_key in resources:
            gpu_numbers = [str(x) for x in resources[self.gpu_resource_key]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_numbers)
