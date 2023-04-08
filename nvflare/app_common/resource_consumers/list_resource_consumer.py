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
from abc import ABC, abstractmethod

from nvflare.apis.resource_manager_spec import ResourceConsumerSpec


class _Consumer(ABC):
    @abstractmethod
    def consume(self, resources: list):
        pass


class _GPUConsumer(_Consumer):
    def __init__(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    def consume(self, resources: list):
        """Consumes resources.

        Note that this class did not check physically if those GPUs exist.
        """
        gpu_numbers = [str(x) for x in resources]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_numbers)


class ListResourceConsumer(ResourceConsumerSpec):
    def __init__(self):
        """This class can be used with ListResourceManager.

        Users can add custom _Consumer in the resource_consumer_map to handle new resource type.
        """
        super().__init__()
        self.resource_consumer_map = {"gpu": _GPUConsumer()}

    def consume(self, resources: dict):
        for key, consumer in self.resource_consumer_map.items():
            if key in resources:
                consumer.consume(resources[key])
