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

from nvflare.app_common.resource_managers.auto_clean_resource_manager import AutoCleanResourceManager
from nvflare.fuel.utils.gpu_utils import get_host_gpu_ids, get_host_gpu_memory_total


def _check_non_negative(prop, prop_name: str):
    if not isinstance(prop, int):
        raise TypeError(f"{prop_name} should be of type int, but got {type(prop)}.")
    if prop < 0:
        raise ValueError(f"{prop_name} should be greater than 0.")


class GPUResource:
    def __init__(self, gpu_id: int, gpu_memory: int):
        self.id = gpu_id
        self.memory = gpu_memory

    def to_dict(self):
        return {"gpu_id": self.id, "memory": self.memory}


class GPUResourceManager(AutoCleanResourceManager):
    def __init__(
        self,
        num_of_gpus: int,
        mem_per_gpu_in_GiB: int,
        num_gpu_key: str = "num_of_gpus",
        gpu_mem_key: str = "mem_per_gpu_in_GiB",
        expiration_period: int = 30,
    ):
        """Resource manager for GPUs.

        Args:
            num_of_gpus (int): Number of GPUs.
            mem_per_gpu_in_GiB (int): Memory for each GPU.
            expiration_period (int): Number of seconds to hold the resources reserved.
                If check_resources is called but after "expiration_period" no allocate resource is called,
                then the reserved resources will be released.
        """
        _check_non_negative(num_of_gpus, "num_of_gpus")
        _check_non_negative(mem_per_gpu_in_GiB, "mem_per_gpu_in_GiB")
        _check_non_negative(expiration_period, "expiration_period")

        num_host_gpus = len(get_host_gpu_ids())
        if num_of_gpus > num_host_gpus:
            raise ValueError(f"num_of_gpus specified {num_of_gpus} is greater than available GPUs: {num_host_gpus}.")

        host_gpu_mem = get_host_gpu_memory_total()
        for i in host_gpu_mem:
            if mem_per_gpu_in_GiB * 1024 > i:
                raise ValueError(
                    f"mem_per_gpu_in_GiB specified {mem_per_gpu_in_GiB * 1024} is greater than available GPU memory: {i}"
                )

        self.num_gpu_key = num_gpu_key
        self.gpu_mem_key = gpu_mem_key
        resources = {i: GPUResource(gpu_id=i, gpu_memory=mem_per_gpu_in_GiB) for i in range(num_of_gpus)}

        super().__init__(resources=resources, expiration_period=expiration_period)

    def _deallocate(self, resources: dict):
        for k, v in resources.items():
            self.resources[k].memory += v

    def _check_required_resource_available(self, resource_requirement: dict) -> bool:
        if not resource_requirement:
            return True

        if self.num_gpu_key not in resource_requirement:
            raise ValueError(f"resource_requirement is missing num_gpu_key {self.num_gpu_key}.")

        check_result = False
        num_gpu = resource_requirement[self.num_gpu_key]
        gpu_mem = resource_requirement.get(self.gpu_mem_key, 0)

        satisfied = 0
        for k in self.resources:
            r: GPUResource = self.resources[k]
            if r.memory >= gpu_mem:
                satisfied += 1
            if satisfied >= num_gpu:
                check_result = True
                break
        return check_result

    def _reserve_resource(self, resource_requirement: dict) -> dict:
        if not resource_requirement:
            return {}

        if self.num_gpu_key not in resource_requirement:
            raise ValueError(f"resource_requirement is missing num_gpu_key {self.num_gpu_key}.")

        reserved_resources = {}
        num_gpu = resource_requirement[self.num_gpu_key]
        gpu_mem = resource_requirement.get(self.gpu_mem_key, 0)
        reserved = 0
        for k in self.resources:
            r: GPUResource = self.resources[k]
            if r.memory >= gpu_mem:
                r.memory -= gpu_mem
                reserved_resources[k] = gpu_mem
                reserved += 1
            if reserved == num_gpu:
                break
        return reserved_resources

    def _resource_to_dict(self) -> dict:
        return {
            "resources": [self.resources[k].to_dict() for k in self.resources],
            "reserved_resources": self.reserved_resources,
        }
