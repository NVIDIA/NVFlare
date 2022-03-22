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
from functools import total_ordering
from typing import Dict


@total_ordering
class Resources:
    def __init__(self, gpu=1, cpu=1, cpu_mem=8):
        self.gpu = gpu
        self.cpu = cpu
        self.cpu_mem = cpu_mem

    def to_tuple(self):
        return self.gpu, self.cpu, self.cpu_mem

    def __ge__(self, other):
        if self.gpu < other.gpu:
            return False
        if self.cpu < other.cpu:
            return False
        if self.cpu_mem < other.cpu_mem:
            return False
        return True

    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()

    def __add__(self, other):
        return Resources(self.gpu + other.gpu, self.cpu + other.cpu, self.cpu_mem + other.cpu_mem)

    def __sub__(self, other):
        return Resources(self.gpu - other.gpu, self.cpu - other.cpu, self.cpu_mem - other.cpu_mem)


class Site:
    def __init__(self, name):
        self.name = name
        # TODO:: do we need to have both capacity and upper limit here?
        self.resources = None

    def set_resources(self, resources: Resources):
        self.resources = resources

    def get_name(self) -> str:
        return self.name

    def get_resources(self) -> Resources:
        return self.resources


class RunManagerSpec(ABC):
    def __init__(self):
        self.sites = None

    @abstractmethod
    def get_sites(self) -> Dict[str, Site]:
        """Return site resources."""
        pass

    @abstractmethod
    def update_site_resources(self, site_name: str, resource: Resources):
        """Update site resources."""
        pass

    @abstractmethod
    def fetch_sites_resources(self):
        """Fetch site resources."""
        pass
