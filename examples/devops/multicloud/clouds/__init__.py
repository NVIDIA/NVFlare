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

from .aws import AwsProvider
from .azure import AzureProvider
from .gcp import GcpProvider
from .kubernetes import KubernetesProvider

CLOUD_ORDER = ("gcp", "aws", "azure", "kubernetes")

PROVIDERS = {
    "gcp": GcpProvider(),
    "aws": AwsProvider(),
    "azure": AzureProvider(),
    "kubernetes": KubernetesProvider(),
}


def get_provider(name: str):
    try:
        return PROVIDERS[name]
    except KeyError:
        raise ValueError(f"unknown cloud: {name}") from None
