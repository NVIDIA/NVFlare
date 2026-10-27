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

"""Validation of CIFAR partition requests that cannot produce usable shards."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize(
    "num_samples,num_sites,alpha",
    [(100, 0, 0.5), (100, 11, 0.5), (0, 1, 0.5), (100, 2, 0), (100, 2, float("nan")), (100, 2, float("inf"))],
)
def test_invalid_partition_request_fails_before_sampling(num_samples, num_sites, alpha):
    pytest.importorskip("torchvision")
    source = Path(__file__).resolve().parents[3] / "examples/advanced/collab/pt_async_cifar10/cifar10_data.py"
    spec = importlib.util.spec_from_file_location("_async_cifar_data", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError):
        module._partition(np.arange(num_samples) % 10, num_sites, alpha, seed=0)
