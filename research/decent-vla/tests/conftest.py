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

"""Shared fixtures. Everything here is CPU-only."""
from __future__ import annotations

import pytest
import torch

from decent_vla.adapters._mock.backend import MockBackend
from decent_vla.configs.schemas import BackendConfig, DataConfig, OptimConfig


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def backend_cfg() -> BackendConfig:
    return BackendConfig(
        name="mock",
        optim=OptimConfig(lr=1e-3, weight_decay=0.0),
        extra={"model_seed": 0, "data_seed": 42},
    )


@pytest.fixture
def data_cfg() -> DataConfig:
    return DataConfig(name="mock", batch_size=8, num_clients=1, seed=42)


@pytest.fixture
def backend(backend_cfg, device) -> MockBackend:
    return MockBackend.from_config(backend_cfg, device)


@pytest.fixture
def backend_factory(backend_cfg, device):
    def factory():
        return MockBackend.from_config(backend_cfg, device)

    return factory
