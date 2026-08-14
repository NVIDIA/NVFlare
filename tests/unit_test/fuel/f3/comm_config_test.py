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

import pytest

from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.utils.config_service import ConfigService


@pytest.fixture
def configurator(monkeypatch):
    ConfigService.reset()
    CommConfigurator.reset()
    monkeypatch.delenv("NVFLARE_STREAMING_MAX_BLOB_SIZE", raising=False)
    monkeypatch.setattr(ConfigService, "load_configuration", lambda file_basename: None)
    yield CommConfigurator()
    ConfigService.reset()
    CommConfigurator.reset()


def test_streaming_max_blob_size_defaults_to_four_gib(configurator):
    assert configurator.get_streaming_max_blob_size() == 4 * 1024 * 1024 * 1024
