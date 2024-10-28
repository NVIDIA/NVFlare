# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import pytest

from nvflare.fuel.f3 import communicator

# Setup custom driver path before communicator module initialization
from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.utils.config_service import ConfigService


class TestCustomDriver:
    @pytest.fixture
    def manager(self):
        CommConfigurator.reset()
        rel_path = "../../../data/custom_drivers/config"
        config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), rel_path))
        ConfigService.initialize({}, [config_path])
        communicator.load_comm_drivers()

        return communicator.driver_mgr

    def test_custom_driver_loading(self, manager):
        driver_class = manager.find_driver_class("warp")
        assert driver_class.__name__ == "WarpDriver"
