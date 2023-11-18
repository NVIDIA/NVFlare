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

from nvflare.fuel.f3 import drivers
from nvflare.fuel.f3.drivers.aio_http_driver import AioHttpDriver
from nvflare.fuel.f3.drivers.aio_tcp_driver import AioTcpDriver
from nvflare.fuel.f3.drivers.driver_manager import DriverManager
from nvflare.fuel.f3.drivers.tcp_driver import TcpDriver


class TestDriverManager:
    @pytest.fixture
    def manager(self):
        driver_manager = DriverManager()
        driver_manager.search_folder(os.path.dirname(drivers.__file__), drivers.__package__)
        return driver_manager

    @pytest.mark.parametrize(
        "scheme, expected",
        [
            ("tcp", TcpDriver),
            ("stcp", TcpDriver),
            ("http", AioHttpDriver),
            ("https", AioHttpDriver),
            ("ws", AioHttpDriver),
            ("wss", AioHttpDriver),
            ("atcp", AioTcpDriver),
            ("satcp", AioTcpDriver),
        ],
    )
    def test_driver_loading(self, manager, scheme, expected):
        driver_class = manager.find_driver_class(scheme)
        assert driver_class == expected
