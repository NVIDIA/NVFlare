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

from unittest.mock import MagicMock

from nvflare.fuel.f3.cellnet.connector_manager import ConnectorManager
from nvflare.fuel.f3.drivers.driver_params import DriverParams


def test_internal_listener_host_overrides_default_resources():
    comm_configurator = MagicMock()
    comm_configurator.get_config.return_value = None

    manager = ConnectorManager(
        communicator=MagicMock(),
        secure=False,
        comm_configurator=comm_configurator,
        internal_listener_host="127.0.0.1",
    )

    assert manager.int_resources[DriverParams.HOST.value] == "127.0.0.1"
    assert manager.int_resources[DriverParams.LISTEN_HOST.value] == "127.0.0.1"
