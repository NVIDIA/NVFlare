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

from nvflare.apis.overseer_spec import SP
from nvflare.private.fed.client.fed_client_base import FederatedClientBase


class TestFederatedClientBase:
    def test_overseer_callback_logs_warning_when_primary_sp_update_fails(self):
        client = FederatedClientBase.__new__(FederatedClientBase)
        client.logger = MagicMock()
        client.engine = MagicMock()
        client.set_primary_sp = MagicMock(side_effect=RuntimeError("Failed to get engine after 30 seconds"))

        sp = SP(name="server-1", fl_port="8002", admin_port="8003", service_session_id="ssid", primary=True)
        overseer_agent = MagicMock()
        overseer_agent.is_shutdown.return_value = False
        overseer_agent.get_primary_sp.return_value = sp

        client.overseer_callback(overseer_agent)

        client.set_primary_sp.assert_called_once_with(sp)
        client.logger.warning.assert_called_once()
        warning_msg = client.logger.warning.call_args[0][0]
        assert "Could not complete primary SP update from overseer" in warning_msg
        assert "The client will keep running" in warning_msg
        assert "Failed to get engine after 30 seconds" in warning_msg
