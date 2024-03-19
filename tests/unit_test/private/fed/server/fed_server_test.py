# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.shareable import Shareable
from nvflare.private.defs import CellMessageHeaderKeys, new_cell_message
from nvflare.private.fed.server.fed_server import FederatedServer
from nvflare.private.fed.server.server_state import ColdState, HotState


class TestFederatedServer:
    @pytest.mark.parametrize("server_state, expected", [(HotState(), ["extra_job"]), (ColdState(), [])])
    def test_heart_beat_abort_jobs(self, server_state, expected):
        with patch("nvflare.private.fed.server.fed_server.ServerEngine") as mock_engine:

            server = FederatedServer(
                project_name="project_name",
                min_num_clients=1,
                max_num_clients=100,
                cmd_modules=None,
                heart_beat_timeout=600,
                args=MagicMock(),
                secure_train=False,
                snapshot_persistor=MagicMock(),
                overseer_agent=MagicMock(),
            )

            server.server_state = server_state
            request = new_cell_message(
                {
                    CellMessageHeaderKeys.TOKEN: "token",
                    CellMessageHeaderKeys.SSID: "ssid",
                    CellMessageHeaderKeys.CLIENT_NAME: "client_name",
                    CellMessageHeaderKeys.PROJECT_NAME: "task_name",
                    CellMessageHeaderKeys.JOB_IDS: ["extra_job"],
                },
                Shareable(),
            )

            result = server.client_heartbeat(request)
            assert result.get_header(CellMessageHeaderKeys.ABORT_JOBS, []) == expected
