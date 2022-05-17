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


from unittest.mock import MagicMock, Mock, patch

import nvflare.private.fed.protos.federated_pb2 as fed_msg
from nvflare.private.fed.server.fed_server import FederatedServer


class TestFederatedServer:
    def test_heart_beat_abort_jobs(self):
        with patch("nvflare.private.fed.server.fed_server.ServerEngine") as mock_engine:
            server = FederatedServer(
                project_name="project_name",
                min_num_clients=1,
                max_num_clients=100,
                wait_after_min_clients=60,
                cmd_modules=None,
                heart_beat_timeout=600,
                args=MagicMock(),
                secure_train=False,
                enable_byoc=True,
                snapshot_persistor=MagicMock(),
                overseer_agent=MagicMock(),
            )

            request = Mock(token="token", jobs=["extra_job"])
            context = MagicMock()
            expected = fed_msg.FederatedSummary(abort_jobs=["extra_job"])
            assert server.Heartbeat(request, context) == expected
