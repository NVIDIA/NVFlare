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

from unittest.mock import patch

from nvflare.app_common.ccwf.common import StatusReport
from nvflare.app_common.ccwf.swarm_server_ctl import SwarmServerController


def test_swarm_projects_client_round_status_to_server_progress():
    controller = SwarmServerController(
        num_rounds=3,
        participating_clients=["site-1", "site-2"],
        starting_client="site-1",
    )

    with patch("nvflare.app_common.ccwf.server_ctl.log_progress") as progress:
        controller._log_round_progress(StatusReport(last_round=0, action="start_learn_task"))
        controller._log_round_progress(StatusReport(last_round=0, action="finished_learn_task"))
        controller._log_round_progress(StatusReport(last_round=1, action="start_learn_task"))

    messages = [call.args[1] for call in progress.call_args_list]
    assert len(messages) == 2
    assert "ROUND 1 / 3" in messages[0]
    assert "ROUND 2 / 3" in messages[1]
    assert all("Swarm learning" in message for message in messages)
