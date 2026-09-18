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

from unittest.mock import MagicMock, patch

from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_opt.xgboost.histogram_based_v2.controller import ClientStatus, XGBController


def test_collective_xgboost_reports_server_visible_phases():
    controller = XGBController(
        adaptor_component_id="adaptor",
        num_rounds=4,
        data_split_mode=0,
        secure_training=False,
        xgb_params={"objective": "binary:logistic"},
    )
    controller.adaptor = MagicMock()
    controller.participating_clients = ["site-1", "site-2"]
    controller.client_statuses = {name: ClientStatus() for name in controller.participating_clients}
    for status in controller.client_statuses.values():
        status.xgb_done = True

    with (
        patch.object(controller, "_configure_clients", return_value=True),
        patch.object(controller, "_start_clients", return_value=True),
        patch.object(controller, "_check_job_status", return_value=True),
        patch("nvflare.app_opt.xgboost.histogram_based_v2.controller.log_progress") as progress,
    ):
        controller.control_flow(Signal(), FLContext())

    messages = [call.args[1] for call in progress.call_args_list]
    assert "Horizontal XGBoost · 2 clients · 4 rounds" in messages[0]
    assert messages[1:] == ["  Training collectively…", "\n  ✓ XGBoost training completed"]
