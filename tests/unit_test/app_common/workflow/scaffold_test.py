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

import numpy as np

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AlgorithmConstants, AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.workflows.scaffold import Scaffold


def test_scaffold_publishes_standard_round_start_before_client_work():
    controller = Scaffold(num_clients=2, num_rounds=1, persistor_id="")
    controller.fl_ctx = FLContext()
    controller.model = FLModel(params={"weight": np.array([1.0])}, meta={})
    controller._global_ctrl_weights = {"weight": np.array([0.0])}
    aggregate_result = FLModel(
        params={"weight": np.array([2.0])},
        meta={AlgorithmConstants.SCAFFOLD_CTRL_DIFF: {"weight": np.array([0.5])}},
    )
    calls = []

    with (
        patch.object(controller, "info"),
        patch.object(controller, "event", side_effect=lambda event: calls.append(event)),
        patch.object(controller, "sample_clients", return_value=["site-1", "site-2"]),
        patch.object(controller, "send_model_and_wait", side_effect=lambda **_: calls.append("send") or []),
        patch.object(controller, "aggregate", return_value=aggregate_result),
        patch.object(controller, "update_model", return_value=controller.model),
        patch.object(controller, "save_model"),
        patch.object(controller, "_maybe_cleanup_memory"),
    ):
        controller._run_rounds()

    assert calls == [AppEventType.ROUND_STARTED, "send", AppEventType.ROUND_DONE]
    assert controller.fl_ctx.get_prop(AppConstants.CURRENT_ROUND) == 0
    assert controller.fl_ctx.get_prop(AppConstants.NUM_ROUNDS) == 1
