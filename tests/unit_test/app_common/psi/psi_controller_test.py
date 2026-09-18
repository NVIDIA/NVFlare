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
from nvflare.app_common.psi.psi_controller import PSIController


def test_psi_controller_reports_privacy_safe_phases():
    controller = PSIController("psi_workflow")
    controller.psi_workflow = MagicMock()
    fl_ctx = FLContext()
    engine = MagicMock()
    engine.get_clients.return_value = [MagicMock(), MagicMock(), MagicMock()]
    fl_ctx.get_engine = MagicMock(return_value=engine)

    with patch("nvflare.app_common.psi.psi_controller.log_progress") as progress:
        controller.control_flow(Signal(), fl_ctx)

    messages = [call.args[1] for call in progress.call_args_list]
    assert "Private set intersection · 3 clients" in messages[0]
    assert messages[1:] == [
        "  Computing encrypted intersection…",
        "\n  ✓ Private set intersection completed",
    ]
    assert all("size" not in message.lower() and "item" not in message.lower() for message in messages)


def test_psi_controller_does_not_report_completion_after_abort():
    controller = PSIController("psi_workflow")
    controller.psi_workflow = MagicMock()
    abort_signal = Signal()
    controller.psi_workflow.post_process.side_effect = lambda _: abort_signal.trigger("stopped")
    fl_ctx = FLContext()
    engine = MagicMock()
    engine.get_clients.return_value = [MagicMock(), MagicMock()]
    fl_ctx.get_engine = MagicMock(return_value=engine)

    with patch("nvflare.app_common.psi.psi_controller.log_progress") as progress:
        result = controller.control_flow(abort_signal, fl_ctx)

    assert result is False
    assert all("completed" not in call.args[1] for call in progress.call_args_list)
