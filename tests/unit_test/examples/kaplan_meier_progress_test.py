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

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

from nvflare.apis.signal import Signal

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVER_PATH = REPO_ROOT / "examples" / "advanced" / "kaplan-meier-he" / "server.py"


def _load_server_module():
    spec = importlib.util.spec_from_file_location("kaplan_meier_server", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_controller(module, final_results):
    controller = module.KM(min_clients=2)
    controller.abort_signal = Signal()
    controller._current_num_targets = 2
    controller.start_fl_collect_hist = MagicMock(return_value=[MagicMock()])
    controller.aggr_hist = MagicMock(return_value=({0: 1}, {0: 0}))
    controller.distribute_global_hist = MagicMock(return_value=final_results)
    return controller


def test_survival_completion_requires_every_target_response():
    module = _load_server_module()
    controller = _make_controller(module, [MagicMock()])

    with patch.object(module, "log_progress") as progress:
        controller.run()

    assert not any("completed" in call.args[1] for call in progress.call_args_list)


def test_survival_completion_is_reported_after_every_target_responds():
    module = _load_server_module()
    controller = _make_controller(module, [MagicMock(), MagicMock()])

    with patch.object(module, "log_progress") as progress:
        controller.run()

    assert any("✓ Survival analysis completed" in call.args[1] for call in progress.call_args_list)


def test_survival_completion_is_not_reported_after_abort():
    module = _load_server_module()
    controller = _make_controller(module, [MagicMock(), MagicMock()])
    controller.abort_signal.trigger(True)

    with patch.object(module, "log_progress") as progress:
        controller.run()

    assert not any("completed" in call.args[1] for call in progress.call_args_list)
