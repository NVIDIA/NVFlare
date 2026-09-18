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
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.signal import Signal

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "advanced" / "kaplan-meier-he"


def _load_module(file_name):
    module_name = f"kaplan_meier_{Path(file_name).stem}"
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLE_ROOT / file_name)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"tenseal": MagicMock()}):
        spec.loader.exec_module(module)
    return module


def _make_controller(module, final_results):
    if hasattr(module, "KM_HE"):
        controller = module.KM_HE(min_clients=2, he_context_path="context")
        controller.start_fl_collect_max_idx = MagicMock(return_value=[MagicMock()])
        controller.aggr_max_idx = MagicMock(return_value=1)
        controller.distribute_max_idx_collect_enc_stats = MagicMock(return_value=[MagicMock()])
        controller.aggr_he_hist = MagicMock(return_value=({0: 1}, {0: 0}))
    else:
        controller = module.KM(min_clients=2)
        controller.start_fl_collect_hist = MagicMock(return_value=[MagicMock()])
        controller.aggr_hist = MagicMock(return_value=({0: 1}, {0: 0}))
    controller.abort_signal = Signal()
    controller.sample_clients = MagicMock(return_value=["site-1", "site-2"])
    controller.distribute_global_hist = MagicMock(return_value=final_results)
    return controller


@pytest.mark.parametrize("file_name", ["server.py", "server_he.py"])
def test_survival_completion_requires_every_target_response(file_name):
    module = _load_module(file_name)
    controller = _make_controller(module, [MagicMock()])

    with patch.object(module, "log_progress") as progress:
        controller.run()

    assert not any("completed" in call.args[1] for call in progress.call_args_list)


@pytest.mark.parametrize("file_name", ["server.py", "server_he.py"])
def test_survival_completion_is_reported_after_every_target_responds(file_name):
    module = _load_module(file_name)
    controller = _make_controller(module, [MagicMock(), MagicMock()])

    with patch.object(module, "log_progress") as progress:
        controller.run()

    assert any("completed" in call.args[1] for call in progress.call_args_list)


@pytest.mark.parametrize("file_name", ["server.py", "server_he.py"])
def test_survival_completion_is_not_reported_after_abort(file_name):
    module = _load_module(file_name)
    controller = _make_controller(module, [MagicMock(), MagicMock()])
    controller.abort_signal.trigger(True)

    with patch.object(module, "log_progress") as progress:
        controller.run()

    assert not any("completed" in call.args[1] for call in progress.call_args_list)
