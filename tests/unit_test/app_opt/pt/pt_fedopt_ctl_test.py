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
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

from nvflare.app_common.abstract.fl_model import FLModel

_MODULE_NAME = "_nvflare_pt_fedopt_ctl_for_test"


class _FakeTorchWeight:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeOptimizer:
    param_groups = [{"lr": 0.1}]


class _FakeLRScheduler:
    pass


def _load_pt_fedopt_ctl(monkeypatch):
    """Load PT fedopt_ctl.py directly so the test does not require importing the whole PT package."""
    if "torch" not in sys.modules:
        fake_torch = ModuleType("torch")
        fake_torch.nn = SimpleNamespace(Module=object)
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
        fake_torch.device = lambda device: device
        fake_torch.tensor = lambda value: value
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

    module_path = Path(__file__).parents[4] / "nvflare" / "app_opt" / "pt" / "fedopt_ctl.py"
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_MODULE_NAME, None)
        raise

    return module


def test_pt_fedopt_update_model_preserves_aggregated_metrics(monkeypatch):
    module = _load_pt_fedopt_ctl(monkeypatch)

    try:
        controller = object.__new__(module.FedOpt)
        controller.current_round = 3
        controller.device = "cpu"
        controller.optimizer = _FakeOptimizer()
        controller.lr_scheduler = _FakeLRScheduler()
        controller.info = MagicMock()
        controller.optimizer_update = MagicMock(
            side_effect=lambda model_diff: ({"trainable": _FakeTorchWeight(1.5)}, ["trainable"])
        )

        global_model = FLModel(
            params={"trainable": 1.0, "batch_norm": 10.0},
            metrics={"old_metric": -1.0},
            meta={"old_meta": "before"},
        )
        aggr_result = FLModel(
            params={"trainable": 0.5, "batch_norm": 2.0},
            metrics={"loss": 0.25, "accuracy": 0.75},
            meta={"nr_aggregated": 2},
        )

        updated = controller.update_model(global_model, aggr_result)

        assert updated.params["trainable"] == 1.5
        assert updated.params["batch_norm"] == 12.0
        assert updated.meta == aggr_result.meta
        assert updated.metrics == aggr_result.metrics

        clear_metrics_result = FLModel(
            params={"trainable": 0.5, "batch_norm": 2.0},
            metrics=None,
            meta={"nr_aggregated": 2},
        )

        updated = controller.update_model(global_model, clear_metrics_result)

        assert updated.metrics is None
    finally:
        sys.modules.pop(_MODULE_NAME, None)
