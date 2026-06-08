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

import numpy as np

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType

_MODULE_NAME = "_nvflare_tf_fedopt_ctl_for_test"


class _FakeWeight:
    def __init__(self, shape):
        self.shape = shape


class _FakeKerasModel:
    trainable_weights = [_FakeWeight((1,))]

    def get_weights(self):
        return [np.array([1.5])]


class _FakeLearningRate:
    def numpy(self):
        return 0.1


class _FakeOptimizer:
    def __init__(self):
        self.learning_rate = _FakeLearningRate()
        self.applied_gradients = None

    def apply_gradients(self, gradients):
        self.applied_gradients = list(gradients)


class _FakeLRScheduler:
    pass


def _load_tf_fedopt_ctl(monkeypatch):
    """Load TF fedopt_ctl.py directly so TensorFlow is not required for this unit test."""
    if "tensorflow" not in sys.modules:
        monkeypatch.setitem(sys.modules, "tensorflow", ModuleType("tensorflow"))

    module_path = Path(__file__).parents[4] / "nvflare" / "app_opt" / "tf" / "fedopt_ctl.py"
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_MODULE_NAME, None)
        raise

    return module


def test_tf_fedopt_update_model_preserves_aggregated_metrics(monkeypatch):
    module = _load_tf_fedopt_ctl(monkeypatch)

    try:
        controller = object.__new__(module.FedOpt)
        controller.current_round = 3
        controller.persistor = SimpleNamespace(model=_FakeKerasModel())
        controller.optimizer = _FakeOptimizer()
        controller.lr_scheduler = _FakeLRScheduler()
        controller.info = MagicMock()
        controller._to_tf_params_list = MagicMock(return_value=["gradient"])

        global_model = FLModel(
            params={
                "trainable": np.array([1.0]),
                "batch_norm": np.array([10.0]),
            },
            metrics={"old_metric": -1.0},
            meta={"old_meta": "before"},
        )
        aggr_result = FLModel(
            params={
                "trainable": np.array([1.25]),
                "batch_norm": np.array([12.0]),
            },
            params_type=ParamsType.FULL,
            metrics={"loss": 0.25, "accuracy": 0.75},
            meta={"nr_aggregated": 2},
        )

        updated = controller.update_model(global_model, aggr_result)

        np.testing.assert_array_equal(updated.params["trainable"], np.array([1.5]))
        np.testing.assert_array_equal(updated.params["batch_norm"], np.array([12.0]))
        assert updated.meta == aggr_result.meta
        assert updated.metrics == aggr_result.metrics

        clear_metrics_result = FLModel(
            params={
                "trainable": np.array([1.25]),
                "batch_norm": np.array([12.0]),
            },
            params_type=ParamsType.FULL,
            metrics=None,
            meta={"nr_aggregated": 2},
        )

        updated = controller.update_model(global_model, clear_metrics_result)

        assert updated.metrics is None
    finally:
        sys.modules.pop(_MODULE_NAME, None)
