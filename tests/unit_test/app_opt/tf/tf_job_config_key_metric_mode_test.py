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

"""Tests for key_metric_mode pass-through in the legacy TensorFlow BaseFedJob wrapper."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from nvflare.apis.job_def import SERVER_SITE_NAME
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector

_TF_BASE_FED_JOB_MODULE = "_nvflare_tf_base_fed_job_for_test"


def load_tf_base_fed_job(monkeypatch):
    """Load only tf/job_config/base_fed_job.py so this unit test does not require TensorFlow."""
    fake_tf = ModuleType("tensorflow")
    fake_keras = ModuleType("tensorflow.keras")
    fake_keras.Model = type("Model", (), {})
    fake_tf.keras = fake_keras
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

    module_path = Path(__file__).parents[4] / "nvflare" / "app_opt" / "tf" / "job_config" / "base_fed_job.py"
    spec = importlib.util.spec_from_file_location(_TF_BASE_FED_JOB_MODULE, module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, _TF_BASE_FED_JOB_MODULE, module)
    spec.loader.exec_module(module)
    return module.BaseFedJob


def get_model_selector(job):
    server_app = job._deploy_map[SERVER_SITE_NAME]
    return server_app.app_config.components.get("model_selector")


class TestTFBaseFedJobKeyMetricMode:
    def test_min_mode_sets_negate_on_selector(self, monkeypatch):
        TFBaseFedJob = load_tf_base_fed_job(monkeypatch)
        job = TFBaseFedJob(key_metric="val_loss", key_metric_mode="min")

        model_selector = get_model_selector(job)
        assert isinstance(model_selector, IntimeModelSelector)
        assert model_selector.key_metric == "val_loss"
        assert model_selector.negate_key_metric is True

    def test_default_mode_is_max(self, monkeypatch):
        TFBaseFedJob = load_tf_base_fed_job(monkeypatch)
        job = TFBaseFedJob(key_metric="accuracy")

        model_selector = get_model_selector(job)
        assert isinstance(model_selector, IntimeModelSelector)
        assert model_selector.negate_key_metric is False

    def test_invalid_mode_raises(self, monkeypatch):
        TFBaseFedJob = load_tf_base_fed_job(monkeypatch)
        with pytest.raises(ValueError, match="key_metric_mode"):
            TFBaseFedJob(key_metric="accuracy", key_metric_mode="bad")
