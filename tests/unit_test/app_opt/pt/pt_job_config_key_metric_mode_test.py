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

"""Tests for key_metric_mode pass-through in the legacy PyTorch job config wrappers."""

import pytest
import torch.nn as nn

from nvflare.apis.job_def import SERVER_SITE_NAME
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.app_opt.pt.job_config.fed_sag_mlflow import SAGMLFlowJob


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)


def get_model_selector(job):
    server_app = job._deploy_map[SERVER_SITE_NAME]
    return server_app.app_config.components.get("model_selector")


class TestPTBaseFedJobKeyMetricMode:
    def test_min_mode_sets_negate_on_selector(self):
        job = BaseFedJob(key_metric="val_loss", key_metric_mode="min")

        model_selector = get_model_selector(job)
        assert isinstance(model_selector, IntimeModelSelector)
        assert model_selector.key_metric == "val_loss"
        assert model_selector.negate_key_metric is True

    def test_default_mode_is_max(self):
        job = BaseFedJob(key_metric="accuracy")

        model_selector = get_model_selector(job)
        assert isinstance(model_selector, IntimeModelSelector)
        assert model_selector.negate_key_metric is False

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="key_metric_mode"):
            BaseFedJob(key_metric="accuracy", key_metric_mode="bad")


class TestPTFedAvgJobKeyMetricMode:
    def test_min_mode_sets_negate_on_selector(self):
        job = FedAvgJob(
            initial_model=SimpleModel(),
            n_clients=2,
            num_rounds=1,
            key_metric="val_loss",
            key_metric_mode="min",
        )

        model_selector = get_model_selector(job)
        assert isinstance(model_selector, IntimeModelSelector)
        assert model_selector.key_metric == "val_loss"
        assert model_selector.negate_key_metric is True


class TestSAGMLFlowJobKeyMetricMode:
    def test_min_mode_sets_negate_on_selector(self):
        job = SAGMLFlowJob(
            initial_model=SimpleModel(),
            n_clients=2,
            num_rounds=1,
            key_metric="val_loss",
            key_metric_mode="min",
        )

        model_selector = get_model_selector(job)
        assert isinstance(model_selector, IntimeModelSelector)
        assert model_selector.key_metric == "val_loss"
        assert model_selector.negate_key_metric is True
