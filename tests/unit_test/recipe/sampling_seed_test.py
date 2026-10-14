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

"""Tests for the sampling_seed recipe option."""

from unittest.mock import patch

import pytest
import torch.nn as nn

from nvflare.apis.job_def import SERVER_SITE_NAME
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe


class SimpleTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(10, 10)

    def forward(self, x):
        return self.lin(x)


@pytest.fixture
def mock_file_system():
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.path.isdir", return_value=True),
        patch("os.path.exists", return_value=True),
    ):
        yield


def get_controller(recipe):
    server_app = recipe._job._deploy_map[SERVER_SITE_NAME]
    for workflow in server_app.app_config.workflows:
        if hasattr(workflow.controller, "_sampling_seed"):
            return workflow.controller
    return None


class TestSamplingSeedRecipeOption:
    def test_default_is_unseeded(self, mock_file_system):
        recipe = FedAvgRecipe(name="t", model=SimpleTestModel(), min_clients=2, num_rounds=1, train_script="client.py")
        controller = get_controller(recipe)
        assert controller is not None
        assert controller._sampling_seed is None
        assert controller._sampling_rng is None

    def test_seed_reaches_the_controller(self, mock_file_system):
        recipe = FedAvgRecipe(
            name="t", model=SimpleTestModel(), min_clients=2, num_rounds=1, train_script="client.py", sampling_seed=7
        )
        assert recipe.sampling_seed == 7
        assert get_controller(recipe)._sampling_seed == 7

    def test_seed_reaches_the_controller_through_fedprox(self, mock_file_system):
        from nvflare.app_opt.pt.recipes.fedprox import FedProxRecipe

        recipe = FedProxRecipe(
            name="t", model=SimpleTestModel(), min_clients=2, num_rounds=1, train_script="client.py", sampling_seed=7
        )
        assert get_controller(recipe)._sampling_seed == 7

    def test_seed_reaches_the_controller_through_fedce(self, mock_file_system):
        from nvflare.app_opt.pt.recipes.fedce import FedCERecipe

        recipe = FedCERecipe(
            name="t", model=SimpleTestModel(), min_clients=2, num_rounds=1, train_script="client.py", sampling_seed=7
        )
        assert get_controller(recipe)._sampling_seed == 7

    def test_seed_reaches_the_controller_through_tensorflow(self, mock_file_system):
        pytest.importorskip("tensorflow", reason="TensorFlow not available")
        from nvflare.app_opt.tf.recipes.fedavg import FedAvgRecipe as TFFedAvgRecipe

        recipe = TFFedAvgRecipe(name="t", min_clients=2, num_rounds=1, train_script="train.py", sampling_seed=7)
        assert get_controller(recipe)._sampling_seed == 7
