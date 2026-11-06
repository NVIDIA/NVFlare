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
"""Tests that the TensorFlow FedAvg recipe forwards sampling_seed to the controller."""

from unittest.mock import patch

import pytest

# The TF recipe imports TF components at the module level, so skip the whole
# file if TF is not installed.
pytest.importorskip("tensorflow", reason="TensorFlow not available")

from nvflare.apis.job_def import SERVER_SITE_NAME  # noqa: E402
from nvflare.app_common.abstract.model_persistor import ModelPersistor  # noqa: E402
from nvflare.app_opt.tf.recipes.fedavg import FedAvgRecipe  # noqa: E402


class DummyPersistor(ModelPersistor):
    """Minimal persistor so the recipe has a model source."""

    def load_model(self, fl_ctx):
        return {}

    def save_model(self, model, fl_ctx):
        return None


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


class TestTFFedAvgRecipeSamplingSeed:
    def test_default_is_unseeded(self, mock_file_system):
        recipe = FedAvgRecipe(
            name="t", min_clients=2, num_rounds=1, train_script="train.py", model_persistor=DummyPersistor()
        )
        controller = get_controller(recipe)
        assert controller is not None
        assert controller._sampling_seed is None

    def test_seed_reaches_the_controller(self, mock_file_system):
        recipe = FedAvgRecipe(
            name="t",
            min_clients=2,
            num_rounds=1,
            train_script="train.py",
            model_persistor=DummyPersistor(),
            sampling_seed=7,
        )
        assert recipe.sampling_seed == 7
        assert get_controller(recipe)._sampling_seed == 7
