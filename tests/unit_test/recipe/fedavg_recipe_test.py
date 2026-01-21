# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch.nn as nn

from nvflare.apis.job_def import SERVER_SITE_NAME
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
from nvflare.app_common.np.recipes import NumpyFedAvgRecipe
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe


class SimpleTestModel(nn.Module):
    """A simple PyTorch model for testing purposes."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(10, 10)

    def forward(self, x):
        x = self.lin(x)
        return x


class MyAggregator(ModelAggregator):
    """Custom aggregator for testing FedAvgRecipe with custom aggregator support."""

    def __init__(self):
        super().__init__()
        self.sum = {}
        self.count = 0

    def accept_model(self, model: FLModel):
        """Accept submitted model and add to the sum."""
        self.info(f"##### MyAggregator: Accepting model with {len(model.params)} variables #####")
        for key, value in model.params.items():
            if key not in self.sum:
                self.sum[key] = 0
            self.sum[key] += value
        self.count += 1

    def aggregate_model(self) -> FLModel:
        """Aggregate the collected models."""
        self.info(f"##### MyAggregator: Aggregating {self.count} models #####")

        # compute the average
        for key in self.sum:
            self.sum[key] = self.sum[key] / self.count

        return FLModel(params=self.sum)

    def reset_stats(self):
        """Reset the aggregator state."""
        self.info("##### MyAggregator: Resetting #####")
        # reset the sum and count
        self.sum = {}
        self.count = 0


class InvalidAggregator:
    """Invalid aggregator that doesn't inherit from Aggregator."""

    def __init__(self):
        pass


@pytest.fixture
def mock_file_system():
    """Mock file system operations for all tests."""
    with patch("os.path.isfile", return_value=True), patch("os.path.exists", return_value=True):
        yield


@pytest.fixture
def custom_aggregator():
    """Create a custom aggregator for testing."""
    return MyAggregator()


@pytest.fixture
def simple_model():
    """Create a simple test model."""
    return SimpleTestModel()


@pytest.fixture
def base_recipe_params():
    """Base parameters for creating FedAvgRecipe instances."""
    return {
        "train_script": "mock_train_script.py",
        "train_args": "--epochs 10",
        "min_clients": 2,
        "num_rounds": 5,
    }


def assert_recipe_basics(recipe, expected_name, expected_params):
    """Helper to assert basic recipe properties."""
    assert recipe.name == expected_name
    assert recipe.train_script == expected_params.get("train_script", "mock_train_script.py")
    assert recipe.train_args == expected_params.get("train_args", "--epochs 10")
    assert recipe.min_clients == expected_params.get("min_clients", 2)
    assert recipe.num_rounds == expected_params.get("num_rounds", 5)
    assert recipe.job is not None
    assert recipe.job.name == expected_name


def get_model_selector(recipe):
    server_app = recipe.job._deploy_map[SERVER_SITE_NAME]
    return server_app.app_config.components.get("model_selector")


class TestFedAvgRecipe:
    """Test cases for FedAvgRecipe class."""

    def test_default_aggregator_initialization(self, mock_file_system, base_recipe_params):
        """Test FedAvgRecipe initialization with default aggregator."""
        recipe = FedAvgRecipe(name="test_fedavg", **base_recipe_params)

        assert_recipe_basics(recipe, "test_fedavg", base_recipe_params)
        assert recipe.initial_model is None
        assert isinstance(recipe.aggregator, Aggregator)

    def test_key_metric_passthrough_pt(self, mock_file_system, base_recipe_params):
        key_metric = "val_auc"
        recipe = FedAvgRecipe(name="test_fedavg_key_metric", key_metric=key_metric, **base_recipe_params)

        model_selector = get_model_selector(recipe)
        assert isinstance(model_selector, IntimeModelSelector)
        assert model_selector.key_metric == key_metric

    def test_custom_aggregator_initialization(self, mock_file_system, base_recipe_params, custom_aggregator):
        """Test FedAvgRecipe initialization with custom aggregator."""
        params = {**base_recipe_params, "min_clients": 1, "num_rounds": 3}
        recipe = FedAvgRecipe(name="test_fedavg_custom", aggregator=custom_aggregator, **params)

        assert_recipe_basics(recipe, "test_fedavg_custom", params)
        assert recipe.aggregator is custom_aggregator
        assert isinstance(recipe.aggregator, MyAggregator)
        assert isinstance(recipe.aggregator, Aggregator)

    def test_initial_model_configuration(self, mock_file_system, base_recipe_params, custom_aggregator, simple_model):
        """Test FedAvgRecipe with initial model."""
        params = {**base_recipe_params, "min_clients": 1, "num_rounds": 3}
        recipe = FedAvgRecipe(
            name="test_fedavg_initial_model", initial_model=simple_model, aggregator=custom_aggregator, **params
        )

        assert_recipe_basics(recipe, "test_fedavg_initial_model", params)
        assert recipe.initial_model == simple_model

    @pytest.mark.parametrize(
        "min_clients,num_rounds,train_args",
        [
            (1, 1, ""),  # Minimum configuration
            (2, 3, "--epochs 5"),  # Standard configuration
            (5, 10, "--lr 0.01 --batch_size 32"),  # Complex configuration
        ],
    )
    def test_recipe_configurations(self, mock_file_system, min_clients, num_rounds, train_args):
        """Test various FedAvgRecipe configurations using parametrized tests."""
        recipe = FedAvgRecipe(
            name=f"test_config_{min_clients}_{num_rounds}",
            train_script="mock_train_script.py",
            train_args=train_args,
            min_clients=min_clients,
            num_rounds=num_rounds,
        )

        expected_params = {
            "train_script": "mock_train_script.py",
            "train_args": train_args,
            "min_clients": min_clients,
            "num_rounds": num_rounds,
        }
        assert_recipe_basics(recipe, f"test_config_{min_clients}_{num_rounds}", expected_params)


class TestFedAvgRecipeKeyMetricVariants:
    """Test key_metric passthrough for NumPy FedAvg recipes."""

    def test_key_metric_passthrough_numpy(self, mock_file_system):
        key_metric = "val_loss"
        recipe = NumpyFedAvgRecipe(
            name="test_numpy_key_metric",
            initial_model=[1.0, 2.0, 3.0],
            min_clients=2,
            train_script="mock_train_script.py",
            key_metric=key_metric,
        )

        model_selector = get_model_selector(recipe)
        assert isinstance(model_selector, IntimeModelSelector)
        assert model_selector.key_metric == key_metric
