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

from nvflare.apis.dxo import DataKind
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

    def test_launch_once_default(self, mock_file_system, base_recipe_params):
        """Test that launch_once defaults to True."""
        recipe = FedAvgRecipe(name="test_launch_once_default", **base_recipe_params)

        assert recipe.launch_once is True
        assert recipe.shutdown_timeout == 0.0

    @pytest.mark.parametrize(
        "launch_once,shutdown_timeout",
        [
            (True, 0.0),  # Default values
            (False, 0.0),  # launch_once=False with default timeout
            (True, 10.0),  # launch_once=True with custom timeout
            (False, 15.0),  # launch_once=False with custom timeout
        ],
    )
    def test_launch_once_and_shutdown_timeout(
        self, mock_file_system, base_recipe_params, launch_once, shutdown_timeout
    ):
        """Test FedAvgRecipe with various launch_once and shutdown_timeout configurations."""
        recipe = FedAvgRecipe(
            name="test_launch_config",
            launch_external_process=True,
            launch_once=launch_once,
            shutdown_timeout=shutdown_timeout,
            **base_recipe_params,
        )

        assert recipe.launch_once == launch_once
        assert recipe.shutdown_timeout == shutdown_timeout
        assert recipe.launch_external_process is True

    def test_launch_once_per_site_config(self, mock_file_system, base_recipe_params):
        """Test per-site configuration with different launch_once settings."""
        per_site_config = {
            "site-1": {
                "launch_once": False,
                "shutdown_timeout": 15.0,
            },
            "site-2": {
                "launch_once": True,
                "shutdown_timeout": 5.0,
            },
        }

        recipe = FedAvgRecipe(
            name="test_per_site_launch",
            launch_external_process=True,
            launch_once=True,  # Default
            shutdown_timeout=10.0,  # Default
            per_site_config=per_site_config,
            **base_recipe_params,
        )

        # Check default values are stored
        assert recipe.launch_once is True
        assert recipe.shutdown_timeout == 10.0
        assert recipe.per_site_config == per_site_config

    def test_launch_once_in_process_mode(self, mock_file_system, base_recipe_params):
        """Test that launch_once and shutdown_timeout can be set even with in-process execution."""
        recipe = FedAvgRecipe(
            name="test_in_process",
            launch_external_process=False,  # In-process mode
            launch_once=False,
            shutdown_timeout=10.0,
            **base_recipe_params,
        )

        # Values should be stored even though they won't be used in in-process mode
        assert recipe.launch_once is False
        assert recipe.shutdown_timeout == 10.0
        assert recipe.launch_external_process is False


class TestFedAvgRecipeKeyMetricVariants:
    """Test key_metric passthrough for NumPy FedAvg recipes."""

    def test_key_metric_passthrough_numpy(self, mock_file_system):
        key_metric = "val_loss"
        recipe = NumpyFedAvgRecipe(
            name="test_numpy_key_metric",
            min_clients=2,
            train_script="mock_train_script.py",
            key_metric=key_metric,
        )

        model_selector = get_model_selector(recipe)
        assert isinstance(model_selector, IntimeModelSelector)
        assert model_selector.key_metric == key_metric
