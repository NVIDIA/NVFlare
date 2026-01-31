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

"""Tests for FedOptRecipe with initial_ckpt support."""

from unittest.mock import patch

import pytest
import torch.nn as nn


class SimpleTestModel(nn.Module):
    """A simple PyTorch model for testing purposes."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(10, 10)

    def forward(self, x):
        return self.lin(x)


@pytest.fixture
def mock_file_system():
    """Mock file system operations for all tests."""
    with patch("os.path.isfile", return_value=True), patch("os.path.exists", return_value=True):
        yield


@pytest.fixture
def simple_model():
    """Create a simple test model."""
    return SimpleTestModel()


@pytest.fixture
def base_recipe_params():
    """Base parameters for creating FedOptRecipe instances."""
    return {
        "train_script": "mock_train_script.py",
        "train_args": "--epochs 10",
        "min_clients": 2,
        "num_rounds": 5,
    }


class TestPTFedOptRecipe:
    """Test cases for PyTorch FedOptRecipe."""

    def test_basic_initialization(self, mock_file_system, base_recipe_params, simple_model):
        """Test PT FedOptRecipe basic initialization."""
        from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe

        recipe = FedOptRecipe(
            name="test_fedopt",
            initial_model=simple_model,
            **base_recipe_params,
        )

        assert recipe.name == "test_fedopt"
        assert recipe.initial_model == simple_model
        assert recipe.job is not None

    def test_initial_ckpt_parameter_accepted(self, mock_file_system, base_recipe_params, simple_model):
        """Test that initial_ckpt parameter is accepted."""
        from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe

        recipe = FedOptRecipe(
            name="test_fedopt_ckpt",
            initial_model=simple_model,
            initial_ckpt="/abs/path/to/model.pt",
            **base_recipe_params,
        )

        assert recipe.initial_ckpt == "/abs/path/to/model.pt"

    def test_dict_model_config_accepted(self, mock_file_system, base_recipe_params):
        """Test that dict model config is accepted."""
        from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe

        model_config = {
            "path": "my_module.models.SimpleNet",
            "args": {"input_size": 10},
        }
        recipe = FedOptRecipe(
            name="test_fedopt_dict",
            initial_model=model_config,
            **base_recipe_params,
        )

        assert recipe.initial_model == model_config

    def test_with_optimizer_args(self, mock_file_system, base_recipe_params, simple_model):
        """Test FedOptRecipe with optimizer arguments."""
        from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe

        optimizer_args = {
            "path": "torch.optim.SGD",
            "args": {"lr": 1.0, "momentum": 0.6},
        }
        recipe = FedOptRecipe(
            name="test_fedopt_optim",
            initial_model=simple_model,
            optimizer_args=optimizer_args,
            **base_recipe_params,
        )

        assert recipe.optimizer_args == optimizer_args


class TestTFFedOptRecipe:
    """Test cases for TensorFlow FedOptRecipe."""

    def test_basic_initialization(self, mock_file_system, base_recipe_params):
        """Test TF FedOptRecipe basic initialization."""
        pytest.importorskip("tensorflow")
        from nvflare.app_opt.tf.recipes.fedopt import FedOptRecipe

        recipe = FedOptRecipe(
            name="test_tf_fedopt",
            initial_model=None,
            **base_recipe_params,
        )

        assert recipe.name == "test_tf_fedopt"
        assert recipe.job is not None

    def test_initial_ckpt_parameter_accepted(self, mock_file_system, base_recipe_params):
        """Test that initial_ckpt parameter is accepted (TF can load without model)."""
        pytest.importorskip("tensorflow")
        from nvflare.app_opt.tf.recipes.fedopt import FedOptRecipe

        recipe = FedOptRecipe(
            name="test_tf_fedopt_ckpt",
            initial_model=None,
            initial_ckpt="/abs/path/to/model.h5",
            **base_recipe_params,
        )

        assert recipe.initial_ckpt == "/abs/path/to/model.h5"
