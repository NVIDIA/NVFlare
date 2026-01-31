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

"""Tests for ScaffoldRecipe with initial_ckpt support."""

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
    """Base parameters for creating ScaffoldRecipe instances."""
    return {
        "train_script": "mock_train_script.py",
        "train_args": "--epochs 10",
        "min_clients": 2,
        "num_rounds": 5,
    }


class TestPTScaffoldRecipe:
    """Test cases for PyTorch ScaffoldRecipe."""

    def test_basic_initialization(self, mock_file_system, base_recipe_params, simple_model):
        """Test PT ScaffoldRecipe basic initialization."""
        from nvflare.app_opt.pt.recipes.scaffold import ScaffoldRecipe

        recipe = ScaffoldRecipe(
            name="test_scaffold",
            initial_model=simple_model,
            **base_recipe_params,
        )

        assert recipe.name == "test_scaffold"
        assert recipe.initial_model == simple_model
        assert recipe.job is not None

    def test_initial_ckpt_parameter_accepted(self, mock_file_system, base_recipe_params, simple_model):
        """Test that initial_ckpt parameter is accepted."""
        from nvflare.app_opt.pt.recipes.scaffold import ScaffoldRecipe

        recipe = ScaffoldRecipe(
            name="test_scaffold_ckpt",
            initial_model=simple_model,
            initial_ckpt="/abs/path/to/model.pt",
            **base_recipe_params,
        )

        assert recipe.initial_ckpt == "/abs/path/to/model.pt"

    def test_dict_model_config_accepted(self, mock_file_system, base_recipe_params):
        """Test that dict model config is accepted."""
        from nvflare.app_opt.pt.recipes.scaffold import ScaffoldRecipe

        model_config = {
            "path": "my_module.models.SimpleNet",
            "args": {"input_size": 10},
        }
        recipe = ScaffoldRecipe(
            name="test_scaffold_dict",
            initial_model=model_config,
            **base_recipe_params,
        )

        assert recipe.initial_model == model_config


class TestTFScaffoldRecipe:
    """Test cases for TensorFlow ScaffoldRecipe."""

    def test_basic_initialization(self, mock_file_system, base_recipe_params):
        """Test TF ScaffoldRecipe basic initialization."""
        pytest.importorskip("tensorflow")
        from nvflare.app_opt.tf.recipes.scaffold import ScaffoldRecipe

        recipe = ScaffoldRecipe(
            name="test_tf_scaffold",
            initial_model=None,
            **base_recipe_params,
        )

        assert recipe.name == "test_tf_scaffold"
        assert recipe.job is not None

    def test_initial_ckpt_parameter_accepted(self, mock_file_system, base_recipe_params):
        """Test that initial_ckpt parameter is accepted (TF can load without model)."""
        pytest.importorskip("tensorflow")
        from nvflare.app_opt.tf.recipes.scaffold import ScaffoldRecipe

        recipe = ScaffoldRecipe(
            name="test_tf_scaffold_ckpt",
            initial_model=None,
            initial_ckpt="/abs/path/to/model.h5",
            **base_recipe_params,
        )

        assert recipe.initial_ckpt == "/abs/path/to/model.h5"
