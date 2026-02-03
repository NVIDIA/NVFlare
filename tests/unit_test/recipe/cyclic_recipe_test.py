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

"""Tests for CyclicRecipe and framework-specific variants with initial_ckpt support."""

from unittest.mock import patch

import pytest
import torch.nn as nn

from nvflare.recipe.cyclic import CyclicRecipe as BaseCyclicRecipe


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
    """Base parameters for creating CyclicRecipe instances."""
    return {
        "train_script": "mock_train_script.py",
        "train_args": "--epochs 10",
        "min_clients": 2,
        "num_rounds": 5,
    }


class TestBaseCyclicRecipe:
    """Test cases for base CyclicRecipe class.

    Note: Base CyclicRecipe doesn't directly support nn.Module or dict config.
    Use framework-specific recipes (PTCyclicRecipe, TFCyclicRecipe) for those.
    """

    def test_initial_ckpt_must_be_absolute(self):
        """Test that relative paths are rejected (no mock - validation must run)."""
        # Don't use mock_file_system fixture - we need real os.path.isabs check
        with patch("os.path.isfile", return_value=True), patch("os.path.exists", return_value=True):
            with pytest.raises(ValueError, match="must be an absolute path"):
                BaseCyclicRecipe(
                    name="test_relative",
                    initial_model=None,
                    initial_ckpt="relative/path/model.pt",
                    train_script="/abs/train.py",
                    min_clients=2,
                    num_rounds=2,
                )

    def test_requires_model_or_checkpoint(self, base_recipe_params):
        """Test that at least initial_model or initial_ckpt must be provided."""
        with pytest.raises(ValueError, match="Must provide either initial_model or initial_ckpt"):
            BaseCyclicRecipe(
                name="test_no_model",
                initial_model=None,
                initial_ckpt=None,
                **base_recipe_params,
            )


class TestPTCyclicRecipe:
    """Test cases for PyTorch CyclicRecipe."""

    def test_pt_cyclic_initial_ckpt(self, mock_file_system, base_recipe_params, simple_model):
        """Test PT CyclicRecipe with initial_ckpt."""
        from nvflare.app_opt.pt.recipes.cyclic import CyclicRecipe as PTCyclicRecipe

        recipe = PTCyclicRecipe(
            name="test_pt_cyclic",
            initial_model=simple_model,
            initial_ckpt="/abs/path/to/model.pt",
            **base_recipe_params,
        )

        assert recipe.name == "test_pt_cyclic"
        assert recipe.job is not None


class TestTFCyclicRecipe:
    """Test cases for TensorFlow CyclicRecipe."""

    def test_tf_cyclic_initial_ckpt(self, mock_file_system, base_recipe_params):
        """Test TF CyclicRecipe with initial_ckpt (TF can load without model)."""
        pytest.importorskip("tensorflow")
        from nvflare.app_opt.tf.recipes.cyclic import CyclicRecipe as TFCyclicRecipe

        recipe = TFCyclicRecipe(
            name="test_tf_cyclic",
            initial_model=None,
            initial_ckpt="/abs/path/to/model.h5",
            **base_recipe_params,
        )

        assert recipe.name == "test_tf_cyclic"
        assert recipe.job is not None
