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

from nvflare.app_opt.pt.job_config.model import PTModel
from nvflare.fuel.utils.constants import FrameworkType
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
    """Test cases for base CyclicRecipe class."""

    def test_initial_ckpt_must_exist_for_relative_path(self):
        """Test that non-existent relative paths are rejected (no mock - validation must run)."""
        with pytest.raises(ValueError, match="does not exist locally"):
            BaseCyclicRecipe(
                name="test_relative",
                model=None,
                initial_ckpt="relative/path/model.pt",
                train_script="/abs/train.py",
                min_clients=2,
                num_rounds=2,
            )

    def test_requires_model_or_checkpoint(self, base_recipe_params):
        """Test that at least model or initial_ckpt must be provided."""
        with pytest.raises(ValueError, match="Must provide either model or initial_ckpt"):
            BaseCyclicRecipe(
                name="test_no_model",
                model=None,
                initial_ckpt=None,
                **base_recipe_params,
            )

    def test_supports_dict_model_for_pytorch(self, mock_file_system, base_recipe_params):
        """Base CyclicRecipe should accept dict model config for PyTorch framework."""
        recipe = BaseCyclicRecipe(
            name="test_base_pt_dict",
            model={"class_path": "torch.nn.Linear", "args": {"in_features": 10, "out_features": 2}},
            framework=FrameworkType.PYTORCH,
            **base_recipe_params,
        )

        server_app = recipe.job._deploy_map.get("server")
        components = server_app.app_config.components
        assert "persistor" in components
        assert "locator" in components

    def test_rejects_pytorch_checkpoint_without_model(self, mock_file_system, base_recipe_params):
        """PyTorch checkpoints need model architecture and must fail fast when model is missing."""
        with pytest.raises(ValueError, match="FrameworkType.PYTORCH requires 'model'"):
            BaseCyclicRecipe(
                name="test_base_pt_ckpt_no_model",
                model=None,
                initial_ckpt="/abs/path/to/model.pt",
                framework=FrameworkType.PYTORCH,
                **base_recipe_params,
            )

    def test_rejects_ckpt_only_for_default_framework(self, mock_file_system, base_recipe_params):
        """Fail fast for ckpt-only usage when no supported framework/wrapper is provided."""
        with pytest.raises(ValueError, match="Unsupported framework"):
            BaseCyclicRecipe(
                name="test_ckpt_only_default_framework",
                model=None,
                initial_ckpt="/abs/path/to/model.pt",
                **base_recipe_params,
            )

    def test_applies_initial_ckpt_to_wrapper_model(self, mock_file_system, base_recipe_params, simple_model):
        """When wrapper model is used, recipe-level initial_ckpt should be applied to persistor."""
        recipe = BaseCyclicRecipe(
            name="test_wrapper_ckpt",
            model=PTModel(model=simple_model),
            initial_ckpt="/abs/path/to/model.pt",
            framework=FrameworkType.PYTORCH,
            **base_recipe_params,
        )

        server_app = recipe.job._deploy_map.get("server")
        persistor = server_app.app_config.components.get("persistor")
        assert persistor is not None
        assert getattr(persistor, "source_ckpt_file_full_name", None) == "/abs/path/to/model.pt"

    def test_rejects_unsupported_framework_without_wrapper(self, mock_file_system, base_recipe_params):
        """Fail fast for unsupported base framework/model persistence combinations."""
        with pytest.raises(ValueError, match="Unsupported framework"):
            BaseCyclicRecipe(
                name="test_unsupported_framework",
                model={"class_path": "torch.nn.Linear", "args": {"in_features": 10, "out_features": 2}},
                framework=FrameworkType.NUMPY,
                **base_recipe_params,
            )


class TestPTCyclicRecipe:
    """Test cases for PyTorch CyclicRecipe."""

    def test_pt_cyclic_initial_ckpt(self, mock_file_system, base_recipe_params, simple_model):
        """Test PT CyclicRecipe with initial_ckpt."""
        from nvflare.app_opt.pt.recipes.cyclic import CyclicRecipe as PTCyclicRecipe

        recipe = PTCyclicRecipe(
            name="test_pt_cyclic",
            model=simple_model,
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
            model=None,
            initial_ckpt="/abs/path/to/model.h5",
            **base_recipe_params,
        )

        assert recipe.name == "test_tf_cyclic"
        assert recipe.job is not None
