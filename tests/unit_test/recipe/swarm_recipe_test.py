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

"""Tests for Swarm Learning recipes."""

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def mock_file_system():
    """Mock file system operations for all tests."""
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.path.isdir", return_value=True),
        patch("os.path.exists", return_value=True),
    ):
        yield


@pytest.fixture
def simple_pt_model():
    """Create a simple PyTorch model for testing."""
    import torch.nn as nn

    return nn.Linear(10, 2)


class TestSimpleSwarmLearningRecipe:
    """Test cases for SimpleSwarmLearningRecipe."""

    def test_import_from_new_location(self, mock_file_system, simple_pt_model):
        """Test importing from new location (app_opt/pt/recipes)."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        recipe = SimpleSwarmLearningRecipe(
            name="test_swarm",
            model=simple_pt_model,
            num_rounds=5,
            train_script="train.py",
        )

        assert recipe.job is not None

    def test_import_from_old_location_backward_compat(self, mock_file_system, simple_pt_model):
        """Test importing from old location (backward compatibility)."""
        from nvflare.app_common.ccwf.recipes.swarm import SimpleSwarmLearningRecipe

        recipe = SimpleSwarmLearningRecipe(
            name="test_swarm",
            model=simple_pt_model,
            num_rounds=5,
            train_script="train.py",
        )

        assert recipe.job is not None

    def test_initial_ckpt_accepted(self, mock_file_system, simple_pt_model):
        """Test that initial_ckpt parameter is accepted."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        recipe = SimpleSwarmLearningRecipe(
            name="test_swarm_ckpt",
            model=simple_pt_model,
            num_rounds=5,
            train_script="train.py",
            initial_ckpt="/abs/path/to/model.pt",
        )

        assert recipe.job is not None

    def test_relative_path_accepted_if_exists(self, mock_file_system, simple_pt_model):
        """Test that existing relative paths are accepted and bundled."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        # This should not raise since relative paths are now supported
        recipe = SimpleSwarmLearningRecipe(
            name="test_swarm",
            model=simple_pt_model,
            num_rounds=5,
            train_script="train.py",
            initial_ckpt="relative/path/model.pt",
        )
        assert recipe is not None

    def test_cross_site_eval_option(self, mock_file_system, simple_pt_model):
        """Test with cross-site evaluation enabled."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        recipe = SimpleSwarmLearningRecipe(
            name="test_swarm_cse",
            model=simple_pt_model,
            num_rounds=5,
            train_script="train.py",
            do_cross_site_eval=True,
            cross_site_eval_timeout=600,
        )

        assert recipe.job is not None

    def test_dict_model_config_accepted(self, mock_file_system):
        """Test that dict model config is accepted."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        recipe = SimpleSwarmLearningRecipe(
            name="test_swarm_dict",
            model={"path": "torch.nn.Linear", "args": {"in_features": 10, "out_features": 2}},
            num_rounds=5,
            train_script="train.py",
        )

        assert recipe.job is not None

    def test_dict_model_config_with_ckpt(self, mock_file_system):
        """Test dict model config with initial checkpoint."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        recipe = SimpleSwarmLearningRecipe(
            name="test_swarm_dict_ckpt",
            model={"path": "torch.nn.Linear", "args": {"in_features": 10, "out_features": 2}},
            num_rounds=5,
            train_script="train.py",
            initial_ckpt="/abs/path/to/model.pt",
        )

        assert recipe.job is not None

    def test_dict_model_missing_path_rejected(self, mock_file_system):
        """Test that dict model without 'path' key is rejected."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        with pytest.raises(ValueError, match="must contain 'path' key"):
            SimpleSwarmLearningRecipe(
                name="test_swarm_bad_dict",
                model={"args": {"in_features": 10}},  # Missing 'path'
                num_rounds=5,
                train_script="train.py",
            )

    def test_train_args_reserved_keys_rejected(self, mock_file_system, simple_pt_model):
        """Test that train_args with reserved keys are rejected."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        with pytest.raises(ValueError, match="reserved keys"):
            SimpleSwarmLearningRecipe(
                name="test_swarm_bad_args",
                model=simple_pt_model,
                num_rounds=5,
                train_script="train.py",
                train_args={"script": "other.py"},  # 'script' is reserved
            )

    def test_train_args_valid_keys_accepted(self, mock_file_system, simple_pt_model):
        """Test that valid train_args are accepted."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        recipe = SimpleSwarmLearningRecipe(
            name="test_swarm_args",
            model=simple_pt_model,
            num_rounds=5,
            train_script="train.py",
            train_args={"script_args": "--batch_size 32"},  # valid key
        )

        assert recipe.job is not None
