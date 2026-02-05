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

"""Tests for NumPy Logistic Regression recipe."""

from unittest.mock import patch

import pytest


@pytest.fixture
def mock_file_system():
    """Mock file system operations for all tests."""
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.path.isdir", return_value=True),
        patch("os.path.exists", return_value=True),
    ):
        yield


class TestFedAvgLrRecipe:
    """Test cases for FedAvgLrRecipe (NumPy Logistic Regression)."""

    def test_basic_initialization(self, mock_file_system):
        """Test FedAvgLrRecipe basic initialization."""
        from nvflare.app_common.np.recipes.lr.fedavg import FedAvgLrRecipe

        recipe = FedAvgLrRecipe(
            name="test_lr",
            train_script="train.py",
            min_clients=2,
            num_rounds=5,
            num_features=13,
        )

        assert recipe.job is not None

    def test_initial_ckpt_accepted(self, mock_file_system):
        """Test that initial_ckpt parameter is accepted."""
        from nvflare.app_common.np.recipes.lr.fedavg import FedAvgLrRecipe

        recipe = FedAvgLrRecipe(
            name="test_lr_ckpt",
            train_script="train.py",
            min_clients=2,
            num_rounds=5,
            num_features=13,
            initial_ckpt="/abs/path/to/lr_weights.npy",
        )

        assert recipe.job is not None

    def test_relative_path_rejected(self, mock_file_system):
        """Test that relative paths are rejected."""
        from nvflare.app_common.np.recipes.lr.fedavg import FedAvgLrRecipe

        with pytest.raises(ValueError, match="must be an absolute path"):
            FedAvgLrRecipe(
                name="test_lr",
                train_script="train.py",
                min_clients=2,
                num_rounds=5,
                num_features=13,
                initial_ckpt="relative/path/model.npy",
            )

    def test_custom_damping_factor(self, mock_file_system):
        """Test that custom damping_factor is accepted."""
        from nvflare.app_common.np.recipes.lr.fedavg import FedAvgLrRecipe

        recipe = FedAvgLrRecipe(
            name="test_lr_damping",
            train_script="train.py",
            min_clients=2,
            num_rounds=5,
            num_features=13,
            damping_factor=0.5,
        )

        assert recipe.job is not None
        assert recipe.damping_factor == 0.5

    def test_with_train_args(self, mock_file_system):
        """Test that train_args parameter is accepted."""
        from nvflare.app_common.np.recipes.lr.fedavg import FedAvgLrRecipe

        recipe = FedAvgLrRecipe(
            name="test_lr_args",
            train_script="train.py",
            min_clients=2,
            num_rounds=5,
            num_features=13,
            train_args="--data_root /tmp/data --batch_size 32",
        )

        assert recipe.job is not None
        assert recipe.train_args == "--data_root /tmp/data --batch_size 32"

    def test_with_external_process(self, mock_file_system):
        """Test launch_external_process parameter."""
        from nvflare.app_common.np.recipes.lr.fedavg import FedAvgLrRecipe

        recipe = FedAvgLrRecipe(
            name="test_lr_external",
            train_script="train.py",
            min_clients=2,
            num_rounds=5,
            num_features=13,
            launch_external_process=True,
            command="python3 -u",
        )

        assert recipe.job is not None
        assert recipe.launch_external_process is True

    def test_invalid_min_clients_rejected(self, mock_file_system):
        """Test that invalid min_clients (zero or negative) is rejected."""
        from pydantic import ValidationError

        from nvflare.app_common.np.recipes.lr.fedavg import FedAvgLrRecipe

        with pytest.raises(ValidationError):
            FedAvgLrRecipe(
                name="test_lr",
                train_script="train.py",
                min_clients=0,  # Invalid: must be positive
                num_rounds=5,
                num_features=13,
            )

    def test_invalid_num_features_rejected(self, mock_file_system):
        """Test that invalid num_features (zero or negative) is rejected."""
        from pydantic import ValidationError

        from nvflare.app_common.np.recipes.lr.fedavg import FedAvgLrRecipe

        with pytest.raises(ValidationError):
            FedAvgLrRecipe(
                name="test_lr",
                train_script="train.py",
                min_clients=2,
                num_rounds=5,
                num_features=0,  # Invalid: must be positive
            )

    def test_recipe_attributes_set_correctly(self, mock_file_system):
        """Test that recipe attributes are set correctly from parameters."""
        from nvflare.app_common.np.recipes.lr.fedavg import FedAvgLrRecipe

        recipe = FedAvgLrRecipe(
            name="test_lr_attrs",
            train_script="my_train.py",
            min_clients=4,
            num_rounds=10,
            num_features=20,
            damping_factor=0.9,
        )

        assert recipe.name == "test_lr_attrs"
        assert recipe.min_clients == 4
        assert recipe.num_rounds == 10
        assert recipe.num_features == 20
        assert recipe.damping_factor == 0.9
        assert recipe.train_script == "my_train.py"
