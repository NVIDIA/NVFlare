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
