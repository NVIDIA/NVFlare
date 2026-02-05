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

"""Tests for Sklearn recipes (FedAvg, KMeans, SVM) with initial_ckpt support."""

from unittest.mock import patch

import pytest


@pytest.fixture
def mock_file_system():
    """Mock file system operations for all tests."""
    with patch("os.path.isfile", return_value=True), patch("os.path.exists", return_value=True):
        yield


@pytest.fixture
def base_recipe_params():
    """Base parameters for creating SklearnFedAvgRecipe instances."""
    return {
        "train_script": "mock_train_script.py",
        "train_args": "--epochs 10",
        "min_clients": 2,
        "num_rounds": 5,
    }


class TestSklearnFedAvgRecipe:
    """Test cases for SklearnFedAvgRecipe."""

    def test_basic_initialization(self, mock_file_system, base_recipe_params):
        """Test SklearnFedAvgRecipe basic initialization."""
        from nvflare.app_opt.sklearn.recipes.fedavg import SklearnFedAvgRecipe

        model_params = {
            "n_classes": 2,
            "learning_rate": "constant",
            "eta0": 1e-4,
        }
        recipe = SklearnFedAvgRecipe(
            name="test_sklearn",
            model_params=model_params,
            **base_recipe_params,
        )

        assert recipe.name == "test_sklearn"
        assert recipe.job is not None

    def test_initial_ckpt_parameter_accepted(self, mock_file_system, base_recipe_params):
        """Test that initial_ckpt parameter is accepted."""
        from nvflare.app_opt.sklearn.recipes.fedavg import SklearnFedAvgRecipe

        model_params = {"n_classes": 2}
        recipe = SklearnFedAvgRecipe(
            name="test_sklearn_ckpt",
            model_params=model_params,
            initial_ckpt="/abs/path/to/model.joblib",
            **base_recipe_params,
        )

        # The initial_ckpt is passed to the persistor, not stored on recipe directly
        assert recipe.job is not None

    def test_initial_ckpt_only_without_model_params(self, mock_file_system, base_recipe_params):
        """Test that initial_ckpt works without model_params."""
        from nvflare.app_opt.sklearn.recipes.fedavg import SklearnFedAvgRecipe

        recipe = SklearnFedAvgRecipe(
            name="test_sklearn_ckpt_only",
            model_params=None,
            initial_ckpt="/abs/path/to/pretrained.joblib",
            **base_recipe_params,
        )

        assert recipe.job is not None

    def test_with_per_site_config(self, mock_file_system, base_recipe_params):
        """Test SklearnFedAvgRecipe with per-site configuration."""
        from nvflare.app_opt.sklearn.recipes.fedavg import SklearnFedAvgRecipe

        per_site_config = {
            "site-1": {"train_args": "--data /path/to/site1"},
            "site-2": {"train_args": "--data /path/to/site2"},
        }
        recipe = SklearnFedAvgRecipe(
            name="test_sklearn_per_site",
            model_params={"n_classes": 2},
            per_site_config=per_site_config,
            **base_recipe_params,
        )

        assert recipe.per_site_config == per_site_config


class TestKMeansFedAvgRecipe:
    """Test cases for KMeansFedAvgRecipe with initial_ckpt support."""

    def test_basic_initialization(self, mock_file_system):
        """Test KMeansFedAvgRecipe basic initialization."""
        from nvflare.app_opt.sklearn.recipes.kmeans import KMeansFedAvgRecipe

        recipe = KMeansFedAvgRecipe(
            name="test_kmeans",
            n_clusters=3,
            train_script="train.py",
            min_clients=2,
            num_rounds=5,
        )

        assert recipe.name == "test_kmeans"
        assert recipe.job is not None

    def test_initial_ckpt_accepted(self, mock_file_system):
        """Test that initial_ckpt parameter is accepted."""
        from nvflare.app_opt.sklearn.recipes.kmeans import KMeansFedAvgRecipe

        recipe = KMeansFedAvgRecipe(
            name="test_kmeans_ckpt",
            n_clusters=3,
            train_script="train.py",
            min_clients=2,
            num_rounds=5,
            initial_ckpt="/abs/path/to/kmeans.joblib",
        )

        assert recipe.job is not None

    def test_relative_path_rejected(self, mock_file_system):
        """Test that relative paths are rejected."""
        from nvflare.app_opt.sklearn.recipes.kmeans import KMeansFedAvgRecipe

        with pytest.raises(ValueError, match="must be an absolute path"):
            KMeansFedAvgRecipe(
                name="test_kmeans",
                n_clusters=3,
                train_script="train.py",
                min_clients=2,
                num_rounds=5,
                initial_ckpt="relative/path/model.joblib",
            )


class TestSVMFedAvgRecipe:
    """Test cases for SVMFedAvgRecipe with initial_ckpt support."""

    def test_basic_initialization(self, mock_file_system):
        """Test SVMFedAvgRecipe basic initialization."""
        from nvflare.app_opt.sklearn.recipes.svm import SVMFedAvgRecipe

        recipe = SVMFedAvgRecipe(
            name="test_svm",
            train_script="train.py",
            min_clients=2,
            kernel="rbf",
        )

        assert recipe.name == "test_svm"
        assert recipe.job is not None

    def test_initial_ckpt_accepted(self, mock_file_system):
        """Test that initial_ckpt parameter is accepted."""
        from nvflare.app_opt.sklearn.recipes.svm import SVMFedAvgRecipe

        recipe = SVMFedAvgRecipe(
            name="test_svm_ckpt",
            train_script="train.py",
            min_clients=2,
            kernel="rbf",
            initial_ckpt="/abs/path/to/svm.joblib",
        )

        assert recipe.job is not None
