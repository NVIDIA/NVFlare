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

"""Integration tests for XGBBaggingRecipe (tree-based, supports bagging and cyclic modes).

These are smoke tests to verify that XGBBaggingRecipe works without errors.
Tests verify:
- Recipe can be instantiated with both bagging and cyclic modes
- Job completes successfully
- Refactored API (add_dataloader, to_clients pattern) works correctly

Tests do NOT verify:
- Model accuracy or convergence
- Detailed XGBoost training behavior
- Comparison between bagging and cyclic performance

NOTE: These tests are currently NOT triggered by any automated test suite.
They require XGBoost and use synthetic data for testing.

To run manually:
    cd tests/integration_test
    pytest test_xgb_bagging_recipe.py -v
"""

import os
import tempfile

import numpy as np
import pytest
import xgboost as xgb

from nvflare.app_opt.xgboost.data_loader import XGBDataLoader
from nvflare.app_opt.xgboost.recipes import XGBBaggingRecipe
from nvflare.recipe import SimEnv


class MockXGBDataLoader(XGBDataLoader):
    """Mock data loader for testing that generates synthetic data."""

    def __init__(self, n_samples=100, n_features=10):
        self.n_samples = n_samples
        self.n_features = n_features

    def load_data(self):
        """Generate synthetic binary classification data."""
        # Generate random features
        X_train = np.random.rand(self.n_samples, self.n_features)
        y_train = np.random.randint(0, 2, self.n_samples)

        X_val = np.random.rand(self.n_samples // 4, self.n_features)
        y_val = np.random.randint(0, 2, self.n_samples // 4)

        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        return dtrain, dval


class TestXGBBaggingRecipe:
    """Smoke tests for XGBBaggingRecipe.

    These tests verify that the recipe can be instantiated and run without errors
    in both bagging and cyclic modes. They use synthetic data and minimal training
    rounds for speed.
    """

    def test_bagging_mode(self):
        """Test bagging mode (federated Random Forest) completes successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = SimEnv(num_clients=3, workspace_root=os.path.join(tmpdir, "test_bagging"))

            recipe = XGBBaggingRecipe(
                name="test_bagging",
                min_clients=3,
                training_mode="bagging",
                num_rounds=1,  # Default for bagging
                num_local_parallel_tree=3,
                local_subsample=0.8,
                max_depth=3,
                learning_rate=0.1,
            )

            # Add mock data loaders using new API
            dataloader = MockXGBDataLoader(n_samples=50, n_features=5)
            # Configure per-site data loaders
            per_site_config = {f"site-{site_id}": {"data_loader": dataloader} for site_id in range(1, 4)}
            recipe.per_site_config = per_site_config
            recipe.job = recipe.configure()

            # Run and verify completion
            run = recipe.execute(env)
            assert run.get_result() is not None
            assert os.path.exists(run.get_result())

    def test_cyclic_mode(self):
        """Test cyclic mode (sequential training) completes successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = SimEnv(num_clients=3, workspace_root=os.path.join(tmpdir, "test_cyclic"))

            recipe = XGBBaggingRecipe(
                name="test_cyclic",
                min_clients=3,
                training_mode="cyclic",
                num_rounds=3,  # Few rounds for speed
                max_depth=3,
                learning_rate=0.1,
            )

            # Add mock data loaders
            per_site_config = {
                f"site-{site_id}": {"data_loader": MockXGBDataLoader(n_samples=50, n_features=5)}
                for site_id in range(1, 4)
            }
            recipe.per_site_config = per_site_config
            recipe.job = recipe.configure()

            # Run and verify completion
            run = recipe.execute(env)
            assert run.get_result() is not None
            assert os.path.exists(run.get_result())

    def test_default_num_rounds(self):
        """Test that num_rounds defaults correctly based on training_mode."""
        # Bagging should default to 1 round
        recipe_bagging = XGBBaggingRecipe(
            name="test_default_bagging",
            min_clients=2,
            training_mode="bagging",
        )
        assert recipe_bagging.num_rounds == 1

        # Cyclic should default to 100 rounds
        recipe_cyclic = XGBBaggingRecipe(
            name="test_default_cyclic",
            min_clients=2,
            training_mode="cyclic",
        )
        assert recipe_cyclic.num_rounds == 100

    def test_invalid_training_mode_raises_error(self):
        """Test that invalid training_mode raises ValueError."""
        with pytest.raises(ValueError, match="training_mode must be 'bagging' or 'cyclic'"):
            XGBBaggingRecipe(
                name="test_invalid",
                min_clients=2,
                training_mode="invalid_mode",
            )

    def test_uniform_lr_mode(self):
        """Test bagging with uniform learning rate mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = SimEnv(num_clients=2, workspace_root=os.path.join(tmpdir, "test_uniform_lr"))

            recipe = XGBBaggingRecipe(
                name="test_uniform_lr",
                min_clients=2,
                training_mode="bagging",
                lr_mode="uniform",
                num_rounds=1,
            )

            # Add data loaders
            for site_id in range(1, 3):
                dataloader = MockXGBDataLoader(n_samples=50, n_features=5)
                recipe.add_dataloader(dataloader, site_name=f"site-{site_id}")

            run = recipe.execute(env)
            assert run.get_result() is not None

    def test_scaled_lr_mode(self):
        """Test bagging with scaled learning rate mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = SimEnv(num_clients=2, workspace_root=os.path.join(tmpdir, "test_scaled_lr"))

            recipe = XGBBaggingRecipe(
                name="test_scaled_lr",
                min_clients=2,
                training_mode="bagging",
                lr_mode="scaled",
                num_rounds=1,
            )

            # Add data loaders and customize lr_scale per client
            for site_id in range(1, 3):
                dataloader = MockXGBDataLoader(n_samples=50, n_features=5)
                recipe.add_dataloader(dataloader, site_name=f"site-{site_id}")
                # Customize lr_scale for this client
                recipe.add_executor_to_client(f"site-{site_id}", lr_scale=0.5)

            run = recipe.execute(env)
            assert run.get_result() is not None

    def test_add_dataloader_to_all_clients(self):
        """Test adding the same dataloader to all clients at once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = SimEnv(num_clients=3, workspace_root=os.path.join(tmpdir, "test_all_clients"))

            recipe = XGBBaggingRecipe(
                name="test_all_clients",
                min_clients=3,
                training_mode="bagging",
                num_rounds=1,
            )

            # Add dataloader to all clients at once (no site_name specified)
            dataloader = MockXGBDataLoader(n_samples=50, n_features=5)
            recipe.add_dataloader(dataloader)  # Will be added to all clients

            run = recipe.execute(env)
            assert run.get_result() is not None

    def test_custom_xgb_params(self):
        """Test that custom XGBoost parameters work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = SimEnv(num_clients=2, workspace_root=os.path.join(tmpdir, "test_custom_params"))

            recipe = XGBBaggingRecipe(
                name="test_custom_params",
                min_clients=2,
                training_mode="bagging",
                num_rounds=1,
                max_depth=5,
                learning_rate=0.05,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                nthread=4,
            )

            # Verify params are stored
            assert recipe.max_depth == 5
            assert recipe.learning_rate == 0.05
            assert recipe.objective == "binary:logistic"

            # Add data loaders and run
            dataloader = MockXGBDataLoader(n_samples=50, n_features=5)
            recipe.add_dataloader(dataloader)

            run = recipe.execute(env)
            assert run.get_result() is not None

    def test_multiple_clients_bagging(self):
        """Test bagging mode works with many clients (5)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            num_clients = 5
            env = SimEnv(num_clients=num_clients, workspace_root=os.path.join(tmpdir, "test_multi_bagging"))

            recipe = XGBBaggingRecipe(
                name="test_multi_bagging",
                min_clients=num_clients,
                training_mode="bagging",
                num_rounds=1,
                num_local_parallel_tree=2,
            )

            # Add data loaders for all clients
            dataloader = MockXGBDataLoader(n_samples=30, n_features=5)
            recipe.add_dataloader(dataloader)

            run = recipe.execute(env)
            assert run.get_result() is not None
            assert os.path.exists(run.get_result())

    def test_refactored_api_consistency(self):
        """Test that the new refactored API (to_clients, add_dataloader) is working."""
        recipe = XGBBaggingRecipe(
            name="test_api",
            min_clients=2,
            training_mode="bagging",
            num_rounds=1,
        )

        # Verify executor is added to all clients via configure()
        # The job should already have the executor configured for all clients
        assert recipe.job is not None

        # Verify add_dataloader method exists and works
        dataloader = MockXGBDataLoader()
        result = recipe.add_dataloader(dataloader, site_name="site-1")
        assert result == recipe  # Should return self for chaining

        # Verify add_dataloader without site_name works
        result = recipe.add_dataloader(dataloader)
        assert result == recipe
