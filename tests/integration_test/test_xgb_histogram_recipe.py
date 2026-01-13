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

"""Integration tests for XGBHistogramRecipe.

These are smoke tests to verify that XGBHistogramRecipe works without errors.
Tests verify:
- Recipe can be instantiated with different algorithms
- Job completes successfully
- TensorBoard tracking is configured

Tests do NOT verify:
- Model accuracy or convergence
- Detailed XGBoost training behavior
- TensorBoard file contents

NOTE: These tests are currently NOT triggered by any automated test suite.
They require XGBoost and use synthetic data for testing.

To run manually:
    cd tests/integration_test
    pytest test_xgb_histogram_recipe.py -v
"""

import os
import tempfile

import numpy as np
import pytest
import xgboost as xgb

from nvflare.app_opt.xgboost.data_loader import XGBDataLoader
from nvflare.app_opt.xgboost.recipes import XGBHistogramRecipe
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


class TestXGBHistogramRecipe:
    """Smoke tests for XGBHistogramRecipe.

    These tests verify that the recipe can be instantiated and run without errors.
    They use synthetic data and minimal training rounds for speed.
    """

    def test_histogram_v2_algorithm(self):
        """Test histogram_v2 algorithm (recommended) completes successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = SimEnv(num_clients=2, workspace_root=os.path.join(tmpdir, "test_histogram_v2"))

            recipe = XGBHistogramRecipe(
                name="test_histogram_v2",
                min_clients=2,
                num_rounds=2,  # Minimal rounds for speed
                algorithm="histogram_v2",
                early_stopping_rounds=1,
                xgb_params={
                    "max_depth": 3,
                    "eta": 0.1,
                    "objective": "binary:logistic",
                    "eval_metric": "auc",
                },
            )

            # Add mock data loaders to clients
            for site_id in range(1, 3):
                data_loader = MockXGBDataLoader(n_samples=50, n_features=5)
                recipe.add_to_client(f"site-{site_id}", data_loader)

            # Run and verify completion
            run = recipe.execute(env)
            assert run.get_result() is not None
            assert os.path.exists(run.get_result())

    def test_histogram_algorithm(self):
        """Test original histogram algorithm completes successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = SimEnv(num_clients=2, workspace_root=os.path.join(tmpdir, "test_histogram"))

            recipe = XGBHistogramRecipe(
                name="test_histogram",
                min_clients=2,
                num_rounds=2,
                algorithm="histogram",
                early_stopping_rounds=1,
                xgb_params={
                    "max_depth": 3,
                    "eta": 0.1,
                    "objective": "binary:logistic",
                    "eval_metric": "auc",
                },
            )

            # Add mock data loaders
            for site_id in range(1, 3):
                data_loader = MockXGBDataLoader(n_samples=50, n_features=5)
                recipe.add_to_client(f"site-{site_id}", data_loader)

            # Run and verify completion
            run = recipe.execute(env)
            assert run.get_result() is not None
            assert os.path.exists(run.get_result())

    def test_invalid_algorithm_raises_error(self):
        """Test that invalid algorithm raises ValueError."""
        with pytest.raises(ValueError, match="algorithm must be 'histogram' or 'histogram_v2'"):
            XGBHistogramRecipe(
                name="test_invalid",
                min_clients=2,
                num_rounds=2,
                algorithm="invalid_algo",
            )

    def test_custom_xgb_params(self):
        """Test that custom XGBoost parameters are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = SimEnv(num_clients=2, workspace_root=os.path.join(tmpdir, "test_custom_params"))

            custom_params = {
                "max_depth": 5,
                "eta": 0.05,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "nthread": 4,
            }

            recipe = XGBHistogramRecipe(
                name="test_custom_params",
                min_clients=2,
                num_rounds=1,
                algorithm="histogram_v2",
                xgb_params=custom_params,
            )

            # Verify params are stored
            assert recipe.xgb_params == custom_params

            # Add data loaders and run
            for site_id in range(1, 3):
                data_loader = MockXGBDataLoader(n_samples=50, n_features=5)
                recipe.add_to_client(f"site-{site_id}", data_loader)

            run = recipe.execute(env)
            assert run.get_result() is not None

    def test_multiple_clients(self):
        """Test recipe works with more than 2 clients."""
        with tempfile.TemporaryDirectory() as tmpdir:
            num_clients = 5
            env = SimEnv(num_clients=num_clients, workspace_root=os.path.join(tmpdir, "test_multi_client"))

            recipe = XGBHistogramRecipe(
                name="test_multi_client",
                min_clients=num_clients,
                num_rounds=1,
                algorithm="histogram_v2",
            )

            # Add data loaders for all clients
            for site_id in range(1, num_clients + 1):
                data_loader = MockXGBDataLoader(n_samples=30, n_features=5)
                recipe.add_to_client(f"site-{site_id}", data_loader)

            run = recipe.execute(env)
            assert run.get_result() is not None
            assert os.path.exists(run.get_result())

    def test_tensorboard_tracking_configured(self):
        """Test that TensorBoard tracking components are configured."""
        recipe = XGBHistogramRecipe(
            name="test_tb",
            min_clients=2,
            num_rounds=1,
            algorithm="histogram_v2",
        )

        # Verify TensorBoard receiver is added to server
        server_components = recipe.job.get_server_components()
        assert "tb_receiver" in server_components

        # Add a client and verify TensorBoard writer is configured
        data_loader = MockXGBDataLoader()
        recipe.add_to_client("site-1", data_loader)

        # Check that metrics_writer and event_to_fed are added
        client_components = recipe.job.get_client_components("site-1")
        assert "metrics_writer" in client_components
        assert "event_to_fed" in client_components
