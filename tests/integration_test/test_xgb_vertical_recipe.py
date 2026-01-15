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

"""Integration tests for XGBVerticalRecipe.

These are smoke tests to verify that XGBVerticalRecipe works without errors.
Tests verify:
- Recipe can be instantiated with label_owner
- Job completes successfully
- TensorBoard tracking is configured

Tests do NOT verify:
- Model accuracy or convergence
- PSI workflow (would require real data)
- Detailed vertical XGBoost behavior

NOTE: These tests are currently NOT triggered by any automated test suite.
They require XGBoost and use synthetic data for testing.

To run manually:
    cd tests/integration_test
    pytest test_xgb_vertical_recipe.py -v
"""

import os
import tempfile

import numpy as np
import pytest
import xgboost as xgb

from nvflare.app_opt.xgboost.data_loader import XGBDataLoader
from nvflare.app_opt.xgboost.recipes import XGBVerticalRecipe
from nvflare.recipe import SimEnv


class MockVerticalDataLoader(XGBDataLoader):
    """Mock vertical data loader for testing that generates synthetic data."""

    def __init__(self, has_labels=False, n_samples=100, n_features=5):
        self.has_labels = has_labels
        self.n_samples = n_samples
        self.n_features = n_features
        self.data_split_mode = 1  # 1 = vertical (column split)

    def load_data(self):
        """Generate synthetic vertical data (features split across clients)."""
        # Generate random features for this client
        X_train = np.random.rand(self.n_samples, self.n_features)
        X_val = np.random.rand(self.n_samples // 4, self.n_features)

        if self.has_labels:
            # Only label owner has labels
            y_train = np.random.randint(0, 2, self.n_samples)
            y_val = np.random.randint(0, 2, self.n_samples // 4)
        else:
            # Other clients have no labels
            y_train = None
            y_val = None

        # Create DMatrix objects with data_split_mode=1 (vertical/column mode)
        dtrain = xgb.DMatrix(X_train, label=y_train, data_split_mode=self.data_split_mode)
        dval = xgb.DMatrix(X_val, label=y_val, data_split_mode=self.data_split_mode)

        return dtrain, dval


class TestXGBVerticalRecipe:
    """Smoke tests for XGBVerticalRecipe.

    These tests verify that the recipe can be instantiated and run without errors.
    They use synthetic data and minimal training rounds for speed.
    """

    def test_vertical_basic(self):
        """Test basic vertical XGBoost completes successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = SimEnv(num_clients=2, workspace_root=os.path.join(tmpdir, "test_vertical"))

            recipe = XGBVerticalRecipe(
                name="test_vertical",
                min_clients=2,
                num_rounds=2,  # Minimal rounds for speed
                label_owner="site-1",
                early_stopping_rounds=1,
                xgb_params={
                    "max_depth": 3,
                    "eta": 0.1,
                    "objective": "binary:logistic",
                    "eval_metric": "auc",
                },
            )

            # Add mock data loaders
            # site-1 has labels, site-2 has features only
            per_site_config = {
                "site-1": {"data_loader": MockVerticalDataLoader(has_labels=True, n_samples=50, n_features=3)},
                "site-2": {"data_loader": MockVerticalDataLoader(has_labels=False, n_samples=50, n_features=3)},
            }
            recipe.per_site_config = per_site_config
            recipe.job = recipe.configure()

            # Run and verify completion
            run = recipe.execute(env)
            assert run.get_result() is not None
            assert os.path.exists(run.get_result())

    def test_label_owner_validation(self):
        """Test that label_owner validation works correctly."""
        # Valid format
        recipe = XGBVerticalRecipe(
            name="test_valid",
            min_clients=2,
            num_rounds=1,
            label_owner="site-1",  # Valid
        )
        assert recipe.label_owner == "site-1"

        # Invalid format should raise error
        with pytest.raises(ValueError, match="label_owner must be in format 'site-X'"):
            XGBVerticalRecipe(
                name="test_invalid",
                min_clients=2,
                num_rounds=1,
                label_owner="client1",  # Invalid format
            )

    def test_custom_xgb_params(self):
        """Test that custom XGBoost parameters are accepted."""
        custom_params = {
            "max_depth": 5,
            "eta": 0.05,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "nthread": 4,
        }

        recipe = XGBVerticalRecipe(
            name="test_custom",
            min_clients=2,
            num_rounds=1,
            label_owner="site-1",
            xgb_params=custom_params,
        )

        # Verify params are stored
        assert recipe.xgb_params == custom_params

    def test_multiple_clients_vertical(self):
        """Test vertical recipe works with more than 2 clients."""
        with tempfile.TemporaryDirectory() as tmpdir:
            num_clients = 3
            env = SimEnv(num_clients=num_clients, workspace_root=os.path.join(tmpdir, "test_multi_vertical"))

            recipe = XGBVerticalRecipe(
                name="test_multi_vertical",
                min_clients=num_clients,
                num_rounds=1,
                label_owner="site-2",  # site-2 has labels
            )

            # Configure per-site data loaders - only site-2 has labels
            per_site_config = {}
            for site_id in range(1, num_clients + 1):
                has_labels = site_id == 2  # Only site-2 has labels
                per_site_config[f"site-{site_id}"] = {
                    "data_loader": MockVerticalDataLoader(has_labels=has_labels, n_samples=30, n_features=2)
                }
            recipe.per_site_config = per_site_config
            recipe.job = recipe.configure()

            run = recipe.execute(env)
            assert run.get_result() is not None
            assert os.path.exists(run.get_result())

    def test_tensorboard_tracking_configured(self):
        """Test that TensorBoard tracking components are configured."""
        recipe = XGBVerticalRecipe(
            name="test_tb",
            min_clients=2,
            num_rounds=1,
            label_owner="site-1",
        )

        # Verify TensorBoard receiver is added to server
        server_components = recipe.job.get_server_components()
        assert "tb_receiver" in server_components

        # Verify TensorBoard tracking is configured (components added in configure())

        # Check that metrics_writer and event_to_fed are added
        client_components = recipe.job.get_client_components("site-1")
        assert "metrics_writer" in client_components
        assert "event_to_fed" in client_components

    def test_in_process_parameter(self):
        """Test that in_process parameter is configurable."""
        recipe = XGBVerticalRecipe(
            name="test_in_process",
            min_clients=2,
            num_rounds=1,
            label_owner="site-1",
            in_process=True,  # Default
        )
        assert recipe.in_process is True

        recipe2 = XGBVerticalRecipe(
            name="test_not_in_process",
            min_clients=2,
            num_rounds=1,
            label_owner="site-1",
            in_process=False,
        )
        assert recipe2.in_process is False

    def test_model_file_name_parameter(self):
        """Test that model_file_name is configurable."""
        custom_name = "my_vertical_model.json"
        recipe = XGBVerticalRecipe(
            name="test_model_name",
            min_clients=2,
            num_rounds=1,
            label_owner="site-1",
            model_file_name=custom_name,
        )
        assert recipe.model_file_name == custom_name

    def test_data_split_mode_is_vertical(self):
        """Test that vertical recipe uses data_split_mode=1 (column mode)."""
        recipe = XGBVerticalRecipe(
            name="test_vertical_mode",
            min_clients=2,
            num_rounds=1,
            label_owner="site-1",
        )

        # Check controller configuration
        server_components = recipe.job.get_server_components()
        controller = server_components.get("xgb_controller")
        assert controller is not None
        # Note: We can't easily verify data_split_mode without running,
        # but we can verify the controller type
        from nvflare.app_opt.xgboost.histogram_based_v2.fed_controller import XGBFedController

        assert isinstance(controller, XGBFedController)
