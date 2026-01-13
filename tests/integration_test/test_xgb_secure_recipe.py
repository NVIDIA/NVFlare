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

import os
import tempfile

import numpy as np
import xgboost as xgb

from nvflare.app_opt.xgboost.data_loader import XGBDataLoader
from nvflare.app_opt.xgboost.recipes import XGBHistogramRecipe, XGBVerticalRecipe
from nvflare.recipe import SimEnv


class MockXGBDataLoader(XGBDataLoader):
    """Mock data loader for testing."""

    def load_data(self):
        # Generate small synthetic dataset
        np.random.seed(42)
        n_samples = 100
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        return xgb.DMatrix(X, label=y)


class TestXGBSecureRecipes:
    """Integration tests for secure XGBoost recipes."""

    def test_horizontal_secure_job_completes(self):
        """Test that secure horizontal XGBoost job completes successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = SimEnv(num_clients=2, workspace_root=os.path.join(tmpdir, "test_horizontal_secure"))

            recipe = XGBHistogramRecipe(
                name="test_horizontal_secure",
                min_clients=2,
                num_rounds=1,
                algorithm="histogram_v2",
                secure=True,  # Enable secure training
                xgb_params={"objective": "binary:logistic", "eval_metric": "auc", "nthread": 1},
            )

            for i in range(1, 3):
                recipe.add_to_client(f"site-{i}", MockXGBDataLoader())

            run = recipe.execute(env)
            assert run.get_result() is not None
            assert os.path.exists(run.get_result())

    def test_vertical_secure_job_completes(self):
        """Test that secure vertical XGBoost job completes successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = SimEnv(num_clients=2, workspace_root=os.path.join(tmpdir, "test_vertical_secure"))

            recipe = XGBVerticalRecipe(
                name="test_vertical_secure",
                min_clients=2,
                num_rounds=1,
                label_owner="site-1",
                secure=True,  # Enable secure training
                xgb_params={"objective": "binary:logistic", "eval_metric": "auc", "nthread": 1},
            )

            for i in range(1, 3):
                recipe.add_to_client(f"site-{i}", MockXGBDataLoader())

            run = recipe.execute(env)
            assert run.get_result() is not None
            assert os.path.exists(run.get_result())

    def test_horizontal_secure_with_custom_client_ranks(self):
        """Test secure horizontal XGBoost with custom client ranks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = SimEnv(num_clients=3, workspace_root=os.path.join(tmpdir, "test_custom_ranks"))

            custom_ranks = {"site-1": 0, "site-2": 1, "site-3": 2}

            recipe = XGBHistogramRecipe(
                name="test_custom_ranks",
                min_clients=3,
                num_rounds=1,
                algorithm="histogram_v2",
                secure=True,
                client_ranks=custom_ranks,
                xgb_params={"objective": "binary:logistic", "eval_metric": "auc", "nthread": 1},
            )

            for i in range(1, 4):
                recipe.add_to_client(f"site-{i}", MockXGBDataLoader())

            run = recipe.execute(env)
            assert run.get_result() is not None
            assert os.path.exists(run.get_result())

    def test_vertical_secure_with_custom_client_ranks(self):
        """Test secure vertical XGBoost with custom client ranks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = SimEnv(num_clients=3, workspace_root=os.path.join(tmpdir, "test_vertical_custom_ranks"))

            custom_ranks = {"site-1": 0, "site-2": 1, "site-3": 2}

            recipe = XGBVerticalRecipe(
                name="test_vertical_custom_ranks",
                min_clients=3,
                num_rounds=1,
                label_owner="site-1",
                secure=True,
                client_ranks=custom_ranks,
                xgb_params={"objective": "binary:logistic", "eval_metric": "auc", "nthread": 1},
            )

            for i in range(1, 4):
                recipe.add_to_client(f"site-{i}", MockXGBDataLoader())

            run = recipe.execute(env)
            assert run.get_result() is not None
            assert os.path.exists(run.get_result())

    def test_horizontal_secure_auto_generates_client_ranks(self):
        """Test that secure horizontal XGBoost auto-generates client ranks when not provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = SimEnv(num_clients=2, workspace_root=os.path.join(tmpdir, "test_auto_ranks"))

            recipe = XGBHistogramRecipe(
                name="test_auto_ranks",
                min_clients=2,
                num_rounds=1,
                algorithm="histogram_v2",
                secure=True,  # Enable secure, but don't provide client_ranks
                xgb_params={"objective": "binary:logistic", "eval_metric": "auc", "nthread": 1},
            )

            # Verify that client_ranks were auto-generated
            assert recipe.client_ranks == {"site-1": 0, "site-2": 1}

            for i in range(1, 3):
                recipe.add_to_client(f"site-{i}", MockXGBDataLoader())

            run = recipe.execute(env)
            assert run.get_result() is not None
            assert os.path.exists(run.get_result())

    def test_vertical_secure_auto_generates_client_ranks(self):
        """Test that secure vertical XGBoost auto-generates client ranks when not provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = SimEnv(num_clients=2, workspace_root=os.path.join(tmpdir, "test_vertical_auto_ranks"))

            recipe = XGBVerticalRecipe(
                name="test_vertical_auto_ranks",
                min_clients=2,
                num_rounds=1,
                label_owner="site-1",
                secure=True,  # Enable secure, but don't provide client_ranks
                xgb_params={"objective": "binary:logistic", "eval_metric": "auc", "nthread": 1},
            )

            # Verify that client_ranks were auto-generated
            assert recipe.client_ranks == {"site-1": 0, "site-2": 1}

            for i in range(1, 3):
                recipe.add_to_client(f"site-{i}", MockXGBDataLoader())

            run = recipe.execute(env)
            assert run.get_result() is not None
            assert os.path.exists(run.get_result())

    def test_histogram_algorithm_with_secure(self):
        """Test secure training with original histogram algorithm."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = SimEnv(num_clients=2, workspace_root=os.path.join(tmpdir, "test_histogram_secure"))

            recipe = XGBHistogramRecipe(
                name="test_histogram_secure",
                min_clients=2,
                num_rounds=1,
                algorithm="histogram",  # Original algorithm
                secure=True,
                xgb_params={"objective": "binary:logistic", "eval_metric": "auc", "nthread": 1},
            )

            for i in range(1, 3):
                recipe.add_to_client(f"site-{i}", MockXGBDataLoader())

            run = recipe.execute(env)
            assert run.get_result() is not None
            assert os.path.exists(run.get_result())

    def test_secure_false_does_not_add_client_ranks(self):
        """Test that non-secure training does not require client_ranks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = SimEnv(num_clients=2, workspace_root=os.path.join(tmpdir, "test_non_secure"))

            recipe = XGBHistogramRecipe(
                name="test_non_secure",
                min_clients=2,
                num_rounds=1,
                algorithm="histogram_v2",
                secure=False,  # Non-secure training
                xgb_params={"objective": "binary:logistic", "eval_metric": "auc", "nthread": 1},
            )

            # client_ranks should be empty for non-secure
            assert recipe.client_ranks == {}

            for i in range(1, 3):
                recipe.add_to_client(f"site-{i}", MockXGBDataLoader())

            run = recipe.execute(env)
            assert run.get_result() is not None
            assert os.path.exists(run.get_result())
