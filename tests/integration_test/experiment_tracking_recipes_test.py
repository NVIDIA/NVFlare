# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Integration tests for experiment tracking with recipes.

These are smoke tests to verify that add_experiment_tracking() works with recipes without errors.
Tests verify:
- add_experiment_tracking() can be called successfully
- The job completes and produces a workspace

Tests do NOT verify:
- Tracking files are actually created (TensorBoard events, MLflow runs, etc.)
- Metrics are logged correctly
- Tracking output content or format

NOTE: These tests are currently NOT triggered by any automated test suite.
They use CIFAR-10 dataset and run real training.

To run manually:
    cd tests/integration_test
    pytest test_experiment_tracking_recipes.py -v

TODO: Decide if these should be added to an existing test category (e.g., CIFAR integration tests)
or run in a separate recipe test suite (takes ~1-2 minutes).
"""

import os

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv
from nvflare.recipe.utils import add_experiment_tracking


class TestExperimentTrackingRecipes:
    """Smoke tests for experiment tracking integration with Recipe API.

    These tests verify that experiment tracking can be added to recipes without errors
    and that jobs complete successfully. They do not verify tracking output.
    """

    @property
    def client_script_path(self):
        """Get absolute path to client.py script from tensorboard example."""
        # Use the tensorboard example's client script for testing
        test_dir = os.path.dirname(__file__)
        repo_root = os.path.dirname(os.path.dirname(test_dir))
        return os.path.join(repo_root, "examples/advanced/experiment-tracking/tensorboard/client.py")

    def test_tensorboard_tracking_integration(self):
        """Test TensorBoard tracking can be added and job completes."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            env = SimEnv(num_clients=2, workspace_root=os.path.join(tmpdir, "test_tensorboard"))
            recipe = FedAvgRecipe(
                name="test_tensorboard",
                min_clients=2,
                num_rounds=1,
                train_script=self.client_script_path,
            )

            # Add TensorBoard tracking
            add_experiment_tracking(recipe, "tensorboard")

            # Run and verify completion
            run = recipe.execute(env)
            assert run.get_result() is not None
            assert os.path.exists(run.get_result())

    def test_mlflow_tracking_integration(self):
        """Test MLflow tracking can be added and job completes."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow_dir = os.path.join(tmpdir, "test_mlflow")
            env = SimEnv(num_clients=2, workspace_root=mlflow_dir)
            recipe = FedAvgRecipe(
                name="test_mlflow",
                min_clients=2,
                num_rounds=1,
                train_script=self.client_script_path,
            )

            # Add MLflow tracking
            mlflow_uri = os.path.join(mlflow_dir, "mlruns")
            add_experiment_tracking(
                recipe,
                "mlflow",
                tracking_config={
                    "tracking_uri": f"file:///{mlflow_uri}",
                    "kw_args": {"experiment_name": "test", "run_name": "test"},
                },
            )

            # Run and verify completion
            run = recipe.execute(env)
            assert run.get_result() is not None
            assert os.path.exists(run.get_result())
