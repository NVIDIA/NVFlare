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
    pytest slow/experiment_tracking_recipes_test.py -v

TODO: Decide if these should be added to an existing test category (e.g., CIFAR integration tests)
or run in a separate recipe test suite (takes ~1-2 minutes).
"""

import os

import pytest

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv
from nvflare.recipe.utils import add_experiment_tracking

INTEGRATION_TEST_ROOT = os.path.dirname(os.path.dirname(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(INTEGRATION_TEST_ROOT))


@pytest.fixture(scope="module")
def cifar10_data_root(tmp_path_factory):
    """Download CIFAR-10 once so simulated clients do not race on the same download/extract path."""
    from torchvision.datasets import CIFAR10

    data_root = str(tmp_path_factory.mktemp("cifar10_data"))
    CIFAR10(root=data_root, train=True, download=True)
    CIFAR10(root=data_root, train=False, download=True)
    return data_root


class TestExperimentTrackingRecipes:
    """Smoke tests for experiment tracking integration with Recipe API.

    These tests verify that experiment tracking can be added to recipes without errors
    and that jobs complete successfully. They do not verify tracking output.
    """

    @property
    def client_script_path(self):
        """Get absolute path to client.py script from tensorboard example."""
        return os.path.join(REPO_ROOT, "examples/advanced/experiment-tracking/tensorboard/client.py")

    @property
    def client_script_dir(self):
        return os.path.dirname(self.client_script_path)

    @property
    def model_path(self):
        return os.path.join(self.client_script_dir, "model.py")

    def _add_model_to_apps(self, recipe):
        recipe.job.add_file_to_server(self.model_path)
        recipe.job.add_file_to_clients(self.model_path)

    def _per_site_config(self, dataset_path: str, model_dir: str) -> dict[str, dict[str, str]]:
        return {
            site_name: {
                "train_args": (
                    f"--dataset_path {dataset_path} "
                    f"--model_path {os.path.join(model_dir, site_name + '_cifar_net.pth')} "
                    "--batch_size 512 --num_workers 0 --local_epochs 1"
                )
            }
            for site_name in ("site-1", "site-2")
        }

    def test_tensorboard_tracking_integration(self, cifar10_data_root):
        """Test TensorBoard tracking can be added and job completes."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            env = SimEnv(num_clients=2, workspace_root=os.path.join(tmpdir, "test_tensorboard"))
            recipe = FedAvgRecipe(
                name="test_tensorboard",
                min_clients=2,
                num_rounds=1,
                model={"class_path": "model.SimpleNetwork", "args": {}},
                train_script=self.client_script_path,
                per_site_config=self._per_site_config(cifar10_data_root, tmpdir),
            )
            self._add_model_to_apps(recipe)

            # Add TensorBoard tracking
            add_experiment_tracking(recipe, "tensorboard")

            # Run and verify completion
            run = recipe.execute(env)
            assert run.get_result() is not None
            assert os.path.exists(run.get_result())

    def test_mlflow_tracking_integration(self, cifar10_data_root):
        """Test MLflow tracking can be added and job completes."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow_dir = os.path.join(tmpdir, "test_mlflow")
            env = SimEnv(num_clients=2, workspace_root=mlflow_dir)
            recipe = FedAvgRecipe(
                name="test_mlflow",
                min_clients=2,
                num_rounds=1,
                model={"class_path": "model.SimpleNetwork", "args": {}},
                train_script=self.client_script_path,
                per_site_config=self._per_site_config(cifar10_data_root, tmpdir),
            )
            self._add_model_to_apps(recipe)

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
