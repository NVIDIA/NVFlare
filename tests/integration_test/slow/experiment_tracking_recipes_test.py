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

"""Slow integration tests for experiment tracking with recipes.

These are smoke tests to verify that add_experiment_tracking() works with recipes without errors.
Tests verify:
- add_experiment_tracking() can be called successfully
- The job completes and produces a workspace

Tests do NOT verify:
- Tracking files are actually created (TensorBoard events, MLflow runs, etc.)
- Metrics are logged correctly
- Tracking output content or format

These tests use the CIFAR-10 dataset and run real training.

To run manually:
    cd tests/integration_test
    pytest slow/experiment_tracking_recipes_test.py -v
"""

import json
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
    from tests.integration_test.tools.prepare_cifar10 import prepare_cifar10

    data_root = str(tmp_path_factory.mktemp("cifar10_data"))
    prepare_cifar10(roots=[data_root])
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

    @property
    def mlflow_client_script_path(self):
        return os.path.join(
            REPO_ROOT,
            "examples/advanced/experiment-tracking/mlflow/hello-pt-mlflow-client/client.py",
        )

    @property
    def mlflow_client_model_path(self):
        return os.path.join(os.path.dirname(self.mlflow_client_script_path), "model.py")

    def _add_model_to_apps(self, recipe, model_path=None):
        model_path = model_path or self.model_path
        recipe.job.add_file_to_server(model_path)
        for site_name in ("site-1", "site-2"):
            recipe.job.add_file_to(model_path, site_name)

    def _assert_exported_client_configs_have_executors(self, recipe, export_root: str):
        recipe.export(export_root)
        job_root = os.path.join(export_root, recipe.name)
        assert os.path.exists(os.path.join(job_root, "app_server", "custom", "model.py"))
        for site_name in ("site-1", "site-2"):
            client_config_path = os.path.join(job_root, f"app_{site_name}", "config", "config_fed_client.json")
            with open(client_config_path, "r") as f:
                client_config = json.load(f)
            assert client_config.get("executors"), f"{client_config_path} has no executors"
            assert os.path.exists(os.path.join(job_root, f"app_{site_name}", "custom", "model.py"))

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

    def test_per_site_export_preserves_client_executors(self, tmp_path):
        """Adding shared model files must not replace per-site client apps."""
        recipe = FedAvgRecipe(
            name="test_export",
            min_clients=2,
            num_rounds=1,
            model={"class_path": "model.SimpleNetwork", "args": {}},
            train_script=self.client_script_path,
            per_site_config=self._per_site_config(str(tmp_path / "data"), str(tmp_path)),
        )
        self._add_model_to_apps(recipe)

        # Exercise the same mutation path as the runtime tests.
        add_experiment_tracking(recipe, "tensorboard")

        self._assert_exported_client_configs_have_executors(recipe, str(tmp_path / "export"))

    def test_tensorboard_tracking_integration(self, cifar10_data_root):
        """Test TensorBoard tracking can be added and job completes."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            env = SimEnv(clients=["site-1", "site-2"], workspace_root=os.path.join(tmpdir, "test_tensorboard"))
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
            env = SimEnv(clients=["site-1", "site-2"], workspace_root=mlflow_dir)
            recipe = FedAvgRecipe(
                name="test_mlflow",
                min_clients=2,
                num_rounds=1,
                model={"class_path": "model.SimpleNetwork", "args": {}},
                train_script=self.mlflow_client_script_path,
                per_site_config=self._per_site_config(cifar10_data_root, tmpdir),
            )
            self._add_model_to_apps(recipe, self.mlflow_client_model_path)

            # Exercise the zero-config MLflow path: local storage and names derived from the recipe.
            add_experiment_tracking(recipe, "mlflow")

            # Run and verify completion plus the actual MLflow experiment and runs.
            run = recipe.execute(env)
            assert run.get_result() is not None
            assert os.path.exists(run.get_result())

            mlflow_db = os.path.join(run.get_result(), "server", "simulate_job", "mlflow.db")
            assert os.path.isfile(mlflow_db)

            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=f"sqlite:///{mlflow_db}")
            experiment = client.get_experiment_by_name("test_mlflow-experiment")
            assert experiment is not None

            runs = client.search_runs([experiment.experiment_id])
            assert len(runs) == 2
            assert all(run.info.run_name.endswith("test_mlflow-Client") for run in runs)

    def test_mlflow_client_tracking_defaults_integration(self, cifar10_data_root):
        """Test zero-config client-side MLflow tracking creates one local store per site."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = os.path.join(tmpdir, "test_mlflow_client")
            env = SimEnv(clients=["site-1", "site-2"], workspace_root=workspace_root)
            recipe = FedAvgRecipe(
                name="test_mlflow_client",
                min_clients=2,
                num_rounds=1,
                model={"class_path": "model.SimpleNetwork", "args": {}},
                train_script=self.mlflow_client_script_path,
                per_site_config=self._per_site_config(cifar10_data_root, tmpdir),
            )
            self._add_model_to_apps(recipe, self.mlflow_client_model_path)

            add_experiment_tracking(recipe, "mlflow", client_side=True, server_side=False)

            run = recipe.execute(env)
            assert run.get_result() is not None
            assert os.path.exists(run.get_result())

            from mlflow.tracking import MlflowClient

            for site_name in ("site-1", "site-2"):
                mlflow_db = os.path.join(run.get_result(), site_name, "simulate_job", "mlflow.db")
                assert os.path.isfile(mlflow_db)

                client = MlflowClient(tracking_uri=f"sqlite:///{mlflow_db}")
                experiment = client.get_experiment_by_name("test_mlflow_client-experiment")
                assert experiment is not None

                runs = client.search_runs([experiment.experiment_id])
                assert len(runs) == 1
                assert runs[0].info.run_name.endswith("test_mlflow_client-Client")
