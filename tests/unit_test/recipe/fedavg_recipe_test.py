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

import json
import warnings
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from nvflare.apis.dxo import DataKind
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.job_def import ALL_SITES, SERVER_SITE_NAME
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.abstract.model_locator import ModelLocator
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
from nvflare.app_common.app_constant import DefaultCheckpointFileName
from nvflare.app_common.executors.client_api_launcher_executor import ClientAPILauncherExecutor
from nvflare.app_common.executors.launcher_executor import LauncherExecutor
from nvflare.app_common.np.recipes import NumpyFedAvgRecipe
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.widgets.metrics_artifact_writer import MetricsArtifactWriter
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.client.config import ConfigKey, TransferType
from nvflare.client.constants import CLIENT_API_CONFIG
from nvflare.fuel.utils.class_utils import instantiate_class
from nvflare.fuel.utils.secret_utils import UnsupportedSecretRefWarning
from nvflare.job_config.base_fed_job import BaseFedJob
from nvflare.recipe import set_per_site_config
from nvflare.recipe.fedavg import FedAvgRecipe as BaseFedAvgRecipe


class SimpleTestModel(nn.Module):
    """A simple PyTorch model for testing purposes."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(10, 10)

    def forward(self, x):
        x = self.lin(x)
        return x


class MyAggregator(ModelAggregator):
    """Custom aggregator for testing FedAvgRecipe with custom aggregator support."""

    def __init__(self):
        super().__init__()
        self.sum = {}
        self.count = 0

    def accept_model(self, model: FLModel):
        """Accept submitted model and add to the sum."""
        self.info(f"##### MyAggregator: Accepting model with {len(model.params)} variables #####")
        for key, value in model.params.items():
            if key not in self.sum:
                self.sum[key] = 0
            self.sum[key] += value
        self.count += 1

    def aggregate_model(self) -> FLModel:
        """Aggregate the collected models."""
        self.info(f"##### MyAggregator: Aggregating {self.count} models #####")

        # compute the average
        for key in self.sum:
            self.sum[key] = self.sum[key] / self.count

        return FLModel(params=self.sum)

    def reset_stats(self):
        """Reset the aggregator state."""
        self.info("##### MyAggregator: Resetting #####")
        # reset the sum and count
        self.sum = {}
        self.count = 0


class InvalidAggregator:
    """Invalid aggregator that doesn't inherit from Aggregator."""

    def __init__(self):
        pass


class CustomMetricSelector(FLComponent):
    def __init__(self, key_metric):
        super().__init__()
        self.key_metric = key_metric


class DummyPersistor(ModelPersistor):
    """Minimal ModelPersistor used to test custom persistor wiring."""

    def load_model(self, fl_ctx):
        return {}

    def save_model(self, model, fl_ctx):
        return None


class DummyLocator(ModelLocator):
    """Minimal ModelLocator used to test explicit locator registration."""

    def get_model_names(self, fl_ctx):
        return []

    def locate_model(self, model_name, fl_ctx):
        return None


@pytest.fixture
def mock_file_system():
    """Mock file system operations for all tests."""
    with patch("os.path.isfile", return_value=True), patch("os.path.exists", return_value=True):
        yield


@pytest.fixture
def custom_aggregator():
    """Create a custom aggregator for testing."""
    return MyAggregator()


@pytest.fixture
def simple_model():
    """Create a simple test model."""
    return SimpleTestModel()


@pytest.fixture
def base_recipe_params():
    """Base parameters for creating FedAvgRecipe instances."""
    return {
        "train_script": "mock_train_script.py",
        "train_args": "--epochs 10",
        "min_clients": 2,
        "num_rounds": 5,
    }


def assert_recipe_basics(recipe, expected_name, expected_params):
    """Helper to assert basic recipe properties."""
    assert recipe.name == expected_name
    assert recipe.train_script == expected_params.get("train_script", "mock_train_script.py")
    assert recipe.train_args == expected_params.get("train_args", "--epochs 10")
    assert recipe.min_clients == expected_params.get("min_clients", 2)
    assert recipe.num_rounds == expected_params.get("num_rounds", 5)
    assert recipe._job is not None
    assert recipe._job.name == expected_name


def get_model_selector(recipe):
    server_app = recipe._job._deploy_map[SERVER_SITE_NAME]
    return server_app.app_config.components.get("model_selector")


def get_server_component(recipe, component_id):
    server_app = recipe._job._deploy_map[SERVER_SITE_NAME]
    return server_app.app_config.components.get(component_id)


def get_server_component_from_job(job, component_id):
    server_app = job._deploy_map[SERVER_SITE_NAME]
    return server_app.app_config.components.get(component_id)


def get_server_controller(recipe):
    server_app = recipe._job._deploy_map[SERVER_SITE_NAME]
    return server_app.app_config.workflows[0].controller


def get_client_executor(recipe, site_name):
    client_app = recipe._job._deploy_map[site_name]
    return client_app.app_config.executors[0].executor


def _get_train_executor_config(client_config):
    for executor_entry in client_config.get("executors", []):
        executor = executor_entry["executor"]
        if "ClientAPILauncherExecutor" in executor.get("path", ""):
            return executor_entry["executor"]
    raise AssertionError("External-process Client API launcher executor not found in exported client config")


def _make_exported_executor_fl_ctx(config_dir, job_id):
    workspace = MagicMock()
    workspace.get_app_config_dir.return_value = str(config_dir)
    engine = MagicMock()
    engine.get_workspace.return_value = workspace
    engine.get_component.return_value = None
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = engine
    fl_ctx.get_job_id.return_value = job_id
    fl_ctx.get_identity_name.return_value = "site-1"
    return fl_ctx


def _run_exported_external_process_executor_startup(executor_config, config_dir, job_id):
    executor = instantiate_class(executor_config["path"], executor_config.get("args", {}))
    pipe = MagicMock()
    pipe.export.return_value = ("nvflare.fuel.utils.pipe.cell_pipe.CellPipe", {})
    executor.pipe = pipe
    fl_ctx = _make_exported_executor_fl_ctx(config_dir=config_dir, job_id=job_id)

    with (
        patch.object(LauncherExecutor, "initialize", lambda self, fl_ctx: None),
        # Skip only advisory timeout relationship checks; prepare_config_for_launch()
        # still exercises required startup validation before writing subprocess config.
        patch.object(ClientAPILauncherExecutor, "_validate_timeout_config", lambda self, fl_ctx: None),
        patch.object(ClientAPILauncherExecutor, "log_info", lambda self, fl_ctx, msg: None),
        patch.object(ClientAPILauncherExecutor, "log_error", lambda self, fl_ctx, msg: None),
    ):
        executor.initialize(fl_ctx)

    client_api_config_path = config_dir / CLIENT_API_CONFIG
    with open(client_api_config_path, "r") as f:
        return json.load(f)


class TestFedAvgRecipe:
    def test_class_docstrings_are_preserved(self):
        assert BaseFedAvgRecipe.__doc__
        assert FedAvgRecipe.__doc__

    def test_external_command_secret_ref_is_supported(self, mock_file_system, base_recipe_params, simple_model):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UnsupportedSecretRefWarning)
            recipe = FedAvgRecipe(
                name="command_secret_ref",
                model=simple_model,
                **base_recipe_params,
            )
            set_per_site_config(
                recipe,
                {
                    "site-1": {
                        "launch_external_process": True,
                        "command": "env API_TOKEN=${secret:API_TOKEN} python3 -u",
                    },
                    "site-2": {},
                },
            )

    """Test cases for FedAvgRecipe class."""

    def test_default_aggregator_initialization(self, mock_file_system, base_recipe_params, simple_model):
        """Test FedAvgRecipe initialization with default (built-in) aggregation."""
        recipe = FedAvgRecipe(name="test_fedavg", model=simple_model, **base_recipe_params)

        assert_recipe_basics(recipe, "test_fedavg", base_recipe_params)
        assert recipe.model == simple_model
        # When no aggregator is passed, built-in weighted averaging is used
        assert recipe.aggregator is None

    def test_set_per_site_config_prepares_site_runners_before_client_customization(
        self, mock_file_system, base_recipe_params, simple_model
    ):
        recipe = FedAvgRecipe(name="test_helper_per_site", model=simple_model, **base_recipe_params)
        config = {
            "site-1": {"train_args": "--epochs 1"},
            "site-2": {},
        }

        assert recipe._job.clients == []
        set_per_site_config(recipe, config)

        assert recipe.configured_sites() == ["site-1", "site-2"]
        assert recipe._job.clients == []

        recipe.add_client_config({"configured": True})

        assert recipe._job.clients == ["site-1", "site-2"]
        assert ALL_SITES not in recipe._job._deploy_map
        assert get_client_executor(recipe, "site-1")._task_script_args == "--epochs 1"
        assert get_client_executor(recipe, "site-2")._task_script_args == "--epochs 10"

    def test_set_per_site_config_snapshots_overrides_before_deferred_preparation(
        self, mock_file_system, base_recipe_params, simple_model
    ):
        recipe = FedAvgRecipe(name="test_helper_snapshot", model=simple_model, **base_recipe_params)
        config = {"site-1": {"train_args": "--epochs 1"}, "site-2": {}}

        set_per_site_config(recipe, config)
        config["site-1"]["train_args"] = "--epochs 99"

        recipe._ensure_client_apps_prepared()

        assert get_client_executor(recipe, "site-1")._task_script_args == "--epochs 1"

    def test_legacy_constructor_config_delegates_to_helper(self, mock_file_system, base_recipe_params, simple_model):
        config = {"site-1": {"train_args": "--epochs 1"}, "site-2": {}}

        with pytest.warns(FutureWarning, match="set_per_site_config"):
            recipe = FedAvgRecipe(
                name="test_legacy_per_site",
                model=simple_model,
                per_site_config=config,
                **base_recipe_params,
            )

        assert recipe.configured_sites() == ["site-1", "site-2"]
        assert recipe._job.clients == []
        recipe._ensure_client_apps_prepared()
        assert recipe._job.clients == ["site-1", "site-2"]
        assert get_client_executor(recipe, "site-1")._task_script_args == "--epochs 1"

    def test_failed_per_site_config_leaves_topology_unprepared_and_can_retry(
        self, mock_file_system, base_recipe_params, simple_model
    ):
        recipe = FedAvgRecipe(name="test_per_site_rollback", model=simple_model, **base_recipe_params)

        with pytest.raises(ValueError, match="Framework invalid unsupported"):
            set_per_site_config(recipe, {"site-1": {}, "site-2": {"framework": "invalid"}})

        assert recipe._job.clients == []
        assert recipe.configured_sites() == []
        assert recipe.per_site_config is None

        set_per_site_config(recipe, {"site-1": {}, "site-2": {}})
        assert recipe._job.clients == []
        recipe._ensure_client_apps_prepared()
        assert recipe._job.clients == ["site-1", "site-2"]

    @pytest.mark.parametrize(("fedprox_mu", "expected"), [(None, None), (0.0, None), (0.2, 0.2)])
    def test_fedprox_mu_configures_pt_controller(
        self, mock_file_system, base_recipe_params, simple_model, fedprox_mu, expected
    ):
        recipe = FedAvgRecipe(
            name="test_fedprox",
            model=simple_model,
            fedprox_mu=fedprox_mu,
            **base_recipe_params,
        )

        assert recipe.fedprox_mu == expected
        assert get_server_controller(recipe).fedprox_mu == expected

    @pytest.mark.parametrize("fedprox_mu", [-0.1, float("inf"), float("nan"), True, "0.1"])
    def test_fedprox_mu_rejects_invalid_values(self, mock_file_system, base_recipe_params, simple_model, fedprox_mu):
        with pytest.raises((TypeError, ValueError), match="finite non-negative number"):
            FedAvgRecipe(
                name="test_invalid_fedprox",
                model=simple_model,
                fedprox_mu=fedprox_mu,
                **base_recipe_params,
            )

    def test_tensor_disk_offload_warns_when_server_format_is_not_pytorch(
        self, mock_file_system, base_recipe_params, simple_model
    ):
        """Tensor disk offload only applies to PyTorch tensor payloads."""
        with pytest.warns(UserWarning, match="only applies to streamed PyTorch tensors"):
            FedAvgRecipe(
                name="test_fedavg_offload_numpy_warning",
                model=simple_model,
                enable_tensor_disk_offload=True,
                **base_recipe_params,
            )

    def test_key_metric_passthrough_pt(self, mock_file_system, base_recipe_params, simple_model):
        key_metric = "val_auc"
        recipe = FedAvgRecipe(
            name="test_fedavg_key_metric", model=simple_model, key_metric=key_metric, **base_recipe_params
        )

        model_selector = get_model_selector(recipe)
        assert isinstance(model_selector, IntimeModelSelector)
        assert model_selector.key_metric == key_metric
        metrics_writer = get_server_component(recipe, "metrics_artifact_writer")
        assert isinstance(metrics_writer, MetricsArtifactWriter)
        assert not hasattr(metrics_writer, "key_metric")

    def test_metrics_writer_does_not_copy_policy_from_custom_selector(self):
        key_metric = "custom_score"
        job = BaseFedJob(
            name="test_custom_selector_policy", min_clients=2, model_selector=CustomMetricSelector(key_metric)
        )

        metrics_writer = get_server_component_from_job(job, "metrics_artifact_writer")
        assert isinstance(metrics_writer, MetricsArtifactWriter)
        assert not hasattr(metrics_writer, "key_metric")

    def test_best_model_filename_passthrough_pt(self, mock_file_system, base_recipe_params, simple_model):
        """best_model_filename should configure the generated PT persistor's best model artifact."""
        recipe = FedAvgRecipe(
            name="test_fedavg_best_filename",
            model=simple_model,
            best_model_filename="custom_best_model.pt",
            **base_recipe_params,
        )

        persistor = get_server_component(recipe, "persistor")
        assert recipe.best_model_filename == "custom_best_model.pt"
        assert recipe.save_filename == "custom_best_model.pt"
        assert persistor.best_global_model_file_name == "custom_best_model.pt"

    def test_default_controller_save_filename_preserved_for_no_persistor(self, mock_file_system, base_recipe_params):
        """The base no-persistor fallback must keep the legacy default save filename."""
        from nvflare.recipe import FedAvgRecipe as UnifiedFedAvgRecipe

        recipe = UnifiedFedAvgRecipe(
            name="test_default_fallback_filename",
            model={"class_path": "model.SimpleNetwork", "args": {}},
            **base_recipe_params,
        )

        controller = get_server_controller(recipe)
        assert recipe.best_model_filename == DefaultCheckpointFileName.BEST_GLOBAL_MODEL
        assert recipe.save_filename == DefaultCheckpointFileName.GLOBAL_MODEL
        assert controller.save_filename == DefaultCheckpointFileName.GLOBAL_MODEL

    def test_best_model_filename_configures_no_persistor_controller_fallback(
        self, mock_file_system, base_recipe_params
    ):
        """Explicit best_model_filename should still affect the controller fallback path."""
        from nvflare.recipe import FedAvgRecipe as UnifiedFedAvgRecipe

        recipe = UnifiedFedAvgRecipe(
            name="test_custom_fallback_filename",
            model={"class_path": "model.SimpleNetwork", "args": {}},
            best_model_filename="custom_best_model.pt",
            **base_recipe_params,
        )

        controller = get_server_controller(recipe)
        assert recipe.best_model_filename == "custom_best_model.pt"
        assert recipe.save_filename == "custom_best_model.pt"
        assert controller.save_filename == "custom_best_model.pt"

    def test_tensorflow_best_model_filename_warns_api_compatibility(self, mock_file_system, base_recipe_params):
        """TF currently accepts best_model_filename without wiring it into the default persistor."""
        from nvflare.fuel.utils.constants import FrameworkType
        from nvflare.recipe import FedAvgRecipe as UnifiedFedAvgRecipe

        with (
            patch("nvflare.job_config.script_runner.optional_import", return_value=(None, True)),
            pytest.warns(UserWarning, match="default persistors do not currently create"),
        ):
            recipe = UnifiedFedAvgRecipe(
                name="test_tf_best_filename_warning",
                framework=FrameworkType.TENSORFLOW,
                model_persistor=DummyPersistor(),
                best_model_filename="custom_tf_best_model.keras",
                **base_recipe_params,
            )

        assert recipe.best_model_filename == "custom_tf_best_model.keras"
        assert recipe.save_filename == "custom_tf_best_model.keras"

    def test_custom_aggregator_initialization(
        self, mock_file_system, base_recipe_params, custom_aggregator, simple_model
    ):
        """Test FedAvgRecipe initialization with custom aggregator."""
        params = {**base_recipe_params, "min_clients": 1, "num_rounds": 3}
        recipe = FedAvgRecipe(name="test_fedavg_custom", model=simple_model, aggregator=custom_aggregator, **params)

        assert_recipe_basics(recipe, "test_fedavg_custom", params)
        assert recipe.aggregator is custom_aggregator
        assert isinstance(recipe.aggregator, MyAggregator)

    def test_model_configuration(self, mock_file_system, base_recipe_params, custom_aggregator, simple_model):
        """Test FedAvgRecipe with initial model."""
        params = {**base_recipe_params, "min_clients": 1, "num_rounds": 3}
        recipe = FedAvgRecipe(name="test_fedavg_model", model=simple_model, aggregator=custom_aggregator, **params)

        assert_recipe_basics(recipe, "test_fedavg_model", params)
        assert recipe.model == simple_model

    @pytest.mark.parametrize(
        "min_clients,num_rounds,train_args",
        [
            (1, 1, ""),  # Minimum configuration
            (2, 3, "--epochs 5"),  # Standard configuration
            (5, 10, "--lr 0.01 --batch_size 32"),  # Complex configuration
        ],
    )
    def test_recipe_configurations(self, mock_file_system, simple_model, min_clients, num_rounds, train_args):
        """Test various FedAvgRecipe configurations using parametrized tests."""
        recipe = FedAvgRecipe(
            name=f"test_config_{min_clients}_{num_rounds}",
            model=simple_model,
            train_script="mock_train_script.py",
            train_args=train_args,
            min_clients=min_clients,
            num_rounds=num_rounds,
        )

        expected_params = {
            "train_script": "mock_train_script.py",
            "train_args": train_args,
            "min_clients": min_clients,
            "num_rounds": num_rounds,
        }
        assert_recipe_basics(recipe, f"test_config_{min_clients}_{num_rounds}", expected_params)


class TestFedAvgRecipeKeyMetricVariants:
    """Test key_metric passthrough for NumPy FedAvg recipes."""

    def test_key_metric_passthrough_numpy(self, mock_file_system):
        key_metric = "val_loss"
        recipe = NumpyFedAvgRecipe(
            name="test_numpy_key_metric",
            model=[1.0, 2.0, 3.0],
            min_clients=2,
            train_script="mock_train_script.py",
            key_metric=key_metric,
        )

        model_selector = get_model_selector(recipe)
        assert isinstance(model_selector, IntimeModelSelector)
        assert model_selector.key_metric == key_metric


class TestNumpyFedAvgRecipe:
    """Test NumpyFedAvgRecipe with new FedAvg features."""

    def test_numpy_recipe_basic_initialization(self, mock_file_system):
        """Test NumpyFedAvgRecipe basic initialization."""
        recipe = NumpyFedAvgRecipe(
            name="test_numpy",
            model=[[1, 2, 3], [4, 5, 6]],
            min_clients=2,
            num_rounds=3,
            train_script="client.py",
        )

        assert recipe.name == "test_numpy"
        assert recipe.min_clients == 2
        assert recipe.num_rounds == 3
        assert recipe._job is not None

    def test_numpy_recipe_with_early_stopping(self, mock_file_system):
        """Test NumpyFedAvgRecipe with early stopping configuration."""
        recipe = NumpyFedAvgRecipe(
            name="test_numpy_early_stop",
            model=[1.0, 2.0, 3.0],
            min_clients=2,
            num_rounds=10,
            train_script="client.py",
            stop_cond="accuracy >= 95",
            patience=3,
        )

        assert recipe.stop_cond == "accuracy >= 95"
        assert recipe.patience == 3

    def test_numpy_recipe_with_aggregation_weights(self, mock_file_system):
        """Test NumpyFedAvgRecipe with per-client aggregation weights."""
        weights = {"site-1": 2.0, "site-2": 1.0, "site-3": 0.5}
        recipe = NumpyFedAvgRecipe(
            name="test_numpy_weights",
            model=[1.0, 2.0],
            min_clients=3,
            num_rounds=5,
            train_script="client.py",
            aggregation_weights=weights,
        )

        assert recipe.aggregation_weights == weights

    def test_numpy_recipe_with_exclude_vars(self, mock_file_system):
        """Test NumpyFedAvgRecipe with exclude_vars configuration."""
        recipe = NumpyFedAvgRecipe(
            name="test_numpy_exclude",
            model=[1.0, 2.0, 3.0],
            min_clients=2,
            num_rounds=5,
            train_script="client.py",
            exclude_vars="bias.*",
        )

        assert recipe.exclude_vars == "bias.*"

    def test_numpy_recipe_with_save_filename(self, mock_file_system):
        """Test NumpyFedAvgRecipe with custom save filename."""
        with pytest.warns(Warning) as warning_records:
            recipe = NumpyFedAvgRecipe(
                name="test_numpy_save",
                model=[1.0, 2.0, 3.0],
                min_clients=2,
                num_rounds=5,
                train_script="client.py",
                save_filename="numpy_model.pt",
            )

        warning_messages = [str(record.message) for record in warning_records]
        assert any("save_filename is deprecated" in message for message in warning_messages)
        assert any("default persistors do not currently create" in message for message in warning_messages)

        assert recipe.best_model_filename == "numpy_model.pt"
        assert recipe.save_filename == "numpy_model.pt"

    def test_numpy_recipe_with_best_model_filename_warns_api_compatibility(self, mock_file_system):
        """NumPy currently accepts best_model_filename without wiring it into the default persistor."""
        with pytest.warns(UserWarning, match="default persistors do not currently create"):
            recipe = NumpyFedAvgRecipe(
                name="test_numpy_best_filename_warning",
                model=[1.0, 2.0, 3.0],
                min_clients=2,
                train_script="client.py",
                best_model_filename="custom_numpy_best_model.npy",
            )

        assert recipe.best_model_filename == "custom_numpy_best_model.npy"
        assert recipe.save_filename == "custom_numpy_best_model.npy"

    def test_numpy_recipe_with_per_site_config(self, mock_file_system):
        """Test NumpyFedAvgRecipe with per-site configuration."""
        per_site_config = {
            "site-1": {"train_args": "--data /path/to/site1"},
            "site-2": {"train_args": "--data /path/to/site2"},
        }
        with pytest.warns(FutureWarning, match="set_per_site_config"):
            recipe = NumpyFedAvgRecipe(
                name="test_numpy_per_site",
                model=[1.0, 2.0],
                min_clients=2,
                num_rounds=3,
                train_script="client.py",
                per_site_config=per_site_config,
            )

        assert recipe.per_site_config == per_site_config

    def test_numpy_helper_config_preserves_numpy_runner_exchange_format(self, mock_file_system):
        from nvflare.client.config import ExchangeFormat
        from nvflare.fuel.utils.constants import FrameworkType

        recipe = NumpyFedAvgRecipe(
            name="test_numpy_helper_format",
            model=[1.0, 2.0],
            min_clients=2,
            train_script="client.py",
        )
        set_per_site_config(recipe, {"site-1": {}, "site-2": {}})

        # NumPy recipes identify as RAW to Recipe CSE utilities, but their
        # training runners must continue exchanging NumPy parameters.
        assert recipe.framework == FrameworkType.RAW
        recipe._ensure_client_apps_prepared()

        for site in ["site-1", "site-2"]:
            executor = get_client_executor(recipe, site)
            assert executor._params_exchange_format == ExchangeFormat.NUMPY

    def test_numpy_cse_export_preserves_per_site_training_apps(self, tmp_path):
        """Adding CSE must retain each per-site training executor alongside NPValidator."""
        from nvflare.recipe.utils import add_cross_site_evaluation

        train_script = tmp_path / "client.py"
        train_script.write_text("# test client script\n")
        sites = ["site-1", "site-2"]
        recipe = NumpyFedAvgRecipe(
            name="test_numpy_per_site_cse",
            model=[1.0, 2.0],
            min_clients=2,
            train_script=str(train_script),
        )
        set_per_site_config(recipe, {site: {"train_args": f"--site {site}"} for site in sites})

        add_cross_site_evaluation(recipe)
        export_dir = tmp_path / "export"
        recipe.export(job_dir=str(export_dir))

        job_dir = export_dir / recipe.name
        assert not (job_dir / "app").exists()
        for site in sites:
            with open(job_dir / f"app_{site}" / "config" / "config_fed_client.json") as f:
                client_config = json.load(f)

            executors = client_config["executors"]
            assert any("*" in executor["tasks"] for executor in executors)
            assert any(
                "validate" in executor["tasks"] and executor["executor"]["path"].endswith(".NPValidator")
                for executor in executors
            )

    def test_numpy_recipe_with_none_model_raises_error(self, mock_file_system):
        """Test NumpyFedAvgRecipe with no model raises error."""
        with pytest.raises(ValueError, match="Must provide either model"):
            NumpyFedAvgRecipe(
                name="test_numpy_no_model",
                model=None,
                min_clients=2,
                num_rounds=3,
                train_script="client.py",
            )

    def test_numpy_recipe_full_configuration(self, mock_file_system):
        """Test NumpyFedAvgRecipe with all new features."""
        with pytest.warns(Warning) as warning_records:
            recipe = NumpyFedAvgRecipe(
                name="test_numpy_full",
                model=[[1, 2], [3, 4], [5, 6]],
                min_clients=3,
                num_rounds=20,
                train_script="train.py",
                train_args="--epochs 10",
                launch_external_process=True,
                command="python3 -u",
                key_metric="f1_score",
                stop_cond="f1_score >= 0.9",
                patience=5,
                save_filename="best_numpy_model.pt",
                exclude_vars="temp_.*",
                aggregation_weights={"site-1": 1.0, "site-2": 2.0, "site-3": 1.5},
            )

        warning_messages = [str(record.message) for record in warning_records]
        assert any("save_filename is deprecated" in message for message in warning_messages)
        assert any("default persistors do not currently create" in message for message in warning_messages)

        assert recipe.name == "test_numpy_full"
        assert recipe.min_clients == 3
        assert recipe.num_rounds == 20
        assert recipe.train_script == "train.py"
        assert recipe.train_args == "--epochs 10"
        assert recipe.launch_external_process is True
        assert recipe.key_metric == "f1_score"
        assert recipe.stop_cond == "f1_score >= 0.9"
        assert recipe.patience == 5
        assert recipe.best_model_filename == "best_numpy_model.pt"
        assert recipe.save_filename == "best_numpy_model.pt"
        assert recipe.exclude_vars == "temp_.*"
        assert recipe.aggregation_weights == {"site-1": 1.0, "site-2": 2.0, "site-3": 1.5}


class TestFedAvgRecipeEarlyStopping:
    """Test early stopping configuration for FedAvgRecipe."""

    def test_early_stopping_configuration(self, mock_file_system, base_recipe_params, simple_model):
        """Test FedAvgRecipe with early stopping configuration."""
        recipe = FedAvgRecipe(
            name="test_early_stop",
            model=simple_model,
            stop_cond="accuracy >= 80",
            patience=5,
            **base_recipe_params,
        )

        assert_recipe_basics(recipe, "test_early_stop", base_recipe_params)
        assert recipe.stop_cond == "accuracy >= 80"
        assert recipe.patience == 5

    def test_save_filename_configuration(self, mock_file_system, base_recipe_params, simple_model):
        """Test FedAvgRecipe accepts save_filename as a backward-compatible alias."""
        with pytest.warns(FutureWarning, match="save_filename is deprecated"):
            recipe = FedAvgRecipe(
                name="test_save_file",
                model=simple_model,
                save_filename="best_model.pt",
                **base_recipe_params,
            )

        persistor = get_server_component(recipe, "persistor")
        assert recipe.best_model_filename == "best_model.pt"
        assert recipe.save_filename == "best_model.pt"
        assert persistor.best_global_model_file_name == "best_model.pt"

    def test_conflicting_best_model_filename_alias_raises(self, mock_file_system, base_recipe_params, simple_model):
        """best_model_filename and save_filename must not silently disagree."""
        with pytest.raises(ValueError, match="conflicting values"):
            FedAvgRecipe(
                name="test_conflicting_best_file",
                model=simple_model,
                best_model_filename="best_model.pt",
                save_filename="legacy_model.pt",
                **base_recipe_params,
            )

    def test_exclude_vars_configuration(self, mock_file_system, base_recipe_params, simple_model):
        """Test FedAvgRecipe with exclude_vars configuration."""
        recipe = FedAvgRecipe(
            name="test_exclude",
            model=simple_model,
            exclude_vars="bn.*|running_mean|running_var",
            **base_recipe_params,
        )

        assert recipe.exclude_vars == "bn.*|running_mean|running_var"

    def test_aggregation_weights_configuration(self, mock_file_system, base_recipe_params, simple_model):
        """Test FedAvgRecipe with per-client aggregation weights."""
        weights = {"site-1": 2.0, "site-2": 1.0}
        recipe = FedAvgRecipe(
            name="test_weights",
            model=simple_model,
            aggregation_weights=weights,
            **base_recipe_params,
        )

        assert recipe.aggregation_weights == weights


class TestFedAvgRecipeValidation:
    """Test FedAvgRecipe input validation."""

    def test_weight_diff_with_full_transfer_is_valid(self, mock_file_system, base_recipe_params, simple_model):
        recipe = FedAvgRecipe(
            name="test_weight_diff",
            model=simple_model,
            aggregator_data_kind=DataKind.WEIGHT_DIFF,
            params_transfer_type=TransferType.FULL,
            **base_recipe_params,
        )

        assert recipe.aggregator_data_kind == DataKind.WEIGHT_DIFF
        assert recipe.params_transfer_type == TransferType.FULL

    def test_custom_aggregator_declared_data_kind_must_match(self, mock_file_system, base_recipe_params, simple_model):
        aggregator = InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS)

        with pytest.raises(ValueError, match="incompatible server aggregation settings"):
            FedAvgRecipe(
                name="test_custom_aggregator_data_kind",
                model=simple_model,
                aggregator=aggregator,
                aggregator_data_kind=DataKind.WEIGHT_DIFF,
                **base_recipe_params,
            )

    def test_invalid_aggregator_type_raises_validation_error(self, mock_file_system, base_recipe_params):
        """Test that invalid aggregator type raises Pydantic validation error."""
        from pydantic import ValidationError

        invalid_aggregator = InvalidAggregator()

        with pytest.raises(ValidationError, match="should be an instance of Aggregator"):
            FedAvgRecipe(
                name="test_invalid_agg",
                aggregator=invalid_aggregator,  # type: ignore[arg-type]
                **base_recipe_params,
            )

    def test_per_site_config_rejects_reserved_server_target(self, mock_file_system, base_recipe_params, simple_model):
        """Reserved target 'server' must not be allowed in per_site_config."""
        recipe = FedAvgRecipe(name="test_reserved_server_target", model=simple_model, **base_recipe_params)

        with pytest.raises(ValueError, match="reserved target name"):
            set_per_site_config(recipe, {"server": {}})

    def test_per_site_config_rejects_reserved_all_sites_target(
        self, mock_file_system, base_recipe_params, simple_model
    ):
        """Reserved target '@ALL' must not be allowed in per_site_config."""
        recipe = FedAvgRecipe(name="test_reserved_all_sites_target", model=simple_model, **base_recipe_params)

        with pytest.raises(ValueError, match="reserved target name"):
            set_per_site_config(recipe, {ALL_SITES: {}})

    @pytest.mark.parametrize(
        ("site_name", "match"),
        [
            ("", "valid target name"),
            ("site/1", "invalid character"),
            ("site@1", "invalid character"),
        ],
    )
    def test_per_site_config_rejects_invalid_target_name(
        self, mock_file_system, base_recipe_params, simple_model, site_name, match
    ):
        recipe = FedAvgRecipe(name="test_invalid_target_name", model=simple_model, **base_recipe_params)

        with pytest.raises(ValueError, match=match):
            set_per_site_config(recipe, {site_name: {}, "site-2": {}})

        assert recipe.configured_sites() == []
        assert recipe._job.clients == []

    def test_per_site_config_requires_at_least_min_clients(self, mock_file_system, base_recipe_params, simple_model):
        recipe = FedAvgRecipe(name="test_per_site_client_count", model=simple_model, **base_recipe_params)

        with pytest.raises(ValueError, match=r"defines 1 site.*min_clients=2"):
            set_per_site_config(recipe, {"site-1": {}})

        assert recipe._job.clients == []

    def test_per_site_config_rejects_after_export(self, tmp_path, base_recipe_params, simple_model):
        train_script = tmp_path / "train.py"
        train_script.write_text("print('training')\n")
        params = dict(base_recipe_params, train_script=str(train_script))
        recipe = FedAvgRecipe(name="test_deployed_per_site", model=simple_model, **params)
        recipe.export(str(tmp_path))
        assert recipe._job.clients == [ALL_SITES]

        with pytest.raises(RuntimeError, match="immediately after recipe construction"):
            set_per_site_config(recipe, {"site-1": {}, "site-2": {}})

    def test_per_site_empty_command_override_is_preserved(self, mock_file_system, base_recipe_params, simple_model):
        """Falsy per-site override values (e.g. command='') must not be replaced by defaults."""
        recipe = FedAvgRecipe(
            name="test_empty_command_override",
            model=simple_model,
            launch_external_process=True,
            **base_recipe_params,
        )
        set_per_site_config(recipe, {"site-1": {"command": ""}, "site-2": {}})
        recipe._ensure_client_apps_prepared()

        site_app = recipe._job._deploy_map.get("site-1")
        assert site_app is not None
        launcher = site_app.app_config.components.get("launcher")
        assert launcher is not None
        assert "python3 -u" not in launcher._script
        assert launcher._script.startswith(" custom/")

    def test_custom_model_persistor_tracks_persistor_id(self, mock_file_system, base_recipe_params, simple_model):
        """Custom PT persistor path should persist comp_ids['persistor_id'] for later workflows."""
        recipe = FedAvgRecipe(
            name="test_custom_persistor_comp_id",
            model=simple_model,
            model_persistor=DummyPersistor(),
            **base_recipe_params,
        )

        persistor_id = recipe._job.comp_ids.get("persistor_id", "")
        assert persistor_id
        assert "locator_id" not in recipe._job.comp_ids
        server_app = recipe._job._deploy_map.get(SERVER_SITE_NAME)
        assert server_app is not None
        assert persistor_id in server_app.app_config.components

    def test_custom_model_persistor_with_locator_registers_locator(
        self, mock_file_system, base_recipe_params, simple_model
    ):
        """If custom model_locator is provided, it should be registered even on custom persistor path."""
        locator = DummyLocator()
        recipe = FedAvgRecipe(
            name="test_custom_persistor_with_locator",
            model=simple_model,
            model_persistor=DummyPersistor(),
            model_locator=locator,
            **base_recipe_params,
        )

        assert recipe._job.comp_ids.get("persistor_id", "")
        locator_id = recipe._job.comp_ids.get("locator_id", "")
        assert locator_id
        server_app = recipe._job._deploy_map.get(SERVER_SITE_NAME)
        assert server_app is not None
        assert server_app.app_config.components.get(locator_id) is locator

    def test_dict_config_missing_class_path_or_path_raises_error(self, mock_file_system, base_recipe_params):
        """Test that dict config without 'class_path' or 'path' key raises error."""
        with pytest.raises(ValueError, match="must have 'class_path' or 'path' key"):
            FedAvgRecipe(
                name="test_invalid_dict",
                model={"args": {"input_size": 10}},  # Missing 'class_path'/'path'
                **base_recipe_params,
            )

    def test_dict_config_path_not_string_raises_error(self, mock_file_system, base_recipe_params):
        """Test that dict config with non-string 'class_path' raises error."""
        with pytest.raises(ValueError, match="'class_path' must be a string"):
            FedAvgRecipe(
                name="test_invalid_path_type",
                model={"class_path": 123, "args": {}},  # class_path is not string
                **base_recipe_params,
            )


class TestFedAvgRecipeInitialCkpt:
    """Test initial_ckpt parameter for FedAvgRecipe."""

    def test_initial_ckpt_parameter_accepted(self, mock_file_system, base_recipe_params, simple_model):
        """Test that initial_ckpt parameter is accepted."""
        recipe = FedAvgRecipe(
            name="test_initial_ckpt",
            model=simple_model,
            initial_ckpt="/abs/path/to/model.pt",
            **base_recipe_params,
        )

        assert recipe.initial_ckpt == "/abs/path/to/model.pt"
        assert recipe.model == simple_model

    def test_initial_ckpt_with_none_model_not_allowed_for_pt(self, mock_file_system, base_recipe_params):
        """Test that PT FedAvg rejects initial_ckpt with None model (PT needs architecture)."""
        # PyTorch requires model architecture even when loading from checkpoint
        # TensorFlow can load full models, but PT cannot
        with pytest.raises(ValueError, match="FrameworkType.PYTORCH requires 'model' when using initial_ckpt"):
            FedAvgRecipe(
                name="test_ckpt_no_model",
                model=None,
                initial_ckpt="/abs/path/to/model.pt",
                **base_recipe_params,
            )

    def test_initial_ckpt_must_exist_for_relative_path(self, base_recipe_params, simple_model):
        """Test that non-existent relative paths are rejected."""
        with pytest.raises(ValueError, match="does not exist locally"):
            FedAvgRecipe(
                name="test_relative_path",
                model=simple_model,
                initial_ckpt="relative/path/model.pt",
                **base_recipe_params,
            )

    def test_dict_model_config_accepted(self, mock_file_system, base_recipe_params):
        """Test that dict model config (class_path) is accepted and normalized to path for job API."""
        model_config = {
            "class_path": "my_module.models.SimpleNet",
            "args": {"input_size": 10, "output_size": 5},
        }
        recipe = FedAvgRecipe(
            name="test_dict_config",
            model=model_config,
            **base_recipe_params,
        )

        assert recipe.model["path"] == "my_module.models.SimpleNet"
        assert recipe.model["args"] == {"input_size": 10, "output_size": 5}

    def test_dict_model_config_path_alias_accepted(self, mock_file_system, base_recipe_params):
        """Test that dict model config accepts path as an alias for class_path."""
        model_config = {
            "path": "my_module.models.SimpleNet",
            "args": {"input_size": 10, "output_size": 5},
        }
        recipe = FedAvgRecipe(
            name="test_dict_config_path_alias",
            model=model_config,
            **base_recipe_params,
        )

        assert recipe.model["path"] == "my_module.models.SimpleNet"
        assert recipe.model["args"] == {"input_size": 10, "output_size": 5}

    def test_dict_model_config_class_path_takes_precedence(self, mock_file_system, base_recipe_params):
        """Test that class_path is used when both class_path and path are provided."""
        model_config = {
            "class_path": "my_module.models.ClassPathNet",
            "path": "my_module.models.PathNet",
            "args": {"input_size": 10},
        }
        recipe = FedAvgRecipe(
            name="test_dict_config_class_path_precedence",
            model=model_config,
            **base_recipe_params,
        )

        assert recipe.model["path"] == "my_module.models.ClassPathNet"
        assert recipe.model["args"] == {"input_size": 10}

    def test_dict_model_config_explicit_none_class_path_raises(self, mock_file_system, base_recipe_params):
        """Test that explicit class_path=None does not fall through to path alias."""
        model_config = {
            "class_path": None,
            "path": "my_module.models.PathNet",
            "args": {"input_size": 10},
        }
        with pytest.raises(ValueError, match="'class_path' must be a string"):
            FedAvgRecipe(
                name="test_dict_config_none_class_path",
                model=model_config,
                **base_recipe_params,
            )

    def test_dict_model_config_with_initial_ckpt(self, mock_file_system, base_recipe_params):
        """Test that dict model config (class_path) with initial_ckpt is accepted."""
        model_config = {
            "class_path": "my_module.models.SimpleNet",
            "args": {"input_size": 10},
        }
        recipe = FedAvgRecipe(
            name="test_dict_with_ckpt",
            model=model_config,
            initial_ckpt="/abs/path/to/pretrained.pt",
            **base_recipe_params,
        )

        assert recipe.model["path"] == "my_module.models.SimpleNet"
        assert recipe.model["args"] == {"input_size": 10}
        assert recipe.initial_ckpt == "/abs/path/to/pretrained.pt"

    def test_unified_numpy_initial_ckpt_only(self, mock_file_system, base_recipe_params):
        """Test unified FedAvgRecipe supports NumPy initial_ckpt without a custom persistor."""
        from nvflare.app_common.np.np_model_persistor import NPModelPersistor
        from nvflare.client.config import ExchangeFormat
        from nvflare.fuel.utils.constants import FrameworkType
        from nvflare.recipe import FedAvgRecipe as UnifiedFedAvgRecipe

        recipe = UnifiedFedAvgRecipe(
            name="test_unified_numpy_ckpt",
            model=None,
            initial_ckpt="/abs/path/to/model.npy",
            framework=FrameworkType.NUMPY,
            server_expected_format=ExchangeFormat.NUMPY,
            **base_recipe_params,
        )

        assert recipe.initial_ckpt == "/abs/path/to/model.npy"
        server_app = recipe._job._deploy_map[SERVER_SITE_NAME]
        persistor = server_app.app_config.components.get("persistor")
        assert isinstance(persistor, NPModelPersistor)
        assert persistor.source_ckpt_file_full_name == "/abs/path/to/model.npy"

    def test_unified_numpy_array_model(self, mock_file_system, base_recipe_params):
        """Test unified FedAvgRecipe converts NumPy array models for NPModelPersistor."""
        import numpy as np

        from nvflare.app_common.np.np_model_persistor import NPModelPersistor
        from nvflare.client.config import ExchangeFormat
        from nvflare.fuel.utils.constants import FrameworkType
        from nvflare.recipe import FedAvgRecipe as UnifiedFedAvgRecipe

        recipe = UnifiedFedAvgRecipe(
            name="test_unified_numpy_model",
            model=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            framework=FrameworkType.NUMPY,
            server_expected_format=ExchangeFormat.NUMPY,
            **base_recipe_params,
        )

        server_app = recipe._job._deploy_map[SERVER_SITE_NAME]
        persistor = server_app.app_config.components.get("persistor")
        assert isinstance(persistor, NPModelPersistor)
        assert persistor.model == [1.0, 2.0, 3.0]


class TestFedAvgRecipeDictConfigJobExport:
    """Test that dict model config works end-to-end with job export."""

    def test_dict_config_job_export(self, tmp_path):
        """Test that a recipe with dict config can export a valid job."""
        import os

        # Create a real temp train script (don't use mock_file_system - it breaks os.makedirs)
        train_script = str(tmp_path / "train.py")
        with open(train_script, "w") as f:
            f.write("# Dummy train script\n")

        model_config = {
            "class_path": "model.SimpleNetwork",
            "args": {},
        }
        recipe = FedAvgRecipe(
            name="test_dict_export",
            model=model_config,
            train_script=train_script,
            train_args="--epochs 10",
            min_clients=2,
            num_rounds=5,
        )

        # Export the job - this validates the config is properly processed
        job_dir = str(tmp_path / "exported_job")
        recipe.export(job_dir=job_dir)

        # Verify export created the job directory
        assert os.path.exists(job_dir)
        assert os.path.exists(os.path.join(job_dir, "test_dict_export"))

    def test_dict_config_with_ckpt_job_export(self, tmp_path):
        """Test that a recipe with dict config and initial_ckpt can export a valid job."""
        import os

        # Create a real temp train script (don't use mock_file_system - it breaks os.makedirs)
        train_script = str(tmp_path / "train.py")
        with open(train_script, "w") as f:
            f.write("# Dummy train script\n")

        model_config = {
            "class_path": "model.SimpleNetwork",
            "args": {"num_classes": 10},
        }
        recipe = FedAvgRecipe(
            name="test_dict_ckpt_export",
            model=model_config,
            initial_ckpt="/server/path/to/pretrained.pt",
            train_script=train_script,
            train_args="--epochs 10",
            min_clients=2,
            num_rounds=5,
        )

        # Export the job
        job_dir = str(tmp_path / "exported_job_ckpt")
        recipe.export(job_dir=job_dir)

        # Verify export created the job directory
        assert os.path.exists(job_dir)
        assert os.path.exists(os.path.join(job_dir, "test_dict_ckpt_export"))


class TestFedAvgRecipeExternalProcessStartup:
    """Regression coverage for recipe-based external-process executor startup."""

    @pytest.mark.parametrize("top_level_max_resends,expected_max_resends", [(None, 3), (5, 5)])
    def test_exported_external_process_executor_startup_uses_bounded_max_resends(
        self, tmp_path, top_level_max_resends, expected_max_resends
    ):
        """Simulate the rc4 failure path: export recipe config, reload executor args, then initialize."""
        train_script = tmp_path / "train.py"
        train_script.write_text("# Dummy train script\n")
        job_name = f"test_external_process_max_resends_{expected_max_resends}"

        recipe = FedAvgRecipe(
            name=job_name,
            model={"class_path": "model.SimpleNetwork", "args": {}},
            train_script=str(train_script),
            train_args="--epochs 1",
            min_clients=2,
            num_rounds=1,
            launch_external_process=True,
        )
        if top_level_max_resends is not None:
            recipe.add_client_config({ConfigKey.MAX_RESENDS: top_level_max_resends})

        export_dir = tmp_path / "exported_job"
        recipe.export(job_dir=str(export_dir))

        config_dir = export_dir / job_name / "app" / "config"
        client_config_path = config_dir / "config_fed_client.json"
        with open(client_config_path, "r") as f:
            client_config = json.load(f)

        train_executor = _get_train_executor_config(client_config)
        train_executor_args = train_executor.get("args", {})

        assert "PTClientAPILauncherExecutor" in train_executor["path"]
        assert train_executor_args[ConfigKey.MAX_RESENDS] == 3
        if top_level_max_resends is not None:
            assert client_config[ConfigKey.MAX_RESENDS] == top_level_max_resends

        client_api_config = _run_exported_external_process_executor_startup(
            train_executor, config_dir=config_dir, job_id=job_name
        )

        assert client_api_config[ConfigKey.TASK_EXCHANGE][ConfigKey.MAX_RESENDS] == expected_max_resends

    def test_old_rc4_exported_null_max_resends_still_rejects_at_startup(self, tmp_path):
        """Old exported configs with executor max_resends: null must fail instead of running unbounded."""
        train_script = tmp_path / "train.py"
        train_script.write_text("# Dummy train script\n")
        job_name = "test_external_process_null_max_resends"

        recipe = FedAvgRecipe(
            name=job_name,
            model={"class_path": "model.SimpleNetwork", "args": {}},
            train_script=str(train_script),
            train_args="--epochs 1",
            min_clients=2,
            num_rounds=1,
            launch_external_process=True,
        )

        export_dir = tmp_path / "exported_job"
        recipe.export(job_dir=str(export_dir))

        config_dir = export_dir / job_name / "app" / "config"
        client_config_path = config_dir / "config_fed_client.json"
        with open(client_config_path, "r") as f:
            client_config = json.load(f)

        train_executor = _get_train_executor_config(client_config)
        train_executor["args"][ConfigKey.MAX_RESENDS] = None
        client_config_path.write_text(json.dumps(client_config))
        with open(client_config_path, "r") as f:
            reloaded_client_config = json.load(f)
        reloaded_train_executor = _get_train_executor_config(reloaded_client_config)

        with pytest.raises(ValueError, match="max_resends is None"):
            _run_exported_external_process_executor_startup(
                reloaded_train_executor, config_dir=config_dir, job_id=job_name
            )

        assert not (config_dir / CLIENT_API_CONFIG).exists()


class TestNumpyFedAvgRecipeInitialCkpt:
    """Test initial_ckpt parameter for NumpyFedAvgRecipe."""

    def test_numpy_initial_ckpt_accepted(self, mock_file_system):
        """Test that initial_ckpt parameter is accepted for NumPy recipe."""
        recipe = NumpyFedAvgRecipe(
            name="test_numpy_ckpt",
            model=[1.0, 2.0, 3.0],
            initial_ckpt="/abs/path/to/model.npy",
            min_clients=2,
            train_script="client.py",
        )

        assert recipe._np_initial_ckpt == "/abs/path/to/model.npy"

    def test_numpy_initial_ckpt_only(self, mock_file_system):
        """Test that NumPy recipe works with initial_ckpt only (no model)."""
        # NumPy can load model from checkpoint without architecture
        recipe = NumpyFedAvgRecipe(
            name="test_numpy_ckpt_only",
            model=None,
            initial_ckpt="/abs/path/to/model.npy",
            min_clients=2,
            train_script="client.py",
        )

        assert recipe._np_initial_ckpt == "/abs/path/to/model.npy"
        assert recipe._np_model is None
