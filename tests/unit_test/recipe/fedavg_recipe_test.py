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

from unittest.mock import patch

import pytest
import torch.nn as nn

from nvflare.apis.job_def import SERVER_SITE_NAME
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
from nvflare.app_common.np.recipes import NumpyFedAvgRecipe
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe


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
    assert recipe.job is not None
    assert recipe.job.name == expected_name


def get_model_selector(recipe):
    server_app = recipe.job._deploy_map[SERVER_SITE_NAME]
    return server_app.app_config.components.get("model_selector")


class TestFedAvgRecipe:
    """Test cases for FedAvgRecipe class."""

    def test_default_aggregator_initialization(self, mock_file_system, base_recipe_params, simple_model):
        """Test FedAvgRecipe initialization with default (built-in) aggregation."""
        recipe = FedAvgRecipe(name="test_fedavg", initial_model=simple_model, **base_recipe_params)

        assert_recipe_basics(recipe, "test_fedavg", base_recipe_params)
        assert recipe.initial_model == simple_model
        # When no aggregator is passed, built-in weighted averaging is used
        assert recipe.aggregator is None

    def test_key_metric_passthrough_pt(self, mock_file_system, base_recipe_params, simple_model):
        key_metric = "val_auc"
        recipe = FedAvgRecipe(
            name="test_fedavg_key_metric", initial_model=simple_model, key_metric=key_metric, **base_recipe_params
        )

        model_selector = get_model_selector(recipe)
        assert isinstance(model_selector, IntimeModelSelector)
        assert model_selector.key_metric == key_metric

    def test_custom_aggregator_initialization(
        self, mock_file_system, base_recipe_params, custom_aggregator, simple_model
    ):
        """Test FedAvgRecipe initialization with custom aggregator."""
        params = {**base_recipe_params, "min_clients": 1, "num_rounds": 3}
        recipe = FedAvgRecipe(
            name="test_fedavg_custom", initial_model=simple_model, aggregator=custom_aggregator, **params
        )

        assert_recipe_basics(recipe, "test_fedavg_custom", params)
        assert recipe.aggregator is custom_aggregator
        assert isinstance(recipe.aggregator, MyAggregator)

    def test_initial_model_configuration(self, mock_file_system, base_recipe_params, custom_aggregator, simple_model):
        """Test FedAvgRecipe with initial model."""
        params = {**base_recipe_params, "min_clients": 1, "num_rounds": 3}
        recipe = FedAvgRecipe(
            name="test_fedavg_initial_model", initial_model=simple_model, aggregator=custom_aggregator, **params
        )

        assert_recipe_basics(recipe, "test_fedavg_initial_model", params)
        assert recipe.initial_model == simple_model

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
            initial_model=simple_model,
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
            initial_model=[1.0, 2.0, 3.0],
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
            initial_model=[[1, 2, 3], [4, 5, 6]],
            min_clients=2,
            num_rounds=3,
            train_script="client.py",
        )

        assert recipe.name == "test_numpy"
        assert recipe.min_clients == 2
        assert recipe.num_rounds == 3
        assert recipe.job is not None

    def test_numpy_recipe_with_early_stopping(self, mock_file_system):
        """Test NumpyFedAvgRecipe with early stopping configuration."""
        recipe = NumpyFedAvgRecipe(
            name="test_numpy_early_stop",
            initial_model=[1.0, 2.0, 3.0],
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
            initial_model=[1.0, 2.0],
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
            initial_model=[1.0, 2.0, 3.0],
            min_clients=2,
            num_rounds=5,
            train_script="client.py",
            exclude_vars="bias.*",
        )

        assert recipe.exclude_vars == "bias.*"

    def test_numpy_recipe_with_save_filename(self, mock_file_system):
        """Test NumpyFedAvgRecipe with custom save filename."""
        recipe = NumpyFedAvgRecipe(
            name="test_numpy_save",
            initial_model=[1.0, 2.0, 3.0],
            min_clients=2,
            num_rounds=5,
            train_script="client.py",
            save_filename="numpy_model.pt",
        )

        assert recipe.save_filename == "numpy_model.pt"

    def test_numpy_recipe_with_per_site_config(self, mock_file_system):
        """Test NumpyFedAvgRecipe with per-site configuration."""
        per_site_config = {
            "site-1": {"train_args": "--data /path/to/site1"},
            "site-2": {"train_args": "--data /path/to/site2"},
        }
        recipe = NumpyFedAvgRecipe(
            name="test_numpy_per_site",
            initial_model=[1.0, 2.0],
            min_clients=2,
            num_rounds=3,
            train_script="client.py",
            per_site_config=per_site_config,
        )

        assert recipe.per_site_config == per_site_config

    def test_numpy_recipe_with_none_initial_model_raises_error(self, mock_file_system):
        """Test NumpyFedAvgRecipe with no model raises error."""
        with pytest.raises(ValueError, match="Must provide either initial_model"):
            NumpyFedAvgRecipe(
                name="test_numpy_no_model",
                initial_model=None,
                min_clients=2,
                num_rounds=3,
                train_script="client.py",
            )

    def test_numpy_recipe_full_configuration(self, mock_file_system):
        """Test NumpyFedAvgRecipe with all new features."""
        recipe = NumpyFedAvgRecipe(
            name="test_numpy_full",
            initial_model=[[1, 2], [3, 4], [5, 6]],
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

        assert recipe.name == "test_numpy_full"
        assert recipe.min_clients == 3
        assert recipe.num_rounds == 20
        assert recipe.train_script == "train.py"
        assert recipe.train_args == "--epochs 10"
        assert recipe.launch_external_process is True
        assert recipe.key_metric == "f1_score"
        assert recipe.stop_cond == "f1_score >= 0.9"
        assert recipe.patience == 5
        assert recipe.save_filename == "best_numpy_model.pt"
        assert recipe.exclude_vars == "temp_.*"
        assert recipe.aggregation_weights == {"site-1": 1.0, "site-2": 2.0, "site-3": 1.5}


class TestFedAvgRecipeEarlyStopping:
    """Test early stopping configuration for FedAvgRecipe."""

    def test_early_stopping_configuration(self, mock_file_system, base_recipe_params, simple_model):
        """Test FedAvgRecipe with early stopping configuration."""
        recipe = FedAvgRecipe(
            name="test_early_stop",
            initial_model=simple_model,
            stop_cond="accuracy >= 80",
            patience=5,
            **base_recipe_params,
        )

        assert_recipe_basics(recipe, "test_early_stop", base_recipe_params)
        assert recipe.stop_cond == "accuracy >= 80"
        assert recipe.patience == 5

    def test_save_filename_configuration(self, mock_file_system, base_recipe_params, simple_model):
        """Test FedAvgRecipe with custom save filename."""
        recipe = FedAvgRecipe(
            name="test_save_file",
            initial_model=simple_model,
            save_filename="best_model.pt",
            **base_recipe_params,
        )

        assert recipe.save_filename == "best_model.pt"

    def test_exclude_vars_configuration(self, mock_file_system, base_recipe_params, simple_model):
        """Test FedAvgRecipe with exclude_vars configuration."""
        recipe = FedAvgRecipe(
            name="test_exclude",
            initial_model=simple_model,
            exclude_vars="bn.*|running_mean|running_var",
            **base_recipe_params,
        )

        assert recipe.exclude_vars == "bn.*|running_mean|running_var"

    def test_aggregation_weights_configuration(self, mock_file_system, base_recipe_params, simple_model):
        """Test FedAvgRecipe with per-client aggregation weights."""
        weights = {"site-1": 2.0, "site-2": 1.0}
        recipe = FedAvgRecipe(
            name="test_weights",
            initial_model=simple_model,
            aggregation_weights=weights,
            **base_recipe_params,
        )

        assert recipe.aggregation_weights == weights


class TestFedAvgRecipeValidation:
    """Test FedAvgRecipe input validation."""

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

    def test_dict_config_missing_path_raises_error(self, mock_file_system, base_recipe_params):
        """Test that dict config without 'path' key raises error."""
        with pytest.raises(ValueError, match="must have 'path' key"):
            FedAvgRecipe(
                name="test_invalid_dict",
                initial_model={"args": {"input_size": 10}},  # Missing 'path'
                **base_recipe_params,
            )

    def test_dict_config_path_not_string_raises_error(self, mock_file_system, base_recipe_params):
        """Test that dict config with non-string 'path' raises error."""
        with pytest.raises(ValueError, match="'path' must be a string"):
            FedAvgRecipe(
                name="test_invalid_path_type",
                initial_model={"path": 123, "args": {}},  # Path is not string
                **base_recipe_params,
            )


class TestFedAvgRecipeInitialCkpt:
    """Test initial_ckpt parameter for FedAvgRecipe."""

    def test_initial_ckpt_parameter_accepted(self, mock_file_system, base_recipe_params, simple_model):
        """Test that initial_ckpt parameter is accepted."""
        recipe = FedAvgRecipe(
            name="test_initial_ckpt",
            initial_model=simple_model,
            initial_ckpt="/abs/path/to/model.pt",
            **base_recipe_params,
        )

        assert recipe.initial_ckpt == "/abs/path/to/model.pt"
        assert recipe.initial_model == simple_model

    def test_initial_ckpt_with_none_model_not_allowed_for_pt(self, mock_file_system, base_recipe_params):
        """Test that PT FedAvg rejects initial_ckpt with None model (PT needs architecture)."""
        # PyTorch requires model architecture even when loading from checkpoint
        # TensorFlow can load full models, but PT cannot
        with pytest.raises(ValueError, match="Unable to add None to job"):
            FedAvgRecipe(
                name="test_ckpt_no_model",
                initial_model=None,
                initial_ckpt="/abs/path/to/model.pt",
                **base_recipe_params,
            )

    def test_initial_ckpt_must_be_absolute_path(self, base_recipe_params, simple_model):
        """Test that relative paths are rejected (without mock to allow validation)."""
        with pytest.raises(ValueError, match="must be an absolute path"):
            FedAvgRecipe(
                name="test_relative_path",
                initial_model=simple_model,
                initial_ckpt="relative/path/model.pt",
                **base_recipe_params,
            )

    def test_dict_model_config_accepted(self, mock_file_system, base_recipe_params):
        """Test that dict model config is accepted."""
        model_config = {
            "path": "my_module.models.SimpleNet",
            "args": {"input_size": 10, "output_size": 5},
        }
        recipe = FedAvgRecipe(
            name="test_dict_config",
            initial_model=model_config,
            **base_recipe_params,
        )

        assert recipe.initial_model == model_config

    def test_dict_model_config_with_initial_ckpt(self, mock_file_system, base_recipe_params):
        """Test that dict model config with initial_ckpt is accepted."""
        model_config = {
            "path": "my_module.models.SimpleNet",
            "args": {"input_size": 10},
        }
        recipe = FedAvgRecipe(
            name="test_dict_with_ckpt",
            initial_model=model_config,
            initial_ckpt="/abs/path/to/pretrained.pt",
            **base_recipe_params,
        )

        assert recipe.initial_model == model_config
        assert recipe.initial_ckpt == "/abs/path/to/pretrained.pt"


class TestFedAvgRecipeDictConfigJobExport:
    """Test that dict model config works end-to-end with job export."""

    def test_dict_config_job_export(self, mock_file_system, base_recipe_params, tmp_path):
        """Test that a recipe with dict config can export a valid job."""
        model_config = {
            "path": "model.SimpleNetwork",
            "args": {},
        }
        recipe = FedAvgRecipe(
            name="test_dict_export",
            initial_model=model_config,
            **base_recipe_params,
        )

        # Export the job - this validates the config is properly processed
        job_dir = str(tmp_path / "exported_job")
        recipe.export(job_dir=job_dir)

        # Verify export created the job directory
        import os

        assert os.path.exists(job_dir)
        assert os.path.exists(os.path.join(job_dir, "test_dict_export"))

    def test_dict_config_with_ckpt_job_export(self, mock_file_system, base_recipe_params, tmp_path):
        """Test that a recipe with dict config and initial_ckpt can export a valid job."""
        model_config = {
            "path": "model.SimpleNetwork",
            "args": {"num_classes": 10},
        }
        recipe = FedAvgRecipe(
            name="test_dict_ckpt_export",
            initial_model=model_config,
            initial_ckpt="/server/path/to/pretrained.pt",
            **base_recipe_params,
        )

        # Export the job
        job_dir = str(tmp_path / "exported_job_ckpt")
        recipe.export(job_dir=job_dir)

        # Verify export created the job directory
        import os

        assert os.path.exists(job_dir)
        assert os.path.exists(os.path.join(job_dir, "test_dict_ckpt_export"))


class TestNumpyFedAvgRecipeInitialCkpt:
    """Test initial_ckpt parameter for NumpyFedAvgRecipe."""

    def test_numpy_initial_ckpt_accepted(self, mock_file_system):
        """Test that initial_ckpt parameter is accepted for NumPy recipe."""
        recipe = NumpyFedAvgRecipe(
            name="test_numpy_ckpt",
            initial_model=[1.0, 2.0, 3.0],
            initial_ckpt="/abs/path/to/model.npy",
            min_clients=2,
            train_script="client.py",
        )

        assert recipe._np_initial_ckpt == "/abs/path/to/model.npy"

    def test_numpy_initial_ckpt_only(self, mock_file_system):
        """Test that NumPy recipe works with initial_ckpt only (no initial_model)."""
        # NumPy can load model from checkpoint without architecture
        recipe = NumpyFedAvgRecipe(
            name="test_numpy_ckpt_only",
            initial_model=None,
            initial_ckpt="/abs/path/to/model.npy",
            min_clients=2,
            train_script="client.py",
        )

        assert recipe._np_initial_ckpt == "/abs/path/to/model.npy"
        assert recipe._np_initial_model is None
