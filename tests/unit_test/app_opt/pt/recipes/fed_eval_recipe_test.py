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

import warnings
from unittest.mock import patch

import pytest
import torch.nn as nn
from pydantic import ValidationError

from nvflare.apis.job_def import ALL_SITES
from nvflare.app_opt.pt.recipes.fedeval import FedEvalRecipe
from nvflare.client.config import ExchangeFormat
from nvflare.fuel.utils.secret_utils import PotentialSecretWarning, UnsupportedSecretRefWarning
from nvflare.recipe import set_per_site_config


class SimpleTestModel(nn.Module):
    """A simple PyTorch model for testing purposes."""

    def __init__(self, checkpoint=None):
        super().__init__()
        self.lin = nn.Linear(10, 10)
        self.checkpoint = checkpoint

    def forward(self, x):
        x = self.lin(x)
        return x


@pytest.fixture
def mock_file_system():
    """Mock file system operations for all tests."""
    with patch("os.path.isfile", return_value=True), patch("os.path.exists", return_value=True):
        yield


@pytest.fixture
def simple_model(tmp_path):
    """Create a simple test model with a checkpoint."""
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()
    return SimpleTestModel(checkpoint=str(checkpoint_path)), str(checkpoint_path)


@pytest.fixture
def base_recipe_params(simple_model):
    """Base parameters for creating FedEvalRecipe instances (requires checkpoint for validation)."""
    _, checkpoint_path = simple_model
    return {
        "eval_script": "mock_eval_script.py",
        "eval_args": "--batch_size 32",
        "min_clients": 2,
        "eval_ckpt": checkpoint_path,
    }


def assert_recipe_basics(recipe, expected_name, expected_params):
    """Helper to assert basic recipe properties."""
    assert recipe.name == expected_name
    assert recipe.eval_script == expected_params.get("eval_script", "mock_eval_script.py")
    assert recipe.eval_args == expected_params.get("eval_args", "--batch_size 32")
    assert recipe.min_clients == expected_params.get("min_clients", 2)
    assert recipe.job is not None
    assert recipe.job.name == expected_name


def get_client_executor(recipe, site_name):
    return recipe.job._deploy_map[site_name].app_config.executors[0].executor


class TestFedEvalRecipe:
    """Test cases for FedEvalRecipe class."""

    def test_warns_on_secret_in_eval_args(self, mock_file_system, base_recipe_params, simple_model):
        model, _ = simple_model
        params = dict(base_recipe_params)
        params["eval_args"] = "--password hunter22x"

        recipe = FedEvalRecipe(name="secret_eval", model=model, **params)

        with pytest.warns(PotentialSecretWarning, match="eval_args"):
            recipe._warn_potential_secrets_in_params()

    def test_external_command_secret_ref_is_supported(self, mock_file_system, base_recipe_params, simple_model):
        model, _ = simple_model
        with warnings.catch_warnings():
            warnings.simplefilter("error", UnsupportedSecretRefWarning)
            recipe = FedEvalRecipe(
                name="command_secret_ref",
                model=model,
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

    def test_basic_initialization(self, mock_file_system, base_recipe_params, simple_model):
        """Test FedEvalRecipe initialization with default parameters."""
        model, checkpoint_path = simple_model
        recipe = FedEvalRecipe(name="test_fed_eval", model=model, **base_recipe_params)

        assert_recipe_basics(recipe, "test_fed_eval", base_recipe_params)
        assert recipe.model == model
        assert recipe.eval_ckpt == checkpoint_path
        assert recipe.launch_external_process is False
        assert recipe.command == "python3 -u"
        assert recipe.server_expected_format == ExchangeFormat.NUMPY
        assert recipe.validation_timeout == 6000
        assert recipe.per_site_config is None

    def test_set_per_site_config_builds_site_specific_runners(self, mock_file_system, base_recipe_params, simple_model):
        model, _ = simple_model
        recipe = FedEvalRecipe(name="test_helper_per_site", model=model, **base_recipe_params)
        config = {"site-1": {"eval_args": "--batch_size 8"}, "site-2": {}}

        assert recipe.job.clients == [ALL_SITES]
        set_per_site_config(recipe, config)

        assert recipe.configured_sites() == ["site-1", "site-2"]
        assert recipe.job.clients == ["site-1", "site-2"]
        assert ALL_SITES not in recipe.job._deploy_map
        assert get_client_executor(recipe, "site-1")._task_script_args == "--batch_size 8"
        assert get_client_executor(recipe, "site-2")._task_script_args == "--batch_size 32"

    def test_default_job_name(self, mock_file_system, base_recipe_params, simple_model):
        """Test FedEvalRecipe with default job name."""
        model, _ = simple_model
        recipe = FedEvalRecipe(model=model, **base_recipe_params)

        assert recipe.name == "eval"
        assert recipe.job.name == "eval"

    def test_custom_command(self, mock_file_system, base_recipe_params, simple_model):
        """Test FedEvalRecipe with custom command."""
        model, _ = simple_model
        recipe = FedEvalRecipe(
            name="test_custom_cmd",
            model=model,
            command="python3 -m torch.distributed.run",
            **base_recipe_params,
        )

        assert recipe.command == "python3 -m torch.distributed.run"

    def test_launch_external_process(self, mock_file_system, base_recipe_params, simple_model):
        """Test FedEvalRecipe with external process launch."""
        model, _ = simple_model
        recipe = FedEvalRecipe(
            name="test_external",
            model=model,
            launch_external_process=True,
            **base_recipe_params,
        )

        assert recipe.launch_external_process is True

    def test_custom_exchange_format(self, mock_file_system, base_recipe_params, simple_model):
        """Test FedEvalRecipe with custom exchange format."""
        model, _ = simple_model
        recipe = FedEvalRecipe(
            name="test_format",
            model=model,
            server_expected_format=ExchangeFormat.PYTORCH,
            **base_recipe_params,
        )

        assert recipe.server_expected_format == ExchangeFormat.PYTORCH

    def test_custom_timeout(self, mock_file_system, base_recipe_params, simple_model):
        """Test FedEvalRecipe with custom validation timeout."""
        model, _ = simple_model
        recipe = FedEvalRecipe(
            name="test_timeout",
            model=model,
            validation_timeout=3000,
            **base_recipe_params,
        )

        assert recipe.validation_timeout == 3000

    @pytest.mark.parametrize(
        "min_clients,eval_args,validation_timeout",
        [
            (1, "", 6000),  # Minimum configuration
            (2, "--batch_size 64", 3000),  # Standard configuration
            (5, "--batch_size 128 --num_workers 4", 9000),  # Complex configuration
        ],
    )
    def test_recipe_configurations(self, mock_file_system, min_clients, eval_args, validation_timeout, simple_model):
        """Test various FedEvalRecipe configurations using parametrized tests."""
        model, checkpoint_path = simple_model
        recipe = FedEvalRecipe(
            name=f"test_config_{min_clients}",
            model=model,
            eval_ckpt=checkpoint_path,
            eval_script="mock_eval_script.py",
            eval_args=eval_args,
            min_clients=min_clients,
            validation_timeout=validation_timeout,
        )

        expected_params = {
            "eval_script": "mock_eval_script.py",
            "eval_args": eval_args,
            "min_clients": min_clients,
        }
        assert_recipe_basics(recipe, f"test_config_{min_clients}", expected_params)
        assert recipe.validation_timeout == validation_timeout

    def test_per_site_config(self, mock_file_system, base_recipe_params, simple_model):
        """Test FedEvalRecipe with per-site configuration."""
        model, _ = simple_model
        per_site_config = {
            "site-1": {
                "eval_script": "site1_eval.py",
                "eval_args": "--batch_size 16",
                "launch_external_process": True,
                "command": "python3",
                "server_expected_format": ExchangeFormat.PYTORCH,
            },
            "site-2": {
                "eval_args": "--batch_size 64",
                "validation_timeout": 7200,
            },
        }

        with pytest.warns(FutureWarning, match="set_per_site_config"):
            recipe = FedEvalRecipe(
                name="test_per_site",
                model=model,
                per_site_config=per_site_config,
                **base_recipe_params,
            )

        assert recipe.per_site_config == per_site_config

    def test_per_site_config_partial_overrides(self, mock_file_system, base_recipe_params, simple_model):
        """Test per-site configuration with partial overrides."""
        model, _ = simple_model
        per_site_config = {
            "site-1": {
                "eval_args": "--batch_size 16",
            },
            "site-2": {},
        }

        recipe = FedEvalRecipe(name="test_partial_override", model=model, **base_recipe_params)
        set_per_site_config(recipe, per_site_config)

        # Check that default values are stored
        assert recipe.eval_script == "mock_eval_script.py"
        assert recipe.eval_args == "--batch_size 32"
        assert recipe.per_site_config == per_site_config


class TestFedEvalRecipeValidation:
    """Test FedEvalRecipe input validation."""

    def test_per_site_config_requires_at_least_min_clients(self, mock_file_system, base_recipe_params, simple_model):
        model, _ = simple_model
        recipe = FedEvalRecipe(name="test_per_site_client_count", model=model, **base_recipe_params)

        with pytest.raises(ValueError, match=r"defines 1 site.*min_clients=2"):
            set_per_site_config(recipe, {"site-1": {}})

        assert recipe.job.clients == [ALL_SITES]

    def test_invalid_eval_ckpt_raises_error(self, simple_model):
        """Test that relative eval_ckpt raises when path does not exist locally.

        validate_ckpt() allows relative paths if the file exists locally;
        it only raises when the relative path is missing. We do not use
        mock_file_system here so the relative path is not considered existing,
        and the validator raises (Pydantic wraps in ValidationError).
        """
        model, _ = simple_model

        with pytest.raises(
            ValidationError,
            match="relative path does not exist locally|initial_ckpt must be an absolute path",
        ):
            FedEvalRecipe(
                name="test_invalid_ckpt",
                model=model,
                eval_ckpt="relative/path/model.pt",
                min_clients=2,
                eval_script="mock_eval_script.py",
                eval_args="--batch_size 32",
            )

    def test_non_nn_module_raises_error(self, mock_file_system, base_recipe_params, simple_model):
        """Test that non-nn.Module model raises ValueError."""

        class NotAModule:
            pass

        invalid_model = NotAModule()

        with pytest.raises(ValueError, match="model must be nn.Module or dict config"):
            FedEvalRecipe(
                name="test_invalid_model",
                model=invalid_model,
                **base_recipe_params,
            )

    def test_model_with_eval_ckpt(self, mock_file_system, base_recipe_params, simple_model):
        """Test that model with valid eval_ckpt works."""
        model, checkpoint_path = simple_model

        recipe = FedEvalRecipe(
            name="test_valid_checkpoint",
            model=model,
            **base_recipe_params,
        )

        assert recipe.eval_ckpt == checkpoint_path


class TestFedEvalRecipeEdgeCases:
    """Test edge cases for FedEvalRecipe."""

    def test_empty_eval_args(self, mock_file_system, simple_model):
        """Test FedEvalRecipe with empty eval_args."""
        model, checkpoint_path = simple_model
        recipe = FedEvalRecipe(
            name="test_empty_args",
            model=model,
            eval_ckpt=checkpoint_path,
            eval_script="eval.py",
            eval_args="",
            min_clients=1,
        )

        assert recipe.eval_args == ""

    def test_single_client(self, mock_file_system, base_recipe_params, simple_model):
        """Test FedEvalRecipe with single client."""
        model, _ = simple_model
        params = {**base_recipe_params, "min_clients": 1, "eval_script": "eval.py"}
        recipe = FedEvalRecipe(
            name="test_single_client",
            model=model,
            **params,
        )

        assert recipe.min_clients == 1

    def test_large_timeout(self, mock_file_system, base_recipe_params, simple_model):
        """Test FedEvalRecipe with very large timeout."""
        model, _ = simple_model
        recipe = FedEvalRecipe(
            name="test_large_timeout",
            model=model,
            validation_timeout=999999,
            **base_recipe_params,
        )

        assert recipe.validation_timeout == 999999

    @pytest.mark.parametrize(
        "exchange_format",
        [
            ExchangeFormat.NUMPY,
            ExchangeFormat.PYTORCH,
            ExchangeFormat.RAW,
        ],
    )
    def test_all_exchange_formats(self, mock_file_system, base_recipe_params, simple_model, exchange_format):
        """Test FedEvalRecipe with different exchange formats."""
        model, _ = simple_model
        recipe = FedEvalRecipe(
            name="test_format",
            model=model,
            server_expected_format=exchange_format,
            **base_recipe_params,
        )

        assert recipe.server_expected_format == exchange_format
