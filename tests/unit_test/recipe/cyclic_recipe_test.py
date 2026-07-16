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

"""Tests for CyclicRecipe and framework-specific variants with initial_ckpt support."""

import warnings
from unittest.mock import patch

import pytest
import torch.nn as nn

from nvflare.apis.job_def import ALL_SITES
from nvflare.app_opt.pt.job_config.model import PTModel
from nvflare.fuel.utils.constants import FrameworkType
from nvflare.fuel.utils.secret_utils import PotentialSecretWarning, UnsupportedSecretRefWarning
from nvflare.recipe.cyclic import CyclicRecipe as BaseCyclicRecipe


class SimpleTestModel(nn.Module):
    """A simple PyTorch model for testing purposes."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(10, 10)

    def forward(self, x):
        return self.lin(x)


@pytest.fixture
def mock_file_system():
    """Mock file system operations for all tests."""
    with patch("os.path.isfile", return_value=True), patch("os.path.exists", return_value=True):
        yield


@pytest.fixture
def simple_model():
    """Create a simple test model."""
    return SimpleTestModel()


@pytest.fixture
def base_recipe_params():
    """Base parameters for creating CyclicRecipe instances."""
    return {
        "train_script": "mock_train_script.py",
        "train_args": "--epochs 10",
        "min_clients": 2,
        "num_rounds": 5,
    }


class TestBaseCyclicRecipe:
    """Test cases for base CyclicRecipe class."""

    def test_warns_on_secret_in_client_config_overrides(self, mock_file_system, base_recipe_params, simple_model):
        secret = "ghp_" + "Ab1" * 12

        recipe = BaseCyclicRecipe(
            name="secret_override",
            model=PTModel(model=simple_model),
            client_config_overrides={"command": secret},
            framework=FrameworkType.PYTORCH,
            **base_recipe_params,
        )

        with pytest.warns(PotentialSecretWarning) as record:
            recipe._warn_potential_secrets_in_params()

        messages = [str(warning.message) for warning in record]
        assert any("client_config_overrides" in message for message in messages)
        assert all(secret not in message for message in messages)

    def test_external_command_secret_ref_is_supported(self, mock_file_system, base_recipe_params, simple_model):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UnsupportedSecretRefWarning)
            BaseCyclicRecipe(
                name="command_secret_ref",
                model=PTModel(model=simple_model),
                launch_external_process=True,
                client_config_overrides={"command": "env API_TOKEN=${secret:API_TOKEN} python3 -u"},
                framework=FrameworkType.PYTORCH,
                **base_recipe_params,
            )

    def test_initial_ckpt_must_exist_for_relative_path(self):
        """Test that non-existent relative paths are rejected (no mock - validation must run)."""
        with pytest.raises(ValueError, match="does not exist locally"):
            BaseCyclicRecipe(
                name="test_relative",
                model=None,
                initial_ckpt="relative/path/model.pt",
                train_script="/abs/train.py",
                min_clients=2,
                num_rounds=2,
            )

    def test_requires_model_or_checkpoint(self, base_recipe_params):
        """Test that at least model or initial_ckpt must be provided."""
        with pytest.raises(ValueError, match="Must provide either model or initial_ckpt"):
            BaseCyclicRecipe(
                name="test_no_model",
                model=None,
                initial_ckpt=None,
                **base_recipe_params,
            )

    def test_rejects_non_wrapper_model_for_base_recipe(self, mock_file_system, base_recipe_params):
        """Base CyclicRecipe no longer owns PT/TF model persistence for raw model inputs."""
        with pytest.raises(ValueError, match="Use a framework-specific CyclicRecipe subclass"):
            BaseCyclicRecipe(
                name="test_base_pt_dict",
                model={"class_path": "torch.nn.Linear", "args": {"in_features": 10, "out_features": 2}},
                framework=FrameworkType.PYTORCH,
                **base_recipe_params,
            )

    def test_rejects_pytorch_checkpoint_without_model(self, mock_file_system, base_recipe_params):
        """Base recipe requires framework-specific subclass for PT checkpoint-only setup."""
        with pytest.raises(ValueError, match="Use a framework-specific CyclicRecipe subclass"):
            BaseCyclicRecipe(
                name="test_base_pt_ckpt_no_model",
                model=None,
                initial_ckpt="/abs/path/to/model.pt",
                framework=FrameworkType.PYTORCH,
                **base_recipe_params,
            )

    def test_rejects_ckpt_only_for_default_framework(self, mock_file_system, base_recipe_params):
        """Fail fast for ckpt-only usage when no supported framework/wrapper is provided."""
        with pytest.raises(ValueError, match="Unsupported framework"):
            BaseCyclicRecipe(
                name="test_ckpt_only_default_framework",
                model=None,
                initial_ckpt="/abs/path/to/model.pt",
                **base_recipe_params,
            )

    def test_applies_initial_ckpt_to_wrapper_model(self, mock_file_system, base_recipe_params, simple_model):
        """When wrapper model is used, recipe-level initial_ckpt should be applied to persistor."""
        recipe = BaseCyclicRecipe(
            name="test_wrapper_ckpt",
            model=PTModel(model=simple_model),
            initial_ckpt="/abs/path/to/model.pt",
            framework=FrameworkType.PYTORCH,
            **base_recipe_params,
        )

        server_app = recipe.job._deploy_map.get("server")
        persistor = server_app.app_config.components.get("persistor")
        assert persistor is not None
        assert getattr(persistor, "source_ckpt_file_full_name", None) == "/abs/path/to/model.pt"

    def test_rejects_unsupported_framework_without_wrapper(self, mock_file_system, base_recipe_params):
        """Fail fast for unsupported base framework/model persistence combinations."""
        with pytest.raises(ValueError, match="Unsupported framework"):
            BaseCyclicRecipe(
                name="test_unsupported_framework",
                model={"class_path": "torch.nn.Linear", "args": {"in_features": 10, "out_features": 2}},
                framework=FrameworkType.NUMPY,
                **base_recipe_params,
            )


class TestBaseCyclicRecipeAttributes:
    """Test that CyclicRecipe stores validated attributes correctly."""

    def test_min_clients_attribute(self, mock_file_system, base_recipe_params, simple_model):
        """min_clients must be accessible as an instance attribute after construction."""
        recipe = BaseCyclicRecipe(
            name="test_min_clients",
            model=PTModel(model=simple_model),
            framework=FrameworkType.PYTORCH,
            **base_recipe_params,
        )
        assert recipe.min_clients == base_recipe_params["min_clients"]


class TestCyclicRecipeControllerConfig:
    """Test named server/client timeouts and advanced config overrides."""

    def test_pt_recipe_parameters_and_override_precedence(self, mock_file_system, base_recipe_params, simple_model):
        from nvflare.app_opt.pt.recipes.cyclic import CyclicRecipe as PTCyclicRecipe

        recipe = PTCyclicRecipe(
            name="test_pt_cyclic_config",
            model=simple_model,
            launch_external_process=True,
            task_assignment_timeout=30,
            shutdown_timeout=45.0,
            server_config_overrides={"task_assignment_timeout": 60, "task_check_period": 2.0},
            client_config_overrides={"shutdown_timeout": 90.0, "launch_once": False},
            **base_recipe_params,
        )

        controller = recipe.job._deploy_map["server"].app_config.workflows[0].controller
        from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor

        client_api_executor = next(
            entry.executor
            for entry in recipe.job._deploy_map[ALL_SITES].app_config.executors
            if isinstance(entry.executor, ClientAPIExecutor)
        )
        assert controller.task_assignment_timeout == 60
        assert controller._task_check_period == 2.0
        assert client_api_executor._execution_mode == "external_process"
        assert client_api_executor._shutdown_timeout == 90.0
        assert client_api_executor._launch_once is False

        with pytest.raises(ValueError, match="task_check_period"):
            PTCyclicRecipe(
                name="test_invalid_task_check_period",
                model=simple_model,
                server_config_overrides={"task_check_period": 0},
                **base_recipe_params,
            )


class TestPTCyclicRecipe:
    """Test cases for PyTorch CyclicRecipe."""

    def test_pt_cyclic_initial_ckpt(self, mock_file_system, base_recipe_params, simple_model):
        """Test PT CyclicRecipe with initial_ckpt."""
        from nvflare.app_opt.pt.recipes.cyclic import CyclicRecipe as PTCyclicRecipe

        recipe = PTCyclicRecipe(
            name="test_pt_cyclic",
            model=simple_model,
            initial_ckpt="/abs/path/to/model.pt",
            **base_recipe_params,
        )

        assert recipe.name == "test_pt_cyclic"
        assert recipe.job is not None

    def test_pt_cyclic_with_ptmodel_wrapper_returns_persistor_id(
        self, mock_file_system, base_recipe_params, simple_model
    ):
        """PTModel wrapper path must correctly extract persistor_id from dict return."""
        from nvflare.app_opt.pt.recipes.cyclic import CyclicRecipe as PTCyclicRecipe

        recipe = PTCyclicRecipe(
            name="test_pt_wrapper",
            model=PTModel(model=simple_model),
            **base_recipe_params,
        )

        server_app = recipe.job._deploy_map.get("server")
        assert server_app is not None
        assert "persistor" in server_app.app_config.components


class TestTFCyclicRecipe:
    """Test cases for TensorFlow CyclicRecipe."""

    def test_tf_cyclic_initial_ckpt(self, mock_file_system, base_recipe_params):
        """Test TF CyclicRecipe with initial_ckpt (TF can load without model)."""
        pytest.importorskip("tensorflow")
        from nvflare.app_opt.tf.recipes.cyclic import CyclicRecipe as TFCyclicRecipe

        recipe = TFCyclicRecipe(
            name="test_tf_cyclic",
            model=None,
            initial_ckpt="/abs/path/to/model.h5",
            **base_recipe_params,
        )

        assert recipe.name == "test_tf_cyclic"
        assert recipe.job is not None
