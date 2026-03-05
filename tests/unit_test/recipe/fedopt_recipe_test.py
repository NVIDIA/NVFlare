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

"""Tests for FedOptRecipe with initial_ckpt support."""

from unittest.mock import patch

import pytest

try:
    import torch.nn as nn

    class SimpleTestModel(nn.Module):
        """A simple PyTorch model for testing purposes."""

        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(10, 10)

        def forward(self, x):
            return self.lin(x)

except ImportError:
    SimpleTestModel = None  # PyTorch not installed (e.g. TF-only venv)


def _tensorflow_available():
    """Return True if TensorFlow can be imported and loaded (used to skip TF tests when TF is broken)."""
    try:
        import tensorflow  # noqa: F401

        return True
    except Exception:
        return False


TENSORFLOW_AVAILABLE = _tensorflow_available()


@pytest.fixture
def mock_file_system():
    """Mock file system operations for all tests."""
    with patch("os.path.isfile", return_value=True), patch("os.path.exists", return_value=True):
        yield


@pytest.fixture
def simple_model():
    """Create a simple test model (skips when PyTorch not installed)."""
    if SimpleTestModel is None:
        pytest.skip("PyTorch not installed")
    return SimpleTestModel()


@pytest.fixture
def base_recipe_params():
    """Base parameters for creating FedOptRecipe instances."""
    return {
        "train_script": "mock_train_script.py",
        "train_args": "--epochs 10",
        "min_clients": 2,
        "num_rounds": 5,
    }


@pytest.mark.skipif(SimpleTestModel is None, reason="PyTorch not installed")
class TestPTFedOptRecipe:
    """Test cases for PyTorch FedOptRecipe."""

    def test_basic_initialization(self, mock_file_system, base_recipe_params, simple_model):
        """Test PT FedOptRecipe basic initialization."""
        from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe

        recipe = FedOptRecipe(
            name="test_fedopt",
            model=simple_model,
            **base_recipe_params,
        )

        assert recipe.name == "test_fedopt"
        assert recipe.model == simple_model
        assert recipe.job is not None

    def test_initial_ckpt_parameter_accepted(self, mock_file_system, base_recipe_params, simple_model):
        """Test that initial_ckpt parameter is accepted."""
        from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe

        recipe = FedOptRecipe(
            name="test_fedopt_ckpt",
            model=simple_model,
            initial_ckpt="/abs/path/to/model.pt",
            **base_recipe_params,
        )

        assert recipe.initial_ckpt == "/abs/path/to/model.pt"

    def test_dict_model_config_accepted(self, mock_file_system, base_recipe_params, simple_model):
        """Test that dict model config is accepted."""
        from unittest.mock import patch

        from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe

        model_config = {
            "class_path": "my_module.models.SimpleNet",
            "args": {"input_size": 10},
        }

        # Mock instantiate_class since my_module.models.SimpleNet doesn't exist
        with patch("nvflare.fuel.utils.class_utils.instantiate_class") as mock_instantiate:
            mock_instantiate.return_value = simple_model

            recipe = FedOptRecipe(
                name="test_fedopt_dict",
                model=model_config,
                **base_recipe_params,
            )

            assert recipe.model["path"] == "my_module.models.SimpleNet"
            assert recipe.model["args"] == {"input_size": 10}

    def test_optimizer_args_class_path_supported(self, mock_file_system, base_recipe_params, simple_model):
        """Test that optimizer_args with 'class_path' is supported; path is set from class_path."""
        from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe

        optimizer_args = {
            "class_path": "torch.optim.SGD",
            "args": {"lr": 1.0, "momentum": 0.6},
        }
        recipe = FedOptRecipe(
            name="test_fedopt_optim",
            model=simple_model,
            optimizer_args=optimizer_args,
            **base_recipe_params,
        )

        assert recipe.optimizer_args["path"] == optimizer_args["class_path"]
        assert recipe.optimizer_args["args"] == optimizer_args["args"]
        assert recipe.optimizer_args.get("config_type") == "dict"

    def test_optimizer_args_path_supported(self, mock_file_system, base_recipe_params, simple_model):
        """Test that optimizer_args with 'path' (no class_path) is supported for backward compatibility."""
        from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe

        optimizer_args = {
            "path": "torch.optim.SGD",
            "args": {"lr": 0.5},
        }
        recipe = FedOptRecipe(
            name="test_fedopt_optim_path",
            model=simple_model,
            optimizer_args=optimizer_args,
            **base_recipe_params,
        )

        assert recipe.optimizer_args["path"] == "torch.optim.SGD"
        assert recipe.optimizer_args["args"] == {"lr": 0.5}
        assert recipe.optimizer_args.get("config_type") == "dict"

    def test_optimizer_args_path_and_class_path_path_takes_precedence(
        self, mock_file_system, base_recipe_params, simple_model
    ):
        """Test that when both 'path' and 'class_path' are given, 'path' is used (not overwritten)."""
        from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe

        optimizer_args = {
            "path": "torch.optim.Adam",
            "class_path": "torch.optim.SGD",
            "args": {"lr": 0.1},
        }
        recipe = FedOptRecipe(
            name="test_fedopt_both",
            model=simple_model,
            optimizer_args=optimizer_args,
            **base_recipe_params,
        )

        assert recipe.optimizer_args["path"] == "torch.optim.Adam"
        assert recipe.optimizer_args["args"] == {"lr": 0.1}
        assert recipe.optimizer_args.get("config_type") == "dict"

    def test_initial_ckpt_must_exist_for_relative_path(self, base_recipe_params, simple_model):
        """Test that non-existent relative paths are rejected."""
        from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe

        with pytest.raises(ValueError, match="does not exist locally"):
            FedOptRecipe(
                name="test_relative_path",
                model=simple_model,
                initial_ckpt="relative/path/model.pt",
                **base_recipe_params,
            )

    def test_dict_config_missing_path_raises_error(self, mock_file_system, base_recipe_params):
        """Test that dict config without 'class_path' key raises error."""
        from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe

        with pytest.raises(ValueError, match="must have 'class_path' key"):
            FedOptRecipe(
                name="test_invalid_dict",
                model={"args": {"input_size": 10}},  # Missing 'class_path'
                **base_recipe_params,
            )

    def test_dict_config_path_not_string_raises_error(self, mock_file_system, base_recipe_params):
        """Test that dict config with non-string 'class_path' raises error."""
        from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe

        with pytest.raises(ValueError, match="'class_path' must be a string"):
            FedOptRecipe(
                name="test_invalid_path_type",
                model={"class_path": 123, "args": {}},  # class_path is not string
                **base_recipe_params,
            )

    def test_dict_config_instantiates_model(self, mock_file_system, base_recipe_params, simple_model):
        """Test that dict config is instantiated to nn.Module before registration."""
        from unittest.mock import patch

        from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe

        # Patch at the location where it's imported and used
        with patch("nvflare.fuel.utils.class_utils.instantiate_class") as mock_instantiate:
            mock_instantiate.return_value = simple_model

            recipe = FedOptRecipe(
                name="test_dict_instantiation",
                model={"class_path": "mymodule.MyModel", "args": {"input_size": 10}},
                **base_recipe_params,
            )

            # Verify instantiate_class was called with correct arguments
            mock_instantiate.assert_called_once_with("mymodule.MyModel", {"input_size": 10})
            assert recipe.job is not None

    def test_model_none_raises_error(self, mock_file_system, base_recipe_params):
        """Test that model=None raises ValueError."""
        from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe

        with pytest.raises(ValueError, match="FedOpt requires model"):
            FedOptRecipe(
                name="test_no_model",
                model=None,
                **base_recipe_params,
            )


@pytest.mark.skipif(
    not TENSORFLOW_AVAILABLE,
    reason="TensorFlow not installed or failed to load (e.g. broken install)",
)
class TestTFFedOptRecipe:
    """Test cases for TensorFlow FedOptRecipe."""

    def test_basic_initialization(self, mock_file_system, base_recipe_params):
        """Test TF FedOptRecipe basic initialization."""
        from nvflare.app_opt.tf.recipes.fedopt import FedOptRecipe

        recipe = FedOptRecipe(
            name="test_tf_fedopt",
            model=None,
            **base_recipe_params,
        )

        assert recipe.name == "test_tf_fedopt"
        assert recipe.job is not None

    def test_initial_ckpt_parameter_accepted(self, mock_file_system, base_recipe_params):
        """Test that initial_ckpt parameter is accepted (TF can load without model)."""
        from nvflare.app_opt.tf.recipes.fedopt import FedOptRecipe

        recipe = FedOptRecipe(
            name="test_tf_fedopt_ckpt",
            model=None,
            initial_ckpt="/abs/path/to/model.h5",
            **base_recipe_params,
        )

        assert recipe.initial_ckpt == "/abs/path/to/model.h5"

    def test_optimizer_args_class_path_supported(self, mock_file_system, base_recipe_params):
        """Test that TF FedOptRecipe optimizer_args with 'class_path' is supported."""
        from nvflare.app_opt.tf.recipes.fedopt import FedOptRecipe

        optimizer_args = {
            "class_path": "tf.keras.optimizers.SGD",
            "args": {"learning_rate": 1.0, "momentum": 0.6},
        }
        recipe = FedOptRecipe(
            name="test_tf_fedopt_optim",
            model=None,
            optimizer_args=optimizer_args,
            **base_recipe_params,
        )

        assert recipe.optimizer_args["path"] == optimizer_args["class_path"]
        assert recipe.optimizer_args["args"] == optimizer_args["args"]
        assert recipe.optimizer_args.get("config_type") == "dict"

    def test_optimizer_args_path_supported(self, mock_file_system, base_recipe_params):
        """Test that TF FedOptRecipe optimizer_args with 'path' is supported."""
        from nvflare.app_opt.tf.recipes.fedopt import FedOptRecipe

        optimizer_args = {
            "path": "tf.keras.optimizers.SGD",
            "args": {"learning_rate": 0.5},
        }
        recipe = FedOptRecipe(
            name="test_tf_fedopt_optim_path",
            model=None,
            optimizer_args=optimizer_args,
            **base_recipe_params,
        )

        assert recipe.optimizer_args["path"] == "tf.keras.optimizers.SGD"
        assert recipe.optimizer_args["args"] == {"learning_rate": 0.5}
        assert recipe.optimizer_args.get("config_type") == "dict"

    def test_lr_scheduler_args_class_path_supported(self, mock_file_system, base_recipe_params):
        """Test that TF FedOptRecipe lr_scheduler_args with 'class_path' is supported."""
        from nvflare.app_opt.tf.recipes.fedopt import FedOptRecipe

        lr_scheduler_args = {
            "class_path": "tf.keras.optimizers.schedules.CosineDecay",
            "args": {"initial_learning_rate": 1.0, "decay_steps": 100},
        }
        recipe = FedOptRecipe(
            name="test_tf_fedopt_scheduler",
            model=None,
            lr_scheduler_args=lr_scheduler_args,
            **base_recipe_params,
        )

        assert recipe.lr_scheduler_args["path"] == lr_scheduler_args["class_path"]
        assert recipe.lr_scheduler_args["args"] == lr_scheduler_args["args"]
        assert recipe.lr_scheduler_args.get("config_type") == "dict"

    def test_lr_scheduler_args_path_supported(self, mock_file_system, base_recipe_params):
        """Test that TF FedOptRecipe lr_scheduler_args with 'path' is supported."""
        from nvflare.app_opt.tf.recipes.fedopt import FedOptRecipe

        lr_scheduler_args = {
            "path": "tf.keras.optimizers.schedules.CosineDecay",
            "args": {"initial_learning_rate": 0.5, "decay_steps": 50},
        }
        recipe = FedOptRecipe(
            name="test_tf_fedopt_scheduler_path",
            model=None,
            lr_scheduler_args=lr_scheduler_args,
            **base_recipe_params,
        )

        assert recipe.lr_scheduler_args["path"] == "tf.keras.optimizers.schedules.CosineDecay"
        assert recipe.lr_scheduler_args["args"] == lr_scheduler_args["args"]
        assert recipe.lr_scheduler_args.get("config_type") == "dict"
