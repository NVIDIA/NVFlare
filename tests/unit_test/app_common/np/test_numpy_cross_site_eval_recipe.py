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

import numpy as np
import pytest

from nvflare.app_common.np.recipes.cross_site_eval import NumpyCrossSiteEvalRecipe


@pytest.fixture
def numpy_model():
    """Create a simple numpy model for testing."""
    return {"numpy_key": np.array([[1.0, 2.0], [3.0, 4.0]])}


@pytest.fixture
def base_recipe_params():
    """Base parameters for creating NumpyCrossSiteEvalRecipe instances."""
    return {
        "min_clients": 2,
    }


def assert_recipe_basics(recipe, expected_name, expected_params):
    """Helper to assert basic recipe properties."""
    assert recipe.name == expected_name
    assert recipe.min_clients == expected_params.get("min_clients", 2)
    assert recipe.job is not None
    assert recipe.job.name == expected_name


class TestNumpyCrossSiteEvalRecipe:
    """Test cases for NumpyCrossSiteEvalRecipe class."""

    def test_minimal_recipe_initialization(self, base_recipe_params):
        """Test NumpyCrossSiteEvalRecipe initialization with minimal parameters."""
        recipe = NumpyCrossSiteEvalRecipe(name="test_cse", **base_recipe_params)

        assert_recipe_basics(recipe, "test_cse", base_recipe_params)
        assert recipe.initial_model is None
        assert recipe.model_locator_config is None
        assert recipe.cross_val_dir == "cross_site_val"  # Default value
        assert recipe.submit_model_timeout == 600
        assert recipe.validation_timeout == 6000
        assert recipe.participating_clients is None

    def test_recipe_with_initial_model(self, base_recipe_params, numpy_model):
        """Test NumpyCrossSiteEvalRecipe with initial model."""
        recipe = NumpyCrossSiteEvalRecipe(name="test_cse_with_model", initial_model=numpy_model, **base_recipe_params)

        assert_recipe_basics(recipe, "test_cse_with_model", base_recipe_params)
        assert recipe.initial_model == numpy_model
        assert recipe.model_locator_config is None

    def test_recipe_with_model_locator_config(self, base_recipe_params):
        """Test NumpyCrossSiteEvalRecipe with model locator configuration."""
        model_locator_config = {
            "model_dir": "/tmp/nvflare/server_models",
            "model_name": {"server_model_1": "model_1.npy", "server_model_2": "model_2.npy"},
        }
        recipe = NumpyCrossSiteEvalRecipe(
            name="test_cse_with_locator", model_locator_config=model_locator_config, **base_recipe_params
        )

        assert_recipe_basics(recipe, "test_cse_with_locator", base_recipe_params)
        assert recipe.initial_model is None
        assert recipe.model_locator_config == model_locator_config

    def test_recipe_with_custom_timeouts(self, base_recipe_params):
        """Test NumpyCrossSiteEvalRecipe with custom timeout values."""
        recipe = NumpyCrossSiteEvalRecipe(
            name="test_cse_timeouts",
            submit_model_timeout=1200,
            validation_timeout=3000,
            **base_recipe_params,
        )

        assert_recipe_basics(recipe, "test_cse_timeouts", base_recipe_params)
        assert recipe.submit_model_timeout == 1200
        assert recipe.validation_timeout == 3000

    def test_recipe_with_participating_clients(self, base_recipe_params):
        """Test NumpyCrossSiteEvalRecipe with specific participating clients."""
        participating_clients = ["site-1", "site-2", "site-3"]
        recipe = NumpyCrossSiteEvalRecipe(
            name="test_cse_clients", participating_clients=participating_clients, **base_recipe_params
        )

        assert_recipe_basics(recipe, "test_cse_clients", base_recipe_params)
        assert recipe.participating_clients == participating_clients

    def test_recipe_with_custom_client_model_config(self, base_recipe_params):
        """Test NumpyCrossSiteEvalRecipe with custom client model directory and name."""
        recipe = NumpyCrossSiteEvalRecipe(
            name="test_cse_client_config",
            client_model_dir="custom_models",
            client_model_name="my_model.npy",
            **base_recipe_params,
        )

        assert_recipe_basics(recipe, "test_cse_client_config", base_recipe_params)
        assert recipe.client_model_dir == "custom_models"
        assert recipe.client_model_name == "my_model.npy"

    @pytest.mark.parametrize(
        "min_clients,cross_val_dir",
        [
            (1, "cse_results"),  # Minimal: single client
            (2, "cross_site_val"),  # Standard configuration
            (5, "evaluation_results"),  # Multiple clients
        ],
    )
    def test_recipe_configurations(self, min_clients, cross_val_dir):
        """Test various NumpyCrossSiteEvalRecipe configurations using parametrized tests."""
        recipe = NumpyCrossSiteEvalRecipe(
            name=f"test_config_{min_clients}",
            min_clients=min_clients,
            cross_val_dir=cross_val_dir,
        )

        expected_params = {
            "min_clients": min_clients,
        }
        assert_recipe_basics(recipe, f"test_config_{min_clients}", expected_params)
        assert recipe.cross_val_dir == cross_val_dir

    def test_recipe_with_all_parameters(self, base_recipe_params, numpy_model):
        """Test NumpyCrossSiteEvalRecipe with all parameters specified."""
        participating_clients = ["site-1", "site-2"]
        recipe = NumpyCrossSiteEvalRecipe(
            name="test_cse_full",
            min_clients=2,
            initial_model=numpy_model,
            cross_val_dir="custom_cross_val",
            submit_model_timeout=900,
            validation_timeout=4500,
            participating_clients=participating_clients,
            client_model_dir="local_models",
            client_model_name="client_best.npy",
        )

        assert recipe.name == "test_cse_full"
        assert recipe.min_clients == 2
        assert recipe.initial_model == numpy_model
        assert recipe.cross_val_dir == "custom_cross_val"
        assert recipe.submit_model_timeout == 900
        assert recipe.validation_timeout == 4500
        assert recipe.participating_clients == participating_clients
        assert recipe.client_model_dir == "local_models"
        assert recipe.client_model_name == "client_best.npy"

    def test_recipe_default_name(self, base_recipe_params):
        """Test that recipe uses default name when not specified."""
        recipe = NumpyCrossSiteEvalRecipe(**base_recipe_params)

        assert recipe.name == "cross_site_eval"  # Default name
        assert recipe.job.name == "cross_site_eval"
