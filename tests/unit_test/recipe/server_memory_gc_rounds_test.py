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

"""Tests for server_memory_gc_rounds parameter in recipes."""

from unittest.mock import patch

import pytest
import torch.nn as nn

from nvflare.apis.job_def import SERVER_SITE_NAME


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
    """Base parameters for creating recipe instances."""
    return {
        "train_script": "mock_train_script.py",
        "train_args": "--epochs 10",
        "min_clients": 2,
        "num_rounds": 5,
    }


def get_controller(recipe):
    """Extract controller from recipe's job."""
    server_app = recipe.job._deploy_map[SERVER_SITE_NAME]
    # Controller is usually the workflow component
    for comp_id, comp in server_app.app_config.components.items():
        if hasattr(comp, "_memory_gc_rounds") or hasattr(comp, "memory_gc_rounds"):
            return comp
    return None


class TestCyclicRecipeServerMemoryGcRounds:
    """Test server_memory_gc_rounds for CyclicRecipe."""

    def test_cyclic_default_server_memory_gc_rounds(self, mock_file_system, simple_model):
        """Test CyclicRecipe has default server_memory_gc_rounds=1."""
        from nvflare.app_opt.pt.recipes.cyclic import CyclicRecipe

        recipe = CyclicRecipe(
            name="test_cyclic",
            initial_model=simple_model,
            train_script="client.py",
            min_clients=2,
            num_rounds=3,
        )

        assert recipe.server_memory_gc_rounds == 1

    def test_cyclic_custom_server_memory_gc_rounds(self, mock_file_system, simple_model):
        """Test CyclicRecipe accepts custom server_memory_gc_rounds."""
        from nvflare.app_opt.pt.recipes.cyclic import CyclicRecipe

        recipe = CyclicRecipe(
            name="test_cyclic",
            initial_model=simple_model,
            train_script="client.py",
            min_clients=2,
            num_rounds=3,
            server_memory_gc_rounds=5,
        )

        assert recipe.server_memory_gc_rounds == 5

    def test_cyclic_disable_server_memory_gc(self, mock_file_system, simple_model):
        """Test CyclicRecipe can disable server memory GC with 0."""
        from nvflare.app_opt.pt.recipes.cyclic import CyclicRecipe

        recipe = CyclicRecipe(
            name="test_cyclic",
            initial_model=simple_model,
            train_script="client.py",
            min_clients=2,
            num_rounds=3,
            server_memory_gc_rounds=0,
        )

        assert recipe.server_memory_gc_rounds == 0


class TestFedAvgRecipeServerMemoryGcRounds:
    """Test server_memory_gc_rounds for FedAvgRecipe."""

    def test_fedavg_default_server_memory_gc_rounds(self, mock_file_system, base_recipe_params):
        """Test FedAvgRecipe has default server_memory_gc_rounds=0."""
        from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe

        recipe = FedAvgRecipe(name="test_fedavg", **base_recipe_params)

        assert recipe.server_memory_gc_rounds == 0

    def test_fedavg_custom_server_memory_gc_rounds(self, mock_file_system, base_recipe_params):
        """Test FedAvgRecipe accepts custom server_memory_gc_rounds."""
        from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe

        recipe = FedAvgRecipe(
            name="test_fedavg",
            server_memory_gc_rounds=3,
            **base_recipe_params,
        )

        assert recipe.server_memory_gc_rounds == 3


class TestFedOptRecipeServerMemoryGcRounds:
    """Test server_memory_gc_rounds for FedOptRecipe."""

    def test_pt_fedopt_default_server_memory_gc_rounds(self, mock_file_system, simple_model):
        """Test PT FedOptRecipe has default server_memory_gc_rounds=1."""
        from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe

        recipe = FedOptRecipe(
            name="test_fedopt",
            initial_model=simple_model,
            train_script="client.py",
            min_clients=2,
            num_rounds=3,
        )

        assert recipe.server_memory_gc_rounds == 1

    def test_pt_fedopt_custom_server_memory_gc_rounds(self, mock_file_system, simple_model):
        """Test PT FedOptRecipe accepts custom server_memory_gc_rounds."""
        from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe

        recipe = FedOptRecipe(
            name="test_fedopt",
            initial_model=simple_model,
            train_script="client.py",
            min_clients=2,
            num_rounds=3,
            server_memory_gc_rounds=10,
        )

        assert recipe.server_memory_gc_rounds == 10


class TestFedAvgHERecipeServerMemoryGcRounds:
    """Test server_memory_gc_rounds for FedAvgRecipeWithHE."""

    def test_fedavg_he_default_server_memory_gc_rounds(self, mock_file_system, simple_model):
        """Test FedAvgRecipeWithHE has default server_memory_gc_rounds=1."""
        from nvflare.app_opt.pt.recipes.fedavg_he import FedAvgRecipeWithHE

        recipe = FedAvgRecipeWithHE(
            name="test_fedavg_he",
            initial_model=simple_model,
            train_script="client.py",
            min_clients=2,
            num_rounds=3,
        )

        assert recipe.server_memory_gc_rounds == 1

    def test_fedavg_he_custom_server_memory_gc_rounds(self, mock_file_system, simple_model):
        """Test FedAvgRecipeWithHE accepts custom server_memory_gc_rounds."""
        from nvflare.app_opt.pt.recipes.fedavg_he import FedAvgRecipeWithHE

        recipe = FedAvgRecipeWithHE(
            name="test_fedavg_he",
            initial_model=simple_model,
            train_script="client.py",
            min_clients=2,
            num_rounds=3,
            server_memory_gc_rounds=2,
        )

        assert recipe.server_memory_gc_rounds == 2
