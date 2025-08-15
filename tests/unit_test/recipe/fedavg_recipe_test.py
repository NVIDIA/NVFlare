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

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe


class MyAggregator(Aggregator):
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

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        """Accept a shareable from a client."""
        dxo = from_shareable(shareable)
        if dxo.data_kind == DataKind.WEIGHTS:
            # Convert to FLModel format for our custom logic
            model = FLModel(params=dxo.data, params_type=ParamsType.FULL)
            self.accept_model(model)
            return True
        return False

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """Perform aggregation and return result as Shareable."""
        aggregated_model = self.aggregate_model()
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=aggregated_model.params)
        return dxo.to_shareable()

    def reset(self, fl_ctx: FLContext):
        """Reset the aggregator state."""
        self.reset_stats()


class InvalidAggregator:
    """Invalid aggregator that doesn't inherit from Aggregator."""

    def __init__(self):
        pass


class TestFedAvgRecipe:
    """Test cases for FedAvgRecipe class."""

    def test_fedavg_recipe_initialization_with_default_aggregator(self):
        """Test FedAvgRecipe initialization with default aggregator."""
        recipe = FedAvgRecipe(
            name="test_fedavg",
            train_script="tests/unit_test/recipe/test_script.py",
            train_args="--epochs 10",
            num_clients=3,
            min_clients=2,
            num_rounds=5,
        )

        assert recipe.name == "test_fedavg"
        assert recipe.train_script == "tests/unit_test/recipe/test_script.py"
        assert recipe.train_args == "--epochs 10"
        assert recipe.num_clients == 3
        assert recipe.min_clients == 2
        assert recipe.num_rounds == 5
        assert recipe.initial_model is None
        assert recipe.clients is None
        assert isinstance(recipe.aggregator, Aggregator)

    def test_fedavg_recipe_initialization_with_custom_aggregator(self):
        """Test FedAvgRecipe initialization with custom aggregator."""
        custom_aggregator = MyAggregator()

        recipe = FedAvgRecipe(
            name="test_fedavg_custom",
            train_script="tests/unit_test/recipe/test_script.py",
            train_args="--epochs 10",
            num_clients=2,
            min_clients=1,
            num_rounds=3,
            aggregator=custom_aggregator,
        )

        assert recipe.name == "test_fedavg_custom"
        assert recipe.aggregator is custom_aggregator
        assert isinstance(recipe.aggregator, MyAggregator)
        assert isinstance(recipe.aggregator, Aggregator)

    def test_fedavg_recipe_with_custom_clients(self):
        """Test FedAvgRecipe with custom client names."""
        custom_aggregator = MyAggregator()
        clients = ["client1", "client2", "client3"]

        recipe = FedAvgRecipe(
            name="test_fedavg_clients",
            train_script="tests/unit_test/recipe/test_script.py",
            train_args="--epochs 10",
            clients=clients,
            min_clients=2,
            num_rounds=3,
            aggregator=custom_aggregator,
        )

        assert recipe.clients == clients
        assert recipe.num_clients == 3
        assert recipe.min_clients == 2

    def test_fedavg_recipe_validation_inconsistent_clients(self):
        """Test FedAvgRecipe validation with inconsistent client configuration."""
        clients = ["client1", "client2"]

        with pytest.raises(ValueError, match="inconsistent number of clients"):
            FedAvgRecipe(
                name="test_fedavg_inconsistent",
                train_script="tests/unit_test/recipe/test_script.py",
                train_args="--epochs 10",
                clients=clients,
                num_clients=3,  # Inconsistent with len(clients) = 2
                min_clients=1,
            )

    def test_fedavg_recipe_with_initial_model(self):
        """Test FedAvgRecipe with initial model."""
        initial_model = {"layer1.weight": np.array([1.0, 2.0]), "layer1.bias": np.array([0.1])}
        custom_aggregator = MyAggregator()

        recipe = FedAvgRecipe(
            name="test_fedavg_initial_model",
            train_script="tests/unit_test/recipe/test_script.py",
            train_args="--epochs 10",
            initial_model=initial_model,
            num_clients=2,
            min_clients=1,
            num_rounds=3,
            aggregator=custom_aggregator,
        )

        assert recipe.initial_model == initial_model

    def test_fedavg_recipe_job_creation(self):
        """Test that FedAvgRecipe creates a valid job structure."""
        custom_aggregator = MyAggregator()

        recipe = FedAvgRecipe(
            name="test_fedavg_job",
            train_script="tests/unit_test/recipe/test_script.py",
            train_args="--epochs 10",
            num_clients=2,
            min_clients=1,
            num_rounds=3,
            aggregator=custom_aggregator,
        )

        # Verify job was created
        assert recipe.job is not None
        assert recipe.job.name == "test_fedavg_job"

    def test_my_aggregator_functionality(self):
        """Test the MyAggregator custom aggregator functionality."""
        aggregator = MyAggregator()

        # Test initial state
        assert aggregator.sum == {}
        assert aggregator.count == 0

        # Create test models
        model1 = FLModel(
            params={"layer1.weight": np.array([1.0, 2.0]), "layer1.bias": np.array([0.1])}, params_type=ParamsType.FULL
        )
        model2 = FLModel(
            params={"layer1.weight": np.array([3.0, 4.0]), "layer1.bias": np.array([0.2])}, params_type=ParamsType.FULL
        )

        # Test accept_model
        aggregator.accept_model(model1)
        assert aggregator.count == 1
        assert "layer1.weight" in aggregator.sum
        assert "layer1.bias" in aggregator.sum

        aggregator.accept_model(model2)
        assert aggregator.count == 2

        # Test aggregate_model
        aggregated = aggregator.aggregate_model()
        assert isinstance(aggregated, FLModel)
        assert aggregated.params["layer1.weight"][0] == 2.0  # (1.0 + 3.0) / 2
        assert aggregated.params["layer1.weight"][1] == 3.0  # (2.0 + 4.0) / 2
        assert aggregated.params["layer1.bias"][0] == 0.15  # (0.1 + 0.2) / 2

        # Test reset_stats
        aggregator.reset_stats()
        assert aggregator.sum == {}
        assert aggregator.count == 0

    def test_my_aggregator_accept_and_aggregate_methods(self):
        """Test MyAggregator accept and aggregate methods with Shareable objects."""
        aggregator = MyAggregator()
        fl_ctx = FLContext()

        # Create test data
        model_params = {"layer1.weight": np.array([1.0, 2.0]), "layer1.bias": np.array([0.1])}
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=model_params)
        shareable = dxo.to_shareable()

        # Test accept method
        result = aggregator.accept(shareable, fl_ctx)
        assert result is True
        assert aggregator.count == 1

        # Test aggregate method
        aggregated_shareable = aggregator.aggregate(fl_ctx)
        assert isinstance(aggregated_shareable, Shareable)

        # Verify the aggregated result
        aggregated_dxo = from_shareable(aggregated_shareable)
        assert aggregated_dxo.data_kind == DataKind.WEIGHTS
        assert "layer1.weight" in aggregated_dxo.data
        assert "layer1.bias" in aggregated_dxo.data

    def test_my_aggregator_reset_method(self):
        """Test MyAggregator reset method."""
        aggregator = MyAggregator()
        fl_ctx = FLContext()

        # Add some data
        model = FLModel(params={"layer1.weight": np.array([1.0, 2.0])}, params_type=ParamsType.FULL)
        aggregator.accept_model(model)
        assert aggregator.count == 1
        assert len(aggregator.sum) == 1

        # Test reset
        aggregator.reset(fl_ctx)
        assert aggregator.count == 0
        assert len(aggregator.sum) == 0

    def test_fedavg_recipe_edge_cases(self):
        """Test FedAvgRecipe edge cases and boundary conditions."""

        # Test with minimum valid configuration
        recipe = FedAvgRecipe(
            name="minimal", train_script="test.py", train_args="", num_clients=1, min_clients=1, num_rounds=1
        )
        assert recipe.num_clients == 1
        assert recipe.min_clients == 1
        assert recipe.num_rounds == 1

        # Test with zero min_clients (should be valid)
        recipe = FedAvgRecipe(
            name="zero_min_clients", train_script="test.py", train_args="", num_clients=2, min_clients=0, num_rounds=2
        )
        assert recipe.min_clients == 0
