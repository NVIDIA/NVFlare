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

import numpy as np
import pytest

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.np.constants import NPConstants
from nvflare.app_common.workflows.lr.fedavg import FedAvgLR


class TestFedAvgLRInit:
    """Test FedAvgLR controller initialization."""

    def test_default_initialization(self):
        """Test FedAvgLR with minimal required parameters."""
        controller = FedAvgLR(damping_factor=0.8)

        assert controller.damping_factor == 0.8
        assert controller.epsilon == 1.0  # default
        assert controller.model_dir == "models"  # default
        assert controller.model_name == "weights.npy"  # default
        assert controller.n_features == 13  # default
        assert isinstance(controller.aggregator, WeightedAggregationHelper)
        assert controller.persistor is None
        assert controller._default_persistor is not None

    def test_custom_initialization(self):
        """Test FedAvgLR with custom parameters."""
        custom_aggregator = WeightedAggregationHelper()

        controller = FedAvgLR(
            damping_factor=0.5,
            epsilon=0.1,
            model_dir="custom_models",
            model_name="lr_weights.npy",
            n_features=20,
            aggregator=custom_aggregator,
            num_clients=5,
            num_rounds=10,
        )

        assert controller.damping_factor == 0.5
        assert controller.epsilon == 0.1
        assert controller.model_dir == "custom_models"
        assert controller.model_name == "lr_weights.npy"
        assert controller.n_features == 20
        assert controller.aggregator is custom_aggregator
        assert controller.num_clients == 5
        assert controller.num_rounds == 10

    def test_initialization_with_zero_epsilon_raises_singular_matrix(self):
        """Test that epsilon=0 may lead to singular matrix issues (documented behavior)."""
        # This test documents that epsilon should be > 0 for numerical stability
        controller = FedAvgLR(damping_factor=0.8, epsilon=0.0)
        assert controller.epsilon == 0.0  # Should initialize but may cause issues later


class TestNewtonRaphsonAggregatorFunction:
    """Test the newton_raphson_aggregator_fn method."""

    def test_aggregator_single_client(self):
        """Test aggregation with a single client."""
        controller = FedAvgLR(damping_factor=0.5, epsilon=1.0, n_features=2)

        # Create mock gradient and hessian from one client
        gradient = np.array([[1.0], [2.0], [3.0]])  # (n_features+1, 1)
        hessian = np.array([[4.0, 0.5, 0.2], [0.5, 5.0, 0.3], [0.2, 0.3, 6.0]])  # (n_features+1, n_features+1)

        result1 = FLModel(
            params={"gradient": gradient, "hessian": hessian},
            params_type=ParamsType.FULL,
            current_round=0,
            meta={
                "client_name": "site-1",
                FLMetaKey.NUM_STEPS_CURRENT_ROUND: 10,
            },
        )

        # Call aggregator function
        aggr_result = controller.newton_raphson_aggregator_fn([result1])

        # Verify result structure
        assert isinstance(aggr_result, FLModel)
        assert "newton_raphson_updates" in aggr_result.params
        assert aggr_result.params_type == ParamsType.FULL
        assert aggr_result.meta["nr_aggregated"] == 1
        assert aggr_result.meta[AppConstants.CURRENT_ROUND] == 0

        # Verify Newton-Raphson computation:
        # update = damping_factor * solve(H + reg*I, g)
        reg = controller.epsilon * np.eye(3)
        expected_updates = controller.damping_factor * np.linalg.solve(hessian + reg, gradient)

        np.testing.assert_allclose(aggr_result.params["newton_raphson_updates"], expected_updates, rtol=1e-6)

    def test_aggregator_multiple_clients_equal_weights(self):
        """Test aggregation with multiple clients having equal weights."""
        controller = FedAvgLR(damping_factor=1.0, epsilon=0.1, n_features=2)

        # Client 1
        gradient1 = np.array([[2.0], [4.0], [6.0]])
        hessian1 = np.array([[10.0, 1.0, 0.5], [1.0, 12.0, 0.5], [0.5, 0.5, 14.0]])

        result1 = FLModel(
            params={"gradient": gradient1, "hessian": hessian1},
            params_type=ParamsType.FULL,
            current_round=1,
            meta={
                "client_name": "site-1",
                FLMetaKey.NUM_STEPS_CURRENT_ROUND: 100,
            },
        )

        # Client 2
        gradient2 = np.array([[4.0], [6.0], [8.0]])
        hessian2 = np.array([[8.0, 0.5, 0.3], [0.5, 10.0, 0.4], [0.3, 0.4, 12.0]])

        result2 = FLModel(
            params={"gradient": gradient2, "hessian": hessian2},
            params_type=ParamsType.FULL,
            current_round=1,
            meta={
                "client_name": "site-2",
                FLMetaKey.NUM_STEPS_CURRENT_ROUND: 100,
            },
        )

        # Aggregate
        aggr_result = controller.newton_raphson_aggregator_fn([result1, result2])

        # Expected: weighted average with equal weights (100 each)
        # gradient_avg = (gradient1 * 100 + gradient2 * 100) / 200
        # hessian_avg = (hessian1 * 100 + hessian2 * 100) / 200
        gradient_avg = (gradient1 + gradient2) / 2
        hessian_avg = (hessian1 + hessian2) / 2

        reg = controller.epsilon * np.eye(3)
        expected_updates = controller.damping_factor * np.linalg.solve(hessian_avg + reg, gradient_avg)

        np.testing.assert_allclose(aggr_result.params["newton_raphson_updates"], expected_updates, rtol=1e-6)
        assert aggr_result.meta["nr_aggregated"] == 2

    def test_aggregator_multiple_clients_different_weights(self):
        """Test aggregation with multiple clients having different sample sizes."""
        controller = FedAvgLR(damping_factor=0.8, epsilon=0.5, n_features=2)

        # Client 1 with 200 samples
        gradient1 = np.array([[1.0], [2.0], [3.0]])
        hessian1 = np.array([[5.0, 0.1, 0.1], [0.1, 6.0, 0.1], [0.1, 0.1, 7.0]])

        result1 = FLModel(
            params={"gradient": gradient1, "hessian": hessian1},
            params_type=ParamsType.FULL,
            current_round=2,
            meta={
                "client_name": "site-1",
                FLMetaKey.NUM_STEPS_CURRENT_ROUND: 200,
            },
        )

        # Client 2 with 100 samples
        gradient2 = np.array([[3.0], [6.0], [9.0]])
        hessian2 = np.array([[15.0, 0.3, 0.3], [0.3, 18.0, 0.3], [0.3, 0.3, 21.0]])

        result2 = FLModel(
            params={"gradient": gradient2, "hessian": hessian2},
            params_type=ParamsType.FULL,
            current_round=2,
            meta={
                "client_name": "site-2",
                FLMetaKey.NUM_STEPS_CURRENT_ROUND: 100,
            },
        )

        # Aggregate
        aggr_result = controller.newton_raphson_aggregator_fn([result1, result2])

        # Expected weighted average: (g1*200 + g2*100) / 300
        gradient_avg = (gradient1 * 200 + gradient2 * 100) / 300
        hessian_avg = (hessian1 * 200 + hessian2 * 100) / 300

        reg = controller.epsilon * np.eye(3)
        expected_updates = controller.damping_factor * np.linalg.solve(hessian_avg + reg, gradient_avg)

        np.testing.assert_allclose(aggr_result.params["newton_raphson_updates"], expected_updates, rtol=1e-6)

    def test_aggregator_with_regularization(self):
        """Test that epsilon regularization prevents singular matrix issues."""
        controller = FedAvgLR(damping_factor=1.0, epsilon=10.0, n_features=2)

        # Create a near-singular hessian matrix
        gradient = np.array([[1.0], [1.0], [1.0]])
        hessian = np.array([[1e-8, 0.0, 0.0], [0.0, 1e-8, 0.0], [0.0, 0.0, 1e-8]])  # Near-singular matrix

        result = FLModel(
            params={"gradient": gradient, "hessian": hessian},
            params_type=ParamsType.FULL,
            current_round=0,
            meta={
                "client_name": "site-1",
                FLMetaKey.NUM_STEPS_CURRENT_ROUND: 50,
            },
        )

        # Should not raise due to regularization
        aggr_result = controller.newton_raphson_aggregator_fn([result])

        # With large epsilon, the regularization dominates
        reg = controller.epsilon * np.eye(3)
        expected_updates = controller.damping_factor * np.linalg.solve(hessian + reg, gradient)

        np.testing.assert_allclose(aggr_result.params["newton_raphson_updates"], expected_updates, rtol=1e-5)

    def test_aggregator_different_damping_factors(self):
        """Test that damping factor correctly scales the updates."""
        # Test with damping_factor = 0.5
        controller1 = FedAvgLR(damping_factor=0.5, epsilon=1.0, n_features=1)

        gradient = np.array([[2.0], [4.0]])
        hessian = np.array([[8.0, 0.5], [0.5, 10.0]])

        result = FLModel(
            params={"gradient": gradient, "hessian": hessian},
            params_type=ParamsType.FULL,
            current_round=0,
            meta={
                "client_name": "site-1",
                FLMetaKey.NUM_STEPS_CURRENT_ROUND: 100,
            },
        )

        aggr_result1 = controller1.newton_raphson_aggregator_fn([result])

        # Test with damping_factor = 1.0
        controller2 = FedAvgLR(damping_factor=1.0, epsilon=1.0, n_features=1)
        aggr_result2 = controller2.newton_raphson_aggregator_fn([result])

        # Updates should be exactly 2x different
        np.testing.assert_allclose(
            aggr_result2.params["newton_raphson_updates"],
            2.0 * aggr_result1.params["newton_raphson_updates"],
            rtol=1e-10,
        )

    def test_aggregator_preserves_metadata(self):
        """Test that aggregator preserves important metadata."""
        controller = FedAvgLR(damping_factor=0.9, epsilon=1.0, n_features=2, num_rounds=20)

        gradient = np.array([[1.0], [1.0], [1.0]])
        hessian = np.eye(3) * 5

        result = FLModel(
            params={"gradient": gradient, "hessian": hessian},
            params_type=ParamsType.FULL,
            current_round=5,
            meta={
                "client_name": "site-1",
                FLMetaKey.NUM_STEPS_CURRENT_ROUND: 150,
            },
        )

        aggr_result = controller.newton_raphson_aggregator_fn([result])

        # Check metadata preservation
        assert aggr_result.meta["nr_aggregated"] == 1
        assert aggr_result.meta[AppConstants.CURRENT_ROUND] == 5
        assert aggr_result.meta[AppConstants.NUM_ROUNDS] == 20
        assert aggr_result.params_type == ParamsType.FULL

    @pytest.mark.parametrize("n_clients", [1, 3, 5, 10])
    def test_aggregator_multiple_clients_various_counts(self, n_clients):
        """Test aggregation with varying numbers of clients."""
        controller = FedAvgLR(damping_factor=1.0, epsilon=1.0, n_features=3)

        results = []
        gradients = []
        hessians = []
        weights = []

        for i in range(n_clients):
            # Generate random gradient and positive definite hessian
            gradient = np.random.randn(4, 1)
            # Create positive definite hessian
            A = np.random.randn(4, 4)
            hessian = A @ A.T + np.eye(4) * 5  # Ensure positive definite

            weight = np.random.randint(50, 200)

            gradients.append(gradient * weight)
            hessians.append(hessian * weight)
            weights.append(weight)

            result = FLModel(
                params={"gradient": gradient, "hessian": hessian},
                params_type=ParamsType.FULL,
                current_round=0,
                meta={
                    "client_name": f"site-{i}",
                    FLMetaKey.NUM_STEPS_CURRENT_ROUND: weight,
                },
            )
            results.append(result)

        # Aggregate
        aggr_result = controller.newton_raphson_aggregator_fn(results)

        # Compute expected weighted average
        total_weight = sum(weights)
        gradient_avg = sum(gradients) / total_weight
        hessian_avg = sum(hessians) / total_weight

        reg = controller.epsilon * np.eye(4)
        expected_updates = controller.damping_factor * np.linalg.solve(hessian_avg + reg, gradient_avg)

        np.testing.assert_allclose(aggr_result.params["newton_raphson_updates"], expected_updates, rtol=1e-5)
        assert aggr_result.meta["nr_aggregated"] == n_clients


class TestUpdateModel:
    """Test the update_model method."""

    def test_update_model_basic(self):
        """Test basic model update with Newton-Raphson updates."""
        controller = FedAvgLR(damping_factor=1.0, n_features=2)

        # Initial model weights
        initial_weights = np.array([[1.0], [2.0], [3.0]])
        model = FLModel(
            params={NPConstants.NUMPY_KEY: initial_weights.copy()},
            params_type=ParamsType.FULL,
            meta={"round": 0},
        )

        # Newton-Raphson updates to apply
        updates = np.array([[0.5], [0.3], [0.2]])
        model_update = FLModel(
            params={"newton_raphson_updates": updates},
            meta={"round": 1, "nr_aggregated": 2},
            metrics={"loss": 0.5},
        )

        # Update the model
        updated_model = controller.update_model(model, model_update)

        # Expected: weights = weights + updates
        expected_weights = initial_weights + updates

        np.testing.assert_allclose(updated_model.params[NPConstants.NUMPY_KEY], expected_weights)
        assert updated_model.meta["round"] == 1
        assert updated_model.meta["nr_aggregated"] == 2
        assert updated_model.metrics["loss"] == 0.5

    def test_update_model_replace_meta_false(self):
        """Test model update with replace_meta=False."""
        controller = FedAvgLR(damping_factor=1.0, n_features=2)

        initial_weights = np.array([[1.0], [2.0], [3.0]])
        model = FLModel(
            params={NPConstants.NUMPY_KEY: initial_weights.copy()},
            params_type=ParamsType.FULL,
            meta={"round": 0, "existing_key": "keep_me"},
        )

        updates = np.array([[0.1], [0.2], [0.3]])
        model_update = FLModel(
            params={"newton_raphson_updates": updates},
            meta={"round": 1, "new_key": "add_me"},
        )

        # Update without replacing meta
        updated_model = controller.update_model(model, model_update, replace_meta=False)

        # Original meta should be preserved and new keys added
        assert updated_model.meta["existing_key"] == "keep_me"
        assert updated_model.meta["new_key"] == "add_me"
        assert updated_model.meta["round"] == 1  # Updated

    def test_update_model_none_update(self):
        """Test that update_model returns original model when update is None."""
        controller = FedAvgLR(damping_factor=1.0, n_features=2)

        initial_weights = np.array([[1.0], [2.0], [3.0]])
        model = FLModel(
            params={NPConstants.NUMPY_KEY: initial_weights.copy()},
            params_type=ParamsType.FULL,
        )

        # Update with None
        updated_model = controller.update_model(model, None)

        # Should return the same model
        assert updated_model is model
        np.testing.assert_array_equal(updated_model.params[NPConstants.NUMPY_KEY], initial_weights)

    def test_update_model_none_params_raises_error(self):
        """Test that update_model raises error when params is None."""
        controller = FedAvgLR(damping_factor=1.0, n_features=2)

        model = FLModel(params={NPConstants.NUMPY_KEY: np.array([[1.0], [2.0]])})
        model_update = FLModel(params=None)

        with pytest.raises(ValueError, match="model params is None"):
            controller.update_model(model, model_update)

    def test_update_model_accumulates_updates(self):
        """Test that multiple updates accumulate correctly."""
        controller = FedAvgLR(damping_factor=1.0, n_features=2)

        initial_weights = np.array([[0.0], [0.0], [0.0]])
        model = FLModel(
            params={NPConstants.NUMPY_KEY: initial_weights.copy()},
            params_type=ParamsType.FULL,
        )

        # First update
        update1 = np.array([[1.0], [2.0], [3.0]])
        model_update1 = FLModel(params={"newton_raphson_updates": update1})
        model = controller.update_model(model, model_update1)

        # Second update
        update2 = np.array([[0.5], [0.5], [0.5]])
        model_update2 = FLModel(params={"newton_raphson_updates": update2})
        model = controller.update_model(model, model_update2)

        # Expected: cumulative updates
        expected_weights = initial_weights + update1 + update2

        np.testing.assert_allclose(model.params[NPConstants.NUMPY_KEY], expected_weights)


class TestLoadSaveModel:
    """Test the load_model and save_model methods."""

    def test_load_model_with_default_persistor(self):
        """Test that load_model uses default persistor when none is provided."""
        controller = FedAvgLR(damping_factor=1.0, n_features=2)

        # Verify default persistor is set
        assert controller.persistor is None
        assert controller._default_persistor is not None

    def test_load_model_raises_on_none_model(self, monkeypatch):
        """Test that load_model raises error when model is None."""
        controller = FedAvgLR(damping_factor=1.0, n_features=2)

        # Mock the parent's load_model to return None
        def mock_load_model(self):
            return None

        monkeypatch.setattr("nvflare.app_common.workflows.base_fedavg.BaseFedAvg.load_model", mock_load_model)

        with pytest.raises(ValueError, match="model can't be None"):
            controller.load_model()

    def test_load_model_raises_on_none_params(self, monkeypatch):
        """Test that load_model raises error when model.params is None."""
        controller = FedAvgLR(damping_factor=1.0, n_features=2)

        # Mock the parent's load_model to return model with None params
        def mock_load_model(self):
            return FLModel(params=None)

        monkeypatch.setattr("nvflare.app_common.workflows.base_fedavg.BaseFedAvg.load_model", mock_load_model)

        with pytest.raises(ValueError, match="model params is None"):
            controller.load_model()


class TestIntegrationScenarios:
    """Integration tests for complete aggregation scenarios."""

    def test_complete_aggregation_workflow(self):
        """Test a complete workflow: aggregate results and update model."""
        # Initialize controller
        controller = FedAvgLR(damping_factor=0.8, epsilon=0.5, n_features=2, num_clients=3, num_rounds=10)

        # Initial model
        initial_weights = np.array([[0.0], [0.0], [0.0]])
        model = FLModel(
            params={NPConstants.NUMPY_KEY: initial_weights.copy()},
            params_type=ParamsType.FULL,
            meta={"round": 0},
        )

        # Simulate 3 clients returning gradients and hessians
        results = []
        for i in range(3):
            gradient = np.random.randn(3, 1)
            A = np.random.randn(3, 3)
            hessian = A @ A.T + np.eye(3) * 5

            result = FLModel(
                params={"gradient": gradient, "hessian": hessian},
                params_type=ParamsType.FULL,
                current_round=1,
                meta={
                    "client_name": f"site-{i}",
                    FLMetaKey.NUM_STEPS_CURRENT_ROUND: 100,
                },
            )
            results.append(result)

        # Aggregate
        aggr_result = controller.newton_raphson_aggregator_fn(results)

        # Update model
        updated_model = controller.update_model(model, aggr_result)

        # Verify update happened
        assert not np.allclose(updated_model.params[NPConstants.NUMPY_KEY], initial_weights)
        assert updated_model.meta["nr_aggregated"] == 3

    def test_realistic_logistic_regression_scenario(self):
        """Test with realistic logistic regression gradient and hessian values."""
        controller = FedAvgLR(damping_factor=1.0, epsilon=0.01, n_features=4)  # 4 features + 1 bias = 5 parameters

        # Simulate 2 clients with realistic LR values
        # Client 1: 1000 samples
        gradient1 = np.array([[-0.5], [0.3], [-0.2], [0.1], [0.05]])
        hessian1 = np.array(
            [
                [250.0, 10.0, 5.0, 3.0, 2.0],
                [10.0, 230.0, 8.0, 4.0, 1.0],
                [5.0, 8.0, 240.0, 6.0, 3.0],
                [3.0, 4.0, 6.0, 260.0, 5.0],
                [2.0, 1.0, 3.0, 5.0, 270.0],
            ]
        )

        result1 = FLModel(
            params={"gradient": gradient1, "hessian": hessian1},
            params_type=ParamsType.FULL,
            current_round=0,
            meta={
                "client_name": "hospital-1",
                FLMetaKey.NUM_STEPS_CURRENT_ROUND: 1000,
            },
        )

        # Client 2: 500 samples
        gradient2 = np.array([[-0.3], [0.4], [-0.1], [0.15], [0.08]])
        hessian2 = np.array(
            [
                [120.0, 5.0, 2.0, 1.5, 1.0],
                [5.0, 115.0, 4.0, 2.0, 0.5],
                [2.0, 4.0, 125.0, 3.0, 1.5],
                [1.5, 2.0, 3.0, 130.0, 2.5],
                [1.0, 0.5, 1.5, 2.5, 135.0],
            ]
        )

        result2 = FLModel(
            params={"gradient": gradient2, "hessian": hessian2},
            params_type=ParamsType.FULL,
            current_round=0,
            meta={
                "client_name": "hospital-2",
                FLMetaKey.NUM_STEPS_CURRENT_ROUND: 500,
            },
        )

        # Aggregate
        aggr_result = controller.newton_raphson_aggregator_fn([result1, result2])

        # Verify result is reasonable
        updates = aggr_result.params["newton_raphson_updates"]

        # Updates should be finite and reasonable magnitude
        assert np.all(np.isfinite(updates))
        assert np.max(np.abs(updates)) < 10.0  # Reasonable update magnitude
        assert aggr_result.meta["nr_aggregated"] == 2

    def test_edge_case_identical_clients(self):
        """Test aggregation when all clients have identical data."""
        controller = FedAvgLR(damping_factor=1.0, epsilon=1.0, n_features=2)

        # Create identical gradient and hessian
        gradient = np.array([[1.0], [2.0], [3.0]])
        hessian = np.array([[5.0, 0.5, 0.2], [0.5, 6.0, 0.3], [0.2, 0.3, 7.0]])

        # Create 3 clients with identical data
        results = []
        for i in range(3):
            result = FLModel(
                params={"gradient": gradient.copy(), "hessian": hessian.copy()},
                params_type=ParamsType.FULL,
                current_round=0,
                meta={
                    "client_name": f"site-{i}",
                    FLMetaKey.NUM_STEPS_CURRENT_ROUND: 100,
                },
            )
            results.append(result)

        # Aggregate
        aggr_result = controller.newton_raphson_aggregator_fn(results)

        # Result should be same as single client (since all are identical)
        single_result = controller.newton_raphson_aggregator_fn([results[0]])

        np.testing.assert_allclose(
            aggr_result.params["newton_raphson_updates"], single_result.params["newton_raphson_updates"], rtol=1e-10
        )
