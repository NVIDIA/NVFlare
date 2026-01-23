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

from typing import Dict
from unittest.mock import Mock, patch

import numpy as np

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.aggregators.assembler import Assembler
from nvflare.app_common.aggregators.collect_and_assemble_model_aggregator import CollectAndAssembleModelAggregator
from nvflare.app_common.app_constant import AppConstants


class MockAssembler(Assembler):
    """Mock assembler for testing purposes."""

    def __init__(self, data_kind: str = DataKind.WEIGHTS):
        super().__init__(data_kind=data_kind)
        self.assemble_called = False
        self.reset_called = False
        self.get_model_params_calls = []

    def get_model_params(self, dxo: DXO) -> dict:
        """Extract model parameters from DXO."""
        self.get_model_params_calls.append(dxo)
        # For testing, just return the data as-is
        return dxo.data if dxo.data else {}

    def assemble(self, data: Dict[str, dict], fl_ctx: FLContext) -> DXO:
        """Assemble collected data into aggregated result."""
        self.assemble_called = True

        # Simple averaging for testing
        if not data:
            return DXO(data_kind=self.expected_data_kind, data={})

        # Average the weights from all clients
        aggregated = {}
        for client_name, params in data.items():
            for key, value in params.items():
                if key not in aggregated:
                    aggregated[key] = value.copy() if isinstance(value, np.ndarray) else value
                else:
                    aggregated[key] += value

        # Average
        num_clients = len(data)
        for key in aggregated:
            if isinstance(aggregated[key], np.ndarray):
                aggregated[key] /= num_clients
            else:
                aggregated[key] /= num_clients

        return DXO(data_kind=self.expected_data_kind, data=aggregated)

    def reset(self) -> None:
        """Reset for next round."""
        self.reset_called = True
        super().reset()


class TestCollectAndAssembleModelAggregatorInit:
    """Test CollectAndAssembleModelAggregator initialization."""

    def test_initialization(self):
        """Test basic initialization."""
        aggregator = CollectAndAssembleModelAggregator(assembler_id="test_assembler")

        assert aggregator.assembler_id == "test_assembler"
        assert aggregator.assembler is None
        assert aggregator.fl_ctx is None

    def test_initialization_with_different_ids(self):
        """Test initialization with various assembler IDs."""
        test_ids = ["assembler1", "kmeans_assembler", "svm_assembler", "custom_id_123"]

        for assembler_id in test_ids:
            aggregator = CollectAndAssembleModelAggregator(assembler_id=assembler_id)
            assert aggregator.assembler_id == assembler_id


class TestAcceptModel:
    """Test the accept_model method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.assembler_id = "test_assembler"
        self.aggregator = CollectAndAssembleModelAggregator(assembler_id=self.assembler_id)

        # Create mock FL context
        self.fl_ctx = FLContext()
        self.fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 0)

        # Create mock engine with assembler
        self.mock_assembler = MockAssembler(data_kind=DataKind.WEIGHTS)
        mock_engine = Mock()
        mock_engine.get_component.return_value = self.mock_assembler

        # Mock get_engine to return our mock engine
        self.fl_ctx.get_engine = Mock(return_value=mock_engine)

        # Trigger START_RUN event to set fl_ctx
        self.aggregator.handle_event(EventType.START_RUN, self.fl_ctx)

    def test_accept_model_basic(self):
        """Test accepting a basic model."""
        model = FLModel(
            params={"weight": np.array([1.0, 2.0, 3.0])},
            params_type=ParamsType.FULL,
            current_round=0,
            meta={"client_name": "site-1"},
        )

        self.aggregator.accept_model(model)

        # Verify assembler was initialized
        assert self.aggregator.assembler is not None

        # Verify model was added to collection
        assert "site-1" in self.mock_assembler.collection
        assert "weight" in self.mock_assembler.collection["site-1"]
        np.testing.assert_array_equal(self.mock_assembler.collection["site-1"]["weight"], np.array([1.0, 2.0, 3.0]))

    def test_accept_multiple_models(self):
        """Test accepting models from multiple clients."""
        models = [
            FLModel(
                params={"weight": np.array([1.0, 2.0])},
                params_type=ParamsType.FULL,
                current_round=0,
                meta={"client_name": "site-1"},
            ),
            FLModel(
                params={"weight": np.array([3.0, 4.0])},
                params_type=ParamsType.FULL,
                current_round=0,
                meta={"client_name": "site-2"},
            ),
            FLModel(
                params={"weight": np.array([5.0, 6.0])},
                params_type=ParamsType.FULL,
                current_round=0,
                meta={"client_name": "site-3"},
            ),
        ]

        for model in models:
            self.aggregator.accept_model(model)

        # Verify all models were accepted
        assert len(self.mock_assembler.collection) == 3
        assert "site-1" in self.mock_assembler.collection
        assert "site-2" in self.mock_assembler.collection
        assert "site-3" in self.mock_assembler.collection

    def test_accept_model_duplicate_client(self):
        """Test that duplicate contributions from same client are rejected."""
        model1 = FLModel(
            params={"weight": np.array([1.0, 2.0])},
            params_type=ParamsType.FULL,
            current_round=0,
            meta={"client_name": "site-1"},
        )

        model2 = FLModel(
            params={"weight": np.array([3.0, 4.0])},
            params_type=ParamsType.FULL,
            current_round=0,
            meta={"client_name": "site-1"},  # Same client
        )

        self.aggregator.accept_model(model1)
        self.aggregator.accept_model(model2)

        # Only first contribution should be accepted
        assert len(self.mock_assembler.collection) == 1
        np.testing.assert_array_equal(
            self.mock_assembler.collection["site-1"]["weight"], np.array([1.0, 2.0])  # First model's data
        )

    def test_accept_model_with_return_code_error(self):
        """Test that models with error return codes are rejected."""
        model = FLModel(
            params={"weight": np.array([1.0, 2.0])},
            params_type=ParamsType.FULL,
            current_round=0,
            meta={"client_name": "site-1", "return_code": ReturnCode.ERROR},
        )

        # Manually set return code in the model's shareable representation
        from nvflare.app_common.utils.fl_model_utils import FLModelUtils

        shareable = FLModelUtils.to_shareable(model)
        shareable.set_return_code(ReturnCode.ERROR)

        # For this test, we need to patch the conversion
        with patch("nvflare.app_common.utils.fl_model_utils.FLModelUtils.to_shareable") as mock_to_shareable:
            mock_to_shareable.return_value = shareable
            self.aggregator.accept_model(model)

        # Model should not be added to collection
        assert len(self.mock_assembler.collection) == 0

    def test_accept_model_wrong_data_kind(self):
        """Test that models with wrong data kind are rejected."""
        # Create assembler expecting a specific data kind
        assembler_expecting_collection = MockAssembler(data_kind=DataKind.COLLECTION)
        self.aggregator.assembler = assembler_expecting_collection

        # Create model with WEIGHTS data kind (wrong for this assembler)
        model = FLModel(
            params={"weight": np.array([1.0, 2.0])},
            params_type=ParamsType.FULL,
            current_round=0,
            meta={"client_name": "site-1"},
        )

        # Patch from_shareable to return DXO with WEIGHTS data kind
        with patch("nvflare.apis.dxo.from_shareable") as mock_from_shareable:
            mock_dxo = DXO(data_kind=DataKind.WEIGHTS, data={"weight": np.array([1.0, 2.0])})
            mock_from_shareable.return_value = mock_dxo

            self.aggregator.accept_model(model)

        # Model should not be added to collection (wrong data kind)
        assert len(assembler_expecting_collection.collection) == 0

    def test_accept_model_wrong_round(self):
        """Test that models from wrong round are rejected."""
        # Current round is 0
        self.fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 0)

        # Model from round 1
        model = FLModel(
            params={"weight": np.array([1.0, 2.0])},
            params_type=ParamsType.FULL,
            current_round=1,  # Wrong round
            meta={"client_name": "site-1"},
        )

        self.aggregator.accept_model(model)

        # Model should not be added to collection
        assert len(self.mock_assembler.collection) == 0

    def test_accept_model_none_round_accepted(self):
        """Test that models with None round are accepted (for backward compatibility)."""
        model = FLModel(
            params={"weight": np.array([1.0, 2.0])},
            params_type=ParamsType.FULL,
            current_round=None,  # No round info
            meta={"client_name": "site-1"},
        )

        self.aggregator.accept_model(model)

        # Model should be accepted
        assert len(self.mock_assembler.collection) == 1
        assert "site-1" in self.mock_assembler.collection

    def test_accept_model_complex_params(self):
        """Test accepting model with complex parameter structure."""
        model = FLModel(
            params={
                "layer1.weight": np.array([[1.0, 2.0], [3.0, 4.0]]),
                "layer1.bias": np.array([0.1, 0.2]),
                "layer2.weight": np.array([[5.0, 6.0], [7.0, 8.0]]),
                "layer2.bias": np.array([0.3, 0.4]),
            },
            params_type=ParamsType.FULL,
            current_round=0,
            meta={"client_name": "site-1"},
        )

        self.aggregator.accept_model(model)

        # Verify all parameters were stored
        assert "site-1" in self.mock_assembler.collection
        collection = self.mock_assembler.collection["site-1"]
        assert "layer1.weight" in collection
        assert "layer1.bias" in collection
        assert "layer2.weight" in collection
        assert "layer2.bias" in collection


class TestAggregateModel:
    """Test the aggregate_model method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.assembler_id = "test_assembler"
        self.aggregator = CollectAndAssembleModelAggregator(assembler_id=self.assembler_id)

        # Create mock FL context
        self.fl_ctx = FLContext()
        self.fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 0)

        # Create mock engine with assembler
        self.mock_assembler = MockAssembler(data_kind=DataKind.WEIGHTS)
        mock_engine = Mock()
        mock_engine.get_component.return_value = self.mock_assembler

        # Mock get_engine to return our mock engine
        self.fl_ctx.get_engine = Mock(return_value=mock_engine)

        # Trigger START_RUN event
        self.aggregator.handle_event(EventType.START_RUN, self.fl_ctx)

    def test_aggregate_model_single_client(self):
        """Test aggregation with single client."""
        model = FLModel(
            params={"weight": np.array([2.0, 4.0, 6.0])},
            params_type=ParamsType.FULL,
            current_round=0,
            meta={"client_name": "site-1"},
        )

        self.aggregator.accept_model(model)
        aggregated = self.aggregator.aggregate_model()

        # Verify result
        assert isinstance(aggregated, FLModel)
        assert aggregated.params_type == ParamsType.FULL
        assert "weight" in aggregated.params
        np.testing.assert_array_equal(aggregated.params["weight"], np.array([2.0, 4.0, 6.0]))

        # Verify metadata
        assert aggregated.meta["nr_aggregated"] == 1
        assert aggregated.meta["current_round"] == 0

        # Verify assembler was called
        assert self.mock_assembler.assemble_called

    def test_aggregate_model_multiple_clients(self):
        """Test aggregation with multiple clients (averaging)."""
        models = [
            FLModel(
                params={"weight": np.array([1.0, 2.0, 3.0])},
                params_type=ParamsType.FULL,
                current_round=0,
                meta={"client_name": "site-1"},
            ),
            FLModel(
                params={"weight": np.array([3.0, 4.0, 5.0])},
                params_type=ParamsType.FULL,
                current_round=0,
                meta={"client_name": "site-2"},
            ),
            FLModel(
                params={"weight": np.array([5.0, 6.0, 7.0])},
                params_type=ParamsType.FULL,
                current_round=0,
                meta={"client_name": "site-3"},
            ),
        ]

        for model in models:
            self.aggregator.accept_model(model)

        aggregated = self.aggregator.aggregate_model()

        # Expected average: (1+3+5)/3, (2+4+6)/3, (3+5+7)/3 = 3, 4, 5
        expected = np.array([3.0, 4.0, 5.0])

        np.testing.assert_allclose(aggregated.params["weight"], expected)
        assert aggregated.meta["nr_aggregated"] == 3

    def test_aggregate_model_no_contributions(self):
        """Test aggregation when no models were accepted.

        When no models are accepted, the assembler is never initialized
        (lazy initialization happens on first accept_model call), so
        aggregate_model returns an empty FLModel().
        """
        aggregated = self.aggregator.aggregate_model()

        # Should return empty FLModel since assembler was never initialized
        assert isinstance(aggregated, FLModel)
        # Assembler not initialized, so meta is empty
        assert aggregated.meta == {}
        assert aggregated.params is None

    def test_aggregate_model_assembler_not_initialized(self):
        """Test aggregation when assembler is not initialized."""
        # Create aggregator without initializing assembler
        aggregator = CollectAndAssembleModelAggregator(assembler_id="test")

        # Set fl_ctx but don't initialize assembler
        fl_ctx = FLContext()
        fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 0)
        aggregator.handle_event(EventType.START_RUN, fl_ctx)
        aggregator.assembler = None  # Explicitly set to None

        aggregated = aggregator.aggregate_model()

        # Should return empty FLModel (params is None when created with no args)
        assert isinstance(aggregated, FLModel)
        assert aggregated.params is None

    def test_aggregate_model_complex_data(self):
        """Test aggregation with complex multi-layer model data."""
        models = [
            FLModel(
                params={
                    "conv1.weight": np.array([[1.0, 2.0], [3.0, 4.0]]),
                    "conv1.bias": np.array([0.5, 1.0]),
                    "fc.weight": np.array([[2.0, 3.0], [4.0, 5.0]]),
                },
                params_type=ParamsType.FULL,
                current_round=0,
                meta={"client_name": "site-1"},
            ),
            FLModel(
                params={
                    "conv1.weight": np.array([[5.0, 6.0], [7.0, 8.0]]),
                    "conv1.bias": np.array([1.5, 2.0]),
                    "fc.weight": np.array([[6.0, 7.0], [8.0, 9.0]]),
                },
                params_type=ParamsType.FULL,
                current_round=0,
                meta={"client_name": "site-2"},
            ),
        ]

        for model in models:
            self.aggregator.accept_model(model)

        aggregated = self.aggregator.aggregate_model()

        # Verify all layers were aggregated
        assert "conv1.weight" in aggregated.params
        assert "conv1.bias" in aggregated.params
        assert "fc.weight" in aggregated.params

        # Verify averaging
        expected_conv1_weight = np.array([[3.0, 4.0], [5.0, 6.0]])
        expected_conv1_bias = np.array([1.0, 1.5])
        expected_fc_weight = np.array([[4.0, 5.0], [6.0, 7.0]])

        np.testing.assert_allclose(aggregated.params["conv1.weight"], expected_conv1_weight)
        np.testing.assert_allclose(aggregated.params["conv1.bias"], expected_conv1_bias)
        np.testing.assert_allclose(aggregated.params["fc.weight"], expected_fc_weight)


class TestResetStats:
    """Test the reset_stats method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.assembler_id = "test_assembler"
        self.aggregator = CollectAndAssembleModelAggregator(assembler_id=self.assembler_id)

        # Create mock FL context
        self.fl_ctx = FLContext()
        self.fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 0)

        # Create mock engine with assembler
        self.mock_assembler = MockAssembler(data_kind=DataKind.WEIGHTS)
        mock_engine = Mock()
        mock_engine.get_component.return_value = self.mock_assembler

        # Mock get_engine to return our mock engine
        self.fl_ctx.get_engine = Mock(return_value=mock_engine)

        # Trigger START_RUN event
        self.aggregator.handle_event(EventType.START_RUN, self.fl_ctx)

    def test_reset_stats_basic(self):
        """Test that reset_stats clears the collection."""
        # Add some models
        model = FLModel(
            params={"weight": np.array([1.0, 2.0])},
            params_type=ParamsType.FULL,
            current_round=0,
            meta={"client_name": "site-1"},
        )
        self.aggregator.accept_model(model)

        # Verify collection has data
        assert len(self.mock_assembler.collection) > 0

        # Reset
        self.aggregator.reset_stats()

        # Verify collection is cleared
        assert len(self.mock_assembler.collection) == 0
        assert self.mock_assembler.reset_called

    def test_reset_stats_when_assembler_none(self):
        """Test reset_stats when assembler is None (should not crash)."""
        aggregator = CollectAndAssembleModelAggregator(assembler_id="test")
        aggregator.assembler = None

        # Should not raise exception
        aggregator.reset_stats()

    def test_reset_stats_after_aggregation(self):
        """Test typical workflow: accept, aggregate, reset."""
        models = [
            FLModel(
                params={"weight": np.array([1.0, 2.0])},
                params_type=ParamsType.FULL,
                current_round=0,
                meta={"client_name": "site-1"},
            ),
            FLModel(
                params={"weight": np.array([3.0, 4.0])},
                params_type=ParamsType.FULL,
                current_round=0,
                meta={"client_name": "site-2"},
            ),
        ]

        # Accept models
        for model in models:
            self.aggregator.accept_model(model)
        assert len(self.mock_assembler.collection) == 2

        # Aggregate
        self.aggregator.aggregate_model()

        # Reset for next round
        self.aggregator.reset_stats()
        assert len(self.mock_assembler.collection) == 0

        # Can accept new models for next round
        self.fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 1)
        new_model = FLModel(
            params={"weight": np.array([5.0, 6.0])},
            params_type=ParamsType.FULL,
            current_round=1,
            meta={"client_name": "site-1"},
        )
        self.aggregator.accept_model(new_model)
        assert len(self.mock_assembler.collection) == 1


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        self.assembler_id = "test_assembler"
        self.aggregator = CollectAndAssembleModelAggregator(assembler_id=self.assembler_id)

        # Create mock FL context
        self.fl_ctx = FLContext()
        self.fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 0)

        # Create mock engine with assembler
        self.mock_assembler = MockAssembler(data_kind=DataKind.WEIGHTS)
        mock_engine = Mock()
        mock_engine.get_component.return_value = self.mock_assembler

        # Mock get_engine to return our mock engine
        self.fl_ctx.get_engine = Mock(return_value=mock_engine)

        # Trigger START_RUN event
        self.aggregator.handle_event(EventType.START_RUN, self.fl_ctx)

    def test_multi_round_workflow(self):
        """Test complete multi-round aggregation workflow."""
        # Round 0
        self.fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 0)

        models_r0 = [
            FLModel(
                params={"weight": np.array([1.0, 2.0])},
                params_type=ParamsType.FULL,
                current_round=0,
                meta={"client_name": f"site-{i}"},
            )
            for i in range(3)
        ]

        for model in models_r0:
            self.aggregator.accept_model(model)

        result_r0 = self.aggregator.aggregate_model()
        assert result_r0.meta["current_round"] == 0
        assert result_r0.meta["nr_aggregated"] == 3

        # Reset for next round
        self.aggregator.reset_stats()

        # Round 1
        self.fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 1)

        models_r1 = [
            FLModel(
                params={"weight": np.array([2.0, 3.0])},
                params_type=ParamsType.FULL,
                current_round=1,
                meta={"client_name": f"site-{i}"},
            )
            for i in range(3)
        ]

        for model in models_r1:
            self.aggregator.accept_model(model)

        result_r1 = self.aggregator.aggregate_model()
        assert result_r1.meta["current_round"] == 1
        assert result_r1.meta["nr_aggregated"] == 3

    def test_heterogeneous_client_participation(self):
        """Test with different clients participating in different rounds."""
        # Round 0: 3 clients
        self.fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 0)

        for i in range(3):
            model = FLModel(
                params={"weight": np.array([float(i), float(i + 1)])},
                params_type=ParamsType.FULL,
                current_round=0,
                meta={"client_name": f"site-{i}"},
            )
            self.aggregator.accept_model(model)

        result = self.aggregator.aggregate_model()
        assert result.meta["nr_aggregated"] == 3

        # Reset
        self.aggregator.reset_stats()

        # Round 1: only 2 clients
        self.fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 1)

        for i in [0, 2]:  # Only sites 0 and 2
            model = FLModel(
                params={"weight": np.array([float(i + 2), float(i + 3)])},
                params_type=ParamsType.FULL,
                current_round=1,
                meta={"client_name": f"site-{i}"},
            )
            self.aggregator.accept_model(model)

        result = self.aggregator.aggregate_model()
        assert result.meta["nr_aggregated"] == 2

    def test_large_scale_aggregation(self):
        """Test aggregation with many clients."""
        num_clients = 100

        for i in range(num_clients):
            model = FLModel(
                params={"weight": np.random.randn(10)},
                params_type=ParamsType.FULL,
                current_round=0,
                meta={"client_name": f"site-{i}"},
            )
            self.aggregator.accept_model(model)

        result = self.aggregator.aggregate_model()

        assert result.meta["nr_aggregated"] == num_clients
        assert "weight" in result.params
        assert result.params["weight"].shape == (10,)

    def test_edge_case_empty_round(self):
        """Test handling when no clients participate.

        When no models are accepted, the assembler is never initialized,
        so aggregate_model returns an empty FLModel().
        """
        # Don't add any models
        result = self.aggregator.aggregate_model()

        # Should handle gracefully - returns empty FLModel
        assert isinstance(result, FLModel)
        # Assembler not initialized, so meta is empty
        assert result.meta == {}
        assert result.params is None
