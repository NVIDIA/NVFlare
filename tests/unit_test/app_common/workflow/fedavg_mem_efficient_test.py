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

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.workflows.fedavg_mem_efficient import FedAvgMemEfficient


class TestFedAvgMemEfficientInit:
    """Test FedAvgMemEfficient initialization."""

    def test_default_initialization(self):
        """Test FedAvgMemEfficient with default parameters."""
        controller = FedAvgMemEfficient()

        assert controller.num_clients == 3
        assert controller.num_rounds == 5
        assert controller.initial_model is None
        assert controller.aggregator is None

    def test_custom_initialization(self):
        """Test FedAvgMemEfficient with custom parameters."""
        initial_model = {"layer1": [1.0, 2.0, 3.0]}
        controller = FedAvgMemEfficient(
            num_clients=5,
            num_rounds=10,
            initial_model=initial_model,
        )

        assert controller.num_clients == 5
        assert controller.num_rounds == 10
        assert controller.initial_model == initial_model


class TestFedAvgMemEfficientAggregationPyTorch:
    """Test memory-efficient aggregation with PyTorch tensors."""

    def test_aggregate_pytorch_tensors_full_params(self):
        """Test aggregation with PyTorch tensors (FULL params)."""
        import torch

        controller = FedAvgMemEfficient(num_clients=2)

        # Create target model with PyTorch tensors
        target_model = FLModel(
            params={
                "w": torch.tensor([1.0, 2.0, 3.0]),
                "b": torch.tensor([0.5]),
            },
            params_type=ParamsType.FULL,
        )

        # Create client results
        results = [
            FLModel(
                params={
                    "w": torch.tensor([2.0, 4.0, 6.0]),
                    "b": torch.tensor([1.0]),
                },
                params_type=ParamsType.FULL,
                metrics={"accuracy": 0.8},
                meta={"client_name": "site-1", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 10},
            ),
            FLModel(
                params={
                    "w": torch.tensor([4.0, 6.0, 8.0]),
                    "b": torch.tensor([2.0]),
                },
                params_type=ParamsType.FULL,
                metrics={"accuracy": 0.9},
                meta={"client_name": "site-2", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 10},
            ),
        ]

        # Aggregate
        result = controller._aggregate_mem_efficient(target_model, results)

        # Verify aggregated values: (2+4)/2 = 3, (4+6)/2 = 5, (6+8)/2 = 7
        assert torch.allclose(result.params["w"], torch.tensor([3.0, 5.0, 7.0]))
        assert torch.allclose(result.params["b"], torch.tensor([1.5]))

        # Verify results are freed
        assert len(results[0].params) == 0
        assert len(results[1].params) == 0

    def test_aggregate_pytorch_integer_tensors(self):
        """Test aggregation with PyTorch integer tensors."""
        import torch

        controller = FedAvgMemEfficient(num_clients=2)

        target_model = FLModel(
            params={"count": torch.tensor([10, 20, 30], dtype=torch.int64)},
            params_type=ParamsType.FULL,
        )

        results = [
            FLModel(
                params={"count": torch.tensor([1, 2, 3], dtype=torch.int64)},
                params_type=ParamsType.FULL,
                meta={"client_name": "site-1", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 1},
            ),
            FLModel(
                params={"count": torch.tensor([4, 5, 6], dtype=torch.int64)},
                params_type=ParamsType.FULL,
                meta={"client_name": "site-2", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 1},
            ),
        ]

        result = controller._aggregate_mem_efficient(target_model, results)

        # Integer tensors: simple addition (no alpha scaling)
        # Zero out then add: 0 + 1 + 4 = 5, etc
        assert torch.equal(result.params["count"], torch.tensor([5, 7, 9], dtype=torch.int64))

    def test_aggregate_pytorch_diff_params(self):
        """Test aggregation with PyTorch tensors (DIFF params)."""
        import torch

        controller = FedAvgMemEfficient(num_clients=2)

        # Target model starts at [10, 20, 30]
        target_model = FLModel(
            params={"w": torch.tensor([10.0, 20.0, 30.0])},
            params_type=ParamsType.DIFF,
        )

        # Client deltas
        results = [
            FLModel(
                params={"w": torch.tensor([1.0, 2.0, 3.0])},  # delta
                params_type=ParamsType.DIFF,
                meta={"client_name": "site-1", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 10},
            ),
            FLModel(
                params={"w": torch.tensor([2.0, 4.0, 6.0])},  # delta
                params_type=ParamsType.DIFF,
                meta={"client_name": "site-2", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 10},
            ),
        ]

        result = controller._aggregate_mem_efficient(target_model, results)

        # DIFF mode: don't zero, just add deltas
        # 10 + (1+2)/2 = 11.5, 20 + (2+4)/2 = 23, 30 + (3+6)/2 = 34.5
        assert torch.allclose(result.params["w"], torch.tensor([11.5, 23.0, 34.5]))


class TestFedAvgMemEfficientAggregationNumPy:
    """Test memory-efficient aggregation with NumPy arrays."""

    def test_aggregate_numpy_arrays_full_params(self):
        """Test aggregation with NumPy arrays (FULL params)."""
        controller = FedAvgMemEfficient(num_clients=2)

        # Create target model with NumPy arrays
        target_model = FLModel(
            params={
                "w": np.array([1.0, 2.0, 3.0]),
                "b": np.array([0.5]),
            },
            params_type=ParamsType.FULL,
        )

        # Create client results
        results = [
            FLModel(
                params={
                    "w": np.array([2.0, 4.0, 6.0]),
                    "b": np.array([1.0]),
                },
                params_type=ParamsType.FULL,
                metrics={"accuracy": 0.8},
                meta={"client_name": "site-1", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 10},
            ),
            FLModel(
                params={
                    "w": np.array([4.0, 6.0, 8.0]),
                    "b": np.array([2.0]),
                },
                params_type=ParamsType.FULL,
                metrics={"accuracy": 0.9},
                meta={"client_name": "site-2", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 10},
            ),
        ]

        # Aggregate
        result = controller._aggregate_mem_efficient(target_model, results)

        # Verify aggregated values: (2+4)/2 = 3, (4+6)/2 = 5, (6+8)/2 = 7
        np.testing.assert_allclose(result.params["w"], np.array([3.0, 5.0, 7.0]))
        np.testing.assert_allclose(result.params["b"], np.array([1.5]))

        # Verify results are freed
        assert len(results[0].params) == 0
        assert len(results[1].params) == 0

    def test_aggregate_numpy_diff_params(self):
        """Test aggregation with NumPy arrays (DIFF params)."""
        controller = FedAvgMemEfficient(num_clients=2)

        # Target model starts at [10, 20, 30]
        target_model = FLModel(
            params={"w": np.array([10.0, 20.0, 30.0])},
            params_type=ParamsType.DIFF,
        )

        # Client deltas
        results = [
            FLModel(
                params={"w": np.array([1.0, 2.0, 3.0])},
                params_type=ParamsType.DIFF,
                meta={"client_name": "site-1", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 10},
            ),
            FLModel(
                params={"w": np.array([2.0, 4.0, 6.0])},
                params_type=ParamsType.DIFF,
                meta={"client_name": "site-2", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 10},
            ),
        ]

        result = controller._aggregate_mem_efficient(target_model, results)

        # DIFF mode: don't zero, just add deltas
        # 10 + (1+2)/2 = 11.5, 20 + (2+4)/2 = 23, 30 + (3+6)/2 = 34.5
        np.testing.assert_allclose(result.params["w"], np.array([11.5, 23.0, 34.5]))


class TestFedAvgMemEfficientAggregationWeights:
    """Test memory-efficient aggregation with different weights."""

    def test_aggregate_with_different_num_steps(self):
        """Test aggregation with different NUM_STEPS_CURRENT_ROUND."""
        import torch

        controller = FedAvgMemEfficient(num_clients=2)

        target_model = FLModel(
            params={"w": torch.tensor([0.0])},
            params_type=ParamsType.FULL,
        )

        # site-1: 100 steps, site-2: 300 steps
        results = [
            FLModel(
                params={"w": torch.tensor([2.0])},
                params_type=ParamsType.FULL,
                meta={"client_name": "site-1", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 100},
            ),
            FLModel(
                params={"w": torch.tensor([6.0])},
                params_type=ParamsType.FULL,
                meta={"client_name": "site-2", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 300},
            ),
        ]

        result = controller._aggregate_mem_efficient(target_model, results)

        # Weighted average: (2*100 + 6*300) / (100+300) = 2000/400 = 5.0
        assert torch.allclose(result.params["w"], torch.tensor([5.0]))

    def test_aggregate_with_three_clients(self):
        """Test aggregation with three clients."""
        import torch

        controller = FedAvgMemEfficient(num_clients=3)

        target_model = FLModel(
            params={"w": torch.tensor([0.0, 0.0])},
            params_type=ParamsType.FULL,
        )

        results = [
            FLModel(
                params={"w": torch.tensor([1.0, 10.0])},
                params_type=ParamsType.FULL,
                meta={"client_name": "site-1", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 10},
            ),
            FLModel(
                params={"w": torch.tensor([2.0, 20.0])},
                params_type=ParamsType.FULL,
                meta={"client_name": "site-2", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 20},
            ),
            FLModel(
                params={"w": torch.tensor([3.0, 30.0])},
                params_type=ParamsType.FULL,
                meta={"client_name": "site-3", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 30},
            ),
        ]

        result = controller._aggregate_mem_efficient(target_model, results)

        # w[0]: (1*10 + 2*20 + 3*30) / (10+20+30) = 140/60 = 2.333...
        # w[1]: (10*10 + 20*20 + 30*30) / (10+20+30) = 1400/60 = 23.333...
        assert torch.allclose(result.params["w"], torch.tensor([2.333333, 23.333333]), atol=1e-5)


class TestFedAvgMemEfficientMetrics:
    """Test memory-efficient aggregation metrics handling."""

    def test_aggregate_metrics(self):
        """Test that metrics are aggregated correctly."""
        import torch

        controller = FedAvgMemEfficient(num_clients=2)

        target_model = FLModel(
            params={"w": torch.tensor([0.0])},
            params_type=ParamsType.FULL,
        )

        results = [
            FLModel(
                params={"w": torch.tensor([1.0])},
                params_type=ParamsType.FULL,
                metrics={"accuracy": 0.8, "loss": 0.5},
                meta={"client_name": "site-1", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 10},
                current_round=0,
            ),
            FLModel(
                params={"w": torch.tensor([2.0])},
                params_type=ParamsType.FULL,
                metrics={"accuracy": 0.9, "loss": 0.3},
                meta={"client_name": "site-2", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 10},
                current_round=0,
            ),
        ]

        result = controller._aggregate_mem_efficient(target_model, results)

        # Verify metrics are aggregated (weighted average)
        assert result.metrics is not None
        assert "accuracy" in result.metrics
        assert "loss" in result.metrics
        # Equal weights: (0.8 + 0.9) / 2 = 0.85
        assert abs(result.metrics["accuracy"] - 0.85) < 1e-6
        # Equal weights: (0.5 + 0.3) / 2 = 0.4
        assert abs(result.metrics["loss"] - 0.4) < 1e-6

    def test_aggregate_metadata(self):
        """Test that metadata is set correctly."""
        import torch

        controller = FedAvgMemEfficient(num_clients=2)

        target_model = FLModel(
            params={"w": torch.tensor([0.0])},
            params_type=ParamsType.FULL,
        )

        results = [
            FLModel(
                params={"w": torch.tensor([1.0])},
                params_type=ParamsType.FULL,
                meta={"client_name": "site-1"},
                current_round=5,
            ),
            FLModel(
                params={"w": torch.tensor([2.0])},
                params_type=ParamsType.FULL,
                meta={"client_name": "site-2"},
                current_round=5,
            ),
        ]

        result = controller._aggregate_mem_efficient(target_model, results)

        # Verify metadata
        assert result.meta["nr_aggregated"] == 2
        assert result.meta["current_round"] == 5
        assert result.params_type == ParamsType.FULL


class TestFedAvgMemEfficientEdgeCases:
    """Test edge cases for memory-efficient aggregation."""

    def test_aggregate_empty_results(self):
        """Test aggregation with empty results list."""
        import torch

        controller = FedAvgMemEfficient(num_clients=1)

        target_model = FLModel(
            params={"w": torch.tensor([1.0, 2.0])},
            params_type=ParamsType.FULL,
        )

        result = controller._aggregate_mem_efficient(target_model, [])

        # Should return target_model unchanged
        assert result is target_model
        assert torch.equal(result.params["w"], torch.tensor([1.0, 2.0]))

    def test_aggregate_results_with_empty_params(self):
        """Test aggregation when results have empty params."""
        import torch

        controller = FedAvgMemEfficient(num_clients=1)

        target_model = FLModel(
            params={"w": torch.tensor([1.0])},
            params_type=ParamsType.FULL,
        )

        results = [
            FLModel(
                params={},  # Empty params
                params_type=ParamsType.FULL,
                meta={"client_name": "site-1"},
            )
        ]

        result = controller._aggregate_mem_efficient(target_model, results)

        # Should return target_model unchanged
        assert result is target_model

    def test_aggregate_with_missing_param_in_some_results(self):
        """Test aggregation when some results are missing a parameter."""
        import torch

        controller = FedAvgMemEfficient(num_clients=2)

        target_model = FLModel(
            params={"w": torch.tensor([0.0]), "b": torch.tensor([0.0])},
            params_type=ParamsType.FULL,
        )

        results = [
            FLModel(
                params={"w": torch.tensor([2.0]), "b": torch.tensor([1.0])},
                params_type=ParamsType.FULL,
                meta={"client_name": "site-1", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 10},
            ),
            FLModel(
                params={"w": torch.tensor([4.0])},  # Missing 'b'
                params_type=ParamsType.FULL,
                meta={"client_name": "site-2", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 10},
            ),
        ]

        result = controller._aggregate_mem_efficient(target_model, results)

        # w: (2+4)/2 = 3
        assert torch.allclose(result.params["w"], torch.tensor([3.0]))
        # b: only from site-1, so just 1.0
        assert torch.allclose(result.params["b"], torch.tensor([1.0]))


class TestFedAvgMemEfficientFrameworkAgnostic:
    """Test that memory-efficient aggregation works without importing torch when using NumPy."""

    def test_numpy_aggregation_without_torch_import(self):
        """Test NumPy aggregation doesn't require torch import."""
        # This test verifies that the duck-typing approach works
        controller = FedAvgMemEfficient(num_clients=2)

        target_model = FLModel(
            params={"w": np.array([0.0])},
            params_type=ParamsType.FULL,
        )

        results = [
            FLModel(
                params={"w": np.array([1.0])},
                params_type=ParamsType.FULL,
                meta={"client_name": "site-1", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 1},
            ),
            FLModel(
                params={"w": np.array([3.0])},
                params_type=ParamsType.FULL,
                meta={"client_name": "site-2", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 1},
            ),
        ]

        # Verify NumPy arrays don't have torch tensor methods
        assert not hasattr(target_model.params["w"], "add_")
        assert not hasattr(target_model.params["w"], "zero_")

        result = controller._aggregate_mem_efficient(target_model, results)

        # Should still work: (1+3)/2 = 2
        np.testing.assert_allclose(result.params["w"], np.array([2.0]))
