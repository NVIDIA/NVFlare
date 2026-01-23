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
import torch

from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper


class TestWeightedAggregationHelper:
    """Test WeightedAggregationHelper with PyTorch tensors and NumPy arrays."""

    def test_pytorch_float_single_contribution(self):
        """Test single contribution with PyTorch float tensors."""
        helper = WeightedAggregationHelper()

        data = {"w1": torch.tensor([1.0, 2.0, 3.0]), "w2": torch.tensor([4.0, 5.0])}
        helper.add(data, weight=2.0, contributor_name="site-1", contribution_round=0)

        result = helper.get_result()

        # Should be weighted by 2.0, then divided by 2.0 = original values
        assert torch.allclose(result["w1"], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(result["w2"], torch.tensor([4.0, 5.0]))

    def test_pytorch_float_multiple_contributions(self):
        """Test multiple contributions with PyTorch float tensors."""
        helper = WeightedAggregationHelper()

        # First contribution
        data1 = {"w": torch.tensor([2.0, 4.0, 6.0])}
        helper.add(data1, weight=1.0, contributor_name="site-1", contribution_round=0)

        # Second contribution
        data2 = {"w": torch.tensor([3.0, 6.0, 9.0])}
        helper.add(data2, weight=2.0, contributor_name="site-2", contribution_round=0)

        result = helper.get_result()

        # Expected: (2*1 + 3*2) / (1+2) = (2+6)/3 = [8/3, 16/3, 24/3]
        expected = torch.tensor([8.0 / 3, 16.0 / 3, 24.0 / 3])
        assert torch.allclose(result["w"], expected)

    def test_pytorch_int_tensor_no_weighting(self):
        """Test integer tensors are not weighted (just summed)."""
        helper = WeightedAggregationHelper()

        # First contribution (integer tensor)
        data1 = {"count": torch.tensor([10, 20, 30], dtype=torch.long)}
        helper.add(data1, weight=2.0, contributor_name="site-1", contribution_round=0)

        # Second contribution
        data2 = {"count": torch.tensor([5, 10, 15], dtype=torch.long)}
        helper.add(data2, weight=3.0, contributor_name="site-2", contribution_round=0)

        result = helper.get_result()

        # Integer tensors: should be summed (not averaged)
        # Expected: 10+5, 20+10, 30+15 = [15, 30, 45]
        expected = torch.tensor([15, 30, 45], dtype=torch.long)
        assert torch.equal(result["count"], expected)

    def test_pytorch_mixed_float_int_tensors(self):
        """Test mixed float and integer tensors."""
        helper = WeightedAggregationHelper()

        data1 = {
            "weights": torch.tensor([2.0, 4.0]),
            "counts": torch.tensor([10, 20], dtype=torch.long),
        }
        helper.add(data1, weight=1.0, contributor_name="site-1", contribution_round=0)

        data2 = {
            "weights": torch.tensor([3.0, 6.0]),
            "counts": torch.tensor([5, 10], dtype=torch.long),
        }
        helper.add(data2, weight=2.0, contributor_name="site-2", contribution_round=0)

        result = helper.get_result()

        # Float weights: (2*1 + 3*2)/(1+2) = 8/3, (4*1 + 6*2)/(1+2) = 16/3
        assert torch.allclose(result["weights"], torch.tensor([8.0 / 3, 16.0 / 3]))

        # Integer counts: sum only = 15, 30
        assert torch.equal(result["counts"], torch.tensor([15, 30], dtype=torch.long))

    def test_numpy_single_contribution(self):
        """Test single contribution with NumPy arrays."""
        helper = WeightedAggregationHelper()

        data = {"w": np.array([1.0, 2.0, 3.0])}
        helper.add(data, weight=2.0, contributor_name="site-1", contribution_round=0)

        result = helper.get_result()

        # Should be weighted by 2.0, then divided by 2.0 = original values
        np.testing.assert_allclose(result["w"], np.array([1.0, 2.0, 3.0]))

    def test_numpy_multiple_contributions(self):
        """Test multiple contributions with NumPy arrays."""
        helper = WeightedAggregationHelper()

        data1 = {"w": np.array([2.0, 4.0, 6.0])}
        helper.add(data1, weight=1.0, contributor_name="site-1", contribution_round=0)

        data2 = {"w": np.array([3.0, 6.0, 9.0])}
        helper.add(data2, weight=2.0, contributor_name="site-2", contribution_round=0)

        result = helper.get_result()

        # Expected: (2*1 + 3*2) / (1+2) = [8/3, 16/3, 24/3]
        expected = np.array([8.0 / 3, 16.0 / 3, 24.0 / 3])
        np.testing.assert_allclose(result["w"], expected)

    def test_exclude_vars_regex(self):
        """Test exclude_vars regex filtering."""
        helper = WeightedAggregationHelper(exclude_vars="bias")

        data = {
            "layer1.weight": torch.tensor([1.0, 2.0]),
            "layer1.bias": torch.tensor([0.1, 0.2]),  # Should be excluded
            "layer2.weight": torch.tensor([3.0, 4.0]),
        }
        helper.add(data, weight=1.0, contributor_name="site-1", contribution_round=0)

        result = helper.get_result()

        # bias should be excluded
        assert "layer1.weight" in result
        assert "layer2.weight" in result
        assert "layer1.bias" not in result

    def test_weigh_by_local_iter_false(self):
        """Test weigh_by_local_iter=False (equal weighting regardless of weight param)."""
        helper = WeightedAggregationHelper(weigh_by_local_iter=False)

        data1 = {"w": torch.tensor([2.0, 4.0])}
        helper.add(data1, weight=1.0, contributor_name="site-1", contribution_round=0)

        data2 = {"w": torch.tensor([4.0, 8.0])}
        helper.add(data2, weight=100.0, contributor_name="site-2", contribution_round=0)  # Large weight

        result = helper.get_result()

        # With weigh_by_local_iter=False:
        # - First contribution: stored without weighting
        # - Subsequent contributions: added without alpha
        # - Counts still accumulate weights
        # Result = (2+4) / (1+100) = 6/101, (4+8) / (1+100) = 12/101
        expected = torch.tensor([6.0 / 101, 12.0 / 101])
        assert torch.allclose(result["w"], expected)

    def test_reset_stats(self):
        """Test reset_stats clears accumulated data."""
        helper = WeightedAggregationHelper()

        data = {"w": torch.tensor([1.0, 2.0])}
        helper.add(data, weight=1.0, contributor_name="site-1", contribution_round=0)

        assert len(helper.total) == 1
        assert len(helper.history) == 1

        helper.reset_stats()

        assert len(helper.total) == 0
        assert len(helper.counts) == 0
        assert len(helper.history) == 0

    def test_history_tracking(self):
        """Test contribution history is tracked."""
        helper = WeightedAggregationHelper()

        data1 = {"w": torch.tensor([1.0])}
        helper.add(data1, weight=2.0, contributor_name="site-1", contribution_round=0)

        data2 = {"w": torch.tensor([2.0])}
        helper.add(data2, weight=3.0, contributor_name="site-2", contribution_round=1)

        assert len(helper.history) == 2
        assert helper.history[0] == {"contributor_name": "site-1", "round": 0, "weight": 2.0}
        assert helper.history[1] == {"contributor_name": "site-2", "round": 1, "weight": 3.0}

    def test_get_len(self):
        """Test get_len returns number of contributions."""
        helper = WeightedAggregationHelper()

        assert helper.get_len() == 0

        data1 = {"w": torch.tensor([1.0])}
        helper.add(data1, weight=1.0, contributor_name="site-1", contribution_round=0)

        assert helper.get_len() == 1

        data2 = {"w": torch.tensor([2.0])}
        helper.add(data2, weight=1.0, contributor_name="site-2", contribution_round=0)

        assert helper.get_len() == 2

    def test_empty_data(self):
        """Test handling of empty data dict."""
        helper = WeightedAggregationHelper()

        helper.add({}, weight=1.0, contributor_name="site-1", contribution_round=0)

        result = helper.get_result()
        assert len(result) == 0

    def test_thread_safety_simulation(self):
        """Test that multiple additions work (basic thread safety check)."""
        helper = WeightedAggregationHelper()

        # Simulate concurrent additions (though not truly concurrent in test)
        for i in range(10):
            data = {"w": torch.tensor([float(i)])}
            helper.add(data, weight=1.0, contributor_name=f"site-{i}", contribution_round=0)

        result = helper.get_result()

        # Expected: sum(0..9) / 10 = 45/10 = 4.5
        assert torch.allclose(result["w"], torch.tensor([4.5]))

    def test_different_keys_per_contribution(self):
        """Test contributions with different sets of keys."""
        helper = WeightedAggregationHelper()

        data1 = {"w1": torch.tensor([1.0]), "w2": torch.tensor([2.0])}
        helper.add(data1, weight=1.0, contributor_name="site-1", contribution_round=0)

        data2 = {"w2": torch.tensor([3.0]), "w3": torch.tensor([4.0])}
        helper.add(data2, weight=1.0, contributor_name="site-2", contribution_round=0)

        result = helper.get_result()

        # w1: only from site-1 = 1.0
        # w2: from both = (2.0 + 3.0) / 2 = 2.5
        # w3: only from site-2 = 4.0
        assert torch.allclose(result["w1"], torch.tensor([1.0]))
        assert torch.allclose(result["w2"], torch.tensor([2.5]))
        assert torch.allclose(result["w3"], torch.tensor([4.0]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
