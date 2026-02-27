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

from nvflare.app_common.aggregators.weighted_aggregation_helper import (
    WeightedAggregationHelper,
    _is_aggregatable_metric_value,
)


class TestIsAggregatableMetricValue:
    """Test _is_aggregatable_metric_value predicate."""

    def test_none_is_not_aggregatable(self):
        assert _is_aggregatable_metric_value(None) is False

    def test_dict_list_set_tuple_str_not_aggregatable(self):
        assert _is_aggregatable_metric_value({}) is False
        assert _is_aggregatable_metric_value({"a": 1}) is False
        assert _is_aggregatable_metric_value([]) is False
        assert _is_aggregatable_metric_value([1, 2]) is False
        assert _is_aggregatable_metric_value(set()) is False
        assert _is_aggregatable_metric_value((1, 2)) is False
        assert _is_aggregatable_metric_value("") is False
        assert _is_aggregatable_metric_value("hello") is False

    def test_int_float_bool_aggregatable(self):
        assert _is_aggregatable_metric_value(0) is True
        assert _is_aggregatable_metric_value(1) is True
        assert _is_aggregatable_metric_value(1.0) is True
        assert _is_aggregatable_metric_value(True) is True
        assert _is_aggregatable_metric_value(False) is True

    def test_numpy_array_aggregatable(self):
        assert _is_aggregatable_metric_value(np.array([1.0, 2.0])) is True
        assert _is_aggregatable_metric_value(np.float64(3.14)) is True

    def test_torch_tensor_aggregatable(self):
        assert _is_aggregatable_metric_value(torch.tensor([1.0, 2.0])) is True
        assert _is_aggregatable_metric_value(torch.tensor(3.14)) is True

    def test_object_with_shape_and_arithmetic_aggregatable(self):
        class FakeArray:
            shape = (2, 3)

            def __mul__(self, other):
                return self

            def __add__(self, other):
                return self

        assert _is_aggregatable_metric_value(FakeArray()) is True

    def test_object_supporting_mul_add_aggregatable(self):
        class Scalelike:
            def __mul__(self, other):
                return self

            def __add__(self, other):
                return self

        assert _is_aggregatable_metric_value(Scalelike()) is True

    def test_object_not_supporting_mul_not_aggregatable(self):
        class NoMul:
            def __add__(self, other):
                return self

        assert _is_aggregatable_metric_value(NoMul()) is False

    def test_object_not_supporting_add_not_aggregatable(self):
        class NoAdd:
            def __mul__(self, other):
                return self

        assert _is_aggregatable_metric_value(NoAdd()) is False

    def test_numpy_string_array_not_aggregatable(self):
        assert _is_aggregatable_metric_value(np.array(["a"])) is False

    def test_object_raising_value_error_not_aggregatable(self):
        class BadValue:
            def __mul__(self, other):
                raise ValueError("unsupported")

            def __add__(self, other):
                return self

        assert _is_aggregatable_metric_value(BadValue()) is False


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

    def test_pytorch_int_tensor_weighting(self):
        """Test PyTorch integer tensors are weighted the same as float tensors."""
        helper = WeightedAggregationHelper()

        # First contribution (integer tensor)
        data1 = {"count": torch.tensor([10, 20, 30], dtype=torch.long)}
        helper.add(data1, weight=2.0, contributor_name="site-1", contribution_round=0)

        # Second contribution
        data2 = {"count": torch.tensor([5, 10, 15], dtype=torch.long)}
        helper.add(data2, weight=3.0, contributor_name="site-2", contribution_round=0)

        result = helper.get_result()

        expected = torch.tensor([7.0, 14.0, 21.0])
        assert torch.allclose(result["count"], expected)

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
        assert torch.allclose(result["counts"], torch.tensor([20.0 / 3, 40.0 / 3]))

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

    def test_add_metrics_no_op_for_none_or_empty(self):
        """add_metrics with None or empty data does nothing."""
        helper = WeightedAggregationHelper()
        helper.add_metrics(None, 1.0, "site-1", 0)
        helper.add_metrics({}, 1.0, "site-1", 0)
        result = helper.get_result()
        assert len(result) == 0

    def test_add_metrics_filters_non_aggregatable(self):
        """add_metrics only aggregates scalar/array-like values; skips dict, list, str."""
        helper = WeightedAggregationHelper()
        data = {
            "loss": 0.5,
            "accuracy": 0.9,
            "config": {"lr": 0.01},  # skipped
            "tags": ["a", "b"],  # skipped
            "name": "run1",  # skipped
        }
        helper.add_metrics(data, weight=1.0, contributor_name="site-1", contribution_round=0)
        result = helper.get_result()
        assert "loss" in result
        assert "accuracy" in result
        assert result["loss"] == 0.5
        assert result["accuracy"] == 0.9
        assert "config" not in result
        assert "tags" not in result
        assert "name" not in result

    def test_add_metrics_aggregates_like_add(self):
        """add_metrics passes filtered data to add(); weighted average is correct."""
        helper = WeightedAggregationHelper()
        helper.add_metrics(
            {"loss": 0.2, "acc": 0.8},
            weight=2.0,
            contributor_name="site-1",
            contribution_round=0,
        )
        helper.add_metrics(
            {"loss": 0.6, "acc": 0.4},
            weight=1.0,
            contributor_name="site-2",
            contribution_round=0,
        )
        result = helper.get_result()
        # loss: (0.2*2 + 0.6*1)/3 = 1.0/3, acc: (0.8*2 + 0.4*1)/3 = 2.0/3
        assert abs(result["loss"] - 1.0 / 3) < 1e-9
        assert abs(result["acc"] - 2.0 / 3) < 1e-9

    def test_add_metrics_warn_skipped_called(self):
        """warn_skipped is invoked for each skipped metric key (key, type_name)."""
        helper = WeightedAggregationHelper()
        warned = []

        def capture(key, type_name):
            warned.append((key, type_name))

        data = {"loss": 0.5, "meta": {"x": 1}, "label": "train"}
        helper.add_metrics(
            data,
            weight=1.0,
            contributor_name="site-1",
            contribution_round=0,
            warn_skipped=capture,
        )
        assert len(warned) == 2
        assert ("meta", "dict") in warned
        assert ("label", "str") in warned
        result = helper.get_result()
        assert result["loss"] == 0.5

    def test_add_metrics_warned_metric_keys_only_warn_once(self):
        """With warned_metric_keys, each key triggers warn_skipped at most once."""
        helper = WeightedAggregationHelper()
        warned = []
        warned_keys = set()

        def capture(key, type_name):
            warned.append((key, type_name))

        # First call: skip "meta" and "name" -> both warned
        helper.add_metrics(
            {"loss": 0.1, "meta": {}, "name": "x"},
            weight=1.0,
            contributor_name="site-1",
            contribution_round=0,
            warn_skipped=capture,
            warned_metric_keys=warned_keys,
        )
        assert len(warned) == 2
        assert warned_keys == {"meta", "name"}

        # Second call: same keys skipped -> no additional warnings
        helper.add_metrics(
            {"acc": 0.2, "meta": {}, "name": "y"},
            weight=1.0,
            contributor_name="site-2",
            contribution_round=0,
            warn_skipped=capture,
            warned_metric_keys=warned_keys,
        )
        assert len(warned) == 2
        assert warned_keys == {"meta", "name"}

        result = helper.get_result()
        assert result["loss"] == 0.1
        assert result["acc"] == 0.2

    def test_add_metrics_all_skipped_no_add(self):
        """When all values are non-aggregatable, add() is not called (no history)."""
        helper = WeightedAggregationHelper()
        helper.add_metrics(
            {"a": [], "b": "x"},
            weight=1.0,
            contributor_name="site-1",
            contribution_round=0,
        )
        assert len(helper.history) == 0
        result = helper.get_result()
        assert len(result) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
