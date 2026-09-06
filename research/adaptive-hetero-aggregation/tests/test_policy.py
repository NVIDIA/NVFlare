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

from adaptive_hetero.policy import AdaptiveHeterogeneityPolicy, AdaptiveWeightingConfig, project_bounded_simplex


def test_bounded_simplex_respects_sum_and_bounds():
    result = project_bounded_simplex([0.95, 0.02, 0.02, 0.01], lower=0.10, upper=0.40)
    assert np.isclose(result.sum(), 1.0, atol=1e-12)
    assert np.all(result >= 0.10 - 1e-12)
    assert np.all(result <= 0.40 + 1e-12)


def test_bounded_simplex_randomized_stress():
    rng = np.random.default_rng(20260906)
    for client_count in (3, 4, 8, 16):
        lower = 0.01
        upper = max(0.20, 1.5 / client_count)
        for _ in range(100):
            result = project_bounded_simplex(rng.lognormal(0.0, 2.0, client_count), lower, upper)
            assert np.isclose(result.sum(), 1.0, atol=1e-10)
            assert np.all(result >= lower - 1e-10)
            assert np.all(result <= upper + 1e-10)


def test_low_heterogeneity_uses_exact_sample_weighting_fallback():
    result = AdaptiveHeterogeneityPolicy().compute(
        sample_counts=[100, 200, 300],
        descriptors=[[0.50, 0.50], [0.51, 0.49], [0.49, 0.51]],
        client_metrics=[0.80, 0.70, 0.81],
    )
    assert result.blend_factor == 0.0
    assert np.array_equal(result.weights, result.base_weights)


def test_high_heterogeneity_but_tiny_metric_gap_uses_exact_fallback():
    result = AdaptiveHeterogeneityPolicy().compute(
        sample_counts=[800, 100, 100],
        descriptors=[[0.95, 0.05], [0.05, 0.95], [0.50, 0.50]],
        client_metrics=[0.99, 0.98, 1.00],
    )
    assert result.mean_heterogeneity > 0.15
    assert result.metric_gap < 0.05
    assert result.blend_factor == 0.0
    assert np.array_equal(result.weights, result.base_weights)


def test_material_heterogeneity_and_metric_gap_enable_adaptive_weighting():
    policy = AdaptiveHeterogeneityPolicy(
        AdaptiveWeightingConfig(
            min_weight=0.05,
            max_weight=0.60,
            heterogeneity_threshold=0.04,
            heterogeneity_deadband=0.0,
            performance_gap_threshold=0.05,
            performance_gap_deadband=0.0,
            max_blend_factor=0.80,
        )
    )
    result = policy.compute(
        sample_counts=[800, 100, 100],
        descriptors=[[0.95, 0.05], [0.05, 0.95], [0.50, 0.50]],
        client_metrics=[0.90, 0.55, 0.75],
    )
    assert result.blend_factor > 0.0
    assert result.weights[1] > result.base_weights[1]
    assert result.fairness_scores[1] > result.fairness_scores[2] > result.fairness_scores[0]
    assert np.isclose(result.weights.sum(), 1.0)


def test_absolute_fairness_gap_does_not_amplify_one_percent_difference():
    policy = AdaptiveHeterogeneityPolicy(AdaptiveWeightingConfig(performance_gap_deadband=0.0))
    result = policy.compute(
        sample_counts=[100, 100],
        descriptors=[[0.99, 0.01], [0.01, 0.99]],
        client_metrics=[1.00, 0.99],
    )
    assert result.fairness_scores[1] < 1.03


def test_quality_signal_is_neutral_by_default():
    policy = AdaptiveHeterogeneityPolicy()
    common = dict(
        sample_counts=[100, 100],
        descriptors=[[0.99, 0.01], [0.01, 0.99]],
        client_metrics=[0.90, 0.60],
    )
    without_quality = policy.compute(**common)
    with_quality = policy.compute(**common, quality_improvements=[-10.0, 10.0])
    assert np.allclose(without_quality.weights, with_quality.weights)


def test_client_metrics_must_use_normalized_contract():
    with pytest.raises(ValueError, match="normalized"):
        AdaptiveHeterogeneityPolicy().compute(
            sample_counts=[100, 100],
            descriptors=[[0.5, 0.5], [0.5, 0.5]],
            client_metrics=[80.0, 90.0],
        )


def test_descriptor_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same length"):
        AdaptiveHeterogeneityPolicy().compute(
            sample_counts=[100, 100],
            descriptors=[[1.0, 0.0], [1.0, 0.0, 0.0]],
            client_metrics=[0.8, 0.8],
        )


def test_infeasible_bounds_are_rejected():
    with pytest.raises(ValueError, match="infeasible"):
        project_bounded_simplex([0.5, 0.5], lower=0.0, upper=0.4)


def test_invalid_config_values_are_rejected_early():
    with pytest.raises(ValueError, match="invalid weight bounds"):
        AdaptiveHeterogeneityPolicy(AdaptiveWeightingConfig(min_weight=0.5, max_weight=0.2))
    with pytest.raises(ValueError, match="max_blend_factor"):
        AdaptiveHeterogeneityPolicy(AdaptiveWeightingConfig(max_blend_factor=1.1))
    with pytest.raises(ValueError, match="temperatures"):
        AdaptiveHeterogeneityPolicy(AdaptiveWeightingConfig(performance_gap_temperature=0.0))
