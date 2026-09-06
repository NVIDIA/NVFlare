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
            values = rng.lognormal(mean=0.0, sigma=2.0, size=client_count)
            result = project_bounded_simplex(values, lower=lower, upper=upper)
            assert np.isclose(result.sum(), 1.0, atol=1e-10)
            assert np.all(result >= lower - 1e-10)
            assert np.all(result <= upper + 1e-10)


def test_low_heterogeneity_stays_close_to_sample_weighting():
    policy = AdaptiveHeterogeneityPolicy()
    result = policy.compute(
        sample_counts=[100, 200, 300],
        descriptors=[[0.50, 0.50], [0.51, 0.49], [0.49, 0.51]],
        client_metrics=[0.80, 0.79, 0.81],
        quality_improvements=[0.1, 0.1, 0.1],
    )
    assert result.blend_factor < 0.02
    assert np.max(np.abs(result.weights - result.base_weights)) < 0.01


def test_high_heterogeneity_increases_minimax_pressure():
    policy = AdaptiveHeterogeneityPolicy(
        AdaptiveWeightingConfig(min_weight=0.05, max_weight=0.60, heterogeneity_threshold=0.04)
    )
    result = policy.compute(
        sample_counts=[800, 100, 100],
        descriptors=[[0.95, 0.05], [0.05, 0.95], [0.50, 0.50]],
        client_metrics=[0.90, 0.55, 0.75],
        quality_improvements=[0.1, 0.1, 0.1],
    )
    assert result.blend_factor > 0.50
    assert result.weights[1] > result.base_weights[1]
    assert np.isclose(result.weights.sum(), 1.0)


def test_descriptor_length_mismatch_is_rejected():
    policy = AdaptiveHeterogeneityPolicy()
    with pytest.raises(ValueError, match="same length"):
        policy.compute(
            sample_counts=[100, 100],
            descriptors=[[1.0, 0.0], [1.0, 0.0, 0.0]],
            client_metrics=[0.8, 0.8],
        )


def test_infeasible_bounds_are_rejected():
    with pytest.raises(ValueError, match="infeasible"):
        project_bounded_simplex([0.5, 0.5], lower=0.0, upper=0.4)


def test_invalid_config_bounds_are_rejected_early():
    with pytest.raises(ValueError, match="invalid weight bounds"):
        AdaptiveHeterogeneityPolicy(AdaptiveWeightingConfig(min_weight=0.5, max_weight=0.2))
