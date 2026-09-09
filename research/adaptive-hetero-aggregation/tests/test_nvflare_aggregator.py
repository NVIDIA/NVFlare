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

import json

import numpy as np
import torch
from adaptive_hetero.nvflare_aggregator import AdaptiveHeterogeneityAggregator, AdaptiveMetaKey

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe


def _context(round_number=0):
    ctx = FLContext()
    ctx.set_prop(AppConstants.CURRENT_ROUND, round_number, private=True, sticky=False)
    return ctx


def _contribution(name, round_number, value, steps, descriptor, metric, quality=0.0, sample_count=None):
    sample_count = steps if sample_count is None else sample_count
    dxo = DXO(
        data_kind=DataKind.WEIGHT_DIFF,
        data={"weight": np.asarray([value], dtype=np.float32)},
        meta={
            MetaKey.NUM_STEPS_CURRENT_ROUND: steps,
            AdaptiveMetaKey.SAMPLE_COUNT: sample_count,
            AdaptiveMetaKey.DISTRIBUTION_DESCRIPTOR: descriptor,
            AdaptiveMetaKey.CLIENT_METRIC: metric,
            AdaptiveMetaKey.QUALITY_IMPROVEMENT: quality,
        },
    )
    shareable = dxo.to_shareable()
    shareable.set_peer_props({ReservedKey.IDENTITY_NAME: name})
    shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, round_number)
    return shareable


def test_real_nvflare_aggregation_matches_server_side_weights():
    aggregator = AdaptiveHeterogeneityAggregator(
        metric_prior_strength=0.0,
        heterogeneity_threshold=0.0,
        heterogeneity_temperature=0.02,
        heterogeneity_deadband=0.0,
        performance_gap_threshold=0.0,
        performance_gap_deadband=0.0,
        max_blend_factor=0.80,
        activation_warmup_rounds=0,
        activation_patience=1,
        require_stable_cohort=False,
        min_weight=0.10,
        max_weight=0.80,
    )
    ctx = _context()
    first = _contribution("site-1", 0, 1.0, 900, [0.95, 0.05], 0.90, 0.10, sample_count=900)
    second = _contribution("site-2", 0, 3.0, 100, [0.05, 0.95], 0.55, 0.10, sample_count=100)

    assert aggregator.accept(first, ctx)
    assert aggregator.accept(second, ctx)
    result = from_shareable(aggregator.aggregate(ctx))

    weights = aggregator.last_weights
    assert np.isclose(sum(weights.values()), 1.0)
    assert weights["site-2"] > 0.10
    expected = weights["site-1"] * 1.0 + weights["site-2"] * 3.0
    assert np.allclose(result.data["weight"], np.asarray([expected], dtype=np.float32), atol=1e-6)
    assert 0.0 < result.meta[AdaptiveMetaKey.BLEND_FACTOR] <= 0.80
    assert "adaptive_final_weights" not in result.meta


def test_native_fallback_uses_optimizer_steps_not_sample_count():
    aggregator = AdaptiveHeterogeneityAggregator(metric_prior_strength=0.0, activation_warmup_rounds=3)
    ctx = _context()
    assert aggregator.accept(_contribution("site-1", 0, 1.0, 9, [0.5, 0.5], 0.8, sample_count=100), ctx)
    assert aggregator.accept(_contribution("site-2", 0, 3.0, 1, [0.5, 0.5], 0.8, sample_count=100), ctx)
    result = from_shareable(aggregator.aggregate(ctx))

    assert result.meta[AdaptiveMetaKey.BLEND_FACTOR] == 0.0
    assert aggregator.last_weights["site-1"] == 0.9
    assert aggregator.last_weights["site-2"] == 0.1
    assert np.allclose(result.data["weight"], np.asarray([1.2], dtype=np.float32), atol=1e-6)


def test_aggregator_preserves_activation_history_and_resets_on_cohort_change():
    aggregator = AdaptiveHeterogeneityAggregator(
        metric_prior_strength=0.0,
        heterogeneity_threshold=0.0,
        heterogeneity_deadband=0.0,
        performance_gap_threshold=0.0,
        performance_gap_deadband=0.0,
        activation_warmup_rounds=2,
        activation_patience=2,
        require_stable_cohort=True,
        min_weight=0.10,
        max_weight=0.90,
    )

    blends = []
    for round_number in range(3):
        ctx = _context(round_number)
        assert aggregator.accept(
            _contribution("site-1", round_number, 1.0, 10, [0.99, 0.01], 0.90, sample_count=100), ctx
        )
        assert aggregator.accept(
            _contribution("site-2", round_number, 3.0, 10, [0.01, 0.99], 0.50, sample_count=100), ctx
        )
        result = from_shareable(aggregator.aggregate(ctx))
        blends.append(result.meta[AdaptiveMetaKey.BLEND_FACTOR])

    assert blends[0] == 0.0
    assert blends[1] == 0.0
    assert blends[2] > 0.0

    changed_ctx = _context(3)
    assert aggregator.accept(_contribution("site-1", 3, 1.0, 10, [0.99, 0.01], 0.90, sample_count=100), changed_ctx)
    assert aggregator.accept(_contribution("site-3", 3, 3.0, 10, [0.01, 0.99], 0.50, sample_count=100), changed_ctx)
    changed = from_shareable(aggregator.aggregate(changed_ctx))
    assert changed.meta[AdaptiveMetaKey.BLEND_FACTOR] == 0.0
    assert changed.meta[AdaptiveMetaKey.ACTIVATION_STREAK] == 1


def test_default_bounds_support_single_client():
    aggregator = AdaptiveHeterogeneityAggregator()
    ctx = _context()
    assert aggregator.accept(_contribution("site-1", 0, 2.0, 5, [0.5, 0.5], 0.8, sample_count=100), ctx)
    result = from_shareable(aggregator.aggregate(ctx))
    assert aggregator.last_weights == {"site-1": 1.0}
    assert np.allclose(result.data["weight"], np.asarray([2.0], dtype=np.float32))


def test_empty_round_returns_empty_result_instead_of_raising():
    aggregator = AdaptiveHeterogeneityAggregator()
    reply = aggregator.aggregate(_context())
    assert reply.get_return_code() == ReturnCode.EMPTY_RESULT


def test_rejects_missing_metadata_duplicate_and_wrong_round():
    aggregator = AdaptiveHeterogeneityAggregator(min_weight=0.10, max_weight=0.80)
    ctx = _context(round_number=2)

    wrong_round = _contribution("site-1", 1, 1.0, 10, [0.5, 0.5], 0.8)
    assert not aggregator.accept(wrong_round, ctx)

    missing = DXO(
        data_kind=DataKind.WEIGHT_DIFF,
        data={"weight": np.asarray([1.0], dtype=np.float32)},
        meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 10},
    ).to_shareable()
    missing.set_peer_props({ReservedKey.IDENTITY_NAME: "site-1"})
    missing.add_cookie(AppConstants.CONTRIBUTION_ROUND, 2)
    assert not aggregator.accept(missing, ctx)

    valid = _contribution("site-1", 2, 1.0, 10, [0.5, 0.5], 0.8)
    assert aggregator.accept(valid, ctx)
    assert not aggregator.accept(valid, ctx)


def test_rejects_nonfinite_negative_and_mismatched_descriptors():
    aggregator = AdaptiveHeterogeneityAggregator(min_weight=0.10, max_weight=0.80)
    ctx = _context()

    assert not aggregator.accept(_contribution("bad-metric", 0, 1.0, 10, [0.5, 0.5], float("nan")), ctx)
    assert not aggregator.accept(_contribution("negative", 0, 1.0, 10, [0.5, -0.5], 0.8), ctx)

    assert aggregator.accept(_contribution("site-1", 0, 1.0, 10, [0.5, 0.5], 0.8), ctx)
    assert not aggregator.accept(_contribution("site-2", 0, 1.0, 10, [0.2, 0.3, 0.5], 0.8), ctx)


def _custom_aggregator():
    return AdaptiveHeterogeneityAggregator(
        metric_prior_strength=7.0,
        heterogeneity_threshold=0.12,
        max_blend_factor=0.15,
        min_weight=0.03,
        max_weight=0.70,
    )


def test_constructor_arguments_are_exposed_for_fedjob_serialization():
    aggregator = _custom_aggregator()
    assert aggregator.metric_prior_strength == 7.0
    assert aggregator.heterogeneity_threshold == 0.12
    assert aggregator.max_blend_factor == 0.15
    assert aggregator.min_weight == 0.03
    assert aggregator.max_weight == 0.70


def test_fedjob_export_preserves_non_default_aggregator_arguments(tmp_path):
    aggregator = _custom_aggregator()
    recipe = FedOptRecipe(
        name="adaptive-hetero-serialization-test",
        min_clients=2,
        num_rounds=1,
        model=torch.nn.Linear(2, 2),
        train_script=__file__,
        aggregator=aggregator,
        optimizer_args={"path": "torch.optim.SGD", "args": {"lr": 1.0}},
    )
    recipe.export(str(tmp_path))

    config_path = tmp_path / recipe.name / "app" / "config" / "config_fed_server.json"
    server_config = json.loads(config_path.read_text())
    serialized = next(component for component in server_config["components"] if component["id"] == "aggregator")
    assert serialized["args"]["metric_prior_strength"] == 7.0
    assert serialized["args"]["heterogeneity_threshold"] == 0.12
    assert serialized["args"]["max_blend_factor"] == 0.15
    assert serialized["args"]["min_weight"] == 0.03
    assert serialized["args"]["max_weight"] == 0.70


def test_fedopt_recipe_accepts_shareable_adaptive_aggregator():
    aggregator = AdaptiveHeterogeneityAggregator(min_weight=0.10, max_weight=0.80)
    recipe = FedOptRecipe(
        name="adaptive-hetero-fedopt-contract-test",
        min_clients=2,
        num_rounds=1,
        model=torch.nn.Linear(2, 2),
        train_script=__file__,
        aggregator=aggregator,
        optimizer_args={"path": "torch.optim.SGD", "args": {"lr": 1.0}},
    )
    assert recipe.aggregator is aggregator
