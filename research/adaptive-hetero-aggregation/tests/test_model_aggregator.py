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
from adaptive_hetero.model_aggregator import AdaptiveHeterogeneityModelAggregator
from adaptive_hetero.nvflare_aggregator import AdaptiveMetaKey

from nvflare.apis.dxo import DataKind
from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.client.config import TransferType


def _result(name, value, steps, samples, descriptor, metric, round_number=0):
    return FLModel(
        params={"weight": np.asarray([value], dtype=np.float32)},
        params_type=ParamsType.DIFF,
        metrics={"accuracy": metric},
        current_round=round_number,
        meta={
            "client_name": name,
            FLMetaKey.NUM_STEPS_CURRENT_ROUND: steps,
            AdaptiveMetaKey.SAMPLE_COUNT: samples,
            AdaptiveMetaKey.DISTRIBUTION_DESCRIPTOR: descriptor,
            AdaptiveMetaKey.CLIENT_METRIC: metric,
        },
    )


def test_model_aggregator_uses_same_policy_contract_without_broadcasting_weights():
    aggregator = AdaptiveHeterogeneityModelAggregator(
        metric_prior_strength=0.0,
        heterogeneity_threshold=0.0,
        heterogeneity_deadband=0.0,
        performance_gap_threshold=0.0,
        performance_gap_deadband=0.0,
        max_blend_factor=0.20,
        activation_warmup_rounds=0,
        activation_patience=1,
        require_stable_cohort=False,
        min_weight=0.05,
        max_weight=0.80,
    )
    aggregator.accept_model(_result("site-1", 1.0, 9, 900, [0.95, 0.05], 0.90))
    aggregator.accept_model(_result("site-2", 3.0, 1, 100, [0.05, 0.95], 0.50))
    result = aggregator.aggregate_model()

    assert result.params_type == ParamsType.DIFF
    assert np.isclose(sum(aggregator.last_weights.values()), 1.0)
    expected = aggregator.last_weights["site-1"] + 3.0 * aggregator.last_weights["site-2"]
    assert np.allclose(result.params["weight"], np.asarray([expected], dtype=np.float32), atol=1e-6)
    assert result.meta[AdaptiveMetaKey.BLEND_FACTOR] > 0.0
    assert "adaptive_final_weights" not in result.meta


def test_model_aggregator_native_fallback_is_weighted_by_steps():
    aggregator = AdaptiveHeterogeneityModelAggregator(activation_warmup_rounds=3)
    aggregator.accept_model(_result("site-1", 1.0, 9, 100, [0.5, 0.5], 0.8))
    aggregator.accept_model(_result("site-2", 3.0, 1, 100, [0.5, 0.5], 0.8))
    result = aggregator.aggregate_model()

    assert result.meta[AdaptiveMetaKey.BLEND_FACTOR] == 0.0
    assert aggregator.last_weights["site-1"] == 0.9
    assert aggregator.last_weights["site-2"] == 0.1
    assert np.allclose(result.params["weight"], np.asarray([1.2], dtype=np.float32), atol=1e-6)


def test_unified_fedavg_recipe_serializes_adaptive_model_aggregator(tmp_path):
    aggregator = AdaptiveHeterogeneityModelAggregator(
        metric_prior_strength=7.0,
        heterogeneity_threshold=0.12,
        max_blend_factor=0.15,
        min_weight=0.03,
        max_weight=0.70,
    )
    recipe = FedAvgRecipe(
        name="adaptive-hetero-fedavg-model-aggregator-test",
        min_clients=2,
        num_rounds=1,
        model=torch.nn.Linear(2, 2),
        train_script=__file__,
        aggregator=aggregator,
        aggregator_data_kind=DataKind.WEIGHT_DIFF,
        params_transfer_type=TransferType.DIFF,
    )
    recipe.export(str(tmp_path))

    config_path = tmp_path / recipe.name / "app" / "config" / "config_fed_server.json"
    server_config = json.loads(config_path.read_text())
    controller = server_config["workflows"][0]
    serialized = controller["args"]["aggregator"]
    assert serialized["path"].endswith("AdaptiveHeterogeneityModelAggregator")
    assert serialized["args"]["metric_prior_strength"] == 7.0
    assert serialized["args"]["heterogeneity_threshold"] == 0.12
    assert serialized["args"]["max_blend_factor"] == 0.15
    assert serialized["args"]["min_weight"] == 0.03
    assert serialized["args"]["max_weight"] == 0.70
