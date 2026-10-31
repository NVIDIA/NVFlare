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

"""Tests of native FLARE integration and the preserved local training semantics."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper

from src.configs.schemas import StrategyConfig
from src.support.training import train_steps
from src.workflows import VLAFedAvg


def test_sample_weighting_uses_native_aggregation_in_fp32():
    ctrl = VLAFedAvg(persistor_id="")
    ctrl._aggr_helper = WeightedAggregationHelper()
    ctrl._aggr_metrics_helper = WeightedAggregationHelper()
    ctrl._all_metrics = False
    ctrl._received_count, ctrl._expected_count, ctrl.current_round = 0, 2, 0
    ctrl._params_type = None
    ctrl.client_losses, ctrl.client_samples = {}, {}
    ctrl.info = lambda *a: None
    # Large frame counts would overflow fp16 arithmetic if upcast came too late.
    for cid, d, n in (("c0", 1, 100000), ("c1", 3, 300000)):
        ctrl._aggregate_one_result(
            FLModel(
                params={"w": np.array([d], dtype=np.float16)},
                params_type=ParamsType.DIFF,
                meta={"client_name": cid, FLMetaKey.NUM_STEPS_CURRENT_ROUND: n},
            )
        )
    np.testing.assert_allclose(ctrl._aggr_helper.get_result()["w"], [2.5])


@pytest.mark.parametrize("normalize", [True, False])
def test_fedprox_matches_original_live_parameter_gradient(normalize):
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.data.fill_(1)
    backend = SimpleNamespace(model=model, compute_loss=lambda batch: {"loss": model.weight.sum() * 0})
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    strategy = StrategyConfig(name="fedprox", mu=0.5, extra={"normalize": normalize})
    loss, samples = train_steps(
        backend,
        iter([{"x": torch.zeros(3, 2)}]),
        optimizer,
        None,
        {"weight": torch.zeros_like(model.weight)},
        1,
        strategy,
    )
    expected = 0.75 if normalize else 0.5
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, expected))
    assert loss == (0.25 if normalize else 0.5)
    assert samples == 3


@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
def test_native_broadcast_preserves_wire_snapshot_quorum_timeout_and_grace(dtype):
    ctrl = VLAFedAvg(persistor_id="")
    ctrl.cfg = SimpleNamespace(
        fl=SimpleNamespace(
            wire_dtype=dtype,
            min_clients=1,
            round_timeout_s=2.2,
            num_rounds=5,
            nvflare=SimpleNamespace(wait_time_after_min_received=3),
        )
    )
    received = {}
    ctrl.broadcast_model = lambda **kw: received.update(kw)
    master = np.array([1.25], dtype=np.float32)
    ctrl.send_model(data=FLModel(params={"w": master}, current_round=0), targets=["c0", "c1"], callback=lambda m: None)
    assert received["min_responses"] == 1
    assert received["timeout"] == 3
    assert received["wait_time_after_min_received"] == 3
    assert received["blocking"] is False
    assert received["data"].params["w"].dtype == (np.float16 if dtype == "fp16" else np.float32)
    master[0] = 9
    assert received["data"].params["w"][0] == 1.25
