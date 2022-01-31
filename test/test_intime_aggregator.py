# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import random

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.shareable import Shareable
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.app_constant import AppConstants


class TestAggregator:
    @pytest.mark.parametrize("aggregator", [InTimeAccumulateWeightedAggregator])
    @pytest.mark.parametrize(
        "received,expected",
        [
            [
                {"client1": {"weight": 0.5, "iter_number": 1, "aggr_data": {"var1": np.array([2.0, 3.0, 1.1, 0.1])}}},
                {"var1": np.array([2.0, 3.0, 1.1, 0.1])},
            ],
            [
                {"client1": {"weight": 1.0, "iter_number": 1, "aggr_data": {"var1": np.array([2.0, 3.0, 1.1, 0.1])}}},
                {"var1": np.array([2.0, 3.0, 1.1, 0.1])},
            ],
            [
                {
                    "client1": {"weight": 0.5, "iter_number": 1, "aggr_data": {"var1": np.array([2.0, 3.0, 1.1, 0.1])}},
                    "client2": {"weight": 1.0, "iter_number": 1, "aggr_data": {"var1": np.array([1.0, 1.0, 2.1, 0.5])}},
                },
                {
                    "var1": np.array(
                        [
                            (0.5 * 2.0 + 1.0 * 1.0) / (0.5 + 1),
                            (0.5 * 3.0 + 1.0 * 1.0) / (0.5 + 1),
                            (0.5 * 1.1 + 1.0 * 2.1) / (0.5 + 1),
                            (0.5 * 0.1 + 1.0 * 0.5) / (0.5 + 1),
                        ]
                    )
                },
            ],
            [
                {
                    "client1": {"weight": 1.0, "iter_number": 2, "aggr_data": {"var1": np.array([2.0, 3.0, 1.1, 0.1])}},
                    "client2": {"weight": 1.0, "iter_number": 4, "aggr_data": {"var1": np.array([1.0, 1.0, 2.1, 0.5])}},
                },
                {
                    "var1": np.array(
                        [
                            (2 * 2.0 + 4 * 1.0) / (2 + 4),
                            (2 * 3.0 + 4 * 1.0) / (2 + 4),
                            (2 * 1.1 + 4 * 2.1) / (2 + 4),
                            (2 * 0.1 + 4 * 0.5) / (2 + 4),
                        ]
                    )
                },
            ],
        ],
    )
    def test_accum_aggregator(self, aggregator, received, expected):
        aggregation_weights = {k: v["weight"] for k, v in received.items()}
        agg = aggregator(aggregation_weights=aggregation_weights)
        fl_ctx_mgr = FLContextManager(engine=None, identity_name="", run_num=1, public_stickers={}, private_stickers={})
        for k, v in received.items():
            fl_ctx_mgr.identity_name = k
            dxo = DXO(
                DataKind.WEIGHT_DIFF,
                data=v["aggr_data"],
                meta={
                    MetaKey.NUM_STEPS_CURRENT_ROUND: v["iter_number"],
                },
            )

            fl_ctx = FLContext()
            fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 0)
            s = Shareable()
            s.set_peer_props({ReservedKey.IDENTITY_NAME: k})
            s.set_header(AppConstants.CONTRIBUTION_ROUND, 0)
            agg.accept(dxo.update_shareable(s), fl_ctx)

        result = agg.aggregate(fl_ctx)
        np.testing.assert_allclose(result["DXO"]["data"]["var1"], expected["var1"])

    @pytest.mark.parametrize("aggregator", [InTimeAccumulateWeightedAggregator])
    @pytest.mark.parametrize("shape", [(4), (6, 6)])
    @pytest.mark.parametrize("n_clients", [10, 50, 100])
    def test_accum_aggregator_random(self, aggregator, shape, n_clients):
        aggregation_weights = {f"client_{i}": random.random() for i in range(n_clients)}
        agg = aggregator(aggregation_weights=aggregation_weights)
        weighted_sum = np.zeros(shape)
        sum_of_weights = 0
        fl_ctx_mgr = FLContextManager(engine=None, identity_name="", run_num=1, public_stickers={}, private_stickers={})
        for client_name in aggregation_weights:
            iter_number = random.randint(1, 50)
            fl_ctx_mgr.identity_name = client_name
            weights = np.random.random(shape)
            fl_ctx = FLContext()
            fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 0)
            s = Shareable()
            s.set_peer_props({ReservedKey.IDENTITY_NAME: client_name})
            s.set_header(AppConstants.CONTRIBUTION_ROUND, 0)
            dxo = DXO(
                DataKind.WEIGHT_DIFF,
                data={"var1": weights},
                meta={
                    MetaKey.NUM_STEPS_CURRENT_ROUND: iter_number,
                },
            )
            weighted_sum = weighted_sum + (weights * iter_number * aggregation_weights[client_name])
            sum_of_weights = sum_of_weights + (iter_number * aggregation_weights[client_name])
            agg.accept(dxo.update_shareable(s), fl_ctx)

        result = agg.aggregate(fl_ctx)
        result_dxo = from_shareable(result)
        np.testing.assert_allclose(result_dxo.data["var1"], weighted_sum / sum_of_weights)

    @pytest.mark.parametrize("aggregator", [InTimeAccumulateWeightedAggregator])
    def test_accum_aggregator_accept(self, aggregator):
        aggregation_weights = {f"client_{i}": random.random() for i in range(2)}
        agg = aggregator(aggregation_weights=aggregation_weights)
        client_name = "client_0"
        iter_number = 1
        fl_ctx_mgr = FLContextManager(engine=None, identity_name="", run_num=1, public_stickers={}, private_stickers={})
        fl_ctx_mgr.identity_name = client_name
        weights = np.random.random((4))

        fl_ctx = FLContext()
        s = Shareable()
        s.set_peer_props({ReservedKey.IDENTITY_NAME: client_name})
        s.set_header(AppConstants.CONTRIBUTION_ROUND, 1)
        fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 1)
        dxo = DXO(
            DataKind.WEIGHT_DIFF,
            data={"var1": weights},
            meta={
                MetaKey.NUM_STEPS_CURRENT_ROUND: iter_number,
            },
        )
        assert (True) == agg.accept(dxo.update_shareable(s), fl_ctx)

        client_name = "client_0"
        iter_number = 1
        fl_ctx_mgr.identity_name = client_name
        weights = np.random.random((4))
        dxo = DXO(
            DataKind.WEIGHT_DIFF,
            data={"var1": weights},
            meta={
                MetaKey.NUM_STEPS_CURRENT_ROUND: iter_number,
            },
        )
        fl_ctx = FLContext()
        s = Shareable()
        s.set_peer_props({ReservedKey.IDENTITY_NAME: client_name})
        s.set_header(AppConstants.CONTRIBUTION_ROUND, 1)
        fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 2)
        assert (False) == agg.accept(dxo.update_shareable(s), fl_ctx)
