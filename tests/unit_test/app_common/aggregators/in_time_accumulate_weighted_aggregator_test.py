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
import re

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.app_constant import AppConstants


class TestInTimeAccumulateWeightedAggregator:
    @pytest.mark.parametrize(
        "exclude_vars,aggregation_weights,expected_data_kind,error,error_msg",
        [
            (
                2.0,
                None,
                DataKind.WEIGHT_DIFF,
                ValueError,
                f"exclude_vars = 2.0 should be a regex string but got {type(2.0)}.",
            ),
            (
                {"dxo1": 3.0, "dxo2": ""},
                None,
                {"dxo1": DataKind.WEIGHT_DIFF, "dxo2": DataKind.WEIGHT_DIFF},
                ValueError,
                f"exclude_vars[dxo1] = 3.0 should be a regex string but got {type(3.0)}.",
            ),
            (None, None, DataKind.ANALYTIC, ValueError, "expected_data_kind = ANALYTIC is not WEIGHT_DIFF or WEIGHTS"),
            (
                None,
                None,
                {"dxo1": DataKind.WEIGHT_DIFF, "dxo2": DataKind.ANALYTIC},
                ValueError,
                "expected_data_kind[dxo2] = ANALYTIC is not WEIGHT_DIFF or WEIGHTS",
            ),
            (
                None,
                {"dxo1": {"client_0": 1.0, "client_1": 2.0}},
                {"dxo1": DataKind.WEIGHT_DIFF, "dxo2": DataKind.WEIGHT_DIFF},
                ValueError,
                "A dict of dict aggregation_weights should specify aggregation_weights "
                "for every key in expected_data_kind. But missed these keys: ['dxo2']",
            ),
            (
                {"dxo2": ""},
                None,
                {"dxo1": DataKind.WEIGHT_DIFF, "dxo2": DataKind.WEIGHT_DIFF},
                ValueError,
                "A dict exclude_vars should specify exclude_vars for every key in expected_data_kind. "
                "But missed these keys: ['dxo1']",
            ),
        ],
    )
    def test_invalid_create(self, exclude_vars, aggregation_weights, expected_data_kind, error, error_msg):
        with pytest.raises(error, match=re.escape(error_msg)):
            _ = InTimeAccumulateWeightedAggregator(
                exclude_vars=exclude_vars,
                aggregation_weights=aggregation_weights,
                expected_data_kind=expected_data_kind,
            )

    @pytest.mark.parametrize(
        "exclude_vars,aggregation_weights,expected_data_kind,expected_object",
        [
            (
                None,
                None,
                DataKind.WEIGHTS,
                InTimeAccumulateWeightedAggregator(
                    exclude_vars=None, aggregation_weights=None, expected_data_kind=DataKind.WEIGHTS
                ),
            ),
            (
                "hello",
                None,
                {"dxo1": DataKind.WEIGHTS, "dxo2": DataKind.WEIGHT_DIFF},
                InTimeAccumulateWeightedAggregator(
                    exclude_vars={"dxo1": "hello", "dxo2": "hello"},
                    aggregation_weights=None,
                    expected_data_kind={"dxo1": DataKind.WEIGHTS, "dxo2": DataKind.WEIGHT_DIFF},
                ),
            ),
            (
                None,
                {"client_0": 1.0, "client_1": 2.0},
                {"dxo1": DataKind.WEIGHTS, "dxo2": DataKind.WEIGHT_DIFF},
                InTimeAccumulateWeightedAggregator(
                    exclude_vars=None,
                    aggregation_weights={
                        "dxo1": {"client_0": 1.0, "client_1": 2.0},
                        "dxo2": {"client_0": 1.0, "client_1": 2.0},
                    },
                    expected_data_kind={"dxo1": DataKind.WEIGHTS, "dxo2": DataKind.WEIGHT_DIFF},
                ),
            ),
        ],
    )
    def test_create(self, exclude_vars, aggregation_weights, expected_data_kind, expected_object):
        result = InTimeAccumulateWeightedAggregator(
            exclude_vars=exclude_vars, aggregation_weights=aggregation_weights, expected_data_kind=expected_data_kind
        )
        assert result.exclude_vars == expected_object.exclude_vars
        assert result.aggregation_weights == expected_object.aggregation_weights
        assert result.expected_data_kind == expected_object.expected_data_kind

    @pytest.mark.parametrize("current_round,contribution_round,expected", [(1, 1, True), (2, 1, False)])
    def test_accept(self, current_round, contribution_round, expected):
        aggregation_weights = {f"client_{i}": random.random() for i in range(2)}
        agg = InTimeAccumulateWeightedAggregator(aggregation_weights=aggregation_weights)
        client_name = "client_0"
        iter_number = 1
        weights = np.random.random(4)

        fl_ctx = FLContext()
        s = Shareable()
        s.set_peer_props({ReservedKey.IDENTITY_NAME: client_name})
        s.set_header(AppConstants.CONTRIBUTION_ROUND, contribution_round)
        fl_ctx.set_prop(AppConstants.CURRENT_ROUND, current_round)
        dxo = DXO(
            DataKind.WEIGHT_DIFF,
            data={"var1": weights},
            meta={
                MetaKey.NUM_STEPS_CURRENT_ROUND: iter_number,
            },
        )
        assert agg.accept(dxo.update_shareable(s), fl_ctx) == expected

    @pytest.mark.parametrize(
        "received,expected",
        [
            (
                {"client1": {"weight": 0.5, "iter_number": 1, "aggr_data": {"var1": np.array([2.0, 3.0, 1.1, 0.1])}}},
                {"var1": np.array([2.0, 3.0, 1.1, 0.1])},
            ),
            (
                {"client1": {"weight": 1.0, "iter_number": 1, "aggr_data": {"var1": np.array([2.0, 3.0, 1.1, 0.1])}}},
                {"var1": np.array([2.0, 3.0, 1.1, 0.1])},
            ),
            (
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
            ),
            (
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
            ),
        ],
    )
    def test_aggregate(self, received, expected):
        aggregation_weights = {k: v["weight"] for k, v in received.items()}
        agg = InTimeAccumulateWeightedAggregator(aggregation_weights=aggregation_weights)
        fl_ctx = FLContext()
        fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 0)
        for k, v in received.items():
            dxo = DXO(
                DataKind.WEIGHT_DIFF,
                data=v["aggr_data"],
                meta={
                    MetaKey.NUM_STEPS_CURRENT_ROUND: v["iter_number"],
                },
            )

            s = Shareable()
            s.set_peer_props({ReservedKey.IDENTITY_NAME: k})
            s.set_header(AppConstants.CONTRIBUTION_ROUND, 0)
            agg.accept(dxo.update_shareable(s), fl_ctx)

        result = agg.aggregate(fl_ctx)
        np.testing.assert_allclose(result["DXO"]["data"]["var1"], expected["var1"])

    @pytest.mark.parametrize("shape", [4, (6, 6)])
    @pytest.mark.parametrize("n_clients", [10, 50, 100])
    def test_aggregate_random(self, shape, n_clients):
        aggregation_weights = {f"client_{i}": random.random() for i in range(n_clients)}
        agg = InTimeAccumulateWeightedAggregator(aggregation_weights=aggregation_weights)
        weighted_sum = np.zeros(shape)
        sum_of_weights = 0
        fl_ctx = FLContext()
        fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 0)
        for client_name in aggregation_weights:
            iter_number = random.randint(1, 50)
            weights = np.random.random(shape)
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

    @pytest.mark.parametrize("num_dxo", [1, 2, 3])
    @pytest.mark.parametrize("shape", [4, (6, 6)])
    @pytest.mark.parametrize("n_clients", [10, 50, 100])
    def test_aggregate_random_dxos(self, num_dxo, shape, n_clients):
        dxo_names = [f"dxo_{i}" for i in range(num_dxo)]
        client_names = [f"client_{i}" for i in range(n_clients)]
        aggregation_weights = {
            dxo_name: {client_name: random.random() for client_name in client_names} for dxo_name in dxo_names
        }
        agg = InTimeAccumulateWeightedAggregator(
            aggregation_weights=aggregation_weights,
            expected_data_kind={dxo_name: DataKind.WEIGHT_DIFF for dxo_name in dxo_names},
        )
        weighted_sum = {dxo_name: np.zeros(shape) for dxo_name in dxo_names}
        sum_of_weights = {dxo_name: 0 for dxo_name in dxo_names}
        fl_ctx = FLContext()
        fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 0)
        for client_name in client_names:
            iter_number = random.randint(1, 50)

            dxo_collection_data = {}
            for dxo_name in dxo_names:
                values = np.random.random(shape)
                dxo = DXO(
                    data_kind=DataKind.WEIGHT_DIFF,
                    data={"var1": values},
                    meta={
                        MetaKey.NUM_STEPS_CURRENT_ROUND: iter_number,
                    },
                )
                dxo_collection_data[dxo_name] = dxo
                weighted_sum[dxo_name] = (
                    weighted_sum[dxo_name] + values * iter_number * aggregation_weights[dxo_name][client_name]
                )
                sum_of_weights[dxo_name] = (
                    sum_of_weights[dxo_name] + iter_number * aggregation_weights[dxo_name][client_name]
                )

            dxo_collection = DXO(data_kind=DataKind.COLLECTION, data=dxo_collection_data)
            s = Shareable()
            s.set_peer_props({ReservedKey.IDENTITY_NAME: client_name})
            s.set_header(AppConstants.CONTRIBUTION_ROUND, 0)
            agg.accept(dxo_collection.update_shareable(s), fl_ctx)

        result = agg.aggregate(fl_ctx)
        result_dxo = from_shareable(result)
        for dxo_name in dxo_names:
            np.testing.assert_allclose(
                result_dxo.data[dxo_name].data["var1"], weighted_sum[dxo_name] / sum_of_weights[dxo_name]
            )
