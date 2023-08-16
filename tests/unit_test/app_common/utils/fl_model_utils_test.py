# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import pytest

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.app_common.abstract.fl_model import FLModel, FLModelConst, ParamsType
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.utils.fl_model_utils import FLModelUtils

TEST_CASES = [
    ({"hello": 123}, 100, 1),
    ({"cool": 123, "very": 4}, 10, 0),
]

FL_MODEL_TEST_CASES = [
    (FLModel(params={"hello": 123}, params_type=ParamsType.FULL, current_round=0, total_rounds=10), DataKind.WEIGHTS),
    (
        FLModel(params={"hello": 123}, params_type=ParamsType.DIFF, current_round=0, total_rounds=10),
        DataKind.WEIGHT_DIFF,
    ),
    (FLModel(metrics={"loss": 0.79}, current_round=0, total_rounds=10, params_type=None), DataKind.METRICS),
]


class TestFLModelUtils:
    @pytest.mark.parametrize("weights,num_rounds,current_round", TEST_CASES)
    def test_from_shareable(self, weights, num_rounds, current_round):
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=weights, meta={AppConstants.VALIDATE_TYPE: "before_train_validate"})
        shareable = dxo.to_shareable()
        shareable.set_header(AppConstants.NUM_ROUNDS, num_rounds)
        shareable.set_header(AppConstants.CURRENT_ROUND, current_round)
        fl_model = FLModelUtils.from_shareable(shareable=shareable)

        assert fl_model.params == dxo.data
        assert fl_model.params_type == ParamsType.FULL
        assert fl_model.current_round == current_round
        assert fl_model.total_rounds == num_rounds

    @pytest.mark.parametrize("fl_model,expected_data_kind", FL_MODEL_TEST_CASES)
    def test_to_shareable(self, fl_model, expected_data_kind):
        shareable = FLModelUtils.to_shareable(fl_model)
        dxo = from_shareable(shareable)
        assert shareable.get_header(AppConstants.CURRENT_ROUND) == fl_model.current_round
        assert shareable.get_header(AppConstants.NUM_ROUNDS) == fl_model.total_rounds
        assert dxo.data_kind == expected_data_kind
        if expected_data_kind == DataKind.METRICS:
            assert dxo.data == fl_model.metrics
        else:
            assert dxo.data == fl_model.params

    @pytest.mark.parametrize("weights,num_rounds,current_round", TEST_CASES)
    def test_from_to_shareable(self, weights, num_rounds, current_round):
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=weights, meta={AppConstants.VALIDATE_TYPE: "before_train_validate"})
        shareable = dxo.to_shareable()
        shareable.set_header(AppConstants.NUM_ROUNDS, num_rounds)
        shareable.set_header(AppConstants.CURRENT_ROUND, current_round)
        fl_model = FLModelUtils.from_shareable(shareable=shareable)
        result_shareable = FLModelUtils.to_shareable(fl_model)
        assert shareable == result_shareable

    @pytest.mark.parametrize("weights,num_rounds,current_round", TEST_CASES)
    def test_from_dxo(self, weights, num_rounds, current_round):
        dxo = DXO(
            data_kind=DataKind.FL_MODEL,
            data={
                FLModelConst.PARAMS: weights,
                FLModelConst.PARAMS_TYPE: ParamsType.FULL,
                FLModelConst.TOTAL_ROUNDS: num_rounds,
                FLModelConst.CURRENT_ROUND: current_round,
            },
        )
        fl_model = FLModelUtils.from_dxo(dxo)
        assert fl_model.params == weights
        assert fl_model.params_type == ParamsType.FULL
        assert fl_model.current_round == current_round
        assert fl_model.total_rounds == num_rounds

    @pytest.mark.parametrize("weights,num_rounds,current_round", TEST_CASES)
    def test_to_dxo(self, weights, num_rounds, current_round):
        fl_model = FLModel(
            params=weights, params_type=ParamsType.FULL, current_round=current_round, total_rounds=num_rounds
        )
        dxo = FLModelUtils.to_dxo(fl_model)
        assert dxo.data_kind == DataKind.FL_MODEL
        assert dxo.data[FLModelConst.PARAMS] == weights
        assert dxo.data[FLModelConst.PARAMS_TYPE] == ParamsType.FULL
        assert dxo.data[FLModelConst.CURRENT_ROUND] == current_round
        assert dxo.data[FLModelConst.TOTAL_ROUNDS] == num_rounds
