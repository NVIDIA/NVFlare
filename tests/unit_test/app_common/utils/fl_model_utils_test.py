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

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.app_common.abstract.fl_model import FLModel, FLModelConst, TransferType
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.utils.fl_model_utils import FLModelUtils


class TestFLModelUtils:
    def test_from_shareable(self):
        weights = {"hello": 123}
        num_rounds = 100
        current_round = 1
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=weights)
        shareable = dxo.to_shareable()
        shareable.set_header(AppConstants.NUM_ROUNDS, num_rounds)
        shareable.set_header(AppConstants.CURRENT_ROUND, current_round)
        fl_model = FLModelUtils.from_shareable(shareable=shareable)

        assert fl_model.model == weights
        assert fl_model.transfer_type == TransferType.MODEL
        assert fl_model.round == current_round
        assert fl_model.total_rounds == num_rounds

    def test_to_shareable(self):
        weights = {"hello": 123}
        num_rounds = 100
        current_round = 1
        fl_model = FLModel(
            model=weights, transfer_type=TransferType.MODEL, round=current_round, total_rounds=num_rounds
        )
        shareable = FLModelUtils.to_shareable(fl_model)
        dxo = from_shareable(shareable)
        assert shareable.get_header(AppConstants.CURRENT_ROUND) == current_round
        assert shareable.get_header(AppConstants.NUM_ROUNDS) == num_rounds
        assert dxo.data == weights
        assert dxo.data_kind == DataKind.WEIGHTS

    def test_from_dxo(self):
        weights = {"hello": 123}
        num_rounds = 100
        current_round = 1
        dxo = DXO(
            data_kind=DataKind.FL_MODEL,
            data={
                FLModelConst.MODEL: weights,
                FLModelConst.TRANSFER_TYPE: TransferType.MODEL,
                FLModelConst.TOTAL_ROUNDS: num_rounds,
                FLModelConst.ROUND: current_round,
            },
        )
        fl_model = FLModelUtils.from_dxo(dxo)
        assert fl_model.model == weights
        assert fl_model.transfer_type == TransferType.MODEL
        assert fl_model.round == current_round
        assert fl_model.total_rounds == num_rounds

    def test_to_dxo(self):
        weights = {"hello": 123}
        num_rounds = 100
        current_round = 1
        fl_model = FLModel(
            model=weights, transfer_type=TransferType.MODEL, round=current_round, total_rounds=num_rounds
        )
        dxo = FLModelUtils.to_dxo(fl_model)
        assert dxo.data_kind == DataKind.FL_MODEL
        assert dxo.data[FLModelConst.MODEL] == weights
        assert dxo.data[FLModelConst.TRANSFER_TYPE] == TransferType.MODEL
        assert dxo.data[FLModelConst.ROUND] == current_round
        assert dxo.data[FLModelConst.TOTAL_ROUNDS] == num_rounds
