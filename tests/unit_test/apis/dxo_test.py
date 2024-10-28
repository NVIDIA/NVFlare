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

from nvflare.apis.dxo import DXO, DataKind, get_leaf_dxos

TEST_INIT_1 = [DataKind.WEIGHTS, {"data": 1.0}]
TEST_INIT_2 = [DataKind.WEIGHT_DIFF, {"data": 1.0}]
TEST_INIT_3 = [DataKind.METRICS, {"data": 1.0}]
TEST_INIT_4 = [DataKind.STATISTICS, {"data": 1.0}]
TEST_INIT_5 = [
    DataKind.COLLECTION,
    {"dxo1": DXO(DataKind.WEIGHTS, {"data": 1.0}), "dxo2": DXO(DataKind.WEIGHTS, {"data": 2.0})},
]

TEST_INIT_ERROR_1 = [DataKind.WEIGHTS, 1.0]

dxo1 = DXO(DataKind.WEIGHTS, {"data": 1.0})
dxo2 = DXO(DataKind.WEIGHTS, {"data": 2.0})
dxo11 = DXO(DataKind.WEIGHTS, {"data": 3.0})
dxo22 = DXO(DataKind.WEIGHTS, {"data": 4.0})
dxo3 = DXO(DataKind.COLLECTION, {"dxo11": dxo11, "dxo22": dxo22})

TEST_GET_LEAVES_1 = [DataKind.COLLECTION, {"dxo1": dxo1, "dxo2": dxo2}, {"dxo1": dxo1, "dxo2": dxo2}]
TEST_GET_LEAVES_2 = [
    DataKind.COLLECTION,
    {"dxo1": dxo1, "dxo2": dxo2, "dxo3": dxo3},
    {"dxo1": dxo1, "dxo2": dxo2, "dxo3.dxo11": dxo11, "dxo3.dxo22": dxo22},
]


class TestDXO:
    @pytest.mark.parametrize("data_kind, data", [TEST_INIT_1, TEST_INIT_2, TEST_INIT_3, TEST_INIT_4, TEST_INIT_5])
    def test_init(self, data_kind, data):
        dxo = DXO(data_kind=data_kind, data=data)
        # why return empty string as valid ? should be a boolean
        assert dxo.validate() == ""

    @pytest.mark.parametrize("data_kind, data", [TEST_INIT_ERROR_1])
    def test_init_no_dict(self, data_kind, data):
        with pytest.raises(ValueError):
            dxo = DXO(data_kind=data_kind, data=data)
            dxo.validate()

    @pytest.mark.parametrize("data_kind, data, expected", [TEST_GET_LEAVES_2])
    def test_get_leaf_dxos(self, data_kind, data, expected):
        dxo = DXO(data_kind=data_kind, data=data)
        assert dxo.validate() == ""

        result, errors = get_leaf_dxos(dxo, root_name="test")
        assert len(result) == len(expected)
        for _exp_key, _exp_dxo in expected.items():
            _leaf_key = "test." + _exp_key
            assert _leaf_key in result
            assert result.get(_leaf_key).data == _exp_dxo.data
            assert result.get(_leaf_key).data_kind == _exp_dxo.data_kind

        assert not errors
