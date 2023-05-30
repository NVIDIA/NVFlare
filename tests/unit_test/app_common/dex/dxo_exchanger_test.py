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

import tempfile

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.app_common.abstract.fl_model import FLModel, TransferType
from nvflare.app_common.dex.dxo_exchanger import DXOExchanger
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.fuel.utils.pipe.pickle_file_accessor import PickleFileAccessor

TEST_CASES = [
    {"a": 1, "b": 3},
    {},
    {"abc": [1, 2, 3], "d": [4, 5]},
    {"abc": (1, 2, 3), "d": (4, 5)},
    {"hello": b"a string", "cool": 6},
    {f"layer{i}": np.random.rand(256, 256) for i in range(5)},
]


class DXOExchangerTest:
    @pytest.fixture
    def get_file_accessor(self):
        pass

    @pytest.mark.parametrize("weights", TEST_CASES)
    def test_put_get_dxo(self, weights, get_file_accessor):
        data_id = "test_obj"
        dxo = DXO(data=weights, data_kind=DataKind.WEIGHT_DIFF)
        with tempfile.TemporaryDirectory() as root_dir:
            x_dxi = DXOExchanger(pipe_role="x")
            x_dxi.initialize(root_dir, get_file_accessor)
            x_dxi.put(data_id=data_id, data=dxo)
            y_dxi = DXOExchanger(pipe_role="y")
            y_dxi.initialize(root_dir, get_file_accessor)
            result_dxo = y_dxi.get(data_id)
            for k, v in result_dxo.data.items():
                np.testing.assert_array_equal(weights[k], v)
            assert result_dxo.data_kind == dxo.data_kind
            x_dxi.finalize()
            y_dxi.finalize()

    @pytest.mark.parametrize("weights", TEST_CASES)
    def test_put_get_fl_model(self, weights, get_file_accessor):
        data_id = "test_obj"
        fl_model = FLModel(model=weights, transfer_type=TransferType.MODEL)
        dxo = FLModelUtils.to_dxo(fl_model)
        with tempfile.TemporaryDirectory() as root_dir:
            x_dxi = DXOExchanger(pipe_role="x")
            x_dxi.initialize(root_dir, get_file_accessor)
            x_dxi.put(data_id=data_id, data=dxo)
            y_dxi = DXOExchanger(pipe_role="y")
            y_dxi.initialize(root_dir, get_file_accessor)
            result_dxo = y_dxi.get(data_id)
            result = FLModelUtils.from_dxo(result_dxo)
            for k, v in result.model.items():
                np.testing.assert_array_equal(weights[k], v)
            x_dxi.finalize()
            y_dxi.finalize()


class TestPickleDXOExchanger(DXOExchangerTest):
    @pytest.fixture
    def get_file_accessor(self):
        yield PickleFileAccessor()
