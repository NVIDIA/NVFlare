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
from nvflare.apis.utils.decomposers import flare_decomposers
from nvflare.app_common.decomposers import common_decomposers
from nvflare.app_common.dex.dxo_file_exchanger import DXOFileExchanger
from nvflare.fuel.utils.pipe.fobs_file_accessor import FobsFileAccessor

TEST_CASES = [
    {"a": 1, "b": 3},
    {},
    {"abc": [1, 2, 3], "d": [4, 5]},
    {"abc": (1, 2, 3), "d": (4, 5)},
    {"hello": b"a string", "cool": 6},
    {f"layer{i}": np.random.rand(256, 256) for i in range(5)},
]


class DXOFileExchangerTest:
    @pytest.fixture
    def get_file_accessor(self):
        pass

    @pytest.mark.parametrize("weights", TEST_CASES)
    def test_put_get_dxo(self, weights, get_file_accessor):
        dxo = DXO(data=weights, data_kind=DataKind.WEIGHT_DIFF)
        with tempfile.TemporaryDirectory() as root_dir:
            x_dxi = DXOFileExchanger(pipe_role="x")
            x_dxi.initialize(data_exchange_path=root_dir, file_accessor=get_file_accessor)
            _, put_msg_id = x_dxi.send_request(dxo=dxo)
            y_dxi = DXOFileExchanger(pipe_role="y")
            y_dxi.initialize(data_exchange_path=root_dir, file_accessor=get_file_accessor)
            result_dxo, get_msg_id = y_dxi.receive_request()
            assert put_msg_id == get_msg_id
            for k, v in result_dxo.data.items():
                np.testing.assert_array_equal(weights[k], v)
            assert result_dxo.data_kind == dxo.data_kind
            x_dxi.finalize()
            y_dxi.finalize()


class TestFobsDXOFileExchanger(DXOFileExchangerTest):
    @pytest.fixture
    def get_file_accessor(self):
        flare_decomposers.register()
        common_decomposers.register()
        yield FobsFileAccessor()
