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

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.dex.dxo_file_exchanger import DXOFileExchanger
from nvflare.app_common.dex.fl_model_exchanger import FLModelExchanger
from nvflare.fuel.utils.pipe.pickle_file_accessor import PickleFileAccessor

TEST_CASES = [
    {"a": 1, "b": 3},
    {},
    {"abc": [1, 2, 3], "d": [4, 5]},
    {"abc": (1, 2, 3), "d": (4, 5)},
    {"hello": b"a string", "cool": 6},
    {f"layer{i}": np.random.rand(256, 256) for i in range(5)},
]


class TestFLModelExchanger:
    @pytest.mark.parametrize("weights", TEST_CASES)
    def test_put_get_fl_model_with_dxo_file_exchanger(self, weights):
        fl_model = FLModel(params=weights, params_type=ParamsType.FULL)

        with tempfile.TemporaryDirectory() as root_dir:
            x_ex = DXOFileExchanger(pipe_role="x")
            x_ex.initialize(data_exchange_path=root_dir, file_accessor=PickleFileAccessor())
            x_mdx = FLModelExchanger(exchanger=x_ex)
            _, put_msg_id = x_mdx.send_request(model=fl_model)

            y_ex = DXOFileExchanger(pipe_role="y")
            y_ex.initialize(data_exchange_path=root_dir, file_accessor=PickleFileAccessor())
            y_mdx = FLModelExchanger(exchanger=y_ex)
            result_model, get_msg_id = y_mdx.receive_request()
            assert put_msg_id == get_msg_id
            for k, v in result_model.params.items():
                np.testing.assert_array_equal(weights[k], v)

            y_mdx.send_reply(result_model, get_msg_id)
            receive_reply_model = x_mdx.receive_reply(get_msg_id)
            for k, v in receive_reply_model.params.items():
                np.testing.assert_array_equal(receive_reply_model.params[k], result_model.params[k])

            x_ex.finalize()
            y_ex.finalize()
