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

from nvflare.app_opt.h5.data_exchanger import H5DataExchanger

TEST_CASES = [
    {"a": 1, "b": 3},
    {},
    {"abc": [1, 2, 3], "d": [4, 5]},
    {"abc": (1, 2, 3), "d": (4, 5)},
    {"hello": b"a string", "cool": 6},
    {f"layer{i}": np.random.rand(4096, 4096) for i in range(20)},
]


class TestH5DataExchanger:
    @pytest.mark.parametrize("weights", TEST_CASES)
    def test_put_get(self, weights):
        data_id = "test_obj"
        with tempfile.TemporaryDirectory() as root_dir:
            x_dxi = H5DataExchanger(pipe_role="x")
            x_dxi.initialize(root_dir)
            x_dxi.put(data_id=data_id, data=weights)
            y_dxi = H5DataExchanger(pipe_role="y")
            y_dxi.initialize(root_dir)
            result = y_dxi.get(data_id)
            for k, v in weights.items():
                np.testing.assert_array_equal(result[k], v)
            x_dxi.finalize()
            y_dxi.finalize()
