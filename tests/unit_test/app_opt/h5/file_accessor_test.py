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

import os
import tempfile

import numpy as np
import pytest

from nvflare.app_opt.h5.file_accessor import H5FileAccessor

TEST_CASES = [
    {"a": 1, "b": 3},
    {},
    {"abc": [1, 2, 3], "d": [4, 5]},
    {"abc": (1, 2, 3), "d": (4, 5)},
    {"hello": b"a string", "cool": 6},
    {f"layer{i}": np.random.rand(4096, 4096) for i in range(20)},
]


class TestH5FileAccessor:
    def test_read_write_str(self):
        data = b"hello moto"
        with tempfile.TemporaryDirectory() as root_dir:
            x = H5FileAccessor()
            file_path = os.path.join(root_dir, "test_file")
            x.write(data, file_path)
            result = x.read(file_path)
            assert result == data

    @pytest.mark.parametrize("weights", TEST_CASES)
    def test_read_write(self, weights):
        with tempfile.TemporaryDirectory() as root_dir:
            x = H5FileAccessor()
            file_path = os.path.join(root_dir, "test_file")
            x.write(weights, file_path)
            result = x.read(file_path)
            for k, v in weights.items():
                np.testing.assert_array_equal(weights[k], v)
