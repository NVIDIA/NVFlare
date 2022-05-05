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

import pytest

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.filters import ExcludeVars

TEST_CASES = [
    ({"a": 1.0, "b": 2.0}, "a", {"b": 2.0}),
    ({"a": 1.0, "b": 2.0, "c": 3.0}, ["a", "b"], {"c": 3.0}),
    ({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}, ["a", "d"], {"b": 2.0, "c": 3.0}),
    (
        {"conv/a": 1.0, "conv/b": 2.0, "drop/c": 3.0, "conv/d": 4.0},
        ["conv/*"],
        {"conv/a": 1.0, "conv/b": 2.0, "drop/c": 3.0, "conv/d": 4.0},
    ),
    ({"conv/a": 1.0, "conv/b": 2.0, "drop/c": 3.0, "conv/d": 4.0}, "conv/*", {"drop/c": 3.0}),
]


class TestExcludeVars:
    @pytest.mark.parametrize("input_data,exclude_vars,expected_data", TEST_CASES)
    def test_exclude(self, input_data, exclude_vars, expected_data):
        dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data=input_data,
        )
        data = dxo.to_shareable()
        fl_ctx = FLContext()
        f = ExcludeVars(exclude_vars=exclude_vars)
        new_data = f.process(data, fl_ctx)
        new_dxo = from_shareable(new_data)
        assert new_dxo.data == expected_data
