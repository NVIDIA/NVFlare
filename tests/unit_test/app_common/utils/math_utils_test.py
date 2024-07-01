# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import operator

import pytest

from nvflare.app_common.utils.math_utils import parse_compare_criteria

TEST_CASES = [
    ("accuracy >= 50", ("accuracy", 50, operator.ge)),
    ("accuracy <= 50", ("accuracy", 50, operator.le)),
    ("accuracy > 50", ("accuracy", 50, operator.gt)),
    ("accuracy < 50", ("accuracy", 50, operator.lt)),
    ("accuracy = 50", ("accuracy", 50, operator.eq)),
    ("loss < 0.1", ("loss", 0.1, operator.lt)),
    ("50 >= 50", ("50", 50, operator.ge)),
]

INVALID_TEST_CASES = [
    ("50 >= accuracy", None),
    ("accuracy >== 50", None),
    ("accuracy >= accuracy", None),
    (50, None),
]


class TestMathUtils:
    @pytest.mark.parametrize("compare_expr,compare_tuple", TEST_CASES + INVALID_TEST_CASES)
    def test_parse_compare_criteria(self, compare_expr, compare_tuple):
        if compare_tuple is None:
            with pytest.raises(Exception):
                result_tuple = parse_compare_criteria(compare_expr)
        else:
            result_tuple = parse_compare_criteria(compare_expr)
            assert result_tuple == compare_tuple
