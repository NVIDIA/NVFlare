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

from nvflare.fuel.utils.validation_utils import (
    check_non_empty_str,
    check_number_range,
    check_positive_int,
    check_positive_number,
    validate_candidate,
    validate_candidates,
)


class TestValidationUtils:
    @pytest.mark.parametrize(
        "name, num, min_value, max_value",
        [
            ("x", 12.34, None, None),
            ("x", 0.23, -1.0, None),
            ("x", 0, None, 1.0),
            ("x", 0, -0.01, 0.1),
        ],
    )
    def test_check_number_range(self, name, num, min_value, max_value):
        check_number_range(name, num, min_value, max_value)

    @pytest.mark.parametrize(
        "name, num, min_value, max_value",
        [
            ("x", -1.0, 0.0, None),
            ("x", "0", None, None),
            ("x", 2.0, 0.1, 1.0),
            ("x", -5, -10, -6),
            ("x", 0, "-1", None),
            ("x", 0, -1, "-2"),
        ],
    )
    def test_check_number_range_error(self, name, num, min_value, max_value):
        with pytest.raises(Exception):
            check_number_range(name, num, min_value, max_value)

    @pytest.mark.parametrize(
        "name, num",
        [
            ("x", 1),
            ("x", 100),
            ("x", 12345),
        ],
    )
    def test_check_positive_int(self, name, num):
        check_positive_int(name, num)

    @pytest.mark.parametrize(
        "name, num",
        [
            ("x", 0),
            ("x", -1.0),
            ("x", "0"),
            ("x", 2.0),
            ("x", -5),
        ],
    )
    def test_check_positive_int_error(self, name, num):
        with pytest.raises(Exception):
            check_positive_int(name, num)

    @pytest.mark.parametrize(
        "name, num",
        [
            ("x", 1),
            ("x", 100),
            ("x", 12345),
            ("x", 0.001),
            ("x", 1.3e5),
        ],
    )
    def test_check_positive_number(self, name, num):
        check_positive_number(name, num)

    @pytest.mark.parametrize(
        "name, num",
        [("x", 0), ("x", 0.0), ("x", -1.0), ("x", "0"), ("x", -5), ("x", -1.3e5)],
    )
    def test_check_positive_number_error(self, name, num):
        with pytest.raises(Exception):
            check_positive_number(name, num)

    @pytest.mark.parametrize(
        "name, value",
        [
            ("x", "abc"),
            ("x", "  vsd "),
        ],
    )
    def test_check_non_empty_str(self, name, value):
        check_non_empty_str(name, value)

    @pytest.mark.parametrize(
        "name, num",
        [
            ("x", 0),
            ("x", 1.2324),
            ("x", ""),
            ("x", "  "),
        ],
    )
    def test_check_non_empty_str_error(self, name, num):
        with pytest.raises(Exception):
            check_non_empty_str(name, num)

    @pytest.mark.parametrize(
        "var_name, candidate, base, default_policy, allow_none, output",
        [
            ("x", "red", ["red", "blue"], "any", True, "red"),
            ("x", " red  ", ["red", "blue"], "any", True, "red"),
            ("x", "", ["red", "blue"], "any", True, "red"),
            ("x", "", ["red", "blue"], "empty", True, ""),
            ("x", None, ["red", "blue"], "any", True, ""),
            ("x", "@none", ["red", "blue"], "any", True, ""),
        ],
    )
    def test_validate_candidate(self, var_name, candidate, base, default_policy, allow_none, output):
        assert validate_candidate(var_name, candidate, base, default_policy, allow_none) == output

    @pytest.mark.parametrize(
        "var_name, candidate, base, default_policy, allow_none",
        [
            ("x", "red", ["red", "blue"], "bad", True),
            ("x", 2, ["red", "blue"], "any", True),
            ("x", "", ["red", "blue"], "disallow", True),
            ("x", "", ["red", "blue"], "all", True),
            ("x", "yellow", ["red", "blue"], "any", True),
            ("x", None, ["red", "blue"], "any", False),
            ("x", "@none", ["red", "blue"], "any", False),
            ("x", "@all", ["red", "blue"], "any", False),
        ],
    )
    def test_validate_candidate_error(self, var_name, candidate, base, default_policy, allow_none):
        with pytest.raises(ValueError):
            validate_candidate(var_name, candidate, base, default_policy, allow_none)

    @pytest.mark.parametrize(
        "var_name, candidates, base, default_policy, allow_none, output",
        [
            ("x", "red", ["red", "blue"], "any", True, ["red"]),
            ("x", [" red ", "blue", "red"], ["red", "blue", "green"], "any", True, ["red", "blue"]),
            ("x", "", ["red", "blue"], "any", True, ["red"]),
            ("x", "", ["red", "blue"], "all", True, ["red", "blue"]),
            ("x", "", ["red", "blue"], "empty", True, []),
            ("x", "red", ["red", "blue"], "any", True, ["red"]),
            ("x", [], ["red", "blue"], "any", True, ["red"]),
            ("x", [], ["red", "blue"], "empty", True, []),
            ("x", [], ["red", "blue"], "all", True, ["red", "blue"]),
            ("x", None, ["red", "blue"], "any", True, []),
            ("x", "@all", ["red", "blue"], "any", True, ["red", "blue"]),
            ("x", "@none", ["red", "blue"], "any", True, []),
        ],
    )
    def test_validate_candidates(self, var_name, candidates, base, default_policy, allow_none, output):
        assert validate_candidates(var_name, candidates, base, default_policy, allow_none) == output

    @pytest.mark.parametrize(
        "var_name, candidate, base, default_policy, allow_none",
        [
            ("x", "red", ["red", "blue"], "bad", True),
            ("x", 2, ["red", "blue"], "any", True),
            ("x", "", ["red", "blue"], "disallow", True),
            ("x", [], ["red", "blue"], "disallow", True),
            ("x", "yellow", ["red", "blue"], "any", True),
            ("x", None, ["red", "blue"], "any", False),
            ("x", "@none", ["red", "blue"], "any", False),
        ],
    )
    def test_validate_candidates_error(self, var_name, candidate, base, default_policy, allow_none):
        with pytest.raises(ValueError):
            validate_candidates(var_name, candidate, base, default_policy, allow_none)
