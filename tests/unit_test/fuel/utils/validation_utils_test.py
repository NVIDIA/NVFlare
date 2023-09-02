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

from nvflare.fuel.utils.validation_utils import validate_candidate, validate_candidates


class TestValidationUtils:
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
