# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
import re

import pytest

from nvflare.fuel.common.fqn import FQN
from nvflare.fuel.f3.cellnet.fqcn import FQCN, FqcnInfo


class TestFQN:
    @pytest.mark.parametrize(
        "fqn",
        [
            "server",
            "site-1",
            "server.job_abc",
            "site-1.job_abc.worker",
        ],
    )
    def test_validate_accepts_well_formed_names(self, fqn):
        assert FQN.validate(fqn) == ""

    @pytest.mark.parametrize(
        "fqn, err",
        [
            ["", "empty"],
            ["a\tb", "invalid char"],
            ["a b", "invalid char"],
            ["a..b", "empty part"],
            ["a.b.a", "dup 'a'"],
        ],
    )
    def test_validate_rejects_bad_names(self, fqn, err):
        assert FQN.validate(fqn) == err

    @pytest.mark.parametrize(
        "fqn",
        [
            " server",
            "server ",
            " server ",
            "\tserver",
            "server\n",
            " site-1.job_abc ",
        ],
    )
    def test_validate_rejects_names_it_would_have_to_strip(self, fqn):
        # validate reports on the name the caller holds, so a name that only passes
        # after stripping is not valid. Callers that accept padding normalize first.
        assert FQN.validate(fqn) != ""

    @pytest.mark.parametrize(
        "fqn",
        [
            " server",
            "server ",
            " site-1.job_abc ",
            "a b",
            "a\tb",
        ],
    )
    def test_validate_agrees_with_valid_pattern(self, fqn):
        # Anything validate accepts must match the pattern validate advertises.
        assert bool(re.match(FQN.VALID_PATTERN, fqn)) == (FQN.validate(fqn) == "")

    @pytest.mark.parametrize(
        "fqn",
        [
            " server",
            "server ",
            " site-1.job_abc ",
        ],
    )
    def test_normalize_then_validate_accepts_padded_names(self, fqn):
        assert FQN.validate(FQN.normalize(fqn)) == ""


class TestFQCN:
    @pytest.mark.parametrize(
        "fqcn",
        [
            " server",
            "server ",
            " site-1.job_abc ",
        ],
    )
    def test_validate_rejects_padded_names(self, fqcn):
        assert FQCN.validate(fqcn) != ""

    def test_padded_name_describes_a_different_cell(self):
        # The two spellings disagree about the most important fact in the path,
        # which is why a name that needs stripping must not validate.
        assert FqcnInfo("server").is_on_server is True
        assert FqcnInfo(" server").is_on_server is False
