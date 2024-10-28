# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.app_common.statistics.min_count_cleanser import MinCountCleanser

MIN_COUNT_VALIDATION_TEST_CASES = [
    # statistics with age count = 6, failure count = 0 , min_count = 7, count < min_count (6 - 0 < 7), valid = False
    ({"count": {"train": {"age": 6}}, "failure_count": {"train": {"age": 0}}}, 7, {"train": {"age": False}}),
    # statistics with age count=12, failure count=3, min_count=7, effective count > min_count (12 - 3 < 7), valid = True
    ({"count": {"train": {"age": 12}}, "failure_count": {"train": {"age": 3}}}, 7, {"train": {"age": True}}),
    # statistics with age count=9, failure count=3, min_count=7, effective count > min_count (9 - 3 < 7), valid = False
    ({"count": {"train": {"age": 9}}, "failure_count": {"train": {"age": 3}}}, 7, {"train": {"age": False}}),
    # statistics with age count = 8, failure count = 0 , min_count = 7, count > min_count (8 - 0 > 7), valid = True
    ({"count": {"train": {"age": 8}}, "failure_count": {"train": {"age": 0}}}, 7, {"train": {"age": True}}),
    # statistics with age count = 8, edu count = 5,  failure count = 0 , min_count = 7,
    # age count > min_count (8 - 0 > 7), valid = True
    # edu count < min_count (5 - 0 < 7), valid = False
    (
        {"count": {"train": {"age": 8, "edu": 5}}, "failure_count": {"train": {"age": 0, "edu": 0}}},
        7,
        {"train": {"age": True, "edu": False}},
    ),
]

MIN_COUNT_APPLY_TEST_CASES = [
    # statistics with age count = 6, failure count = 0 , min_count = 7, count < min_count (6 - 0 < 7), valid = False
    # all features are removed from result. Modified Flag = True
    (
        {"count": {"train": {"age": 6}}, "failure_count": {"train": {"age": 0}}},
        7,
        ({"count": {"train": {"age": 6}}, "failure_count": {"train": {}}}, True),
    ),
    # statistics with age count = 8, failure count = 1 , min_count = 7, (8 - 0 < 7), valid = True
    # all feature statistics remain. Modified Flag = False
    (
        {"count": {"train": {"age": 8}}, "failure_count": {"train": {"age": 1}}},
        7,
        ({"count": {"train": {"age": 8}}, "failure_count": {"train": {"age": 1}}}, False),
    ),
    (
        {
            "count": {"train": {"age": 8, "edu": 5}},
            "sum": {"train": {"age": 120, "edu": 360}},
            "failure_count": {"train": {"age": 0, "edu": 0}},
        },
        7,
        (
            {
                "count": {"train": {"age": 8, "edu": 5}},
                "sum": {"train": {"age": 120}},
                "failure_count": {"train": {"age": 0}},
            },
            True,
        ),
    ),
]


class TestMinCountChecker:
    @pytest.mark.parametrize("statistics, min_count, statistics_valid", MIN_COUNT_VALIDATION_TEST_CASES)
    def test_min_count_validate(self, statistics, min_count, statistics_valid):
        checker = MinCountCleanser(min_count=min_count)
        results = checker.min_count_validate("site-1", statistics=statistics)
        assert results == statistics_valid

    @pytest.mark.parametrize("statistics, min_count, expected_result", MIN_COUNT_APPLY_TEST_CASES)
    def test_min_count_apply(self, statistics, min_count, expected_result):
        checker = MinCountCleanser(min_count=min_count)
        results = checker.apply(statistics=statistics, client_name="site-1")
        assert results == expected_result
