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

from nvflare.app_common.statistics.min_count_cleanser import MinCountCleanser

MIN_COUNT_VALIDATION_TEST_CASES = [
    ({"count": {"train": {"age": 6}}}, 7, {"train": {"age": False}}),
    ({"count": {"train": {"age": 8}}}, 7, {"train": {"age": True}}),
    ({"count": {"train": {"age": 8, "edu": 5}}}, 7, {"train": {"age": True, "edu": False}}),
]

MIN_COUNT_APPLY_TEST_CASES = [
    ({"count": {"train": {"age": 6}}}, 7, ({"count": {"train": {}}}, True)),
    ({"count": {"train": {"age": 8}}}, 7, ({"count": {"train": {"age": 8}}}, False)),
    ({"count": {"train": {"age": 8, "edu": 5}}}, 7, ({"count": {"train": {"age": 8}}}, True)),
]


class TestMinCountChecker:
    @pytest.mark.parametrize("metrics, min_count, expected_result", MIN_COUNT_VALIDATION_TEST_CASES)
    def test_min_count_validate(self, metrics, min_count, expected_result):
        checker = MinCountCleanser(min_count=min_count)
        results = checker.min_count_validate("site-1", metrics=metrics)
        assert results == expected_result

    @pytest.mark.parametrize("metrics, min_count, expected_result", MIN_COUNT_APPLY_TEST_CASES)
    def test_min_count_apply(self, metrics, min_count, expected_result):
        checker = MinCountCleanser(min_count=min_count)
        results = checker.apply(metrics=metrics, client_name="site-1")
        assert results == expected_result
