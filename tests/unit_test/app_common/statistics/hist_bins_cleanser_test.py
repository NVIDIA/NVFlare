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

from nvflare.app_common.abstract.statistics_spec import Bin, Histogram, HistogramType
from nvflare.app_common.statistics.histogram_bins_cleanser import HistogramBinsCleanser

hist_bins = [Bin(i, i + 1, 100) for i in range(0, 10)]
age_hist = Histogram(hist_type=HistogramType.STANDARD, bins=hist_bins, hist_name=None)
"""
    # case 1: 
        # numbers of bins = 10
        # count = 6
        # max_bins_percent = 10, i.e 10%
        # 6*10% = 0.1*6 = 0.6 ==> round(0.6) ==> 1
        # 10 > 1

    # case 2: 
        # numbers of bins = 10
        # count = 200
        # max_bins_percent = 10, i.e 10%
        # 200*10% = 0.1*200 = 20
        # 10 < 20 
"""
HIST_BINS_VALIDATION_TEST_CASES = [
    (
        {
            "count": {"train": {"age": 6}},
            "failure_count": {"train": {"age": 0}},
            "histogram": {"train": {"age": age_hist}},
        },
        10,
        {"train": {"age": False}},
    ),
    (
        {
            "count": {"train": {"age": 200}},
            "failure_count": {"train": {"age": 0}},
            "histogram": {"train": {"age": age_hist}},
        },
        10,
        {"train": {"age": True}},
    ),
]

HIST_BINS_APPLY_TEST_CASES = [
    (
        {
            "count": {"train": {"age": 6}},
            "failure_count": {"train": {"age": 0}},
            "histogram": {"train": {"age": age_hist}},
        },
        10,
        ({"count": {"train": {"age": 6}}, "failure_count": {"train": {"age": 0}}, "histogram": {"train": {}}}, True),
    ),
    (
        {
            "count": {"train": {"age": 200}},
            "failure_count": {"train": {"age": 0}},
            "histogram": {"train": {"age": age_hist}},
        },
        10,
        (
            {
                "count": {"train": {"age": 200}},
                "failure_count": {"train": {"age": 0}},
                "histogram": {"train": {"age": age_hist}},
            },
            False,
        ),
    ),
]


class TestHistBinsCleanser:
    @pytest.mark.parametrize("statistics, max_bins_percent, expected_result", HIST_BINS_VALIDATION_TEST_CASES)
    def test_hist_bins_validate(self, statistics, max_bins_percent, expected_result):
        checker = HistogramBinsCleanser(max_bins_percent=max_bins_percent)
        results = checker.hist_bins_validate("site-1", statistics=statistics)
        assert results == expected_result

    @pytest.mark.parametrize("statistics, max_bins_percent, expected_result", HIST_BINS_APPLY_TEST_CASES)
    def test_hist_bins_apply(self, statistics, max_bins_percent, expected_result):
        checker = HistogramBinsCleanser(max_bins_percent=max_bins_percent)
        results = checker.apply(statistics=statistics, client_name="site-1")
        assert results == expected_result
