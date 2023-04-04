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

from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.statistics.min_max_cleanser import AddNoiseToMinMax

MAX_TEST_CASES = [
    (100, (0.1, 0.3), (100 * 1.1, 100 * 1.3)),
    (0, (0.1, 0.3), (1e-5, 1e-5 * 1.3)),
    (1e-4, (0.1, 0.3), (1e-4, 1e-4 * 1.3)),
    (0.6 * 1e-3, (0.1, 0.3), (0.6 * 1e-3, 0.6 * 1.3)),
    (-0.6 * 1e-3, (0.1, 0.3), (-0.6 * 1e-3, -0.6 * 1e-3 * 0.7)),
    (-1e-3, (0.1, 0.3), (-1e-3, -1e-3 * 0.7)),
    (-100, (0.1, 0.3), (-100, -100 * 0.7)),
]

MIN_TEST_CASES = [
    (100, (0.1, 0.3), (100 * 0.7, 100 * 0.9)),
    (0, (0.1, 0.3), (-1e-5, 0)),
    (-100, (0.1, 0.3), (-100 * 1.3, -100 * 0.9)),
    (0.6 * 1e-3, (0.1, 0.3), (0.6 * 1e-3 * 0.7, 0.6 * 1e-3 * 0.9)),
    (-0.6 * 1e-3, (0.1, 0.3), (-0.6 * 1e-3 * 1.3, -0.6 * 1e-3 * 0.9)),
    (-1e-3, (0.1, 0.3), (-1e-3 * 1.3, -1e-3 * 0.9)),
    (-1e-4, (0.1, 0.3), (-1e-4 * 1.3, -1e-4 * 0.9)),
]

NOISE_TEST_CASES = [
    (
        {"min": {"train": {"age": 0, "edu": 4}}, "max": {"train": {"age": 120, "edu": 13}}},
        (0.1, 0.3),
        {"min": {"train": {"age": -1e-5 * 0.7, "edu": 4 * 0.9}}, "max": {"train": {"age": 120 * 1.1, "edu": 4 * 1.1}}},
    )
]


class TestAddNoiseToMinMax:
    @pytest.mark.parametrize("value, noise_level, compare_result", MAX_TEST_CASES)
    def test_get_max_value(self, value, noise_level, compare_result):
        value_with_noise = AddNoiseToMinMax._get_max_value(value, noise_level)
        assert value_with_noise > compare_result[0]
        assert value_with_noise <= compare_result[1]

    @pytest.mark.parametrize("value, noise_level, compare_result", MIN_TEST_CASES)
    def test_get_min_value(self, value, noise_level, compare_result):
        value_with_noise = AddNoiseToMinMax._get_min_value(value, noise_level)
        assert value_with_noise > compare_result[0]
        assert value_with_noise <= compare_result[1]

    @pytest.mark.parametrize("statistics, noise_level, compare_result", NOISE_TEST_CASES)
    def test_min_value_noise_generator(self, statistics, noise_level, compare_result):
        gen = AddNoiseToMinMax(noise_level[0], noise_level[1])
        statistic = StC.STATS_MIN
        statistics_with_noise = gen.generate_noise(statistics, statistic)
        min_statistics = statistics_with_noise[statistic]
        for ds in min_statistics:
            for feature in min_statistics[ds]:
                assert min_statistics[ds][feature] <= compare_result[statistic][ds][feature]

    @pytest.mark.parametrize("statistics, noise_level, compare_result", NOISE_TEST_CASES)
    def test_max_value_noise_generator(self, statistics, noise_level, compare_result):
        gen = AddNoiseToMinMax(noise_level[0], noise_level[1])
        statistic = StC.STATS_MAX
        statistics_with_noise = gen.generate_noise(statistics, statistic)
        max_statistics = statistics_with_noise[statistic]
        for ds in max_statistics:
            for feature in max_statistics[ds]:
                assert max_statistics[ds][feature] > compare_result[statistic][ds][feature]
