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
from nvflare.app_common.statistics.min_count_cleanser import MinCountCleanser
from nvflare.app_common.statistics.numeric_stats import exclude_withheld_counts, get_global_stats, get_min_or_max_values

TEST_CASE_1 = [
    (
        {
            "site-1": {
                "train": {
                    "Age": 630,
                    "fnlwgt": 3673746,
                    "Education-Num": 177,
                    "Capital Gain": 16258,
                    "Capital Loss": 0,
                    "Hours per week": 631,
                },
            }
        },
        {
            "train": {
                "Age": 630,
                "fnlwgt": 3673746,
                "Education-Num": 177,
                "Capital Gain": 16258,
                "Capital Loss": 0,
                "Hours per week": 631,
            }
        },
    )
]


class TestNumericStats:
    @pytest.mark.parametrize("client_stats, expected_global_stats", TEST_CASE_1)
    def test_accumulate_metrics(self, client_stats, expected_global_stats):
        from nvflare.app_common.statistics.numeric_stats import accumulate_metrics

        global_stats = {}
        for client_name in client_stats:
            global_stats = accumulate_metrics(
                metrics=client_stats[client_name], global_metrics=global_stats, precision=4
            )

        assert global_stats.keys() == expected_global_stats.keys()
        assert global_stats == expected_global_stats

    def test_get_min_or_max_values(self):
        client_statistics = {
            "site-1": {"train": {"Age": 0}, "test": {"Age": 2}},
            "site-2": {"train": {"Age": 1}, "test": {"Age": 3}},
        }

        global_statistics: dict[str, dict[str, int]] = {}
        for client in client_statistics:
            statistics = client_statistics[client]
            print("get_min_or_max_values =", global_statistics)
            global_statistics = get_min_or_max_values(statistics, global_statistics, min)

        assert global_statistics == {"test": {"Age": 0}, "train": {"Age": 0}}

        global_statistics: dict[str, dict[str, int]] = {}
        for client in client_statistics:
            statistics = client_statistics[client]
            print("get_min_or_max_values =", global_statistics)
            global_statistics = get_min_or_max_values(statistics, global_statistics, max)

        assert global_statistics == {"test": {"Age": 3}, "train": {"Age": 3}}


def _client_first_round_stats(values, min_count=10):
    """First-round statistics of one client, after the privacy cleanser, as the server receives them."""
    count = len(values)
    total = float(sum(values))
    statistics = {
        StC.STATS_COUNT: {"train": {"x": count}},
        StC.STATS_FAILURE_COUNT: {"train": {"x": 0}},
        StC.STATS_SUM: {"train": {"x": total}},
        StC.STATS_MEAN: {"train": {"x": total / count}},
    }
    cleansed, _ = MinCountCleanser(min_count=min_count).apply(statistics, "site")
    return cleansed


def _by_metric(clients):
    return {metric: {name: stats[metric] for name, stats in clients.items()} for metric in next(iter(clients.values()))}


class TestWithheldCounts:
    def test_exclude_withheld_counts(self):
        counts = {"train": {"a": 5, "b": 1}, "test": {"a": 2}}
        sums = {"train": {"a": 10.0}, "test": {}}

        assert exclude_withheld_counts(counts, sums) == {"train": {"a": 5}, "test": {}}

    def test_exclude_withheld_counts_keeps_everything_when_sums_are_not_requested(self):
        counts = {"train": {"a": 5}}

        assert exclude_withheld_counts(counts, None) == counts

    def test_global_mean_ignores_site_withheld_by_min_count(self):
        clients = {
            "site-1": _client_first_round_stats([11.0]),
            "site-2": _client_first_round_stats([9.0] * 40),
            "site-3": _client_first_round_stats([11.0] * 60),
        }
        assert clients["site-1"][StC.STATS_SUM] == {"train": {}}

        result = get_global_stats({}, _by_metric(clients), StC.STATS_1st_STATISTICS, {})

        assert result[StC.STATS_COUNT]["train"]["x"] == 100
        assert result[StC.STATS_SUM]["train"]["x"] == 1020.0
        assert result[StC.STATS_MEAN]["train"]["x"] == 10.2

    def test_global_count_keeps_every_site_when_sum_is_not_requested(self):
        clients = {
            "site-1": _client_first_round_stats([11.0]),
            "site-2": _client_first_round_stats([9.0] * 40),
        }
        count_only = {StC.STATS_COUNT: {name: stats[StC.STATS_COUNT] for name, stats in clients.items()}}

        result = get_global_stats({}, count_only, StC.STATS_1st_STATISTICS, {})

        assert result[StC.STATS_COUNT]["train"]["x"] == 41

    def test_global_mean_when_no_site_is_withheld(self):
        clients = {
            "site-1": _client_first_round_stats([11.0] * 10),
            "site-2": _client_first_round_stats([9.0] * 40),
        }

        result = get_global_stats({}, _by_metric(clients), StC.STATS_1st_STATISTICS, {})

        assert result[StC.STATS_COUNT]["train"]["x"] == 50
        assert result[StC.STATS_MEAN]["train"]["x"] == 9.4
