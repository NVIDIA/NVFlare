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

from typing import Dict

import pytest

from nvflare.app_common.statistics.numeric_stats import get_min_or_max_values

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

        global_statistics: Dict[str, Dict[str, int]] = {}
        for client in client_statistics:
            statistics = client_statistics[client]
            print("get_min_or_max_values =", global_statistics)
            global_statistics = get_min_or_max_values(statistics, global_statistics, min)

        assert global_statistics == {"test": {"Age": 0}, "train": {"Age": 0}}

        global_statistics: Dict[str, Dict[str, int]] = {}
        for client in client_statistics:
            statistics = client_statistics[client]
            print("get_min_or_max_values =", global_statistics)
            global_statistics = get_min_or_max_values(statistics, global_statistics, max)

        assert global_statistics == {"test": {"Age": 3}, "train": {"Age": 3}}
