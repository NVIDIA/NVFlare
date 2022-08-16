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

from typing import Dict

from nvflare.app_common.statistics.numeric_stats import get_min_or_max_values


class TestNumericStats:
    def test_get_min_or_max_values(self):
        client_metrics = {
            "site-1": {"train": {"Age": 0}, "test": {"Age": 2}},
            "site-2": {"train": {"Age": 1}, "test": {"Age": 3}},
        }

        global_metrics: Dict[str, Dict[str, int]] = {}
        for client in client_metrics:
            metrics = client_metrics[client]
            print("get_min_or_max_values =", global_metrics)
            global_metrics = get_min_or_max_values(metrics, global_metrics, min)

        assert global_metrics == {"test": {"Age": 0}, "train": {"Age": 0}}

        global_metrics: Dict[str, Dict[str, int]] = {}
        for client in client_metrics:
            metrics = client_metrics[client]
            print("get_min_or_max_values =", global_metrics)
            global_metrics = get_min_or_max_values(metrics, global_metrics, max)

        assert global_metrics == {"test": {"Age": 3}, "train": {"Age": 3}}
