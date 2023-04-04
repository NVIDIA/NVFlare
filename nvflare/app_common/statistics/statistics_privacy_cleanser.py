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

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from nvflare.app_common.app_constant import StatisticsConstants as StC


class StatisticsPrivacyCleanser(ABC):
    @abstractmethod
    def apply(self, statistics: dict, client_name: str) -> Tuple[dict, bool]:
        pass

    def cleanse(
        self, statistics: dict, statistic_keys: List[str], validation_result: Dict[str, Dict[str, bool]]
    ) -> (dict, bool):
        """
        Args:
            statistics: original client local metrics
            statistic_keys: statistic keys need to be cleansed
            validation_result: local metrics privacy validation result
        Returns:
            filtered metrics with feature metrics that violating the privacy policy be removed from the original metrics

        """
        statistics_modified = False
        for key in statistic_keys:
            if key != StC.STATS_COUNT:
                for ds_name in list(statistics[key].keys()):
                    for feature in list(statistics[key][ds_name].keys()):
                        if not validation_result[ds_name][feature]:
                            statistics[key][ds_name].pop(feature, None)
                            statistics_modified = True

        return statistics, statistics_modified
