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

from typing import Dict, Tuple

from nvflare.apis.fl_component import FLComponent
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.statistics.statistics_privacy_cleanser import StatisticsPrivacyCleanser


class MinCountCleanser(FLComponent, StatisticsPrivacyCleanser):
    def __init__(self, min_count: float):
        """
        min_count:  minimum of data records (or tabular data rows) that required in order to perform statistics
                    calculation this is part the data privacy policy.
        """
        super().__init__()
        self.min_count = min_count
        self.validate_inputs()

    def validate_inputs(self):
        if self.min_count < 0:
            raise ValueError(f"min_count must be positive, but {self.min_count} is provided. ")

    def min_count_validate(self, client_name: str, statistics: Dict) -> Dict[str, Dict[str, bool]]:
        feature_statistics_valid = {}
        if StC.STATS_COUNT in statistics:
            count_statistics = statistics[StC.STATS_COUNT]
            for ds_name in count_statistics:
                feature_statistics_valid[ds_name] = {}
                feature_counts = statistics[StC.STATS_COUNT][ds_name]
                feature_failure_counts = statistics[StC.STATS_FAILURE_COUNT][ds_name]
                for feature in feature_counts:
                    count = feature_counts[feature]
                    failure_count = feature_failure_counts[feature]
                    effective_count = count - failure_count
                    feature_statistics_valid[ds_name][feature] = True
                    if effective_count < self.min_count:
                        feature_statistics_valid[ds_name][feature] = False
                        self.logger.info(
                            f"dataset {ds_name} feature '{feature}' item count is "
                            f"less than required minimum count {self.min_count} for client {client_name} "
                        )
        return feature_statistics_valid

    def apply(self, statistics: dict, client_name) -> Tuple[dict, bool]:
        self.logger.info(f"apply MinCountCheck for client {client_name}")
        validation_result = self.min_count_validate(client_name, statistics)
        statistics_keys = list(statistics.keys())
        return super().cleanse(statistics, statistics_keys, validation_result)
