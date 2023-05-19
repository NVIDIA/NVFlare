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
from nvflare.app_common.abstract.statistics_spec import Histogram
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.statistics.statistics_privacy_cleanser import StatisticsPrivacyCleanser


class HistogramBinsCleanser(FLComponent, StatisticsPrivacyCleanser):
    def __init__(self, max_bins_percent):
        """
        max_bins_percent:   max number of bins allowed in terms of percent of local data size.
                            Set this number to avoid number of bins equal or close equal to the
                            data size, which can lead to data leak.
                            for example: max_bins_percent = 10, means 10%
                            number of bins < max_bins_percent /100 * local count
        """
        super().__init__()
        self.max_bins_percent = max_bins_percent
        self.validate_inputs()

    def validate_inputs(self):
        if self.max_bins_percent < 0 or self.max_bins_percent > 100:
            raise ValueError(f"max_bins_percent {self.max_bins_percent} is not within (0, 100) ")

    def hist_bins_validate(self, client_name: str, statistics: Dict) -> Dict[str, Dict[str, bool]]:
        result = {}
        if StC.STATS_HISTOGRAM in statistics:
            hist_statistics = statistics[StC.STATS_HISTOGRAM]
            for ds_name in hist_statistics:
                result[ds_name] = {}
                feature_item_counts = statistics[StC.STATS_COUNT][ds_name]
                feature_item_failure_counts = statistics[StC.STATS_FAILURE_COUNT][ds_name]
                feature_statistics = hist_statistics[ds_name]
                for feature in feature_statistics:
                    hist: Histogram = feature_statistics[feature]
                    num_of_bins: int = len(hist.bins)
                    item_count = feature_item_counts[feature]
                    item_failure_count = feature_item_failure_counts[feature]
                    effective_count = item_count - item_failure_count
                    result[ds_name][feature] = True
                    limit_count = round(effective_count * self.max_bins_percent / 100)
                    if num_of_bins >= limit_count:
                        result[ds_name][feature] = False
                        self.logger.info(
                            f"number of bins: '{num_of_bins}' needs to be smaller than: {limit_count}], which"
                            f" is '{self.max_bins_percent}' percent of ( total count - failure count) '{effective_count}'"
                            f" for feature '{feature}' in dataset '{ds_name}' for client {client_name}"
                        )
        return result

    def apply(self, statistics: dict, client_name: str) -> Tuple[dict, bool]:
        self.logger.info(f"HistogramBinCheck for client {client_name}")
        if StC.STATS_HISTOGRAM in statistics:
            validation_result = self.hist_bins_validate(client_name, statistics)
            statistics_keys = [StC.STATS_HISTOGRAM]
            return super().cleanse(statistics, statistics_keys, validation_result)
        else:
            return statistics, False
