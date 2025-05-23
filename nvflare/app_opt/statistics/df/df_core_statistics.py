# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from abc import ABC
from math import sqrt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.core.series import Series

from nvflare.app_common.abstract.statistics_spec import BinRange, Feature, Histogram, HistogramType, Statistics
from nvflare.app_common.app_constant import StatisticsConstants
from nvflare.app_common.statistics.numpy_utils import dtype_to_data_type, get_std_histogram_buckets
from nvflare.fuel.utils.import_utils import optional_import


class DFStatisticsCore(Statistics, ABC):
    def __init__(self, max_bin=None):
        # assumption: the data can be loaded and cached in the memory
        self.data: Optional[Dict[str, pd.DataFrame]] = None
        super(DFStatisticsCore, self).__init__()
        self.max_bin = max_bin

    def features(self) -> Dict[str, List[Feature]]:
        results: Dict[str, List[Feature]] = {}
        for ds_name in self.data:
            df = self.data[ds_name]
            results[ds_name] = []
            for feature_name in df:
                data_type = dtype_to_data_type(df[feature_name].dtype)
                results[ds_name].append(Feature(feature_name, data_type))

        return results

    def count(self, dataset_name: str, feature_name: str) -> int:
        df: pd.DataFrame = self.data[dataset_name]
        return df[feature_name].count()

    def sum(self, dataset_name: str, feature_name: str) -> float:
        df: pd.DataFrame = self.data[dataset_name]
        return df[feature_name].sum().item()

    def mean(self, dataset_name: str, feature_name: str) -> float:

        count: int = self.count(dataset_name, feature_name)
        sum_value: float = self.sum(dataset_name, feature_name)
        return sum_value / count

    def stddev(self, dataset_name: str, feature_name: str) -> float:
        df = self.data[dataset_name]
        return df[feature_name].std().item()

    def variance_with_mean(
        self, dataset_name: str, feature_name: str, global_mean: float, global_count: float
    ) -> float:
        df = self.data[dataset_name]
        tmp = (df[feature_name] - global_mean) * (df[feature_name] - global_mean)
        variance = tmp.sum() / (global_count - 1)
        return variance.item()

    def histogram(
        self, dataset_name: str, feature_name: str, num_of_bins: int, global_min_value: float, global_max_value: float
    ) -> Histogram:

        num_of_bins: int = num_of_bins

        df = self.data[dataset_name]
        feature: Series = df[feature_name]
        flattened = feature.ravel()
        flattened = flattened[flattened != np.array(None)]
        buckets = get_std_histogram_buckets(flattened, num_of_bins, BinRange(global_min_value, global_max_value))
        return Histogram(HistogramType.STANDARD, buckets)

    def max_value(self, dataset_name: str, feature_name: str) -> float:
        """this is needed for histogram calculation, not used for reporting"""

        df = self.data[dataset_name]
        return df[feature_name].max()

    def min_value(self, dataset_name: str, feature_name: str) -> float:
        """this is needed for histogram calculation, not used for reporting"""

        df = self.data[dataset_name]
        return df[feature_name].min()

    def quantiles(self, dataset_name: str, feature_name: str, percents: List) -> Dict:
        TDigest, flag = optional_import("fastdigest", name="TDigest")
        results = {}
        if not flag:
            results[StatisticsConstants.STATS_QUANTILE] = {}
            return results

        df = self.data[dataset_name]
        data = df[feature_name]
        max_bin = self.max_bin if self.max_bin else round(sqrt(len(data)))
        digest = TDigest(data)
        digest.compress(max_bin)

        p_results = {}
        for p in percents:
            v = round(digest.quantile(p), 4)
            p_results[p] = v
        results[StatisticsConstants.STATS_QUANTILE] = p_results

        # Extract the Q-Digest into a dictionary
        results[StatisticsConstants.STATS_DIGEST_COORD] = digest.to_dict()
        return results
