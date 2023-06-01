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

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.core.series import Series

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.statistics_spec import (
    BinRange,
    DataType,
    Feature,
    Histogram,
    HistogramType,
    Statistics,
)
from nvflare.app_common.statistics.numpy_utils import get_std_histogram_buckets


def load_data() -> Dict[str, pd.DataFrame]:
    try:
        train_data = [["tom", 10], ["nick", 15], ["juli", 14], ["tom2", 10], ["nick1", 25], ["juli1", 24]]
        test_data = [["john", 100], ["mary", 25], ["rose", 34], ["tom1", 20], ["nick2", 35], ["juli1", 34]]
        train = pd.DataFrame(train_data, columns=["Name", "Age"])
        test = pd.DataFrame(test_data, columns=["Name", "Age"])

        return {"train": train, "test": test}

    except Exception as e:
        raise Exception(f"Load data failed! {e}")


class MockDFStatistics(Statistics):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.data: Optional[Dict[str, pd.DataFrame]] = None

    def initialize(self, fl_ctx: FLContext):
        self.data = load_data()
        if self.data is None:
            raise ValueError("data is not loaded. make sure the data is loaded")

    def features(self) -> Dict[str, List[Feature]]:
        features = [Feature("Name", DataType.STRING), Feature("Age", DataType.INT)]
        results: Dict[str, List[Feature]] = {"train": features, "test": features}
        return results

    def count(self, dataset_name: str, feature_name: str) -> int:
        df: pd.DataFrame = self.data[dataset_name]
        return df[feature_name].count()

    def sum(self, dataset_name: str, feature_name: str) -> float:
        raise NotImplementedError

    def mean(self, dataset_name: str, feature_name: str) -> float:

        count: int = self.count(dataset_name, feature_name)
        sum_value: float = self.sum(dataset_name, feature_name)
        return sum_value / count

    def stddev(self, dataset_name: str, feature_name: str) -> float:
        raise NotImplementedError

    def variance_with_mean(
        self, dataset_name: str, feature_name: str, global_mean: float, global_count: float
    ) -> float:
        raise NotImplementedError

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
        """this is needed for histogram global max estimation, not used for reporting"""
        df = self.data[dataset_name]
        return df[feature_name].max()

    def min_value(self, dataset_name: str, feature_name: str) -> float:
        """this is needed for histogram global min estimation, not used for reporting"""

        df = self.data[dataset_name]
        return df[feature_name].min()
