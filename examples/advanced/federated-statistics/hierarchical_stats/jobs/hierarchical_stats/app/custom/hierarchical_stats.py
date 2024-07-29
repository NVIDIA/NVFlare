# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.core.series import Series

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.statistics_spec import BinRange, Feature, Histogram, HistogramType, Statistics
from nvflare.app_common.statistics.numpy_utils import dtype_to_data_type, get_std_histogram_buckets


class HierarchicalStats(Statistics):
    def __init__(self):
        super().__init__()
        self.data_root_dir = "/tmp/nvflare/data/hierarchical_stats/"
        self.data: Optional[Dict[str, pd.DataFrame]] = None
        self.data_features = [
            "Pass",
            "Fail",
            "Percentage",
        ]
        self.skip_rows = {}

    def load_data(self, fl_ctx: FLContext) -> Dict[str, pd.DataFrame]:
        client_name = fl_ctx.get_identity_name()
        self.log_info(fl_ctx, f"Load data for client {client_name}")
        try:
            skip_rows = self.skip_rows
            path = Path(self.data_root_dir)
            csv_file_name = client_name + ".csv"
            csv_file = os.path.join(path, client_name, csv_file_name)
            dfs = []
            df = pd.read_csv(csv_file, sep=r"\s*,\s*", skiprows=skip_rows, engine="python", na_values="?")
            return {"default_set": df}
        except Exception as e:
            raise Exception(f"Load data for client {client_name} failed! {e}")

    def initialize(self, fl_ctx: FLContext):
        self.data = self.load_data(fl_ctx)
        if self.data is None:
            raise ValueError("Data is not loaded. make sure the data is loaded")

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
        if feature_name == self.data_features[0] or feature_name == self.data_features[1]:
            # The features `Pass` and `Fail` will only have values `0` and `1` so we only
            # return the count with values `1`.
            return df[feature_name].sum()

        return df[feature_name].count()

    def sum(self, dataset_name: str, feature_name: str) -> float:
        df: pd.DataFrame = self.data[dataset_name]
        return df[feature_name].sum()

    def mean(self, dataset_name: str, feature_name: str) -> float:
        df: pd.DataFrame = self.data[dataset_name]
        return df[feature_name].mean()

    def stddev(self, dataset_name: str, feature_name: str) -> float:
        df: pd.DataFrame = self.data[dataset_name]
        return df[feature_name].std().item()

    def min_value(self, dataset_name: str, feature_name: str) -> float:
        df: pd.DataFrame = self.data[dataset_name]
        return df[feature_name].min()

    def max_value(self, dataset_name: str, feature_name: str) -> float:
        df: pd.DataFrame = self.data[dataset_name]
        return df[feature_name].max()

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

    def variance_with_mean(
        self, dataset_name: str, feature_name: str, global_mean: float, global_count: float
    ) -> float:
        df = self.data[dataset_name]
        tmp = (df[feature_name] - global_mean) * (df[feature_name] - global_mean)
        variance = tmp.sum() / (global_count - 1)
        return variance.item()
