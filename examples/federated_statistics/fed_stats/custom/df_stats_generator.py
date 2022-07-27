# Copyright (c) 2022, NVIDIA CORPORATION.
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

from typing import Dict, Optional, List

import pandas as pd
from numpy as np
from pandas.core.series import Series

from load_data_utils import get_app_paths, load_config
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.statistics_spec import Statistics
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.statistics.numeric_stats import (
    get_std_histogram_buckets,
    dtype_to_data_type
)
from nvflare.app_common.statistics.stats_def import BinRange, Histogram, HistogramType, Feature


class DFStatistics(Statistics):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.data: Optional[Dict[str, pd.DataFrame]] = None

    def load_data(self, fl_ctx: FLContext) -> Dict[str, pd.DataFrame]:
        client_name = fl_ctx.get_prop(ReservedKey.CLIENT_NAME)
        self.log_info(fl_ctx, f"load data for client {client_name}")
        try:
            workspace_dir, job_dir, config_path = get_app_paths(fl_ctx)
            config = load_config(config_path)

            features = config["fed_stats.data.features"]
            skip_rows = config[f"fed_stats.data.clients.{client_name}.skiprows"]
            data_path = self.data_path.replace("{workspace_dir}", workspace_dir).replace("{client_name}", client_name)

            # example of load data from CSV
            data: pd.DataFrame = pd.read_csv(
                data_path, names=features, sep=r"\s*,\s*", skiprows=skip_rows, engine="python", na_values="?"
            )
            self.log_info(fl_ctx, f"load data done for client {client_name}")
            return {"train": data}

        except BaseException as e:
            raise Exception(f"Load data for client {client_name} failed! {e}")

    def initialize(self, parts: dict, fl_ctx: FLContext):
        data = self.load_data(fl_ctx)

    def get_features(self, fl_ctx: FLContext) -> Dict[str, List[Feature]]:
        results: Dict[str, List[Feature]] = {}
        for ds_name in self.data:
            df = self.data[ds_name]
            results[ds_name] = []
            for feature_name in df:
                data_type = dtype_to_data_type[df[feature_name]]
                results[ds_name].append(Feature(feature_name, data_type))

        return results

    def get_count(self,
                  dataset_name: str,
                  feature_name: str,
                  inputs: Shareable,
                  fl_cxt: FLContext) -> int:
        df: pd.DataFrame = self.data[dataset_name]
        return df[feature_name].count()

    def get_sum(self,
                dataset_name: str,
                feature_name: str,
                inputs: Shareable,
                fl_ctx: FLContext) -> float:
        df: pd.DataFrame = self.data[dataset_name]
        return df[feature_name].sum()

    def get_mean(self,
                 dataset_name: str,
                 feature_name: str,
                 inputs: Shareable,
                 fl_ctx: FLContext) -> float:

        count: int = self.get_count(dataset_name, feature_name, inputs, fl_ctx)
        sum_value: float = self.get_sum(dataset_name, feature_name, inputs, fl_ctx)
        return sum_value / count

    def get_stddev(self,
                   dataset_name: str,
                   feature_name: str,
                   inputs: Shareable,
                   fl_ctx: FLContext) -> float:
        df = self.data[dataset_name]
        return df[feature_name].std()

    def get_variance_with_mean(self,
                               dataset_name: str,
                               feature_name: str,
                               inputs: Shareable,
                               fl_ctx: FLContext) -> float:

        global_mean = inputs[StC.STATS_GLOBAL_MEAN][dataset_name][feature_name]
        global_count = inputs[StC.STATS_GLOBAL_COUNT][dataset_name][feature_name]

        df = self.data[dataset_name]
        tmp = (df[feature_name] - global_mean) * (df[feature_name] - global_mean)
        variance = tmp.sum()/(global_count - 1)
        return variance

    def get_histogram(self,
                      dataset_name: str,
                      feature_name: str,
                      inputs: Shareable,
                      fl_ctx: FLContext) -> Histogram:

        global_bin_min = inputs[StC.STATS_MIN][dataset_name][feature_name]
        global_bin_max = inputs[StC.STATS_MAX][dataset_name][feature_name]
        num_of_bins: int = inputs[StC.STATS_BINS]

        df = self.data[dataset_name]
        feature: Series = df[feature_name]
        flattened = feature.ravel()
        flattened = flattened[flattened != np.array(None)]
        buckets = get_std_histogram_buckets(flattened, num_of_bins, BinRange(global_bin_min, global_bin_max))
        return Histogram(buckets, HistogramType.STANDARD)

    def get_max_value(self,
                      dataset_name: str,
                      feature_name: str,
                      inputs: Shareable,
                      fl_ctx: FLContext) -> float:
        """ this is needed for histogram calculation, not used for reporting """

        df = self.data[dataset_name]
        return df[feature_name].max()

    def get_min_value(self,
                      dataset_name: str,
                      feature_name: str,
                      inputs: Shareable,
                      fl_ctx: FLContext) -> float:
        """ this is needed for histogram calculation, not used for reporting """

        df = self.data[dataset_name]
        return df[feature_name].min()
