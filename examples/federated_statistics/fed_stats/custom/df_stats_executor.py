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
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.data_spec import Data
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.executors.statistics.statistics_executor import StatisticsExecutor
from nvflare.app_common.statistics.stats_def import BinRange, DataType, Histogram, HistogramType, Bin
from nvflare.app_common.statistics.numeric_stats import get_std_histogram_buckets, dtype_to_data_type

class DFStatistics(StatisticsExecutor):
    def __init__(self,
                 data_path,
                 min_count
                 ):
        super().__init__(data_path, min_count)

    def load_data(self, task_name: str, client_name: str, fl_ctx: FLContext) -> Data[pd.DataFrame]:
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
            return Data[pd.DataFrame](data)

        except BaseException as e:
            raise Exception(f"Load data in task: {task_name} for client {client_name} failed! {e}")

    def client_data_validate(self, client_name: str, shareable: Shareable, fl_ctx: FLContext):
        # make sure all features are numeric features.
        # df: pd.DataFrame = self.data.get_data()
        # for feature in df:
        #     f = df[feature]
        #     data_type = dtype_to_data_type(f.dtype)
        #     if not (data_type == DataType.INT or data_type == DataType.FLOAT):
        #         raise ValueError(f"feature {feature} is not a numerical data type")

        # if you believe your data is all numeric, you can skip this check
        pass

    def get_counts(self, shareable: Shareable, fl_cxt: FLContext) -> Dict[str, int]:
        if len(self.counts) > 0:
            return self.counts
        else:
            df: pd.DataFrame = self.data.get_data()
            results = {}
            for feature_name in df:
                count = df[feature_name].count()
                results[feature_name] = count

            self.counts = results
            return results

    def get_sums(self, inputs: Shareable, fl_ctx: FLContext) -> Dict[str, float]:
        if len(self.sums) > 0:
            return self.sums
        else:
            df: pd.DataFrame = self.data.get_data()
            results = {}
            for feature_name in df:
                sum_value = df[feature_name].sum()
                results[feature_name] = sum_value
            self.sums = results
            return results

    def get_stddevs(self, inputs: Shareable, fl_ctx: FLContext) -> Dict[str, float]:
        df: pd.DataFrame = self.data.get_data()
        results = {}
        for feature_name in df:
            std_dev = df[feature_name].std()
            results[feature_name] = std_dev
        return results

    def get_variances_with_mean(self, inputs: Shareable, fl_ctx: FLContext) -> Dict[str, float]:
        means: Dict[str, float] = inputs[StC.STATS_GLOBAL_MEAN]
        counts: Dict[str, float] = inputs[StC.STATS_GLOBAL_COUNT]
        df = self.data.get_data()
        variances = {}
        for feat in means:
            tmp = (df[feat] - means[feat]) * (df[feat] - means[feat]) / (counts[feat] - 1)
            variances[feat] = tmp.sum()
        return variances

    def get_histograms(self, inputs: Shareable, fl_ctx: FLContext) -> Dict[str, Histogram]:
        bin_range: BinRange = inputs[StC.STATS_BIN_RANGE]
        num_of_bins: int = inputs[StC.STATS_BINS]

        df = self.data.get_data()
        histograms = {}
        for feature_name in df:
            feature: Series = df[feature_name]
            flattened = feature.ravel()
            flattened = flattened[flattened != np.array(None)]
            buckets = get_std_histogram_buckets(flattened, num_of_bins, bin_range)
            histograms[feature_name] = Histogram(buckets, HistogramType.STANDARD)

        return histograms


