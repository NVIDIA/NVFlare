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

from typing import Dict, List, Optional

from monai.fl.client import ClientAlgoStats
from monai.fl.utils.constants import ExtraItems, FlStatistics
from monai.utils.enums import DataStatsKeys, ImageStatsKeys

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.statistics_spec import Bin, DataType, Feature, Histogram, HistogramType, Statistics


class ClientAlgoStatistics(Statistics):
    def __init__(self, client_algo_stats_id):
        """Statistics generator that gets data statistics from ClientAlgoStats.

        Args:
            client_algo_stats_id (str): id pointing to the client_stats object
        Returns:
            a Shareable with the computed local statistics`
        """
        super().__init__()
        self.client_algo_stats_id = client_algo_stats_id
        self.client_name = None
        self.client_algo_stats = None
        self.stats = None
        self.histograms = None
        self.fl_ctx = None

    def initialize(self, parts: dict, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx
        self.client_name = fl_ctx.get_identity_name()
        engine = fl_ctx.get_engine()
        self.client_algo_stats = engine.get_component(self.client_algo_stats_id)
        if not isinstance(self.client_algo_stats, ClientAlgoStats):
            raise TypeError(f"client_stats must be client_stats type. Got: {type(self.client_algo_stats)}")
        self.client_algo_stats.initialize(
            extra={
                ExtraItems.CLIENT_NAME: fl_ctx.get_identity_name(),
                ExtraItems.APP_ROOT: fl_ctx.get_prop(FLContextKey.APP_ROOT),
            }
        )

    def pre_run(
        self,
        statistics: List[str],
        num_of_bins: Optional[Dict[str, Optional[int]]],
        bin_ranges: Optional[Dict[str, Optional[List[float]]]],
    ):

        if num_of_bins:
            req_num_of_bins = list(num_of_bins.values())
        else:
            req_num_of_bins = []

        if bin_ranges:
            req_bin_ranges = list(bin_ranges.values())
        else:
            req_bin_ranges = []

        requested_stats = {
            FlStatistics.STATISTICS: statistics,
            FlStatistics.HIST_BINS: req_num_of_bins,
            FlStatistics.HIST_RANGE: req_bin_ranges,
        }
        self.stats = self.client_algo_stats.get_data_stats(extra=requested_stats).statistics

        # parse histograms
        self.histograms = {}
        for dataset_name in self.stats:
            self.histograms[dataset_name] = {}
            hist_list = self.stats[dataset_name][FlStatistics.DATA_STATS][DataStatsKeys.IMAGE_HISTOGRAM][ImageStatsKeys.HISTOGRAM]
            hist_feature_names = self.stats[dataset_name][FlStatistics.FEATURE_NAMES]
            for _hist_fn, _histo in zip(hist_feature_names, hist_list):
                self.histograms[dataset_name][_hist_fn] = _histo

    def features(self) -> Dict[str, List[Feature]]:
        features = {}
        for ds in self.stats:
            # convert dataset names to str to support FOBS
            features[str(ds)] = []
            for feat_name in self.stats[ds][FlStatistics.FEATURE_NAMES]:
                features[str(ds)].append(Feature(feat_name, DataType.FLOAT))

        return features

    def count(self, dataset_name: str, feature_name: str) -> int:
        if dataset_name in self.stats:
            return self.stats[dataset_name].get(FlStatistics.DATA_COUNT)
        else:
            self.log_warning(self.fl_ctx, f"No such dataset {dataset_name}")
            return 0

    def failure_count(self, dataset_name: str, feature_name: str) -> int:
        if dataset_name in self.stats:
            return self.stats[dataset_name].get(FlStatistics.FAIL_COUNT)
        else:
            self.log_warning(self.fl_ctx, f"No such dataset {dataset_name}")
            return 0

    def histogram(
        self, dataset_name: str, feature_name: str, num_of_bins: int, global_min_value: float, global_max_value: float
    ) -> Histogram:
        if dataset_name in self.stats:
            if feature_name in self.histograms[dataset_name]:
                histo = self.histograms[dataset_name][feature_name]
            else:
                self.log_warning(self.fl_ctx, f"Could not find a matching histogram for feature {feature_name} in dataset {dataset_name}.")
                return Histogram(HistogramType.STANDARD, list())
        else:
            self.log_warning(self.fl_ctx, f"No such dataset {dataset_name}")
            return Histogram(HistogramType.STANDARD, list())

        bin_edges = histo["bin_edges"]
        counts = histo["counts"]
        num_of_bins = len(counts)

        histogram_bins: List[Bin] = []
        for j in range(num_of_bins):
            low_value = bin_edges[j]
            high_value = bin_edges[j + 1]
            bin_sample_count = counts[j]
            histogram_bins.append(Bin(low_value=low_value, high_value=high_value, sample_count=bin_sample_count))

        return Histogram(HistogramType.STANDARD, histogram_bins)
