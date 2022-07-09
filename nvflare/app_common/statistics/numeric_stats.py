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
from math import sqrt
from typing import Dict, Optional, List

from numpy as np

from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.statistics.stats_def import (
    BinRange,
    NumericStatistics,
    FeatureStatistics,
    DatasetStatistics,
)
from nvflare.app_common.statistics.stats_def import Histogram, HistogramType, Bin, DataType


def get_client_dataset_stats(client_metrics: dict, client_data_types: dict) -> List[DatasetStatistics]:
    results = []
    for client_name in client_metrics:
        metrics = client_metrics[client_name]
        feature_data_types = client_data_types[client_name]

        feature_counts = metrics[StC.STATS_COUNT]
        feature_means = metrics[StC.STATS_MEAN]
        feature_sums = metrics[StC.STATS_SUM]
        feature_stddev = metrics[StC.STATS_STDDEV]
        feature_histograms = metrics[StC.STATS_HISTOGRAM]

        sample_count = list(feature_counts.values())[0]
        feature_stats = []

        for f in feature_means:
            numeric_stats = \
                NumericStatistics(
                    feature_means[f],
                    feature_sums[f],
                    feature_stddev[f],
                    feature_counts[f],
                    feature_histograms[f]
                )
            feature_stats.append(FeatureStatistics(f, feature_data_types[f], numeric_stats))

        results.append(DatasetStatistics(client_name, sample_count, feature_stats))

        return results


def get_global_dataset_stats(dataset_name: str, metrics: dict, feature_data_types: dict) -> DatasetStatistics:
    feature_means = metrics[StC.STATS_MEAN]
    feature_counts = metrics[StC.STATS_COUNT]
    feature_sums = metrics[StC.STATS_SUM]
    feature_stddev = metrics[StC.STATS_STDDEV]
    feature_histograms = metrics[StC.STATS_HISTOGRAM]

    sample_count = list(feature_counts.values())[0]
    feature_stats = []

    for f in feature_means:
        numeric_stats = \
            NumericStatistics(
                feature_means[f],
                feature_sums[f],
                feature_stddev[f],
                feature_counts[f],
                feature_histograms[f]
            )
        feature_stats.append(FeatureStatistics(f, feature_data_types[f], numeric_stats))

    return DatasetStatistics(dataset_name, sample_count, feature_stats)


def get_global_feature_data_types(client_feature_dts: dict) -> dict:
    global_feature_dts = {}
    for client_name in client_feature_dts:
        feature_dts = client_feature_dts[client_name]
        for f in feature_dts:
            if f not in global_feature_dts:
                data_type = feature_dts[f]
                global_feature_dts[f] = data_type

    return global_feature_dts


def get_global_stats(client_metrics: dict) -> dict:
    global_metrics = {}

    for metric in client_metrics:
        stats = client_metrics[metric]
        global_metrics[metric] = {}
        if metric == StC.STATS_COUNT or metric == StC.STATS_SUM:
            for client_name in stats:
                global_metrics[metric] = accumulate_metrics(stats[client_name], global_metrics[metric])
        elif metric == StC.STATS_MEAN:
            global_metrics[metric] = get_means(global_metrics[StC.STATS_SUM], global_metrics[StC.STATS_COUNT])
        elif metric == StC.STATS_HISTOGRAM:
            for client_name in stats:
                global_metrics[metric] = accumulate_hists(stats[client_name], global_metrics[metric])
        elif metric == StC.STATS_VAR:
            for client_name in stats:
                global_metrics[metric] = accumulate_metrics(stats[client_name], global_metrics[metric])
        elif metric == StC.STATS_STDDEV:
            feature_vars = global_metrics[StC.STATS_VAR]
            feature_stddev = {}
            for feature in feature_vars:
                feature_stddev[feature] = sqrt(vars[feature])
            global_metrics[StC.STATS_STDDEV] = feature_stddev

    return global_metrics


def accumulate_metrics(metrics: Dict[str, int], global_metrics: Dict[str, int]) -> Dict[str, int]:
    for feature in metrics:
        global_metrics[feature] += metrics[feature]
    return global_metrics


def bins_to_dict(bins: List[Bin]) -> Dict[BinRange, float]:
    buckets = {}
    for bucket in bins:
        bucket_range = BinRange(bucket.low_value, bucket.high_value)
        buckets[bucket_range] = bucket.sample_count
    return buckets


def accumulate_hists(metrics: Dict[str, Histogram], global_hists: Dict[str, Histogram]) -> Dict[str, Histogram]:
    for feature in metrics:
        hist: Histogram = metrics[feature]
        if feature not in global_hists:
            g_bins = []
            for bucket in hist.bins:
                g_bins.append(Bin(bucket.low_value, bucket.high_value, bucket.sample_count))
            g_hist = Histogram(g_bins, HistogramType.STANDARD)
            global_hists[feature] = g_hist
        else:
            g_hist = global_hists[feature]
            g_buckets = bins_to_dict(g_hist.bins)
            for bucket in hist.bins:
                bin_range = BinRange(bucket.low_value, bucket.high_value)
                if bin_range in g_buckets:
                    g_buckets[bin_range] += bucket.sample_count
                else:
                    g_buckets[bin_range] = bucket.sample_count

            # update ordered bins
            updated_bins = []
            for gb in g_hist.bins:
                bin_range = BinRange(gb.low_value, db.high_value)
                updated_bins.append(Bin(gb.low_value, gb.high_value, g_buckets[bin_range]))

            global_hists[feature] = Histogram(updated_bins, g_hist.hist_type)

    return global_hists


def get_means(sums: Dict[str, float], counts: Dict[str, int]) -> Dict[str, float]:
    means = {}
    for feature in sums:
        means[feature] = sums[feature] / counts[feature]
    return means


def dtype_to_data_type(dtype) -> DataType:
    if dtype.char in np.typecodes["AllFloat"]:
        return DataType.FLOAT
    elif dtype.char in np.typecodes["AllInteger"] or dtype == bool:
        return DataType.INT
    elif np.issubdtype(dtype, np.datetime64) or np.issubdtype(dtype, np.timedelta64):
        return DataType.DATETIME
    else:
        return DataType.STRING


def get_std_histogram_buckets(nums: np.ndarray, num_bins: int = 10, br: Optional[BinRange] = None):
    num_posinf = len(nums[np.isposinf(nums)])
    num_neginf = len(nums[np.isneginf(nums)])
    if br:
        counts, buckets = np.histogram(nums, bins=num_bins, range=(br.min_value, br.max_value))
    else:
        counts, buckets = np.histogram(nums, bins=num_bins)

    histogram_buckets: List[Bin] = []
    for bucket_count in range(len(counts)):
        # Add any negative or positive infinities to the first and last
        # buckets in the histogram.
        bucket_low_value = buckets[bucket_count]
        bucket_high_value = buckets[bucket_count + 1]
        bucket_sample_count = counts[bucket_count]
        if bucket_count == 0 and num_neginf > 0:
            bucket_low_value = float("-inf")
            bucket_sample_count += num_neginf
        elif bucket_count == len(counts) - 1 and num_posinf > 0:
            bucket_high_value = float("inf")
            bucket_sample_count += num_posinf

        histogram_buckets.append(
            Bin(low_value=bucket_low_value, high_value=bucket_high_value, sample_count=bucket_sample_count)
        )

    if buckets is not None and len(buckets) > 0:
        bucket = None
        if num_neginf:
            bucket = Bin(low_value=float("-inf"), high_value=float("-inf"), sample_count=num_neginf)
        if num_posinf:
            bucket = Bin(low_value=float("inf"), high_value=float("inf"), sample_count=num_posinf)

        if bucket:
            histogram_buckets.append(bucket)

    return histogram_buckets
