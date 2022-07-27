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
from typing import Dict, Optional, List, Generic
from typing import (
    List,
    Optional,
    TypeVar,
    NamedTuple,
    Generic,
    Dict
)
T = TypeVar('T')

from numpy as np


from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.statistics.stats_def import (
    BinRange,
    NumericStatistics,
    FeatureStatistics,
    DatasetStatistics, Feature,
)
from nvflare.app_common.statistics.stats_def import Histogram, HistogramType, Bin, DataType


def get_client_dataset_stats(client_metrics: dict,
                             client_data_types: Dict[str, Dict[str, List[Feature]]]) -> \
                             List[DatasetStatistics]:
    results = []
    for client_name in client_metrics:
        metrics : Dict[str, Dict[str, Generic[T] ]] = client_metrics[client_name]
        feature_data_types : Dict[str, List[Feature]] = client_data_types[client_name]
        for ds_name in feature_data_types:
            dataset_name = f"{client_name}_{ds_name}"
            feature_types : List[Feature] = feature_data_types[ds_name]
            feature_counts = metrics[StC.STATS_COUNT][ds_name]
            feature_means = metrics[StC.STATS_MEAN][ds_name]
            feature_sums = metrics[StC.STATS_SUM][ds_name]
            feature_stddev = metrics[StC.STATS_STDDEV][ds_name]
            feature_histograms = metrics[StC.STATS_HISTOGRAM][ds_name]
            feature_stats = []
            for f in feature_types:
                numeric_stats = \
                        NumericStatistics(
                            feature_means[f.feature_name],
                            feature_sums[f.feature_name],
                            feature_stddev[f.feature_name],
                            feature_counts[f.feature_name],
                            feature_histograms[f.feature_name]
                        )
                feature_stats.append(FeatureStatistics(f.feature_name,f.data_type, numeric_stats))

            # assume all features have the same count
            sample_count = feature_stats[0].num_stats.count
            results.append(DatasetStatistics(dataset_name, sample_count, feature_stats))

        return results


def get_global_dataset_stats(ds_name_prefix: str, metrics: dict, feature_data_types: dict) -> List[DatasetStatistics]:
    results = []
    for ds_name in feature_data_types:
        dataset_name = f"{ds_name_prefix}_{ds_name}"
        feature_types = feature_data_types[ds_name]
        feature_means = metrics[StC.STATS_MEAN][ds_name]
        feature_counts = metrics[StC.STATS_COUNT][ds_name]
        feature_sums = metrics[StC.STATS_SUM][ds_name]
        feature_stddev = metrics[StC.STATS_STDDEV][ds_name]
        feature_histograms = metrics[StC.STATS_HISTOGRAM][ds_name]
        feature_stats = []
        for f in feature_types:
            numeric_stats = \
                NumericStatistics(
                    feature_means[f],
                    feature_sums[f],
                    feature_stddev[f],
                    feature_counts[f],
                    feature_histograms[f]
                )
            feature_stats.append(FeatureStatistics(f, feature_types[f], numeric_stats))

        # all features should have the sample count
        sample_count = feature_stats[0].num_stats.count
        results.append(DatasetStatistics(dataset_name, sample_count, feature_stats))
    return results


def get_global_feature_data_types(client_feature_dts: Dict[str, Dict[str, List[Feature]]]) ->\
                                  Dict[str, Dict[str, DataType]]:
    global_feature_data_types = {}
    for client_name in client_feature_dts:
        ds_features: Dict[str,List[Feature]] = client_feature_dts[client_name]
        for ds_name in ds_features:
            global_feature_data_types[ds_name] = {}
            features = ds_features[ds_name]
            for f in features:
                if f.feature_name not in global_feature_data_types:
                    global_feature_data_types[ds_name][f.feature_name] = f.data_type

    return global_feature_data_types


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
        elif metric == StC.STATS_MAX:
            for client_name in stats:
                global_metrics[metric] = get_metric_values(stats[client_name], global_metrics[metric], max)
        elif metric == StC.STATS_MIN:
            for client_name in stats:
                global_metrics[metric] = get_metric_values(stats[client_name], global_metrics[metric], min)
        elif metric == StC.STATS_HISTOGRAM:
            for client_name in stats:
                global_metrics[metric] = accumulate_hists(stats[client_name], global_metrics[metric])
        elif metric == StC.STATS_VAR:
            for client_name in stats:
                global_metrics[metric] = accumulate_metrics(stats[client_name], global_metrics[metric])
        elif metric == StC.STATS_STDDEV:
            ds_vars = global_metrics[StC.STATS_VAR]
            ds_stddev = {}
            for ds_name in ds_vars:
                ds_stddev[ds_name] = {}
                feature_vars = ds_vars[ds_name]
                for feature in feature_vars:
                    ds_stddev[ds_name][feature] = sqrt(feature_vars[feature])

                global_metrics[StC.STATS_STDDEV] = ds_stddev

    return global_metrics


def accumulate_metrics(metrics: Dict[str, Dict[str, int]],
                       global_metrics: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    for ds_name in metrics:
        feature_metrics = metrics[ds_name]
        for feature_name in feature_metrics:
            global_metrics[ds_name][feature_name] += feature_metrics[feature_name]
    return global_metrics


def get_metric_values(metrics: Dict[str, Dict[str, int]],
                   global_metrics: Dict[str, Dict[str, int]],
                   fn2) -> Dict[str, Dict[str, int]]:
    """
        use 2 argument function to calculate fn2(global, client), example, min, max
    :param metrics:
    :param global_metrics:
    :param fn2:
    :return:
    """

    for ds_name in metrics:
        feature_metrics = metrics[ds_name]
        for feature_name in feature_metrics:
            global_metrics[ds_name][feature_name] = fn2(global_metrics[ds_name][feature_name], feature_metrics[feature_name])
    return global_metrics

def bins_to_dict(bins: List[Bin]) -> Dict[BinRange, float]:
    buckets = {}
    for bucket in bins:
        bucket_range = BinRange(bucket.low_value, bucket.high_value)
        buckets[bucket_range] = bucket.sample_count
    return buckets


def accumulate_hists(metrics: Dict[str, Dict[str, Histogram]],
                     global_hists: Dict[str, Dict[str, Histogram]]) -> Dict[str, Dict[str,Histogram]]:
    for ds_name in metrics:
        feature_hists = metrics[ds_name]
        if ds_name not in global_hists:
            global_hists[ds_name] = {}

        for feature in feature_hists:
            hist: Histogram = feature_hists[feature]
            if feature not in global_hists:
                g_bins = []
                for bucket in hist.bins:
                    g_bins.append(Bin(bucket.low_value, bucket.high_value, bucket.sample_count))
                g_hist = Histogram(g_bins, HistogramType.STANDARD)
                global_hists[ds_name][feature] = g_hist
            else:
                g_hist = global_hists[ds_name][feature]
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
                    bin_range = BinRange(gb.low_value, gb.high_value)
                    updated_bins.append(Bin(gb.low_value, gb.high_value, g_buckets[bin_range]))

                global_hists[ds_name][feature] = Histogram(updated_bins, g_hist.hist_type)

    return global_hists


def get_means(sums: Dict[str, Dict[str, float]], counts: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    means = {}
    for ds_name in sums:
        means[ds_name] = {}
        feature_sums = sums[ds_name]
        feature_counts = counts[ds_name]
        for feature in feature_sums:
            means[ds_name][feature] = feature_sums[feature] / feature_counts[feature]

    return means

def filter_numeric_features(ds_features: Dict[str, List[Feature]] ) -> Dict[str, List[Feature]]:
    numeric_ds_features = {}
    for ds_name in ds_features:
        features: List[Feature] = numeric_ds_features[ds_name]
        n_features = [f for f in features if (f.data_type == DataType.INT or f.data_type == DataType.FLOAT)]
        numeric_ds_features[ds_name] = n_features

    return numeric_ds_features

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
