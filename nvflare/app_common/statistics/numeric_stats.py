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

from math import sqrt
from typing import Dict, List, TypeVar

from nvflare.app_common.abstract.statistics_spec import Bin, BinRange, DataType, Feature, Histogram, HistogramType
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_opt.statistics.quantile_stats import get_quantiles
from nvflare.fuel.utils.log_utils import get_module_logger

T = TypeVar("T")

logger = get_module_logger(name=__name__)


def get_global_feature_data_types(
    client_feature_dts: Dict[str, Dict[str, List[Feature]]],
) -> Dict[str, Dict[str, DataType]]:
    global_feature_data_types = {}
    for client_name in client_feature_dts:
        ds_features: Dict[str, List[Feature]] = client_feature_dts[client_name]
        for ds_name in ds_features:
            global_feature_data_types[ds_name] = {}
            features = ds_features[ds_name]
            for f in features:
                if f.feature_name not in global_feature_data_types:
                    global_feature_data_types[ds_name][f.feature_name] = f.data_type

    return global_feature_data_types


def get_global_stats(
    global_metrics: dict, client_metrics: dict, metric_task: str, statistic_configs: Dict[str, dict], precision: int = 4
) -> dict:
    # we need to calculate the metrics in specified order
    ordered_target_metrics = StC.ordered_statistics[metric_task]
    ordered_metrics = [metric for metric in ordered_target_metrics if metric in client_metrics]

    for metric in ordered_metrics:
        if metric not in global_metrics:
            global_metrics[metric] = {}

        stats = client_metrics[metric]
        if metric == StC.STATS_COUNT or metric == StC.STATS_FAILURE_COUNT or metric == StC.STATS_SUM:
            for client_name in stats:
                global_metrics[metric] = accumulate_metrics(stats[client_name], global_metrics[metric], precision)
        elif metric == StC.STATS_MEAN:
            global_metrics[metric] = get_means(
                global_metrics[StC.STATS_SUM], global_metrics[StC.STATS_COUNT], precision
            )
        elif metric == StC.STATS_MAX:
            for client_name in stats:
                global_metrics[metric] = get_min_or_max_values(
                    stats[client_name], global_metrics[metric], max, precision
                )
        elif metric == StC.STATS_MIN:
            for client_name in stats:
                global_metrics[metric] = get_min_or_max_values(
                    stats[client_name], global_metrics[metric], min, precision
                )
        elif metric == StC.STATS_HISTOGRAM:
            for client_name in stats:
                global_metrics[metric] = accumulate_hists(stats[client_name], global_metrics[metric])
        elif metric == StC.STATS_VAR:
            for client_name in stats:
                global_metrics[metric] = accumulate_metrics(stats[client_name], global_metrics[metric], precision)
        elif metric == StC.STATS_STDDEV:
            ds_vars = global_metrics[StC.STATS_VAR]
            ds_stddev = {}
            for ds_name in ds_vars:
                ds_stddev[ds_name] = {}
                feature_vars = ds_vars[ds_name]
                for feature in feature_vars:
                    ds_stddev[ds_name][feature] = round(sqrt(feature_vars[feature]), precision)

                global_metrics[StC.STATS_STDDEV] = ds_stddev
        elif metric == StC.STATS_QUANTILE:
            global_metrics[metric] = get_quantiles(stats, statistic_configs, precision)

    return global_metrics


def accumulate_metrics(metrics: dict, global_metrics: dict, precision: int) -> dict:
    for ds_name in metrics:
        if ds_name not in global_metrics:
            global_metrics[ds_name] = {}

        feature_metrics = metrics[ds_name]
        for feature_name in feature_metrics:
            if feature_metrics[feature_name] is not None:
                if feature_name not in global_metrics[ds_name]:
                    global_metrics[ds_name][feature_name] = round(feature_metrics[feature_name], precision)
                else:
                    global_metrics[ds_name][feature_name] = round(
                        global_metrics[ds_name][feature_name] + feature_metrics[feature_name], precision
                    )

    return global_metrics


def get_min_or_max_values(metrics: dict, global_metrics: dict, fn2, precision: int = 4) -> dict:
    """Use 2 argument function to calculate fn2(global, client), for example, min or max.

    .. note::

        The global min/max values are min/max of all clients and all datasets.

    Args:
        metrics: client's metric
        global_metrics: global metrics
        fn2: two-argument function such as min or max
        precision: decimal number precision

    Returns: Dict[dataset, Dict[feature, int]]

    """
    for ds_name in metrics:
        if ds_name not in global_metrics:
            global_metrics[ds_name] = {}

        feature_metrics = metrics[ds_name]
        for feature_name in feature_metrics:
            if feature_name not in global_metrics[ds_name]:
                global_metrics[ds_name][feature_name] = round(feature_metrics[feature_name], precision)
            else:
                global_metrics[ds_name][feature_name] = round(
                    fn2(global_metrics[ds_name][feature_name], feature_metrics[feature_name]), precision
                )

    results = {}
    for ds_name in global_metrics:
        for feature_name in global_metrics[ds_name]:
            if feature_name not in results:
                results[feature_name] = round(global_metrics[ds_name][feature_name], precision)
            else:
                results[feature_name] = round(
                    fn2(results[feature_name], global_metrics[ds_name][feature_name]), precision
                )

    for ds_name in global_metrics:
        for feature_name in global_metrics[ds_name]:
            global_metrics[ds_name][feature_name] = results[feature_name]

    return global_metrics


def bins_to_dict(bins: List[Bin]) -> Dict[BinRange, float]:
    buckets = {}
    for bucket in bins:
        bucket_range = BinRange(bucket.low_value, bucket.high_value)
        buckets[bucket_range] = bucket.sample_count
    return buckets


def accumulate_hists(
    metrics: Dict[str, Dict[str, Histogram]], global_hists: Dict[str, Dict[str, Histogram]], precision: int = 4
) -> Dict[str, Dict[str, Histogram]]:
    for ds_name in metrics:
        feature_hists = metrics[ds_name]
        if ds_name not in global_hists:
            global_hists[ds_name] = {}

        for feature in feature_hists:
            hist: Histogram = feature_hists[feature]
            if feature not in global_hists[ds_name]:
                g_bins = []
                for bucket in hist.bins:
                    g_bins.append(
                        Bin(
                            round(bucket.low_value, precision), round(bucket.high_value, precision), bucket.sample_count
                        )
                    )
                g_hist = Histogram(HistogramType.STANDARD, g_bins)
                global_hists[ds_name][feature] = g_hist
            else:
                g_hist = global_hists[ds_name][feature]
                g_buckets = bins_to_dict(g_hist.bins)
                for bucket in hist.bins:
                    bin_range = BinRange(round(bucket.low_value, precision), round(bucket.high_value, precision))
                    if bin_range in g_buckets:
                        g_buckets[bin_range] += bucket.sample_count
                    else:
                        g_buckets[bin_range] = bucket.sample_count

                # update ordered bins
                updated_bins = []
                for gb in g_hist.bins:
                    bin_range = BinRange(round(gb.low_value, precision), round(gb.high_value, precision))
                    updated_bins.append(
                        Bin(round(gb.low_value, precision), round(gb.high_value, precision), g_buckets[bin_range])
                    )

                global_hists[ds_name][feature] = Histogram(g_hist.hist_type, updated_bins)

    return global_hists


def get_means(sums: dict, counts: dict, precision: int = 4) -> dict:
    means = {}
    for ds_name in sums:
        means[ds_name] = {}
        feature_sums = sums[ds_name]
        feature_counts = counts[ds_name]
        for feature in feature_sums:
            means[ds_name][feature] = round(feature_sums[feature] / feature_counts[feature], precision)
    return means


def filter_numeric_features(ds_features: Dict[str, List[Feature]]) -> Dict[str, List[Feature]]:
    numeric_ds_features = {}
    for ds_name in ds_features:
        features: List[Feature] = ds_features[ds_name]
        n_features = [f for f in features if (f.data_type == DataType.INT or f.data_type == DataType.FLOAT)]
        numeric_ds_features[ds_name] = n_features

    return numeric_ds_features
