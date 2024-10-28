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

import copy
from math import sqrt
from typing import Dict, List, TypeVar

from nvflare.app_common.abstract.statistics_spec import Bin, BinRange, DataType, Feature, Histogram, HistogramType
from nvflare.app_common.app_constant import StatisticsConstants as StC

T = TypeVar("T")


def get_initial_structure(client_metrics: dict, ordered_metrics: dict) -> dict:
    """Calculate initial output structure that is common at all the hierarchical levels.

    Args:
        client_metrics: Local stats for each client.
        ordered_metrics: Ordered target metrics.

    Returns:
        A dict containing initial output structure.
    """
    stats = {}
    for metric in ordered_metrics:
        stats[metric] = {}
        for stat in client_metrics:
            for site in client_metrics[stat]:
                for ds in client_metrics[stat][site]:
                    stats[metric][ds] = {}
                    for feature in client_metrics[stat][site][ds]:
                        stats[metric][ds][feature] = 0
    return stats


def create_output_structure(
    client_metrics: dict, metric_task: str, ordered_metrics: dict, hierarchy_config: dict
) -> dict:
    """Recursively calculate the hierarchical global stats structure from the given hierarchy config.

    Args:
        client_metrics: Local stats for each client.
        metric_task: Statistics task.
        ordered_metrics: Ordered target metrics.
        hierarchy_config: Hierarchy configuration for the global stats.

    Returns:
        A dict containing hierarchical global stats structure.
    """

    def recursively_add_values(structure: dict, value_json: dict, metric_task: str, ordered_metrics: dict):
        if isinstance(structure, dict):
            new_items = {}
            for key, value in list(structure.items()):
                if key == StC.NAME:
                    continue
                if isinstance(value, list):
                    if key not in new_items:
                        new_items[StC.GLOBAL] = get_initial_structure(value_json, ordered_metrics)
                    for i, item in enumerate(value):
                        if isinstance(item, str):
                            value[i] = {
                                StC.NAME: item,
                                StC.LOCAL: get_initial_structure(value_json, ordered_metrics),
                            }
                        else:
                            recursively_add_values(item, value_json, metric_task, ordered_metrics)
                else:
                    recursively_add_values(value, value_json, metric_task, ordered_metrics)
            structure.update(new_items)
        elif isinstance(structure, list):
            for item in structure:
                recursively_add_values(item, value_json, metric_task, ordered_metrics)
        return structure

    filled_structure = copy.deepcopy(hierarchy_config)
    final_strcture = recursively_add_values(filled_structure, client_metrics, metric_task, ordered_metrics)
    return final_strcture


def get_output_structure(client_metrics: dict, metric_task: str, ordered_metrics: dict, hierarchy_config: dict) -> dict:
    """Create required global statistics hierarchical output structure.

    Args:
        client_metrics: Local stats for each client.
        metric_task: Statistics task.
        ordered_metrics: Ordered target metrics.
        hierarchy_config: Hierarchy configuration for the global stats.

    Returns:
        A dict containing hierarchical global stats structure that also includes
        top level global stats structure.
    """
    top_strcture = get_initial_structure(client_metrics, ordered_metrics)
    output_structure = {
        StC.GLOBAL: top_strcture,
        **create_output_structure(client_metrics, metric_task, ordered_metrics, hierarchy_config),
    }
    return output_structure


def update_output_strcture(
    client_metrics: dict,
    metric_task: str,
    ordered_metrics: dict,
    global_metrics: dict,
) -> None:
    """Update global statistics hierarchical output structure with the new ordered metrics.

    Args:
        client_metrics: Local stats for each client.
        metric_task: Statistics task.
        ordered_metrics: Ordered target metrics.
        global_metrics: The current global metrics.

    Returns:
        A dict containing updated hierarchical global stats.
    """
    if isinstance(global_metrics, dict):
        for key, value in list(global_metrics.items()):
            if key == StC.NAME:
                continue
            elif key == StC.GLOBAL:
                global_metrics[key].update(get_initial_structure(client_metrics, ordered_metrics))
            elif key == StC.LOCAL:
                global_metrics[key].update(get_initial_structure(client_metrics, ordered_metrics))
                return
            elif isinstance(value, list):
                update_output_strcture(client_metrics, metric_task, ordered_metrics, value)
    elif isinstance(global_metrics, list):
        for item in global_metrics:
            update_output_strcture(client_metrics, metric_task, ordered_metrics, item)


def get_global_stats(global_metrics: dict, client_metrics: dict, metric_task: str, hierarchy_config: dict) -> dict:
    """Get global hierarchical statistics for the given hierarchy config.

    Args:
        global_metrics: The current global metrics.
        client_metrics: Local stats for each client.
        metric_task: Statistics task.
        hierarchy_config: Hierarchy configuration for the global stats.


    Returns:
        A dict containing global hierarchical statistics.
    """
    # create stats structure
    ordered_target_metrics = StC.ordered_statistics[metric_task]
    ordered_metrics = [metric for metric in ordered_target_metrics if metric in client_metrics]

    # Create hierarchical output structure
    if StC.GLOBAL not in global_metrics:
        global_metrics = get_output_structure(client_metrics, metric_task, ordered_metrics, hierarchy_config)
    else:
        update_output_strcture(client_metrics, metric_task, ordered_metrics, global_metrics)

    for metric in ordered_metrics:
        stats = client_metrics[metric]
        if metric == StC.STATS_COUNT or metric == StC.STATS_FAILURE_COUNT or metric == StC.STATS_SUM:
            for client_name in stats:
                global_metrics = accumulate_hierarchical_metrics(
                    metric, client_name, stats[client_name], global_metrics, hierarchy_config
                )
        elif metric == StC.STATS_MAX or metric == StC.STATS_MIN:
            for client_name in stats:
                global_metrics = get_hierarchical_mins_or_maxs(
                    metric, client_name, stats[client_name], global_metrics, hierarchy_config
                )
        elif metric == StC.STATS_MEAN:
            global_metrics = get_hierarchical_means(metric, global_metrics)
        elif metric == StC.STATS_HISTOGRAM:
            for client_name in stats:
                global_metrics = get_hierarchical_histograms(
                    metric, client_name, stats[client_name], global_metrics, hierarchy_config
                )
        elif metric == StC.STATS_VAR:
            for client_name in stats:
                global_metrics = accumulate_hierarchical_metrics(
                    metric, client_name, stats[client_name], global_metrics, hierarchy_config
                )
        elif metric == StC.STATS_STDDEV:
            global_metrics = get_hierarchical_stddevs(global_metrics)

    return global_metrics


def accumulate_hierarchical_metrics(
    metric: str, client_name: str, metrics: dict, global_metrics: dict, hierarchy_config: dict
) -> dict:
    """Accumulate matrics at each hierarchical level.

    Args:
        metric: Metric to accumulate.
        client_name: Client name.
        metrics: Client metrics.
        global_metrics: The current global metrics.
        hierarchy_config:  Hierarchy configuration for the global stats.

    Returns:
        A dict containing accumulated hierarchical global statistics.
    """

    def recursively_accumulate_hierarchical_metrics(
        metric: str, client_name: str, metrics: dict, global_metrics: dict, dataset: str, feature: str, org: list
    ) -> dict:
        if isinstance(global_metrics, dict):
            for key, value in global_metrics.items():
                if key == StC.GLOBAL and StC.NAME not in global_metrics:
                    global_metrics[StC.GLOBAL][metric][dataset][feature] += metrics[dataset][feature]
                    continue
                if key == StC.NAME:
                    if org and value in org:
                        # The client belongs to this org so update current global matrics before sending it further
                        global_metrics[StC.GLOBAL][metric][dataset][feature] += metrics[dataset][feature]
                    elif value == client_name:
                        # This is a client local metrics update
                        global_metrics[StC.LOCAL][metric][dataset][feature] += metrics[dataset][feature]
                    else:
                        break
                if isinstance(value, list):
                    for item in value:
                        recursively_accumulate_hierarchical_metrics(
                            metric, client_name, metrics, item, dataset, feature, org
                        )

    client_org = get_client_hierarchy(copy.deepcopy(hierarchy_config), client_name)
    for dataset in metrics:
        for feature in metrics[dataset]:
            recursively_accumulate_hierarchical_metrics(
                metric, client_name, metrics, global_metrics, dataset, feature, client_org
            )

    return global_metrics


def get_hierarchical_mins_or_maxs(
    metric: str, client_name: str, metrics: dict, global_metrics: dict, hierarchy_config: dict
) -> dict:
    """Calculate min or max at each hierarchical level.

    Args:
        metric: Metric to accumulate.
        client_name: Client name.
        metrics: Client metrics.
        global_metrics: The current global metrics.
        hierarchy_config:  Hierarchy configuration for the global stats.

    Returns:
        A dict containing updated hierarchical global statistics with
        accumulated mins or maxs.
    """

    def recursively_update_org_mins_or_maxs(
        metric: str,
        client_name: str,
        metrics: dict,
        global_metrics: dict,
        dataset: str,
        feature: str,
        org: list,
        op: str,
    ) -> dict:
        if isinstance(global_metrics, dict):
            for key, value in global_metrics.items():
                if key == StC.GLOBAL and StC.NAME not in global_metrics:
                    if global_metrics[StC.GLOBAL][metric][dataset][feature]:
                        global_metrics[StC.GLOBAL][metric][dataset][feature] = op(
                            global_metrics[StC.GLOBAL][metric][dataset][feature], metrics[dataset][feature]
                        )
                    else:
                        global_metrics[StC.GLOBAL][metric][dataset][feature] = metrics[dataset][feature]
                    continue
                if key == StC.NAME:
                    if org and value in org:
                        # The client belongs to this org so update current global matrics before sending it further
                        if global_metrics[StC.GLOBAL][metric][dataset][feature]:
                            global_metrics[StC.GLOBAL][metric][dataset][feature] = op(
                                global_metrics[StC.GLOBAL][metric][dataset][feature], metrics[dataset][feature]
                            )
                        else:
                            global_metrics[StC.GLOBAL][metric][dataset][feature] = metrics[dataset][feature]
                    elif value == client_name:
                        # This is a client local metrics update
                        global_metrics[StC.LOCAL][metric][dataset][feature] = metrics[dataset][feature]
                    else:
                        break
                if isinstance(value, list):
                    for item in value:
                        recursively_update_org_mins_or_maxs(
                            metric, client_name, metrics, item, dataset, feature, org, op
                        )

    if metric == "min":
        op = min
    else:
        op = max
    client_org = get_client_hierarchy(copy.deepcopy(hierarchy_config), client_name)
    for dataset in metrics:
        for feature in metrics[dataset]:
            recursively_update_org_mins_or_maxs(
                metric, client_name, metrics, global_metrics, dataset, feature, client_org, op
            )

    return global_metrics


def get_hierarchical_means(metric: str, global_metrics: dict) -> dict:
    """Calculate means at each hierarchical level.

    Args:
        metric: Metric to accumulate.
        global_metrics: The current global metrics.

    Returns:
        A dict containing updated hierarchical global statistics with
        accumulated means.
    """

    def recursively_update_org_means(metrics: dict, global_metrics: dict, dataset: str, feature: str) -> dict:
        if isinstance(global_metrics, dict):
            for key, value in global_metrics.items():
                if key == StC.GLOBAL:
                    global_metrics[StC.GLOBAL][metric][dataset][feature] = (
                        global_metrics[StC.GLOBAL][StC.STATS_SUM][dataset][feature]
                        / global_metrics[StC.GLOBAL][StC.STATS_COUNT][dataset][feature]
                    )
                if key == StC.LOCAL:
                    global_metrics[StC.LOCAL][metric][dataset][feature] = (
                        global_metrics[StC.LOCAL][StC.STATS_SUM][dataset][feature]
                        / global_metrics[StC.LOCAL][StC.STATS_COUNT][dataset][feature]
                    )
                if isinstance(value, list):
                    for item in value:
                        recursively_update_org_means(metrics, item, dataset, feature)

    #  Iterate each hierarchical level and calculate 'mean' from 'sum' and 'count'.
    for dataset in global_metrics[StC.GLOBAL][StC.STATS_COUNT]:
        for feature in global_metrics[StC.GLOBAL][StC.STATS_COUNT][dataset]:
            recursively_update_org_means(metric, global_metrics, dataset, feature)

    return global_metrics


def get_hierarchical_histograms(
    metric: str, client_name: str, metrics: dict, global_metrics: dict, hierarchy_config: dict
) -> dict:
    """Calculate histograms at each hierarchical level.

    Args:
        metric: Metric to accumulate.
        client_name: Client name.
        metrics: Client metrics.
        global_metrics: The current global metrics.
        hierarchy_config:  Hierarchy configuration for the global stats.

    Returns:
        A dict containing updated hierarchical global statistics with
        accumulated histograms.
    """

    def recursively_accumulate_org_histograms(
        metric: str,
        client_name: str,
        metrics: dict,
        global_metrics: dict,
        dataset: str,
        feature: str,
        org: list,
        histogram: dict,
    ) -> dict:
        if isinstance(global_metrics, dict):
            for key, value in global_metrics.items():
                if key == StC.GLOBAL and StC.NAME not in global_metrics:
                    if (
                        feature not in global_metrics[StC.GLOBAL][metric][dataset]
                        or not global_metrics[StC.GLOBAL][metric][dataset][feature]
                    ):
                        g_bins = []
                        for bucket in histogram.bins:
                            g_bins.append(Bin(bucket.low_value, bucket.high_value, bucket.sample_count))
                        g_hist = Histogram(HistogramType.STANDARD, g_bins)
                        global_metrics[StC.GLOBAL][metric][dataset][feature] = g_hist
                    else:
                        g_hist = global_metrics[StC.GLOBAL][metric][dataset][feature]
                        g_buckets = bins_to_dict(g_hist.bins)
                        for bucket in histogram.bins:
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
                        global_metrics[StC.GLOBAL][metric][dataset][feature] = Histogram(g_hist.hist_type, updated_bins)
                    continue
                if key == StC.NAME:
                    if org and value in org:
                        # The client belongs to this org so update current global matrics before sending it further
                        if (
                            feature not in global_metrics[StC.GLOBAL][metric][dataset]
                            or not global_metrics[StC.GLOBAL][metric][dataset][feature]
                        ):
                            g_bins = []
                            for bucket in histogram.bins:
                                g_bins.append(Bin(bucket.low_value, bucket.high_value, bucket.sample_count))
                            g_hist = Histogram(HistogramType.STANDARD, g_bins)
                            global_metrics[StC.GLOBAL][metric][dataset][feature] = g_hist
                        else:
                            g_hist = global_metrics[StC.GLOBAL][metric][dataset][feature]
                            g_buckets = bins_to_dict(g_hist.bins)
                            for bucket in histogram.bins:
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
                            global_metrics[StC.GLOBAL][metric][dataset][feature] = Histogram(
                                g_hist.hist_type, updated_bins
                            )
                    elif value == client_name:
                        # This is a client local metrics update
                        if (
                            feature not in global_metrics[StC.LOCAL][metric][dataset]
                            or not global_metrics[StC.LOCAL][metric][dataset][feature]
                        ):
                            g_bins = []
                            for bucket in histogram.bins:
                                g_bins.append(Bin(bucket.low_value, bucket.high_value, bucket.sample_count))
                            g_hist = Histogram(HistogramType.STANDARD, g_bins)
                            global_metrics[StC.LOCAL][metric][dataset][feature] = g_hist
                        else:
                            g_hist = global_metrics[StC.LOCAL][metric][dataset][feature]
                            g_buckets = bins_to_dict(g_hist.bins)
                            for bucket in histogram.bins:
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
                            global_metrics[StC.LOCAL][metric][dataset][feature] = Histogram(
                                g_hist.hist_type, updated_bins
                            )
                    else:
                        break
                if isinstance(value, list):
                    for item in value:
                        recursively_accumulate_org_histograms(
                            metric, client_name, metrics, item, dataset, feature, org, histogram
                        )

    client_org = get_client_hierarchy(copy.deepcopy(hierarchy_config), client_name)
    for dataset in metrics:
        for feature in metrics[dataset]:
            histogram = metrics[dataset][feature]
            recursively_accumulate_org_histograms(
                metric, client_name, metrics, global_metrics, dataset, feature, client_org, histogram
            )

    return global_metrics


def get_hierarchical_stddevs(global_metrics: dict) -> dict:
    """Calculate stddevs at each hierarchical level.

    Args:
        global_metrics: The current global metrics.

    Returns:
        A dict containing updated hierarchical global statistics with
        accumulated stddevs.
    """

    def recursively_update_org_stddevs(global_metrics: dict, dataset: str, feature: str) -> dict:
        if isinstance(global_metrics, dict):
            for key, value in global_metrics.items():
                if key == StC.GLOBAL:
                    global_metrics[StC.GLOBAL][StC.STATS_STDDEV][dataset][feature] = sqrt(
                        global_metrics[StC.GLOBAL][StC.STATS_VAR][dataset][feature]
                    )
                if key == StC.LOCAL:
                    global_metrics[StC.LOCAL][StC.STATS_STDDEV][dataset][feature] = sqrt(
                        global_metrics[StC.LOCAL][StC.STATS_VAR][dataset][feature]
                    )
                if isinstance(value, list):
                    for item in value:
                        recursively_update_org_stddevs(item, dataset, feature)

    for dataset in global_metrics[StC.GLOBAL][StC.STATS_VAR]:
        for feature in global_metrics[StC.GLOBAL][StC.STATS_VAR][dataset]:
            recursively_update_org_stddevs(global_metrics, dataset, feature)

    return global_metrics


def get_hierarchical_levels(data: dict, level: int = 0, levels_dict: dict = None) -> dict:
    """Calculate number of hierarchical levels from the given hierarchy config.

    Args:
        data: Hierarchy configuration for the global stats.
        level: The current hierarchical level (used for recursive calls).
        levels_dict: The accumulated levels dict (used for recursive calls).

    Returns:
        A dict containing containing hierarchical levels.
    """
    if levels_dict is None:
        levels_dict = {}

    if isinstance(data, list):
        for item in data:
            get_hierarchical_levels(item, level, levels_dict)
    elif isinstance(data, dict):
        for key, value in data.items():
            if key == StC.NAME:
                continue
            if key not in levels_dict:
                levels_dict[key] = level
            get_hierarchical_levels(value, level + 1, levels_dict)

    return levels_dict


def get_client_hierarchy(hierarchy_config: dict, client_name: str, path=None) -> list:
    """Calculate hierarchy for the given client name.

    Args:
        hierarchy_config: Hierarchy configuration for the global stats.
        client_name: Client name.
        path: The accumulated hierarchy path (used for recursive calls).

    Returns:
        A list containing hierarchy levels for the client.
    """
    if path is None:
        path = []

    if isinstance(hierarchy_config, dict):
        for key, value in hierarchy_config.items():
            if isinstance(value, list):
                result = get_client_hierarchy(value, client_name, path)
                if result:
                    return result
    elif isinstance(hierarchy_config, list):
        for item in hierarchy_config:
            if item == client_name:
                return path
            if isinstance(item, dict):
                result = get_client_hierarchy(item, client_name, path + [item.get(StC.NAME)])
                if result:
                    return result

    return None


def bins_to_dict(bins: List[Bin]) -> Dict[BinRange, float]:
    """Convert histogram bins to a 'dict'.

    Args:
        bins: Histogram bins.

    Returns:
        A dict containing histogram bins.
    """
    buckets = {}
    for bucket in bins:
        bucket_range = BinRange(bucket.low_value, bucket.high_value)
        buckets[bucket_range] = bucket.sample_count
    return buckets


def filter_numeric_features(ds_features: Dict[str, List[Feature]]) -> Dict[str, List[Feature]]:
    """Filter numeric features.

    Args:
        ds_features: A features dict.

    Returns:
        A dict containing numeric features.
    """
    numeric_ds_features = {}
    for ds_name in ds_features:
        features: List[Feature] = ds_features[ds_name]
        n_features = [f for f in features if (f.data_type == DataType.INT or f.data_type == DataType.FLOAT)]
        numeric_ds_features[ds_name] = n_features

    return numeric_ds_features
