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
from typing import Dict, List, Optional, Set, TypeVar

from nvflare.app_common.abstract.statistics_spec import Bin, BinRange, DataType, Feature, Histogram, HistogramType
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.statistics.numeric_stats import exclude_withheld_features

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
                    # Union over the sites: a site that reports no features for a dataset must not erase the
                    # features the other sites report.
                    stats[metric].setdefault(ds, {})
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


def get_global_stats(
    global_metrics: dict,
    client_metrics: dict,
    metric_task: str,
    hierarchy_config: dict,
    withheld_features: Optional[Dict[str, Dict[str, Set[str]]]] = None,
) -> dict:
    """Get global hierarchical statistics for the given hierarchy config.

    Args:
        global_metrics: The current global metrics.
        client_metrics: Local stats for each client.
        metric_task: Statistics task.
        hierarchy_config: Hierarchy configuration for the global stats.
        withheld_features: The features whose statistics a client's privacy filter withheld, as
            ``{client: {dataset: {feature, ...}}}``. Such a client contributes nothing to those features at any
            level and has no Local entry for them.

    Returns:
        A dict containing global hierarchical statistics.
    """
    withheld_features = withheld_features or {}
    client_metrics = remove_withheld_features(client_metrics, withheld_features)

    # create stats structure
    ordered_target_metrics = StC.ordered_statistics[metric_task]
    ordered_metrics = [metric for metric in ordered_target_metrics if metric in client_metrics]

    # Create hierarchical output structure
    if StC.GLOBAL not in global_metrics:
        global_metrics = get_output_structure(client_metrics, metric_task, ordered_metrics, hierarchy_config)
    else:
        check_not_withheld_after_contributing(global_metrics, withheld_features)
        update_output_strcture(client_metrics, metric_task, ordered_metrics, global_metrics)
    remove_unreported_from_output(global_metrics, client_metrics, ordered_metrics)

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
            # The clients' var results are taken around the top level mean and count, so their sum is
            # only the var of the top level. Recompute every other level from the sites' own stddev.
            global_metrics = get_hierarchical_vars(global_metrics, client_metrics)
        elif metric == StC.STATS_STDDEV:
            global_metrics = get_hierarchical_stddevs(global_metrics)

    return global_metrics


def remove_withheld_features(client_metrics: dict, withheld_features: Dict[str, Dict[str, Set[str]]]) -> dict:
    """Return the client metrics without the features their privacy filter withheld."""
    if not withheld_features:
        return client_metrics

    return {
        metric: {
            client: exclude_withheld_features(values, withheld_features.get(client))
            for client, values in clients.items()
        }
        for metric, clients in client_metrics.items()
    }


def remove_unreported_from_output(global_metrics, client_metrics: dict, ordered_metrics: List[str]) -> List[str]:
    """Drop unreported placeholders and return the names of the sites under the node.

    The initial structure seeds every feature at every node. Keep an entry only if that site's filtered
    results, or a site's results under that level, contain it. A missing feature/dataset and a privacy-withheld
    feature both contribute nothing. Only prune the current round's metrics; retain the first-round aggregates
    when preparing the second round. Empty dataset dictionaries are retained for the derived-statistics loops.
    """
    sites = []
    if isinstance(global_metrics, list):
        for item in global_metrics:
            sites.extend(remove_unreported_from_output(item, client_metrics, ordered_metrics))
        return sites
    if not isinstance(global_metrics, dict):
        return sites

    for value in global_metrics.values():
        if isinstance(value, list):
            sites.extend(remove_unreported_from_output(value, client_metrics, ordered_metrics))

    if StC.LOCAL in global_metrics:  # a site
        sites = [global_metrics[StC.NAME]]
        node_metrics = global_metrics[StC.LOCAL]
    elif StC.GLOBAL in global_metrics:
        node_metrics = global_metrics[StC.GLOBAL]
    else:
        return sites

    for metric in ordered_metrics:
        reported = {}
        clients = client_metrics[metric]
        # Top-level aggregation includes every reporting client, even clients not listed in the hierarchy.
        contributing_sites = sites if StC.NAME in global_metrics else clients
        for site in contributing_sites:
            for ds_name, features in clients.get(site, {}).items():
                reported.setdefault(ds_name, set()).update(features)
        for ds_name, features in node_metrics[metric].items():
            node_metrics[metric][ds_name] = {
                feature: value for feature, value in features.items() if feature in reported.get(ds_name, ())
            }
    return sites


def check_not_withheld_after_contributing(global_metrics, withheld_features: Dict[str, Dict[str, Set[str]]]) -> None:
    """Reject a site whose statistics were withheld only after it contributed to the 1st round aggregates.

    The privacy filter judges the count minus the failure count, which can change between the rounds (for
    example when records fail to load while the 2nd round runs). The 1st round aggregates already include such
    a site and cannot be undone, while the 2nd round results cannot include it, so the global statistics of the
    feature would no longer describe the same records. A site withheld in the 1st round has no Local count.
    """
    if isinstance(global_metrics, list):
        for item in global_metrics:
            check_not_withheld_after_contributing(item, withheld_features)
    elif isinstance(global_metrics, dict):
        if StC.LOCAL in global_metrics:
            site = global_metrics.get(StC.NAME)
            local_counts = global_metrics[StC.LOCAL].get(StC.STATS_COUNT, {})
            for ds_name, features in withheld_features.get(site, {}).items():
                for feature in sorted(features):
                    if feature in local_counts.get(ds_name, {}):
                        raise ValueError(
                            f"The statistics of feature '{feature}' in dataset '{ds_name}' from site '{site}' were "
                            "withheld by its privacy filter in the 2nd statistics round, after the site contributed "
                            "to the 1st round. The count minus the failure count dropped below the minimum count, "
                            "for example because records failed to load, so the global statistics of the feature "
                            "would be inconsistent."
                        )
        for value in global_metrics.values():
            if isinstance(value, list):
                check_not_withheld_after_contributing(value, withheld_features)


def accumulate_hierarchical_metrics(
    metric: str, client_name: str, metrics: dict, global_metrics: dict, hierarchy_config: dict
) -> dict:
    """Accumulate metrics at each hierarchical level.

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
                        # The client belongs to this org so update current global metrics before sending it further
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
                        # The client belongs to this org so update current global metrics before sending it further
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
                if key == StC.GLOBAL and feature in global_metrics[StC.GLOBAL][StC.STATS_COUNT][dataset]:
                    global_metrics[StC.GLOBAL][metric][dataset][feature] = (
                        global_metrics[StC.GLOBAL][StC.STATS_SUM][dataset][feature]
                        / global_metrics[StC.GLOBAL][StC.STATS_COUNT][dataset][feature]
                    )
                if key == StC.LOCAL and feature in global_metrics[StC.LOCAL][StC.STATS_COUNT][dataset]:
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
                        # The client belongs to this org so update current global metrics before sending it further
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


def get_hierarchical_vars(global_metrics: dict, client_metrics: dict) -> dict:
    """Calculate vars at each hierarchical level below the top level.

    The top level var is left as accumulated from the clients' 'variance_with_mean' results. Every other level
    is pooled from the count n_i and mean m_i (from the 1st statistics round) and the stddev s_i of each site
    under it::

        var = sum((n_i - 1) * s_i^2 + n_i * (m_i - m)^2) / (n - 1)

    where n and m are the count and mean of the level. A site's Local var is s_i^2. A site with a single record
    adds nothing to the first term (its stddev is NaN) and a site without records is skipped.

    Args:
        global_metrics: The current global metrics.
        client_metrics: Local stats for each client, which must include the clients' stddev.

    Returns:
        A dict containing updated hierarchical global statistics with
        pooled vars.
    """

    def get_site_stddev(site: str, dataset: str, feature: str) -> float:
        stddev = client_metrics.get(StC.STATS_STDDEV, {}).get(site, {}).get(dataset, {}).get(feature)
        if stddev is None:
            raise ValueError(
                f"Missing stddev of feature '{feature}' in dataset '{dataset}' from site '{site}'. "
                "Hierarchical var is computed from each site's stddev, so every site must return it "
                "in the 2nd statistics round and the Statistics generator must implement stddev()."
            )
        return stddev

    def recursively_update_org_vars(global_metrics: dict, dataset: str, feature: str) -> list:
        # Returns (count, mean, sum of squared deviations from its own mean) for each site under this level.
        sites = []
        if isinstance(global_metrics, dict):
            if StC.LOCAL in global_metrics:
                local_metrics = global_metrics[StC.LOCAL]
                if feature not in local_metrics[StC.STATS_COUNT][dataset]:
                    return sites  # the site's statistics for this feature were withheld
                count = local_metrics[StC.STATS_COUNT][dataset][feature]
                if not count:
                    local_metrics[StC.STATS_VAR][dataset][feature] = float("nan")
                    return sites
                stddev = get_site_stddev(global_metrics[StC.NAME], dataset, feature)
                local_metrics[StC.STATS_VAR][dataset][feature] = stddev * stddev
                squares = (count - 1) * stddev * stddev if count > 1 else 0.0
                sites.append((count, local_metrics[StC.STATS_MEAN][dataset][feature], squares))
                return sites

            for value in global_metrics.values():
                if isinstance(value, list):
                    for item in value:
                        sites.extend(recursively_update_org_vars(item, dataset, feature))

            if (
                StC.NAME in global_metrics
                and StC.GLOBAL in global_metrics
                and feature in global_metrics[StC.GLOBAL][StC.STATS_COUNT][dataset]
            ):
                level_metrics = global_metrics[StC.GLOBAL]
                count = level_metrics[StC.STATS_COUNT][dataset][feature]
                mean = level_metrics[StC.STATS_MEAN][dataset][feature]
                if count > 1:
                    squares = sum(s + n * (m - mean) * (m - mean) for n, m, s in sites)
                    level_metrics[StC.STATS_VAR][dataset][feature] = squares / (count - 1)
                else:
                    level_metrics[StC.STATS_VAR][dataset][feature] = float("nan")
        return sites

    for dataset in global_metrics[StC.GLOBAL][StC.STATS_VAR]:
        for feature in global_metrics[StC.GLOBAL][StC.STATS_VAR][dataset]:
            recursively_update_org_vars(global_metrics, dataset, feature)

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
                if key == StC.GLOBAL and feature in global_metrics[StC.GLOBAL][StC.STATS_VAR][dataset]:
                    global_metrics[StC.GLOBAL][StC.STATS_STDDEV][dataset][feature] = sqrt(
                        global_metrics[StC.GLOBAL][StC.STATS_VAR][dataset][feature]
                    )
                if key == StC.LOCAL and feature in global_metrics[StC.LOCAL][StC.STATS_VAR][dataset]:
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
