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


import json
import os
from typing import Dict, List, Tuple

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.abstract.statistics_spec import Bin, Histogram, StatisticConfig
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.utils.json_utils import ObjectEncoder
from nvflare.fuel.utils import fobs


def save_to_json(ds_stats: dict, output_path):
    parent_dir = os.path.dirname(output_path)
    os.makedirs(parent_dir, exist_ok=True)
    content = json.dumps(ds_stats, cls=ObjectEncoder)
    with open(output_path, "w") as outfile:
        outfile.write(content)


def combine_all_statistics(
    statistic_configs: Dict[str, dict], global_statistics: Dict, client_statistics: Dict, precision: int = 4
):
    result = {}
    filtered_client_statistics = [statistic for statistic in client_statistics if statistic in statistic_configs]
    filtered_global_statistics = [statistic for statistic in global_statistics if statistic in statistic_configs]

    for statistic in filtered_client_statistics:
        for client in client_statistics[statistic]:
            for ds in client_statistics[statistic][client]:
                for feature_name in client_statistics[statistic][client][ds]:
                    if feature_name not in result:
                        result[feature_name] = {}
                    if statistic not in result[feature_name]:
                        result[feature_name][statistic] = {}

                    if client not in result[feature_name][statistic]:
                        result[feature_name][statistic][client] = {}

                    if ds not in result[feature_name][statistic][client]:
                        result[feature_name][statistic][client][ds] = {}

                    if statistic == StC.STATS_HISTOGRAM:
                        hist: Histogram = client_statistics[statistic][client][ds][feature_name]
                        buckets = apply_histogram_precision(hist.bins, precision)

                        result[feature_name][statistic][client][ds] = buckets
                    else:
                        result[feature_name][statistic][client][ds] = round(
                            client_statistics[statistic][client][ds][feature_name], precision
                        )

    for statistic in filtered_global_statistics:
        for ds in global_statistics[statistic]:
            for feature_name in global_statistics[statistic][ds]:
                if StC.GLOBAL not in result[feature_name][statistic]:
                    result[feature_name][statistic][StC.GLOBAL] = {}

                if ds not in result[feature_name][statistic][StC.GLOBAL]:
                    result[feature_name][statistic][StC.GLOBAL][ds] = {}

                if statistic == StC.STATS_HISTOGRAM:
                    hist: Histogram = global_statistics[statistic][ds][feature_name]
                    buckets = apply_histogram_precision(hist.bins, precision)
                    result[feature_name][statistic][StC.GLOBAL][ds] = buckets
                else:
                    result[feature_name][statistic][StC.GLOBAL].update(
                        {ds: round(global_statistics[statistic][ds][feature_name], precision)}
                    )

    return result


def apply_histogram_precision(bins: List[Bin], precision) -> List[Bin]:
    buckets = []
    for bucket in bins:
        buckets.append(
            Bin(
                round(bucket.low_value, precision),
                round(bucket.high_value, precision),
                bucket.sample_count,
            )
        )
    return buckets


def get_target_statistics(statistic_configs: Dict[str, dict], ordered_statistics: list) -> List[StatisticConfig]:
    """
        for given requested statistic configurations, argument the additional metrics needed
    Args:
        statistic_configs: requested statistic configuration, for each metric, there are corresponding metric config
        ordered_statistics: the order of metric sequences specified.

    Returns:

    """
    targets = []
    if statistic_configs:
        for statistic in statistic_configs:
            # if target statistic has histogram, we are not in 2nd statistic task
            # we only need to estimate the global min/max if we have histogram statistic,
            # If the user provided the global min/max for a specified feature, then we do nothing
            # if the user did not provide the global min/max for the feature, then we need to ask
            # client to provide the local estimated min/max for that feature.
            # then we used the local estimate min/max to estimate global min/max.
            # to do that, we calculate the local min/max in 1st statistic task.
            # in all cases, we will still send the STATS_MIN/MAX tasks, but client executor may or may not
            # delegate to stats generator to calculate the local min/max depends on if the global bin ranges
            # are specified. to do this, we send over the histogram configuration when calculate the local min/max
            if statistic == StC.STATS_HISTOGRAM and statistic not in ordered_statistics:
                targets.append(StatisticConfig(StC.STATS_MIN, statistic_configs[StC.STATS_HISTOGRAM]))
                targets.append(StatisticConfig(StC.STATS_MAX, statistic_configs[StC.STATS_HISTOGRAM]))

            if statistic == StC.STATS_STDDEV and statistic in ordered_statistics:
                targets.append(StatisticConfig(StC.STATS_VAR, {}))

            for rm in ordered_statistics:
                if rm == statistic:
                    targets.append(StatisticConfig(statistic, statistic_configs[statistic]))
    return targets


def prepare_inputs(statistic_configs: Dict[str, dict], statistic_task: str, global_statistics) -> Dict:
    inputs = {}
    target_statistics: List[StatisticConfig] = get_target_statistics(
        statistic_configs, StC.ordered_statistics[statistic_task]
    )

    for tm in target_statistics:
        if tm.name == StC.STATS_HISTOGRAM:
            if StC.STATS_MIN in global_statistics:
                inputs[StC.STATS_MIN] = global_statistics[StC.STATS_MIN]
            if StC.STATS_MAX in global_statistics:
                inputs[StC.STATS_MAX] = global_statistics[StC.STATS_MAX]

        if tm.name == StC.STATS_VAR:
            if StC.STATS_COUNT in global_statistics:
                inputs[StC.STATS_GLOBAL_COUNT] = global_statistics[StC.STATS_COUNT]
            if StC.STATS_MEAN in global_statistics:
                inputs[StC.STATS_GLOBAL_MEAN] = global_statistics[StC.STATS_MEAN]

    inputs[StC.STATISTICS_TASK_KEY] = statistic_task

    inputs[StC.STATS_TARGET_STATISTICS] = fobs.dumps(target_statistics)

    return inputs


def get_client_stats(
    client_results: Dict[str, Dict[str, FLModel]], client_statistics: dict, client_features
) -> Tuple[Dict, Dict]:
    for task_name, client_data in client_results.items():
        for client_name, task_data in client_data.items():
            task_results = task_data.params
            statistics_task = task_results[StC.STATISTICS_TASK_KEY]
            statistics = fobs.loads(task_results[statistics_task])
            for statistic in statistics:
                if statistic not in client_statistics:
                    client_statistics[statistic] = {client_name: statistics[statistic]}
                else:
                    client_statistics[statistic].update({client_name: statistics[statistic]})

            ds_features = task_results.get(StC.STATS_FEATURES, None)
            if ds_features:
                client_features.update({client_name: fobs.loads(ds_features)})

    return client_statistics, client_features
