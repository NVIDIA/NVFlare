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

from typing import Dict

from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.fuel.utils.log_utils import get_module_logger

try:
    from fastdigest import TDigest

    TDIGEST_AVAILABLE = True
except ImportError:
    TDIGEST_AVAILABLE = False


logger = get_module_logger(name="quantile_stats")


def get_quantiles(stats: Dict, statistic_configs: Dict, precision: int):

    logger.info(f"get_quantiles: stats: {TDIGEST_AVAILABLE=}")

    if not TDIGEST_AVAILABLE:
        return {}

    global_digest = {}
    for client_name in stats:
        global_digest = merge_quantiles(stats[client_name], global_digest)

    quantile_config = statistic_configs.get(StC.STATS_QUANTILE)
    return compute_quantiles(global_digest, quantile_config, precision)


def get_target_quantiles(quantile_config: dict, feature_name: str):
    if feature_name in quantile_config:
        percents = quantile_config.get(feature_name)
    elif "*" in quantile_config:
        percents = quantile_config.get("*")
    else:
        raise ValueError(f"feature: {feature_name} target percents are not defined.")

    return percents


def merge_quantiles(metrics: Dict[str, Dict[str, Dict]], g_digest: dict) -> dict:

    if not TDIGEST_AVAILABLE:
        return g_digest

    for ds_name in metrics:
        if ds_name not in g_digest:
            g_digest[ds_name] = {}

        feature_metrics = metrics[ds_name]
        for feature_name in feature_metrics:
            if feature_metrics[feature_name] is not None:
                digest_dict: Dict = feature_metrics[feature_name].get(StC.STATS_DIGEST_COORD)
                feature_digest = TDigest.from_dict(digest_dict)
                if feature_name not in g_digest[ds_name]:
                    g_digest[ds_name][feature_name] = feature_digest
                else:
                    g_digest[ds_name][feature_name] = g_digest[ds_name][feature_name].merge(feature_digest)

    return g_digest


def compute_quantiles(g_digest: dict, quantile_config: Dict, precision: int) -> Dict:
    g_ds_metrics = {}
    if not TDIGEST_AVAILABLE:
        return g_digest

    for ds_name in g_digest:
        if ds_name not in g_ds_metrics:
            g_ds_metrics[ds_name] = {}

        feature_metrics = g_digest[ds_name]
        for feature_name in feature_metrics:
            digest = feature_metrics[feature_name]
            percentiles = get_target_quantiles(quantile_config, feature_name)
            quantile_values = {}
            for percentile in percentiles:
                quantile_values[percentile] = round(digest.quantile(percentile), precision)

            g_ds_metrics[ds_name][feature_name] = quantile_values

    return g_ds_metrics
