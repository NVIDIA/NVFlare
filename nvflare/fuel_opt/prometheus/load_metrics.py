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

import json
import os.path
from typing import Counter, Dict

from prometheus_client import Gauge

metric_types = {
    "gauge": Gauge,
    "counter": Counter,
}


def load_metrics_config(config_file) -> Dict:
    """
    Load the metrics configuration from a JSON file and create Gauge objects.
    Metrics definition config is like the following

    {
      "metrics": [
        {
          "name": "my_gauge_1",
          "description": "Description of my gauge 1"
          "metrics_type": "gauge"
        },
        {
          "name": "my_gauge_2",
          "description": "Description of my gauge 2"
          "metrics_type": "gauge"
        }
      ]
    }

    Args:
        config_file (str): Path to the JSON configuration file.

    Returns:
        dict: A dictionary of Gauge objects keyed by their metric names.
    """

    if not os.path.isfile(config_file):
        raise ValueError(f"file: '{config_file}' does not exists")

    with open(config_file) as f:
        metrics_config = json.load(f)

    metrics_store = {}

    for metric in metrics_config["metrics"]:
        prometheus_metric = metric_types.get(metric["metrics_type"], None)
        if prometheus_metric is None:
            raise ValueError(f"unknown metric type:{metric['metrics_type']}")
        metrics_store[metric["name"]] = prometheus_metric(metric["name"], metric["description"])

    return metrics_store
