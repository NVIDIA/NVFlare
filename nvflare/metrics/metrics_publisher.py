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

import time
from typing import Optional

from nvflare.apis.fl_constant import ReservedTopic
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.metrics.metrics_keys import MetricKeys


def publish_app_metrics(
    metrics: dict, metric_name: str, labels: dict, data_bus: DataBus, timestamp: Optional[int] = None
) -> None:
    metrics_data = []

    for key in metrics:
        metrics_value = metrics.get(key)
        metrics_data.append(
            {
                MetricKeys.metric_name: f"{metric_name}_{key}" if metric_name else key,
                MetricKeys.value: metrics_value,
                MetricKeys.labels: {} if labels is None else labels,
                MetricKeys.timestamp: int(time.time() if timestamp is None else timestamp),
            }
        )
    data_bus.publish([ReservedTopic.APP_METRICS], metrics_data)
