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
import logging
import traceback

from datadog import initialize, statsd

from nvflare.apis.fl_constant import ReservedTopic
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.metrics.metrics_keys import MetricKeys, MetricTypes

# require datalog statsd dependency


class StatsDReporter:
    def __init__(self, site: str = "", host="localhost", port=9125):

        # Initialize the DataDog StatsD client
        initialize(statsd_host=host, statsd_port=port)
        self.metrics = {}
        self.data_bus = DataBus()

        self.data_bus.subscribe([ReservedTopic.APP_METRICS], self.process_metrics)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.site = site

    def process_metrics(self, topic, metrics, data_bus):

        if topic == ReservedTopic.APP_METRICS:
            try:
                for metric in metrics:
                    metric_name = metric.get(MetricKeys.metric_name)
                    metric_value = metric.get(MetricKeys.value)

                    tags = metric.get(MetricKeys.tags, {})
                    metric_tags = []
                    for k, v in tags.items():
                        metric_tags.append(f"{k}:{v}")

                    metric_type = metric.get(MetricKeys.type)
                    metric_timestamp = metric.get(MetricKeys.timestamp)

                    if metric_type == MetricTypes.COUNTER:
                        statsd.increment(metric_name, metric_value, tags=metric_tags)

                    elif metric_type == MetricTypes.GAUGE:
                        statsd.gauge(metric_name, metric_value, tags=metric_tags)
                    elif metric_type == MetricTypes.HISTOGRAM:
                        pass
                    elif metric_type == MetricTypes.SUMMARY:
                        pass
                    else:
                        self.logger.warning(f"Unknown metric type: {metric_type} for metric: {metric_name}")

            except Exception:
                self.logger.warning(f"Failed to process_metrics metrics: {traceback.format_exc()}")
