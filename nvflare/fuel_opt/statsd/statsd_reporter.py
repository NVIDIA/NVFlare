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
import threading
import traceback

from nvflare.apis.fl_constant import ReservedTopic
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.metrics.metrics_keys import MetricKeys, MetricTypes


class StatsDReporter:
    """Publishes metrics to DogStatsd. Importing this class does not require ``datadog`` to be installed.

    ``datadog.initialize`` runs on first ``process_metrics`` call (lazy), so Job API export on a submitter
    machine does not open StatsD or require the monitoring endpoint to be reachable at export time.
    Server and client runtimes still need ``datadog`` installed when metrics are actually emitted.
    """

    def __init__(self, site: str = "", host="localhost", port=9125):
        self.site = site
        self.host = host
        self.port = port
        self.metrics = {}
        self._statsd = None
        self._statsd_disabled = False
        self._statsd_init_lock = threading.Lock()
        self.data_bus = DataBus()

        self.data_bus.subscribe([ReservedTopic.APP_METRICS], self.process_metrics)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _ensure_statsd(self):
        with self._statsd_init_lock:
            if self._statsd is not None:
                return True
            if self._statsd_disabled:
                return False

            try:
                from datadog import initialize, statsd

                initialize(statsd_host=self.host, statsd_port=self.port)
                self._statsd = statsd
                return True
            except Exception:
                self._statsd_disabled = True
                self.logger.warning(
                    "Disabling StatsDReporter for this process after initialization failed: %s",
                    traceback.format_exc(),
                )
                return False

    def process_metrics(self, topic, metrics, data_bus):

        if topic == ReservedTopic.APP_METRICS:
            try:
                if not self._ensure_statsd():
                    return
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
                        self._statsd.increment(metric_name, metric_value, tags=metric_tags)

                    elif metric_type == MetricTypes.GAUGE:
                        self._statsd.gauge(metric_name, metric_value, tags=metric_tags)
                    elif metric_type == MetricTypes.HISTOGRAM:
                        pass
                    elif metric_type == MetricTypes.SUMMARY:
                        pass
                    else:
                        self.logger.warning(f"Unknown metric type: {metric_type} for metric: {metric_name}")

            except Exception:
                self.logger.warning(f"Failed to process_metrics metrics: {traceback.format_exc()}")
