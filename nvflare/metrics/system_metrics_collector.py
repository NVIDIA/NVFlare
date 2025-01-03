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


import time
from typing import List

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.metrics.metrics_keys import MetricKeys, MetricTypes
from nvflare.metrics.metrics_publisher import publish_app_metrics


class SysMetricsCollector(FLComponent):

    def __init__(self, tags: List[str]):
        """
        Args:
            tags: static tags. used to specify server, client, production, test etc.
        """
        super().__init__()
        self.tags = tags
        self.data_bus = DataBus()

        # system events
        self.system_start = 0
        self.system_end = 0
        self.start_run = 0
        self.end_run = 0

        # server-site only events
        self.client_disconnected = 0
        self.client_reconnected = 0
        self.before_check_client_resources = 0
        self.after_check_client_resources = 0

    def handle_event(self, event: str, fl_ctx: FLContext):

        current_time = time.time()
        metric_name = event

        metrics = {"count": 1, MetricKeys.type: MetricTypes.COUNTER, "tags": self.tags}
        duration_metrics = {"time_taken": 0, MetricKeys.type: MetricTypes.GAUGE, "tags": self.tags}

        if event == EventType.SYSTEM_START:
            publish_app_metrics(metrics, metric_name, self.labels, self.data_bus, timestamp=current_time)
            self.system_start = current_time

        elif event == EventType.SYSTEM_END:
            publish_app_metrics(metrics, metric_name, self.labels, self.data_bus, timestamp=current_time)

            time_taken = current_time - self.system_start
            duration_metrics["time_taken"] = time_taken
            metric_name = self.prefix + "_system_time_taken"
            publish_app_metrics(duration_metrics, metric_name, self.labels, self.data_bus, timestamp=current_time)

        elif event == EventType.START_RUN:
            publish_app_metrics(metrics, metric_name, self.labels, self.data_bus, timestamp=current_time)
            self.start_run = current_time

        elif event == EventType.END_RUN:
            publish_app_metrics(metrics, metric_name, self.labels, self.data_bus, timestamp=current_time)

            time_taken = current_time - self.start_run
            duration_metrics["time_taken"] = time_taken
            metric_name = self.prefix + "_run_time_taken"
            publish_app_metrics(duration_metrics, metric_name, self.labels, self.data_bus, timestamp=current_time)

        elif event == EventType.BEFORE_CHECK_CLIENT_RESOURCES:
            publish_app_metrics(metrics, metric_name, self.labels, self.data_bus, timestamp=current_time)
            self.before_check_client_resources = current_time

        elif event == EventType.AFTER_CHECK_CLIENT_RESOURCES:
            publish_app_metrics(metrics, metric_name, self.labels, self.data_bus, timestamp=current_time)

            time_taken = current_time - self.before_check_client_resources
            duration_metrics["time_taken"] = time_taken
            metric_name = self.prefix + "_check_client_resources_time_taken"
            publish_app_metrics(duration_metrics, metric_name, self.labels, self.data_bus, timestamp=current_time)
        else:
            publish_app_metrics(metrics, metric_name, self.labels, self.data_bus, timestamp=current_time)
