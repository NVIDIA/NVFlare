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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.metrics.metrics_keys import MetricKeys, MetricTypes
from nvflare.metrics.metrics_publisher import publish_app_metrics


class SysMetricsCollector(FLComponent):

    def __init__(self, tags: dict):
        """
        Args:
            tags: comma separated static tags. used to specify server, client, production, test etc.
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
        self.before_client_heartbeat = 0
        self.before_client_register = 0
        self.client_disconnected = 0
        self.client_reconnected = 0
        self.before_check_client_resources = 0
        self.after_check_client_resources = 0

    def handle_event(self, event: str, fl_ctx: FLContext):

        current_time = time.time()
        metric_name = event

        metrics = {MetricKeys.count: 1, MetricKeys.type: MetricTypes.COUNTER}
        duration_metrics = {MetricKeys.time_taken: 0, MetricKeys.type: MetricTypes.GAUGE}

        if event == EventType.SYSTEM_START:
            publish_app_metrics(metrics, metric_name, self.tags, self.data_bus)
            self.system_start = current_time

        elif event == EventType.SYSTEM_END:
            publish_app_metrics(metrics, metric_name, self.tags, self.data_bus)

            time_taken = current_time - self.system_start
            duration_metrics[MetricKeys.time_taken] = time_taken
            metric_name = "_system_time_taken"
            publish_app_metrics(duration_metrics, metric_name, self.tags, self.data_bus)

        elif event == EventType.START_RUN:
            publish_app_metrics(metrics, metric_name, self.tags, self.data_bus)
            self.start_run = current_time

        elif event == EventType.END_RUN:
            publish_app_metrics(metrics, metric_name, self.tags, self.data_bus)

            time_taken = current_time - self.start_run
            duration_metrics[MetricKeys.time_taken] = time_taken
            metric_name = "_run_time_taken"
            publish_app_metrics(duration_metrics, metric_name, self.tags, self.data_bus)

        elif event == EventType.BEFORE_CHECK_CLIENT_RESOURCES:
            publish_app_metrics(metrics, metric_name, self.tags, self.data_bus)
            self.before_check_client_resources = current_time

        elif event == EventType.AFTER_CHECK_CLIENT_RESOURCES:
            publish_app_metrics(metrics, metric_name, self.tags, self.data_bus)

            time_taken = current_time - self.before_check_client_resources
            duration_metrics[MetricKeys.time_taken] = time_taken
            metric_name = "_check_client_resources_time_taken"
            publish_app_metrics(duration_metrics, metric_name, self.tags, self.data_bus)
        elif event == EventType.CLIENT_DISCONNECTED:
            publish_app_metrics(metrics, metric_name, self.tags, self.data_bus)
        elif event == EventType.CLIENT_RECONNECTED:
            publish_app_metrics(metrics, metric_name, self.tags, self.data_bus)
        elif event == EventType.BEFORE_CHECK_RESOURCE_MANAGER:
            publish_app_metrics(metrics, metric_name, self.tags, self.data_bus)
        elif event == EventType.BEFORE_SEND_ADMIN_COMMAND:
            publish_app_metrics(metrics, metric_name, self.tags, self.data_bus)
        elif event == EventType.BEFORE_CLIENT_REGISTER:
            publish_app_metrics(metrics, metric_name, self.tags, self.data_bus)
            self.before_client_register = current_time
        elif event == EventType.AFTER_CLIENT_REGISTER:
            publish_app_metrics(metrics, metric_name, self.tags, self.data_bus)
        elif event == EventType.CLIENT_REGISTER_RECEIVED:
            publish_app_metrics(metrics, metric_name, self.tags, self.data_bus)
        elif event == EventType.CLIENT_QUIT:
            publish_app_metrics(metrics, metric_name, self.tags, self.data_bus)
        elif event == EventType.SYSTEM_BOOTSTRAP:
            publish_app_metrics(metrics, metric_name, self.tags, self.data_bus)
        elif event == EventType.BEFORE_CLIENT_HEARTBEAT:
            publish_app_metrics(metrics, metric_name, self.tags, self.data_bus)
            self.before_client_heartbeat = current_time
        elif event == EventType.AFTER_CLIENT_HEARTBEAT:
            publish_app_metrics(metrics, metric_name, self.tags, self.data_bus)

            time_taken = current_time - self.before_client_heartbeat
            duration_metrics[MetricKeys.time_taken] = time_taken
            metric_name = "_client_heart_time_taken"
            publish_app_metrics(duration_metrics, metric_name, self.tags, self.data_bus)
        elif event == EventType.CLIENT_HEARTBEAT_RECEIVED:
            publish_app_metrics(metrics, metric_name, self.tags, self.data_bus)
        elif event == EventType.CLIENT_HEARTBEAT_PROCESSED:
            publish_app_metrics(metrics, metric_name, self.tags, self.data_bus)
        elif event == EventType.BEFORE_JOB_LAUNCH:
            publish_app_metrics(metrics, metric_name, self.tags, self.data_bus)
        else:
            pass
