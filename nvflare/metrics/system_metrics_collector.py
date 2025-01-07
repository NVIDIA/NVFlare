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
from nvflare.metrics.metrics_publisher import collect_metrics


class SysMetricsCollector(FLComponent):

    def __init__(self, tags: dict, streaming_to_server: bool = False):
        """
        Args:
            tags: comma separated static tags. used to specify server, client, production, test etc.
        """
        super().__init__()

        self.tags = tags
        self.data_bus = DataBus()
        self.streaming_to_server = streaming_to_server

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
        self.start = True

        print("Initialize Sys_metrics collector, tags=", tags)

    def handle_event(self, event: str, fl_ctx: FLContext):

        current_time = time.time()
        metric_name = event

        metrics = {MetricKeys.count: 1, MetricKeys.type: MetricTypes.COUNTER}
        duration_metrics = {MetricKeys.time_taken: 0, MetricKeys.type: MetricTypes.GAUGE}

        if event == EventType.SYSTEM_START:
            self.publish_metrics(metrics, metric_name, self.tags, fl_ctx)
            self.system_start = current_time

        elif event == EventType.SYSTEM_END:
            self.publish_metrics(metrics, metric_name, self.tags, fl_ctx)

            time_taken = current_time - self.system_start
            duration_metrics[MetricKeys.time_taken] = time_taken
            metric_name = "_system"
            self.publish_metrics(duration_metrics, metric_name, self.tags, fl_ctx)

        elif event == EventType.START_RUN:
            self.publish_metrics(metrics, metric_name, self.tags, fl_ctx)
            self.start_run = current_time

        elif event == EventType.END_RUN:
            self.publish_metrics(metrics, metric_name, self.tags, fl_ctx)

            time_taken = current_time - self.start_run
            duration_metrics[MetricKeys.time_taken] = time_taken
            metric_name = "_run"
            self.publish_metrics(duration_metrics, metric_name, self.tags, fl_ctx)

        elif event == EventType.BEFORE_CHECK_CLIENT_RESOURCES:
            self.publish_metrics(metrics, metric_name, self.tags, fl_ctx)
            self.before_check_client_resources = current_time

        elif event == EventType.AFTER_CHECK_CLIENT_RESOURCES:
            self.publish_metrics(metrics, metric_name, self.tags, fl_ctx)

            time_taken = current_time - self.before_check_client_resources
            duration_metrics[MetricKeys.time_taken] = time_taken
            metric_name = "_check_client_resources"
            self.publish_metrics(duration_metrics, metric_name, self.tags, fl_ctx)
        elif event == EventType.CLIENT_DISCONNECTED:
            self.publish_metrics(metrics, metric_name, self.tags, fl_ctx)
        elif event == EventType.CLIENT_RECONNECTED:
            self.publish_metrics(metrics, metric_name, self.tags, fl_ctx)
        elif event == EventType.BEFORE_CHECK_RESOURCE_MANAGER:
            self.publish_metrics(metrics, metric_name, self.tags, fl_ctx)
        elif event == EventType.BEFORE_SEND_ADMIN_COMMAND:
            self.publish_metrics(metrics, metric_name, self.tags, fl_ctx)
        elif event == EventType.BEFORE_CLIENT_REGISTER:
            self.publish_metrics(metrics, metric_name, self.tags, fl_ctx)
            self.before_client_register = current_time
        elif event == EventType.AFTER_CLIENT_REGISTER:
            self.publish_metrics(metrics, metric_name, self.tags, fl_ctx)

            time_taken = current_time - self.before_client_register
            duration_metrics[MetricKeys.time_taken] = time_taken
            metric_name = "_client_register"
            self.publish_metrics(duration_metrics, metric_name, self.tags, fl_ctx)

        elif event == EventType.CLIENT_REGISTER_RECEIVED:
            self.publish_metrics(metrics, metric_name, self.tags, fl_ctx)

        elif event == EventType.CLIENT_REGISTER_PROCESSED:
            self.publish_metrics(metrics, metric_name, self.tags, fl_ctx)

        elif event == EventType.CLIENT_QUIT:

            self.publish_metrics(metrics, metric_name, self.tags, fl_ctx)
        elif event == EventType.SYSTEM_BOOTSTRAP:
            self.publish_metrics(metrics, metric_name, self.tags, fl_ctx)
        elif event == EventType.BEFORE_CLIENT_HEARTBEAT:
            self.publish_metrics(metrics, metric_name, self.tags, fl_ctx)
            self.before_client_heartbeat = current_time
        elif event == EventType.AFTER_CLIENT_HEARTBEAT:
            self.publish_metrics(metrics, metric_name, self.tags, fl_ctx)

            time_taken = current_time - self.before_client_heartbeat
            duration_metrics[MetricKeys.time_taken] = time_taken
            metric_name = "_client_heartbeat"
            self.publish_metrics(duration_metrics, metric_name, self.tags, fl_ctx)
        elif event == EventType.CLIENT_HEARTBEAT_RECEIVED:
            self.publish_metrics(metrics, metric_name, self.tags, fl_ctx)
        elif event == EventType.CLIENT_HEARTBEAT_PROCESSED:
            self.publish_metrics(metrics, metric_name, self.tags, fl_ctx)
        elif event == EventType.BEFORE_JOB_LAUNCH:
            self.publish_metrics(metrics, metric_name, self.tags, fl_ctx)
        else:
            pass

    def publish_metrics(self, metrics: dict, metric_name: str, tags: dict, fl_ctx: FLContext):

        collect_metrics(self, self.streaming_to_server, metrics, metric_name, tags, self.data_bus, fl_ctx)
