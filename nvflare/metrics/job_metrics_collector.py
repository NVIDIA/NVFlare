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


class JobMetricsCollector(FLComponent):

    def __init__(self, tags: dict):
        """
        Args:
            tags: comma separated static tags. used to specify server, client, production, test etc.
        """
        super().__init__()
        self.tags: dict = tags
        self.data_bus = DataBus()

        # job events
        self.job_start_workflow = 0
        self.job_started = 0

        # job task events
        self.before_pull_task = 0
        self.before_process_task_request = 0
        self.before_process_submission = 0
        self.before_task_data_filter = 0
        self.before_task_result_filter = 0
        self.before_task_execution = 0
        self.before_send_task_result = 0
        self.before_process_result_of_unknown_task = 0

    def handle_event(self, event: str, fl_ctx: FLContext):

        current_time = time.time()
        job_id = fl_ctx.get_job_id()
        metric_name = event

        tags: dict = self.tags
        tags["job_id"] = job_id

        metrics = {MetricKeys.count: 1, MetricKeys.type: MetricTypes.COUNTER}
        duration_metrics = {MetricKeys.time_taken: 0, MetricKeys.type: MetricTypes.GAUGE}

        if event == EventType.START_WORKFLOW:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)
            self.job_start_workflow = current_time

        elif event == EventType.END_WORKFLOW:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)

            time_taken = current_time - self.job_start_workflow
            duration_metrics[MetricKeys.time_taken] = time_taken
            metric_name = "_workflow_time_taken"
            publish_app_metrics(duration_metrics, metric_name, tags, self.data_bus)

        elif event == EventType.JOB_STARTED:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)
            self.job_started = current_time

        elif event == EventType.JOB_COMPLETED or event == EventType.JOB_ABORTED or event == EventType.JOB_CANCELLED:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)

            time_taken = current_time - self.job_started
            duration_metrics[MetricKeys.time_taken] = time_taken
            metric_name = "_job_time_taken"
            publish_app_metrics(duration_metrics, metric_name, tags, self.data_bus)

        elif event == EventType.BEFORE_PULL_TASK:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)
            self.before_pull_task = current_time

        elif event == EventType.AFTER_PULL_TASK:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)

            time_taken = current_time - self.before_pull_task
            duration_metrics[MetricKeys.time_taken] = time_taken
            metric_name = "_pull_task_time_taken"
            publish_app_metrics(duration_metrics, metric_name, tags, self.data_bus)

        elif event == EventType.BEFORE_PROCESS_TASK_REQUEST:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)
            self.before_process_task_request = current_time

        elif event == EventType.AFTER_PROCESS_TASK_REQUEST:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)

            time_taken = current_time - self.before_process_task_request
            duration_metrics[MetricKeys.time_taken] = time_taken
            metric_name = "_process_task_time_taken"
            publish_app_metrics(duration_metrics, metric_name, tags, self.data_bus)

        elif event == EventType.BEFORE_PROCESS_SUBMISSION:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)
            self.before_process_submission = current_time

        elif event == EventType.AFTER_PROCESS_SUBMISSION:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)

            time_taken = current_time - self.before_process_submission
            duration_metrics[MetricKeys.time_taken] = time_taken
            metric_name = "_process_submission_time_taken"
            publish_app_metrics(duration_metrics, metric_name, tags, self.data_bus)

        elif event == EventType.BEFORE_TASK_DATA_FILTER:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)
            self.before_task_data_filter = current_time

        elif event == EventType.AFTER_TASK_DATA_FILTER:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)

            time_taken = current_time - self.before_task_data_filter
            duration_metrics[MetricKeys.time_taken] = time_taken
            metric_name = "_data_filter_time_taken"
            publish_app_metrics(duration_metrics, metric_name, tags, self.data_bus)

        elif event == EventType.BEFORE_TASK_RESULT_FILTER:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)
            self.before_task_result_filter = current_time

        elif event == EventType.AFTER_TASK_RESULT_FILTER:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)

            time_taken = current_time - self.before_task_result_filter
            duration_metrics[MetricKeys.time_taken] = time_taken
            metric_name = "_result_filter_time_taken"
            publish_app_metrics(duration_metrics, metric_name, tags, self.data_bus)

        elif event == EventType.BEFORE_TASK_EXECUTION:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)
            self.before_task_execution = current_time

        elif event == EventType.AFTER_TASK_EXECUTION:

            time_taken = current_time - self.before_task_execution
            duration_metrics[MetricKeys.time_taken] = time_taken
            metric_name = "_task_execution_time_taken"
            publish_app_metrics(duration_metrics, metric_name, tags, self.data_bus)

        elif event == EventType.ABORT_TASK:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)

            time_taken = current_time - self.before_task_execution
            duration_metrics[MetricKeys.time_taken] = time_taken
            metric_name = "_before_abort_time_taken"
            publish_app_metrics(duration_metrics, metric_name, tags, self.data_bus)

        elif event == EventType.BEFORE_SEND_TASK_RESULT:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)
            self.before_send_task_result = current_time

        elif event == EventType.AFTER_SEND_TASK_RESULT:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)

            time_taken = current_time - self.before_send_task_result
            duration_metrics[MetricKeys.time_taken] = time_taken
            metric_name = "_send_task_result_time_taken"
            publish_app_metrics(duration_metrics, metric_name, tags, self.data_bus)

        elif event == EventType.BEFORE_PROCESS_RESULT_OF_UNKNOWN_TASK:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)
            self.before_process_result_of_unknown_task = current_time

        elif event == EventType.AFTER_PROCESS_RESULT_OF_UNKNOWN_TASK:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)

            time_taken = current_time - self.before_process_result_of_unknown_task
            duration_metrics[MetricKeys.time_taken] = time_taken
            metric_name = "_process_result_of_unknown_task_time_taken"
            publish_app_metrics(duration_metrics, metric_name, tags, self.data_bus)

        elif event == EventType.SUBMIT_JOB:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)

        elif event == EventType.DEPLOY_JOB_TO_SERVER:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)

        elif event == EventType.DEPLOY_JOB_TO_CLIENT:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)

        elif event == EventType.BEFORE_CHECK_RESOURCE_MANAGER:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)

        elif event == EventType.BEFORE_SEND_ADMIN_COMMAND:
            publish_app_metrics(metrics, metric_name, tags, self.data_bus)

        else:
            pass
