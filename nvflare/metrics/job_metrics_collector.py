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



from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.metrics.metrics_collector import MetricsCollector


class JobMetricsCollector(MetricsCollector):

    def __init__(self, tags: dict, streaming_to_server: bool = False):
        """
        Args:
            tags: comma separated static tags. used to specify server, client, production, test etc.
            streaming_to_server: boolean to specify if metrics should be streamed to server
        """
        super().__init__(tags = tags, streaming_to_server = streaming_to_server)
        
        self.single_events = [
            EventType.SUBMIT_JOB,
            EventType.DEPLOY_JOB_TO_SERVER,
            EventType.DEPLOY_JOB_TO_CLIENT,
            EventType.BEFORE_CHECK_RESOURCE_MANAGER,
            EventType.BEFORE_SEND_ADMIN_COMMAND
        ]
    
        self.pair_events = {
            EventType.START_WORKFLOW: "_workflow",
            EventType.END_WORKFLOW: "_workflow",

            EventType.JOB_STARTED: "_job",
            EventType.JOB_COMPLETED: "_job",
            EventType.JOB_ABORTED: "_job",
            EventType.JOB_CANCELLED: "_job",

            EventType.BEFORE_PULL_TASK: "_pull_task",
            EventType.AFTER_PULL_TASK: "_pull_task",
            EventType.BEFORE_PROCESS_TASK_REQUEST: "_process_task",

            EventType.BEFORE_PROCESS_SUBMISSION: "_process_submission",
            EventType.AFTER_PROCESS_SUBMISSION: "_process_submission",

            EventType.BEFORE_TASK_DATA_FILTER: "_data_filter",
            EventType.AFTER_TASK_DATA_FILTER: "_data_filter",

            EventType.BEFORE_TASK_RESULT_FILTER: "_result_filter",
            EventType.AFTER_TASK_RESULT_FILTER: "_result_filter",

            EventType.BEFORE_TASK_EXECUTION: "_task_execution",
            EventType.AFTER_TASK_EXECUTION: "_task_execution",
            EventType.ABORT_TASK: "_task_execution",

            EventType.BEFORE_SEND_TASK_RESULT: "_send_task_result",
            EventType.AFTER_SEND_TASK_RESULT: "_send_task_result",

            EventType.BEFORE_PROCESS_RESULT_OF_UNKNOWN_TASK: "_process_result_of_unknown_task",
            EventType.AFTER_PROCESS_RESULT_OF_UNKNOWN_TASK: "_process_result_of_unknown_task"
        }


    def handle_event(self, event: str, fl_ctx: FLContext):
        print("job event = ", event)
        job_id = fl_ctx.get_job_id()
        tags: dict = self.tags
        tags["job_id"] = job_id
        super().collect_event_metrics(event=event, tags = tags, fl_ctx=fl_ctx)


        # metrics = {MetricKeys.count: 1, MetricKeys.type: MetricTypes.COUNTER}
        # duration_metrics = {MetricKeys.time_taken: 0, MetricKeys.type: MetricTypes.GAUGE}

        # if event == EventType.START_WORKFLOW:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)
        #     self.job_start_workflow = current_time

        # elif event == EventType.END_WORKFLOW:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)

        #     time_taken = current_time - self.job_start_workflow
        #     duration_metrics[MetricKeys.time_taken] = time_taken
        #     metric_name = "_workflow"
        #     self.publish_metrics(duration_metrics, metric_name, tags, fl_ctx)

        # elif event == EventType.JOB_STARTED:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)
        #     self.job_started = current_time

        # elif event == EventType.JOB_COMPLETED or event == EventType.JOB_ABORTED or event == EventType.JOB_CANCELLED:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)

        #     time_taken = current_time - self.job_started
        #     duration_metrics[MetricKeys.time_taken] = time_taken
        #     metric_name = "_job"
        #     self.publish_metrics(duration_metrics, metric_name, tags, fl_ctx)

        # elif event == EventType.BEFORE_PULL_TASK:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)
        #     self.before_pull_task = current_time

        # elif event == EventType.AFTER_PULL_TASK:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)

        #     time_taken = current_time - self.before_pull_task
        #     duration_metrics[MetricKeys.time_taken] = time_taken
        #     metric_name = "_pull_task"
        #     self.publish_metrics(duration_metrics, metric_name, tags, fl_ctx)

        # elif event == EventType.BEFORE_PROCESS_TASK_REQUEST:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)
        #     self.before_process_task_request = current_time

        # elif event == EventType.AFTER_PROCESS_TASK_REQUEST:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)

        #     time_taken = current_time - self.before_process_task_request
        #     duration_metrics[MetricKeys.time_taken] = time_taken
        #     metric_name = "_process_task"
        #     self.publish_metrics(duration_metrics, metric_name, tags, fl_ctx)

        # elif event == EventType.BEFORE_PROCESS_SUBMISSION:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)
        #     self.before_process_submission = current_time

        # elif event == EventType.AFTER_PROCESS_SUBMISSION:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)

        #     time_taken = current_time - self.before_process_submission
        #     duration_metrics[MetricKeys.time_taken] = time_taken
        #     metric_name = "_process_submission"
        #     self.publish_metrics(duration_metrics, metric_name, tags, fl_ctx)

        # elif event == EventType.BEFORE_TASK_DATA_FILTER:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)
        #     self.before_task_data_filter = current_time

        # elif event == EventType.AFTER_TASK_DATA_FILTER:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)

        #     time_taken = current_time - self.before_task_data_filter
        #     duration_metrics[MetricKeys.time_taken] = time_taken
        #     metric_name = "_data_filter"
        #     self.publish_metrics(duration_metrics, metric_name, tags, fl_ctx)

        # elif event == EventType.BEFORE_TASK_RESULT_FILTER:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)
        #     self.before_task_result_filter = current_time

        # elif event == EventType.AFTER_TASK_RESULT_FILTER:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)

        #     time_taken = current_time - self.before_task_result_filter
        #     duration_metrics[MetricKeys.time_taken] = time_taken
        #     metric_name = "_result_filter"
        #     self.publish_metrics(duration_metrics, metric_name, tags, fl_ctx)

        # elif event == EventType.BEFORE_TASK_EXECUTION:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)
        #     self.before_task_execution = current_time

        # elif event == EventType.AFTER_TASK_EXECUTION:

        #     time_taken = current_time - self.before_task_execution
        #     duration_metrics[MetricKeys.time_taken] = time_taken
        #     metric_name = "_task_execution"
        #     self.publish_metrics(duration_metrics, metric_name, tags, fl_ctx)

        # elif event == EventType.ABORT_TASK:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)

        #     time_taken = current_time - self.before_task_execution
        #     duration_metrics[MetricKeys.time_taken] = time_taken
        #     metric_name = "_before_abort"
        #     self.publish_metrics(duration_metrics, metric_name, tags, fl_ctx)

        # elif event == EventType.BEFORE_SEND_TASK_RESULT:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)
        #     self.before_send_task_result = current_time

        # elif event == EventType.AFTER_SEND_TASK_RESULT:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)

        #     time_taken = current_time - self.before_send_task_result
        #     duration_metrics[MetricKeys.time_taken] = time_taken
        #     metric_name = "_send_task_result"
        #     self.publish_metrics(duration_metrics, metric_name, tags, fl_ctx)

        # elif event == EventType.BEFORE_PROCESS_RESULT_OF_UNKNOWN_TASK:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)
        #     self.before_process_result_of_unknown_task = current_time

        # elif event == EventType.AFTER_PROCESS_RESULT_OF_UNKNOWN_TASK:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)

        #     time_taken = current_time - self.before_process_result_of_unknown_task
        #     duration_metrics[MetricKeys.time_taken] = time_taken
        #     metric_name = "_process_result_of_unknown_task"
        #     self.publish_metrics(duration_metrics, metric_name, tags, fl_ctx)

        # elif event == EventType.SUBMIT_JOB:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)

        # elif event == EventType.DEPLOY_JOB_TO_SERVER:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)

        # elif event == EventType.DEPLOY_JOB_TO_CLIENT:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)

        # elif event == EventType.BEFORE_CHECK_RESOURCE_MANAGER:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)

        # elif event == EventType.BEFORE_SEND_ADMIN_COMMAND:
        #     self.publish_metrics(metrics, metric_name, tags, fl_ctx)

        # else:
        #     pass

    # def publish_metrics(self, metrics: dict, metric_name: str, tags: dict, fl_ctx: FLContext):
    #     collect_metrics(self, self.streaming_to_server, metrics, metric_name, tags, self.data_bus, fl_ctx)

    def get_single_events(self):
        return self.single_events

    def get_pair_events(self):
        return self.pair_events
 