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
from nvflare.app_common.app_event_type import AppEventType
from nvflare.metrics.metrics_collector import MetricsCollector
from nvflare.metrics.metrics_publisher import collect_metrics


class JobMetricsCollector(MetricsCollector):
    def __init__(self, tags: dict, streaming_to_server: bool = False):
        """
        Args:
            tags: comma separated static tags. used to specify server, client, production, test etc.
            streaming_to_server: boolean to specify if metrics should be streamed to server
        """
        super().__init__(tags=tags, streaming_to_server=streaming_to_server)

        #  Job events
        self.single_events = [
            EventType.SUBMIT_JOB,
            EventType.DEPLOY_JOB_TO_SERVER,
            EventType.DEPLOY_JOB_TO_CLIENT,
            EventType.BEFORE_CHECK_RESOURCE_MANAGER,
            EventType.BEFORE_SEND_ADMIN_COMMAND,
            # application
            AppEventType.INITIAL_MODEL_LOADED,
            AppEventType.BEFORE_TRAIN_TASK,
            AppEventType.AFTER_AGGREGATION,
        ]

        self.pair_events, self.pair_start_events = self._build_pair_event_maps(
            [
                (EventType.START_WORKFLOW, EventType.END_WORKFLOW, "_workflow"),
                (EventType.START_RUN, EventType.END_RUN, "_run"),
                (
                    EventType.JOB_STARTED,
                    EventType.JOB_COMPLETED,
                    EventType.JOB_ABORTED,
                    EventType.JOB_CANCELLED,
                    "_job",
                ),
                (EventType.BEFORE_PULL_TASK, EventType.AFTER_PULL_TASK, "_pull_task"),
                (
                    EventType.BEFORE_PROCESS_TASK_REQUEST,
                    EventType.AFTER_PROCESS_TASK_REQUEST,
                    "_process_task",
                ),
                (EventType.BEFORE_PROCESS_SUBMISSION, EventType.AFTER_PROCESS_SUBMISSION, "_process_submission"),
                (EventType.BEFORE_TASK_DATA_FILTER, EventType.AFTER_TASK_DATA_FILTER, "_data_filter"),
                (EventType.BEFORE_TASK_RESULT_FILTER, EventType.AFTER_TASK_RESULT_FILTER, "_result_filter"),
                (
                    EventType.BEFORE_TASK_EXECUTION,
                    EventType.AFTER_TASK_EXECUTION,
                    EventType.ABORT_TASK,
                    "_task_execution",
                ),
                (EventType.BEFORE_SEND_TASK_RESULT, EventType.AFTER_SEND_TASK_RESULT, "_send_task_result"),
                (
                    EventType.BEFORE_PROCESS_RESULT_OF_UNKNOWN_TASK,
                    EventType.AFTER_PROCESS_RESULT_OF_UNKNOWN_TASK,
                    "_process_result_of_unknown_task",
                ),
                (AppEventType.BEFORE_AGGREGATION, AppEventType.END_AGGREGATION, "_aggregation"),
                (AppEventType.RECEIVE_BEST_MODEL, "_receive_best_model"),
                (AppEventType.BEFORE_TRAIN, AppEventType.AFTER_TRAIN, "_train"),
                (AppEventType.TRAIN_DONE, "_train_done"),
                (AppEventType.TRAINING_STARTED, AppEventType.TRAINING_FINISHED, "_training"),
                (AppEventType.ROUND_STARTED, AppEventType.ROUND_DONE, "_round"),
            ]
        )

    def handle_event(self, event: str, fl_ctx: FLContext):
        job_id = fl_ctx.get_job_id()
        tags: dict = self.tags
        tags["job_id"] = job_id
        super().collect_event_metrics(event=event, tags=tags, fl_ctx=fl_ctx)

    def publish_metrics(self, metrics: dict, metric_name: str, tags: dict, fl_ctx: FLContext):
        collect_metrics(self, self.streaming_to_server, metrics, metric_name, tags, self.data_bus, fl_ctx)

    def get_single_events(self):
        return self.single_events

    def get_pair_events(self):
        return self.pair_events

    def get_pair_start_events(self):
        return self.pair_start_events
