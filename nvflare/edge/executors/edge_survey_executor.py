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
from typing import Optional

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.edge.constants import EdgeApiStatus
from nvflare.edge.executors.ete import EdgeTaskExecutor, TaskInfo
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.selection_request import SelectionRequest
from nvflare.edge.web.models.selection_response import SelectionResponse
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.task_response import TaskResponse


class EdgeSurveyExecutor(EdgeTaskExecutor):
    """This executor is for test purpose only. It is to be used as the "learner" for the
    HierarchicalAggregationManager.
    """

    def __init__(self, updater_id: str, update_timeout: float, task_duration: float):
        EdgeTaskExecutor.__init__(self, updater_id, update_timeout)
        self.task_duration = task_duration
        self.task = None
        self.task_start_time = None

    def task_started(self, task: TaskInfo, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"got task_started: {task.id} (seq {task.seq})")
        self.task_start_time = time.time()
        self.task = task

    def task_ended(self, task: TaskInfo, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"got task_ended: {task.id} (seq {task.seq})")
        self.task = None
        self.task_start_time = None

    def process_edge_task_request(
        self, request: TaskRequest, current_task: TaskInfo, fl_ctx: FLContext
    ) -> TaskResponse:
        current_task = self.task
        if not current_task:
            return TaskResponse(EdgeApiStatus.RETRY)

        if time.time() - self.task_start_time > self.task_duration:
            self.log_info(fl_ctx, f"task done after {self.task_duration} seconds")
            self.set_task_done(current_task.id, fl_ctx)
            return TaskResponse(EdgeApiStatus.DONE)

        self.log_info(fl_ctx, f"received edge request: {request}")
        self.accept_update(
            task_id=current_task.id,
            update=Shareable({"num_devices": 1}),
            fl_ctx=fl_ctx,
        )
        return TaskResponse(EdgeApiStatus.RETRY, job_id=fl_ctx.get_job_id())

    def process_edge_selection_request(
        self, request: SelectionRequest, current_task: TaskInfo, fl_ctx: FLContext
    ) -> Optional[SelectionResponse]:
        return None

    def process_edge_result_report(
        self, request: ResultReport, current_task: TaskInfo, fl_ctx: FLContext
    ) -> Optional[ResultResponse]:
        return None
