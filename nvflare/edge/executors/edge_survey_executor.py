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
from typing import Any

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.edge.executors.ete import EdgeTaskExecutor, TaskInfo


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

    def process_edge_request(self, request: Any, current_task: TaskInfo, fl_ctx: FLContext) -> Any:
        current_task = self.task
        if not current_task:
            return {"status": "tryAgain", "comment": "no task"}

        if time.time() - self.task_start_time > self.task_duration:
            self.log_info(fl_ctx, f"task done after {self.task_duration} seconds")
            self.set_task_done(current_task.id, fl_ctx)
            return {"status": "tryAgain", "comment": "task done"}

        assert isinstance(request, dict)
        self.log_info(fl_ctx, f"received edge request: {request}")
        self.accept_update(
            task_id=current_task.id,
            update=Shareable({"num_devices": 1}),
            fl_ctx=fl_ctx,
        )
        return {"status": "tryAgain", "comment": f"received {request}"}
