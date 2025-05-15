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
from typing import Optional

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.edge.constants import EdgeApiStatus, MsgKey
from nvflare.edge.executors.ete import EdgeTaskExecutor
from nvflare.edge.executors.hug import TaskInfo
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.selection_request import SelectionRequest
from nvflare.edge.web.models.selection_response import SelectionResponse
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.security.logging import secure_format_exception


class SimpleEdgeExecutor(EdgeTaskExecutor):
    """A very simple edge executor that only does aggregation"""

    def __init__(self, updater_id, update_timeout=60):
        EdgeTaskExecutor.__init__(self, updater_id, update_timeout)
        self.devices = None

    def convert_task(self, task_data: Shareable, current_task: TaskInfo, fl_ctx: FLContext) -> dict:
        """Convert task_data to a plain dict"""
        self.log_debug(fl_ctx, f"Converting task for task: {current_task.id}")
        return {"weights": task_data.get("weights", None), MsgKey.TASK_ID: current_task.id}

    def convert_result(self, result: dict, current_task: TaskInfo, fl_ctx: FLContext) -> Shareable:
        """Convert result from device to shareable"""
        self.log_debug(fl_ctx, f"Converting result for task: {current_task.id}")
        shareable = Shareable(result)
        shareable.set_header(ReservedHeaderKey.TASK_ID, current_task.id)
        return shareable

    def process_edge_task_request(
        self, request: TaskRequest, current_task: TaskInfo, fl_ctx: FLContext
    ) -> TaskResponse:
        """Handle task request from device"""

        device_id = request.get_device_id()
        job_id = fl_ctx.get_job_id()

        # This device already processed current task
        last_task_id = self.devices.get(device_id, None)
        task_id = current_task.id
        if task_id == last_task_id:
            msg = f"Task {task_id} is already processed by this device"
            return TaskResponse(EdgeApiStatus.RETRY, job_id, 30, message=msg)

        task_data = self.convert_task(current_task.task, current_task, fl_ctx)
        self.devices[device_id] = task_id
        return TaskResponse(EdgeApiStatus.OK, job_id, 0, task_id, current_task.name, task_data)

    def process_edge_result_report(
        self, report: ResultReport, current_task: TaskInfo, fl_ctx: FLContext
    ) -> ResultResponse:
        """Handle result report from device
        The report task_id may be different from current task_id. Let HAM deal with it
        """

        try:
            result = self.convert_result(report.result, current_task, fl_ctx)
            self.accept_update(task_id=report.task_id, update=result, fl_ctx=fl_ctx)
            return ResultResponse(EdgeApiStatus.OK, task_id=report.task_id, task_name=report.task_name)
        except Exception as ex:
            msg = f"Error accepting contribution: {secure_format_exception(ex)}"
            self.log_error(fl_ctx, msg)
            return ResultResponse(EdgeApiStatus.ERROR, task_id=report.task_id, task_name=report.task_name, message=msg)

    def task_started(self, task: TaskInfo, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"Got task_started: {task.id} (seq {task.seq})")
        self.devices = {}

    def task_ended(self, task: TaskInfo, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"Got task_ended: {task.id} (seq {task.seq})")
        self.devices = None

    def process_edge_selection_request(
        self, request: SelectionRequest, current_task: TaskInfo, fl_ctx: FLContext
    ) -> Optional[SelectionResponse]:
        return None
