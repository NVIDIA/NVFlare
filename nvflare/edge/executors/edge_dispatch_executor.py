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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, ReturnCode, Shareable, make_reply
from nvflare.edge.constants import EdgeApiStatus, MsgKey
from nvflare.edge.executors.ete import EdgeTaskExecutor
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.fuel.utils.validation_utils import check_non_negative_int, check_non_negative_number
from nvflare.security.logging import secure_format_exception


class EdgeDispatchExecutor(EdgeTaskExecutor):
    """This executor dispatches tasks to edge devices and wait for the response from all devices"""

    def __init__(self, wait_time=300.0, min_devices=0, aggregator_id=None):
        EdgeTaskExecutor.__init__(self)

        check_non_negative_number("wait_time", wait_time)
        check_non_negative_int("min_devices", min_devices)

        self.wait_time = wait_time
        self.min_devices = min_devices
        self.task_sequence = 0
        self.task_name = None
        self.task_id = None
        self.task_data = None
        self.start_time = None
        self.devices = None
        self.num_results = 0
        self.aggregator_id = aggregator_id
        self.aggregator = None
        self.register_event_handler(EventType.START_RUN, self.setup)

    def setup(self, _event_type, fl_ctx: FLContext):
        if not self.aggregator_id:
            raise ValueError("aggregator_id is not set")

        self.aggregator = fl_ctx.get_engine().get_component(self.aggregator_id)
        if not self.aggregator:
            raise RuntimeError(f"Can't find aggregator: {self.aggregator_id}")

    def convert_task(self, task_data: Shareable) -> dict:
        """Convert task_data to a plain dict"""

        return {MsgKey.PAYLOAD: task_data[MsgKey.PAYLOAD], MsgKey.TASK_ID: self.task_id}

    def convert_result(self, result: dict) -> Shareable:
        """Convert result from device to shareable"""
        shareable = Shareable(result)
        shareable.set_header(ReservedHeaderKey.TASK_ID, self.task_id)
        return shareable

    def handle_task_request(self, request: TaskRequest, fl_ctx: FLContext) -> TaskResponse:
        """Handle task request from device"""

        device_id = request.get_device_id()
        job_id = fl_ctx.get_job_id()

        # This device already processed current task
        last_task_id = self.devices.get(device_id, None)
        if self.task_id == last_task_id:
            return TaskResponse("RETRY", job_id, 30, message=f"Task {self.task_id} is already processed by this device")

        task_done = self.current_task.get("task_done")
        task_data = self.convert_task(self.current_task)
        self.devices[device_id] = self.task_id
        status = "DONE" if task_done else "OK"
        return TaskResponse(status, job_id, 0, self.task_id, self.task_name, task_data)

    def handle_result_report(self, report: ResultReport, fl_ctx: FLContext) -> ResultResponse:
        """Handle result report from device"""

        if report.task_id != self.task_id:
            msg = f"Task {report.task_id} is already done, result ignored"
            self.log_warning(fl_ctx, msg)
            # Still returns OK because this late result may be useful in certain cases
            return ResultResponse("OK", task_id=self.task_id, task_name=self.task_name, message=msg)

        try:
            result = self.convert_result(report.result)
            self.aggregator.accept(result, fl_ctx)
            self.num_results += 1
            return ResultResponse(EdgeApiStatus.OK, task_id=self.task_id, task_name=self.task_name)
        except Exception as ex:
            msg = f"exception when 'accept' from aggregator {type(self.aggregator)}: {secure_format_exception(ex)}"
            self.log_error(fl_ctx, msg)
            return ResultResponse(EdgeApiStatus.ERROR, task_id=self.task_id, task_name=self.task_name, message=msg)

    def task_received(self, task_name: str, task_data: Shareable, fl_ctx: FLContext):
        # Reset aggregator
        self.start_time = time.time()
        self.task_name = task_name
        self.task_id = task_data.get_header(ReservedHeaderKey.TASK_ID)
        self.task_data = task_data
        self.task_sequence += 1
        self.devices = {}  # Devices got this task
        self.num_results = 0  # Number of devices reported results

    def is_task_done(self, fl_ctx: FLContext) -> bool:
        return time.time() - self.start_time > self.wait_time or 0 < self.min_devices <= self.num_results

    def process_edge_request(self, request: Any, fl_ctx: FLContext) -> Any:
        self.log_info(fl_ctx, f"Received edge request: {request}")

        if isinstance(request, TaskRequest):
            response = self.handle_task_request(request, fl_ctx)
        elif isinstance(request, ResultReport):
            response = self.handle_result_report(request, fl_ctx)
        else:
            raise RuntimeError(f"Received unknown request type: {type(request)}")

        return {"status": ReturnCode.OK, "response": response}

    def get_task_result(self, fl_ctx: FLContext) -> Shareable:
        try:
            self.log_info(fl_ctx, "return aggregation result")
            return self.aggregator.aggregate(fl_ctx)
        except Exception as ex:
            self.log_error(fl_ctx, f"Error from {type(self.aggregator)}: {secure_format_exception(ex)}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
