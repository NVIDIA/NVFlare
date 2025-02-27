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
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.edge.aggregators.edge_result_accumulator import EdgeResultAccumulator
from nvflare.edge.executors.ete import EdgeTaskExecutor
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.task_response import TaskResponse


class EdgeDispatchExecutor(EdgeTaskExecutor):
    """This executor dispatches tasks to edge devices and wait for the response from all devices
    """

    def __init__(self, wait_time=300.0, min_devices=0, edge_aggregator_id=None):
        EdgeTaskExecutor.__init__(self)
        self.wait_time = wait_time
        self.min_devices = min_devices
        self.task_sequence = None
        self.task_name = None
        self.task_id = None
        self.task_data = None
        self.start_time = None
        self.devices = None
        self.num_results = 0
        self.edge_aggregator_id = edge_aggregator_id
        self.aggregator = None
        self.register_event_handler(EventType.START_RUN, self.setup)

    def setup(self, _, fl_ctx: FLContext):
        if self.edge_aggregator_id:
            self.aggregator = fl_ctx.get_engine().get_component(self.edge_aggregator_id)
        else:
            self.aggregator = EdgeResultAccumulator()

    def task_received(self, task_name: str, task_data: Shareable, fl_ctx: FLContext):
        # Reset aggregator
        self.start_time = time.time()
        self.task_name = task_name
        self.task_id = task_data.get_header("xx")
        self.task_data = task_data
        self.task_sequence += 1
        self.devices = {}  # Devices got this task
        self.num_results = 0  # Number of devices reported results

    def is_task_done(self, fl_ctx: FLContext) -> bool:
        return time.time() - self.start_time > self.wait_time or 0 < self.min_devices < self.num_results

    def process_edge_request(self, request: Any, fl_ctx: FLContext) -> Any:
        self.log_info(fl_ctx, f"Received edge request: {request}")
        device_id = request.get_device_id()
        job_id = fl_ctx.get_job_id()
        if isinstance(request, TaskRequest):
            # Convert task_data to a format suitable for the device
            response = TaskResponse("OK", job_id, self.task_id, self.task_name, self.current_task)
            self.devices[device_id] = request
        elif isinstance(request, ResultReport):
            self.aggregator.accept(Shareable({"weights": request.result}), fl_ctx)
            self.num_results += 1
            response = ResultResponse("OK", task_id=self.task_id, task_name=self.task_name)
        else:
            raise RuntimeError(f"Received unknown request type: {type(request)}")

        return {"status": ReturnCode.OK, "response": response}

    def get_task_result(self, fl_ctx: FLContext) -> Shareable:
        result = make_reply(ReturnCode.OK)
        result.update(self.aggregator.aggregate(fl_ctx))
        return result
