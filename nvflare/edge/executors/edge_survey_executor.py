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
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.edge.executors.ete import EdgeTaskExecutor


class EdgeSurveyExecutor(EdgeTaskExecutor):
    """This executor is for test purpose only. It is to be used as the "learner" for the
    HierarchicalAggregationManager.
    """

    def __init__(self, timeout=10.0):
        EdgeTaskExecutor.__init__(self)
        self.timeout = timeout
        self.num_devices = 0
        self.start_time = None

    def task_received(self, task_name: str, task_data: Shareable, fl_ctx: FLContext):
        self.num_devices = 0
        self.start_time = time.time()

    def is_task_done(self, fl_ctx: FLContext) -> bool:
        return time.time() - self.start_time > self.timeout

    def process_edge_request(self, request: Any, fl_ctx: FLContext) -> Any:
        assert isinstance(request, dict)
        self.log_info(fl_ctx, f"received edge request: {request}")
        self.num_devices += 1
        return {"status": "tryAgain", "comment": f"received {request}"}

    def get_task_result(self, fl_ctx: FLContext) -> Shareable:
        result = make_reply(ReturnCode.OK)
        result["num_devices"] = self.num_devices
        return result
