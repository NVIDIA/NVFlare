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
import json
import traceback
from abc import ABC, abstractmethod
from typing import Optional

from nvflare.edge.constants import EdgeApiStatus
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.selection_request import SelectionRequest
from nvflare.edge.web.models.selection_response import SelectionResponse
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.security.logging import secure_format_exception

from .constants import QueryType
from .edge_api_pb2 import Reply, Request
from .utils import make_reply


class _QueryInfo:

    def __init__(self, device_info, user_info):
        self.device_info = device_info
        self.user_info = user_info


class QueryHandler(ABC):

    def __init__(self):
        self.logger = get_obj_logger(self)

        self.processors = {
            QueryType.JOB_REQUEST: (
                JobRequest.from_dict,
                self.handle_job_request,
            ),
            QueryType.TASK_REQUEST: (
                TaskRequest.from_dict,
                self.handle_task_request,
            ),
            QueryType.SELECTION_REQUEST: (
                SelectionRequest.from_dict,
                self.handle_selection_request,
            ),
            QueryType.RESULT_REPORT: (
                ResultReport.from_dict,
                self.handle_result_report,
            ),
        }

    def _to_dict(self, name: str, data: bytes) -> Optional[dict]:
        try:
            return json.loads(data)
        except Exception as ex:
            self.logger.error(f"error decoding {name} data: {secure_format_exception(ex)}")
            return None

    @abstractmethod
    def handle_job_request(self, request: JobRequest) -> JobResponse:
        pass

    @abstractmethod
    def handle_task_request(self, request: TaskRequest) -> TaskResponse:
        pass

    @abstractmethod
    def handle_selection_request(self, request: SelectionRequest) -> SelectionResponse:
        pass

    @abstractmethod
    def handle_result_report(self, request: ResultReport) -> ResultResponse:
        pass

    def handle_query(self, request: Request) -> Reply:
        p = self.processors.get(request.type)
        if not p:
            self.logger.error(f"received invalid query type: {request.type}")
            return make_reply(EdgeApiStatus.INVALID_REQUEST)

        self.logger.debug(f"received request {request.type}")

        payload = self._to_dict("payload", request.payload)
        if not payload:
            return make_reply(EdgeApiStatus.INVALID_REQUEST)

        try:
            to_request_f, process_f = p
            error, req = to_request_f(payload)
            if error:
                self.logger.error(f"error in request {request.type}: {error}")
                return make_reply(EdgeApiStatus.INVALID_REQUEST)
            resp = process_f(req)
            return make_reply(EdgeApiStatus.OK, resp)
        except Exception as ex:
            traceback.print_exc()
            self.logger.error(f"error processing request: {secure_format_exception(ex)}")
            return make_reply(EdgeApiStatus.ERROR)
