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

from .constants import NONE_DATA, QueryType
from .edge_api_pb2 import Reply, Request


def to_bytes(data: Optional[dict]) -> bytes:
    if not data:
        return NONE_DATA
    str_data = json.dumps(data)
    return str_data.encode("utf-8")


def make_reply(status: str, payload: Optional[dict] = None):
    return Reply(
        status=status,
        payload=to_bytes(payload),
    )


def _request_to_grpc(query_type: str, method: str, req) -> Request:
    payload = {}
    payload.update(req)

    return Request(type=query_type, method=method, header=NONE_DATA, payload=to_bytes(payload))


def _grpc_reply_to_response(reply: Reply, clazz):
    if reply.status != EdgeApiStatus.OK:
        return clazz(status=reply.status)
    if reply.payload != NONE_DATA:
        d = json.loads(reply.payload)
        resp = clazz(EdgeApiStatus.OK)
        resp.update(d)
    else:
        resp = None
    return resp


def job_request_to_grpc_request(request: JobRequest) -> Request:
    return _request_to_grpc(QueryType.JOB_REQUEST, "POST", request)


def grpc_reply_to_job_response(reply: Reply) -> JobResponse:
    return _grpc_reply_to_response(reply, JobResponse)


def task_request_to_grpc_request(request: TaskRequest) -> Request:
    return _request_to_grpc(QueryType.TASK_REQUEST, "GET", request)


def grpc_reply_to_task_response(reply: Reply) -> TaskResponse:
    return _grpc_reply_to_response(reply, TaskResponse)


def selection_request_to_grpc_request(request: SelectionRequest) -> Request:
    return _request_to_grpc(QueryType.SELECTION_REQUEST, "GET", request)


def grpc_reply_to_selection_response(reply: Reply) -> SelectionResponse:
    return _grpc_reply_to_response(reply, SelectionResponse)


def result_report_to_grpc_request(request: ResultReport) -> Request:
    return _request_to_grpc(QueryType.RESULT_REPORT, "POST", request)


def grpc_reply_to_result_response(reply: Reply) -> ResultResponse:
    return _grpc_reply_to_response(reply, ResultResponse)
