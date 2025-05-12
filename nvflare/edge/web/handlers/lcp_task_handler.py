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
import logging
from typing import Any

from nvflare.apis.client_engine_spec import ClientEngineSpec
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.job_def import JobMetaKey
from nvflare.edge.constants import EdgeApiStatus, EdgeContextKey, EdgeEventType, EdgeProtoKey
from nvflare.edge.web.handlers.edge_task_handler import EdgeTaskHandler
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.selection_request import SelectionRequest
from nvflare.edge.web.models.selection_response import SelectionResponse
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.task_response import TaskResponse


class LcpTaskHandler(EdgeTaskHandler):
    def __init__(self):
        self.engine = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_engine(self, engine: ClientEngineSpec):
        self.engine = engine

    def handle_job(self, job_request: JobRequest) -> JobResponse:
        with self.engine.new_context() as fl_ctx:
            fl_ctx.set_prop(EdgeContextKey.EDGE_CAPABILITIES, job_request.capabilities, private=True, sticky=False)

            self.engine.fire_event(EdgeEventType.EDGE_JOB_REQUEST_RECEIVED, fl_ctx)

            reply = fl_ctx.get_prop(EdgeContextKey.REPLY_TO_EDGE)
            assert isinstance(reply, dict)
            status = reply.get(EdgeProtoKey.STATUS)
            if status != EdgeApiStatus.OK:
                response = JobResponse(EdgeApiStatus.RETRY, retry_wait=30)
            else:
                job_id = reply.get(EdgeProtoKey.DATA)
                job_meta = fl_ctx.get_prop(FLContextKey.JOB_META)
                if job_meta:
                    job_name = job_meta.get(JobMetaKey.JOB_NAME)
                else:
                    job_name = "No Name"
                response = JobResponse(EdgeApiStatus.OK, job_id=job_id, job_name=job_name, job_meta=job_meta)

            return response

    def handle_task(self, task_request: TaskRequest) -> TaskResponse:
        reply = self._handle_task_request(task_request)
        status = reply.get(EdgeProtoKey.STATUS)
        if status == EdgeApiStatus.OK:
            data = reply.get(EdgeProtoKey.DATA)
            response = data.get(EdgeProtoKey.RESPONSE)
        elif status == EdgeApiStatus.NO_JOB:
            self.logger.info(f"Job {task_request.job_id} is done")
            response = TaskResponse(EdgeApiStatus.NO_JOB, retry_wait=30)
        elif status == EdgeApiStatus.NO_TASK:
            self.logger.info(f"Job {task_request.job_id} has no task")
            response = TaskResponse(EdgeApiStatus.NO_TASK, retry_wait=30)
        else:
            self.logger.error(f"Task request for {task_request.job_id} failed with status {status}")
            response = TaskResponse(EdgeApiStatus.RETRY, retry_wait=30)

        return response

    def handle_result(self, result_report: ResultReport) -> ResultResponse:
        reply = self._handle_task_request(result_report)
        status = reply.get(EdgeProtoKey.STATUS)
        if status != EdgeApiStatus.OK:
            response = ResultResponse(EdgeApiStatus.RETRY, retry_wait=30)
        else:
            data = reply.get(EdgeProtoKey.DATA)
            response = data.get(EdgeProtoKey.RESPONSE)

        return response

    def handle_selection(self, selection_request: SelectionRequest) -> SelectionResponse:
        reply = self._handle_task_request(selection_request)
        status = reply.get(EdgeProtoKey.STATUS)
        if status != EdgeApiStatus.OK:
            response = SelectionResponse(EdgeApiStatus.RETRY, job_id=selection_request.job_id)
        else:
            data = reply.get(EdgeProtoKey.DATA)
            response = data.get(EdgeProtoKey.RESPONSE)

        return response

    def _handle_task_request(self, request: Any) -> dict:
        with self.engine.new_context() as fl_ctx:
            fl_ctx.set_prop(EdgeContextKey.JOB_ID, request.job_id, private=True, sticky=False)
            fl_ctx.set_prop(EdgeContextKey.REQUEST_FROM_EDGE, request, private=True, sticky=False)

            self.engine.fire_event(EdgeEventType.EDGE_REQUEST_RECEIVED, fl_ctx)

            reply = fl_ctx.get_prop(EdgeContextKey.REPLY_TO_EDGE)
            assert isinstance(reply, dict)
            return reply
