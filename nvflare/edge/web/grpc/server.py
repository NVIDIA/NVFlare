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
import asyncio
import concurrent.futures.thread
import json
import threading
from abc import ABC, abstractmethod
from typing import Optional

import grpc

from nvflare.edge.constants import CookieKey, EdgeApiStatus, EdgeProtoKey, HttpHeaderKey
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.selection_request import SelectionRequest
from nvflare.edge.web.models.selection_response import SelectionResponse
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.edge.web.models.user_info import UserInfo
from nvflare.fuel.f3.drivers.aio_context import AioContext
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.security.logging import secure_format_exception

from .constants import NONE_DATA, QueryType
from .edge_api_pb2 import Reply, Request
from .edge_api_pb2_grpc import EdgeApiServicer, add_EdgeApiServicer_to_server


class QueryHandler(ABC):

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


class _QueryInfo:

    def __init__(self, device_info, user_info, cookie):
        self.device_info = device_info
        self.user_info = user_info
        self.cookie = cookie


class Servicer(EdgeApiServicer):

    def __init__(self, handler: QueryHandler, aio_ctx: AioContext, max_workers=100):
        self.logger = get_obj_logger(self)
        self.handler = handler
        self.aio_ctx = aio_ctx
        self.workers = concurrent.futures.thread.ThreadPoolExecutor(max_workers=max_workers)
        self.processors = {
            QueryType.JOB_REQUEST: (
                None,
                self._to_job_request,
                self.handler.handle_job_request,
            ),
            QueryType.TASK_REQUEST: (
                self._check_task_request,
                self._to_task_request,
                self.handler.handle_task_request,
            ),
            QueryType.SELECTION_REQUEST: (
                self._check_selection_request,
                self._to_selection_request,
                self.handler.handle_selection_request,
            ),
            QueryType.RESULT_REPORT: (
                self._check_result_report,
                self._to_result_report,
                self.handler.handle_result_report,
            ),
        }

    @staticmethod
    def _to_bytes(data: Optional[dict]) -> bytes:
        if not data:
            return NONE_DATA
        str_data = json.dumps(data)
        return str_data.encode("utf-8")

    def _make_reply(self, status: str, payload: Optional[dict] = None, cookie: Optional[dict] = None):
        return Reply(
            status=status,
            payload=self._to_bytes(payload),
            cookie=self._to_bytes(cookie),
        )

    async def Query(self, request: Request, context) -> Reply:
        p = self.processors.get(request.type)
        if not p:
            self.logger.error(f"received invalid query type: {request.type}")
            return self._make_reply(EdgeApiStatus.INVALID_REQUEST)

        self.logger.info(f"received request {request.type}")

        payload = self._to_dict("payload", request.payload)
        if not payload:
            return self._make_reply(EdgeApiStatus.INVALID_REQUEST)

        status, query_info = self._get_query_info(request)
        if status != EdgeApiStatus.OK:
            return self._make_reply(status, NONE_DATA)

        try:
            check_f, to_request_f, process_f = p

            if check_f is not None:
                is_valid = check_f(payload, query_info, request.type)
                if not is_valid:
                    return self._make_reply(EdgeApiStatus.INVALID_REQUEST)

            status, req = to_request_f(payload, query_info)

            if status != EdgeApiStatus.OK:
                return self._make_reply(status)

            # resp = await asyncio.to_thread(process_f, request=req)
            loop = self.aio_ctx.get_event_loop()
            resp = await loop.run_in_executor(self.workers, process_f, req)
            if not resp:
                raise RuntimeError("no result from QueryHandler.")
            cookie = resp.pop("cookie", None)
            return self._make_reply(EdgeApiStatus.OK, resp, cookie)
        except Exception as ex:
            self.logger.error(f"error processing request: {secure_format_exception(ex)}")
            return self._make_reply(EdgeApiStatus.ERROR)

    def _to_dict(self, name: str, data: bytes) -> Optional[dict]:
        try:
            return json.loads(data)
        except Exception as ex:
            self.logger.error(f"error decoding {name} data: {secure_format_exception(ex)}")
            return None

    def _get_query_info(self, request: Request) -> (str, Optional[_QueryInfo]):
        header = self._to_dict("header", request.header)
        if not header:
            return EdgeApiStatus.INVALID_REQUEST, None

        device_id = header.get(HttpHeaderKey.DEVICE_ID, None)
        if not device_id:
            self.logger.error("missing device ID header")
            return EdgeApiStatus.INVALID_REQUEST, None

        device_info = DeviceInfo(device_id)
        device_info_header = header.get(HttpHeaderKey.DEVICE_INFO, None)
        if device_info_header:
            device_info.from_query_string(device_info_header)

        user_info_header = header.get(HttpHeaderKey.USER_INFO, None)
        if user_info_header:
            user_info = UserInfo()
            user_info.from_query_string(user_info_header)
        else:
            user_info = None

        cookie = None
        if request.cookie != NONE_DATA:
            cookie = self._to_dict("cookie", request.cookie)
            if not cookie:
                return EdgeApiStatus.INVALID_REQUEST, None

            if CookieKey.DEVICE_SELECTION_ID in cookie:
                cookie[CookieKey.DEVICE_SELECTION_ID] = int(cookie[CookieKey.DEVICE_SELECTION_ID])

            if CookieKey.MODEL_VERSION in cookie:
                cookie[CookieKey.MODEL_VERSION] = int(cookie[CookieKey.MODEL_VERSION])

        self.logger.info(f"QueryInfo: {device_id=} {device_info=} {user_info=} {cookie=}")

        return EdgeApiStatus.OK, _QueryInfo(device_info, user_info, cookie)

    def _check_payload_keys(self, payload: dict, keys, query_type: str) -> bool:
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            if key not in payload:
                self.logger.error(f"missing {key} from {query_type}")
                return False
        return True

    def _to_job_request(self, payload: dict, info: _QueryInfo) -> (str, Optional[JobRequest]):
        job_req = JobRequest(info.device_info, info.user_info, **payload)
        return EdgeApiStatus.OK, job_req

    def _check_task_request(self, payload: dict, info: _QueryInfo, req_type: str) -> bool:
        if not self._check_payload_keys(payload, EdgeProtoKey.JOB_ID, req_type):
            return False
        else:
            return True

    def _to_task_request(self, payload: dict, info: _QueryInfo) -> (str, Optional[TaskRequest]):
        task_req = TaskRequest(info.device_info, info.user_info, payload.get(EdgeProtoKey.JOB_ID), info.cookie)
        return EdgeApiStatus.OK, task_req

    def _check_selection_request(self, payload: dict, info: _QueryInfo, req_type: str) -> bool:
        if not self._check_payload_keys(payload, EdgeProtoKey.JOB_ID, req_type):
            return False
        else:
            return True

    def _to_selection_request(self, payload: dict, info: _QueryInfo) -> (str, Optional[SelectionRequest]):
        return EdgeApiStatus.OK, SelectionRequest(info.device_info, payload.get(EdgeProtoKey.JOB_ID))

    def _check_result_report(self, payload: dict, info: _QueryInfo, req_type: str) -> bool:
        if not self._check_payload_keys(
            payload,
            [
                EdgeProtoKey.JOB_ID,
                EdgeProtoKey.TASK_ID,
                EdgeProtoKey.TASK_NAME,
                EdgeProtoKey.STATUS,
                EdgeProtoKey.RESULT,
            ],
            req_type,
        ):
            return False
        elif not info.cookie:
            self.logger.error(f"missing cookie from {req_type}")
            return False
        else:
            return True

    def _to_result_report(self, payload: dict, info: _QueryInfo) -> (str, Optional[ResultReport]):
        report = ResultReport(
            info.device_info,
            info.user_info,
            job_id=payload.get(EdgeProtoKey.JOB_ID),
            task_id=payload.get(EdgeProtoKey.TASK_ID),
            task_name=payload.get(EdgeProtoKey.TASK_NAME),
            status=payload.get(EdgeProtoKey.STATUS),
            result=payload.get(EdgeProtoKey.RESULT),
            cookie=info.cookie,
        )
        report.update(payload)
        return EdgeApiStatus.OK, report


class Server:

    def __init__(
        self,
        handler: QueryHandler,
        address: str,
        grpc_options=None,
        max_workers=100,
    ):
        self.aio_ctx = AioContext.get_global_context()
        self.logger = get_obj_logger(self)
        self.handler = handler
        self.address = address
        self.grpc_options = grpc_options
        self.max_workers = max_workers
        self.grpc_server = None
        self.grpc_server_stop_grace = 0.5
        self.waiter = threading.Event()

    async def _start(self):
        self.grpc_server = grpc.aio.server(options=self.grpc_options)
        servicer = Servicer(self.handler, self.aio_ctx, self.max_workers)
        add_EdgeApiServicer_to_server(servicer, self.grpc_server)
        self.grpc_server.add_insecure_port(self.address)
        self.logger.info(f"added insecure port at {self.address}")

        self.logger.info("starting server engine")
        await self.grpc_server.start()
        self.logger.info("started server and wait for termination")
        await self.grpc_server.wait_for_termination()

    async def _shutdown(self):
        try:
            await self.grpc_server.stop(grace=self.grpc_server_stop_grace)

            # Note that self.grpc_server.stop returns immediately. Since we gave 0.5 grace time for RPCs to end,
            # we wait here until RPCs are done or aborted.
            # Without this, we may run into "excepthook" error at the end of the program since the GRPC server isn't
            # properly shutdown.
            await asyncio.sleep(self.grpc_server_stop_grace)
            self.grpc_server = None
            self.logger.debug("Server is stopped!")
        except Exception as ex:
            self.logger.debug(f"exception shutdown server: {secure_format_exception(ex)}")

    def start(self):
        self.aio_ctx.run_coro(self._start())
        self.logger.info("waiting for server to finish")
        self.waiter.wait()
        self.logger.info("server is done")

    def shutdown(self):
        self.aio_ctx.run_coro(self._shutdown())
        self.waiter.set()
        self.logger.info("Shutting Down Server")
