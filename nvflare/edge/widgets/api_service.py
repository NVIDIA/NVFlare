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
import threading

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace
from nvflare.edge.constants import EdgeContextKey, EdgeEventType
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.selection_request import SelectionRequest
from nvflare.edge.web.models.selection_response import SelectionResponse
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.edge.web.rpc.query_handler import QueryHandler
from nvflare.edge.web.rpc.server import EdgeApiServer
from nvflare.widgets.widget import Widget


class ApiService(Widget, QueryHandler):
    def __init__(self, max_workers=100, lcp_mapping_file_name="lcp_map.json"):
        Widget.__init__(self)
        QueryHandler.__init__(self)

        self.lcp_mapping_file_name = lcp_mapping_file_name
        self.max_workers = max_workers
        self.address = None
        self.engine = None
        self.server = None

        self.register_event_handler(EventType.SYSTEM_START, self._startup)
        self.register_event_handler(EventType.SYSTEM_END, self._shutdown)

    def _handle_all_request(self, request, event_type: str):
        with self.engine.new_context() as fl_ctx:
            assert isinstance(fl_ctx, FLContext)
            fl_ctx.set_prop(EdgeContextKey.REQUEST_FROM_EDGE, request, sticky=False, private=True)
            self.fire_event(event_type, fl_ctx)
            result = fl_ctx.get_prop(EdgeContextKey.REPLY_TO_EDGE)
            if not result:
                self.logger.warning(f"no result from ETD for event {event_type}")
            return result

    def handle_job_request(self, request: JobRequest) -> JobResponse:
        return self._handle_all_request(request, EdgeEventType.EDGE_JOB_REQUEST_RECEIVED)

    def handle_task_request(self, request: TaskRequest) -> TaskResponse:
        return self._handle_all_request(request, EdgeEventType.EDGE_TASK_REQUEST_RECEIVED)

    def handle_selection_request(self, request: SelectionRequest) -> SelectionResponse:
        return self._handle_all_request(request, EdgeEventType.EDGE_SELECTION_REQUEST_RECEIVED)

    def handle_result_report(self, request: ResultReport) -> ResultResponse:
        return self._handle_all_request(request, EdgeEventType.EDGE_RESULT_REPORT_RECEIVED)

    def _startup(self, _event_type: str, fl_ctx: FLContext):
        self.engine = fl_ctx.get_engine()
        workspace: Workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        mapping_file = workspace.get_file_path_in_site_config(self.lcp_mapping_file_name)
        with open(mapping_file, "r") as mapping_file:
            mapping = json.load(mapping_file)
        client_name = fl_ctx.get_identity_name()
        client_config = mapping.get(client_name)
        if client_config is None:
            raise ValueError(f"Client {client_name} not found in {mapping_file}")
        host = client_config.get("host")
        port = client_config.get("port")
        self.address = f"{host}:{port}"
        self.server = EdgeApiServer(
            handler=self,
            address=self.address,
            max_workers=self.max_workers,
        )
        t = threading.Thread(target=self.server.start, daemon=True)
        t.start()
        self.log_info(fl_ctx, f"Edge API GRPC Service is started on address {self.address}")

    def _shutdown(self, _event_type: str, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"Edge API GRPC Service on address {self.address} is shutting down")
        if self.server:
            self.server.shutdown()
