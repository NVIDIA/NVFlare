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
import os.path
from typing import Tuple, Union

from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.edge.constants import EdgeApiStatus
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.selection_request import SelectionRequest
from nvflare.edge.web.models.selection_response import SelectionResponse
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.drivers.grpc.utils import get_grpc_client_credentials
from nvflare.fuel.utils.hash_utils import UniformHash
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.security.logging import secure_format_exception

from .client import EdgeApiClient
from .utils import (
    grpc_reply_to_job_response,
    grpc_reply_to_result_response,
    grpc_reply_to_selection_response,
    grpc_reply_to_task_response,
    job_request_to_grpc_request,
    result_report_to_grpc_request,
    selection_request_to_grpc_request,
    task_request_to_grpc_request,
)


class Query:

    def __init__(self, lcp_mapping_file: str = None, ca_cert_file: str = None):
        ssl_credentials = None
        if ca_cert_file:
            if not os.path.isfile(ca_cert_file):
                raise ValueError(f"specified ca_cert_file {ca_cert_file} does not exist or is not a file")
            params = {
                DriverParams.CONNECTION_SECURITY.value: ConnectionSecurity.TLS,
                DriverParams.CA_CERT.value: ca_cert_file,
            }
            ssl_credentials = get_grpc_client_credentials(params)

        self.lcp_list = []
        # TODO: add ssl support
        self.client = EdgeApiClient(ssl_credentials=ssl_credentials)
        # self.client = EdgeApiClient()
        self.logger = get_obj_logger(self)
        if lcp_mapping_file:
            self.load_lcp_map(lcp_mapping_file)

    def _add_lcp(self, name: str, addr: str):
        self.lcp_list.append((name, addr))

    def _map(self, device_id: str) -> Tuple[str, str]:
        uniform_hash = UniformHash(len(self.lcp_list))
        index = uniform_hash.hash(device_id)
        return self.lcp_list[index]

    def load_lcp_map(self, mapping_file: str):
        with open(mapping_file, "r") as f:
            mapping = json.load(f)

        for name, config in mapping.items():
            host = config["host"]
            port = config["port"]
            addr = f"{host}:{port}"
            self._add_lcp(name, addr)

    def _query(
        self,
        request: Union[TaskRequest, JobRequest, SelectionRequest, ResultReport],
        to_grpc_f,
        from_grpc_f,
        default_response,
    ):
        if not self.lcp_list:
            self.logger.error("No LCP configured")
            return default_response

        grpc_req = to_grpc_f(request)
        device_id = request.get_device_id()
        name, addr = self._map(device_id)
        self.logger.debug(f"sending request {type(request)} to {name} at {addr}")

        try:
            grpc_reply = self.client.query(addr, grpc_req)
            resp = from_grpc_f(grpc_reply)
            if not resp:
                resp = default_response
            return resp
        except Exception as ex:
            self.logger.error(f"exception querying grpc service: {secure_format_exception(ex)}")
            return default_response

    def __call__(self, request: Union[TaskRequest, JobRequest, SelectionRequest, ResultReport]):
        if isinstance(request, JobRequest):
            return self._query(
                request, job_request_to_grpc_request, grpc_reply_to_job_response, JobResponse(EdgeApiStatus.RETRY)
            )
        elif isinstance(request, TaskRequest):
            return self._query(
                request, task_request_to_grpc_request, grpc_reply_to_task_response, TaskResponse(EdgeApiStatus.RETRY)
            )
        elif isinstance(request, SelectionRequest):
            return self._query(
                request,
                selection_request_to_grpc_request,
                grpc_reply_to_selection_response,
                SelectionResponse(EdgeApiStatus.RETRY),
            )
        elif isinstance(request, ResultReport):
            return self._query(
                request,
                result_report_to_grpc_request,
                grpc_reply_to_result_response,
                ResultResponse(EdgeApiStatus.RETRY),
            )
        else:
            self.logger.error(f"received invalid request type: {type(request)}")
            return None
