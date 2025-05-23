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
from typing import Union

from flask import Blueprint, request

from nvflare.edge.constants import EdgeApiStatus, HttpHeaderKey
from nvflare.edge.web.models.api_error import ApiError
from nvflare.edge.web.models.base_model import EdgeProtoKey
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.selection_request import SelectionRequest
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.user_info import UserInfo
from nvflare.edge.web.service.query import Query


class APIQuery:

    def __init__(self):
        self.lcp_mapping_file = None
        self.ca_cert_file = None
        self.query = None

    def set_lcp_mapping(self, file_name: str):
        self.lcp_mapping_file = file_name

    def set_ca_cert(self, file_name: str):
        self.ca_cert_file = file_name

    def start(self):
        self.query = Query(lcp_mapping_file=self.lcp_mapping_file, ca_cert_file=self.ca_cert_file)

    def __call__(self, req: Union[TaskRequest, JobRequest, SelectionRequest, ResultReport]):
        return self.query(req)


feg_bp = Blueprint("feg", __name__)
api_query = APIQuery()


def _process_headers() -> dict:
    headers = request.headers
    device_id = headers.get(HttpHeaderKey.DEVICE_ID, None)
    if not device_id:
        raise ApiError(400, EdgeApiStatus.INVALID_REQUEST, "Device ID missing")

    d = {}
    device_info = DeviceInfo(device_id)
    device_info_header = headers.get(HttpHeaderKey.DEVICE_INFO, None)
    if device_info_header:
        device_info.from_query_string(device_info_header)
    d[EdgeProtoKey.DEVICE_INFO] = device_info

    user_info_header = headers.get(HttpHeaderKey.USER_INFO, None)
    if user_info_header:
        user_info = UserInfo()
        user_info.from_query_string(user_info_header)
    else:
        user_info = None
    if user_info:
        d[EdgeProtoKey.USER_INFO] = user_info
    return d


def _update_body(d: dict):
    data = request.get_json()
    d.update(data)


def _update_args(d: dict, keys: dict):
    for k, default_value in keys.items():
        arg_value = request.args.get(k)
        if arg_value is None:
            arg_value = default_value
        if arg_value is not None:
            d[k] = arg_value


def _do_query(req):
    resp = api_query(req)
    if not resp:
        raise ApiError(400, EdgeApiStatus.INVALID_REQUEST, "unknown request type")
    return resp


@feg_bp.route("/job", methods=["POST"])
def job_view():
    d = _process_headers()
    _update_body(d)
    error, req = JobRequest.from_dict(d)
    if error:
        raise ApiError(400, EdgeApiStatus.INVALID_REQUEST, error)
    return _do_query(req)


@feg_bp.route("/task", methods=["POST"])
def task_view():
    d = _process_headers()
    _update_args(d, {EdgeProtoKey.JOB_ID: None})
    _update_body(d)

    error, req = TaskRequest.from_dict(d)
    if error:
        raise ApiError(400, EdgeApiStatus.INVALID_REQUEST, error)
    return _do_query(req)


@feg_bp.route("/result", methods=["POST"])
def result_view():
    d = _process_headers()
    _update_args(
        d,
        {
            EdgeProtoKey.JOB_ID: None,
            EdgeProtoKey.TASK_ID: None,
            EdgeProtoKey.TASK_NAME: None,
            EdgeProtoKey.STATUS: EdgeApiStatus.OK,
        },
    )
    _update_body(d)

    error, req = ResultReport.from_dict(d)
    if error:
        raise ApiError(400, EdgeApiStatus.INVALID_REQUEST, error)
    return _do_query(req)


@feg_bp.route("/selection", methods=["POST"])
def selection_view():
    d = _process_headers()
    _update_args(d, {EdgeProtoKey.JOB_ID: None})
    _update_body(d)

    error, req = SelectionRequest.from_dict(d)
    if error:
        raise ApiError(400, EdgeApiStatus.INVALID_REQUEST, error)
    return _do_query(req)
