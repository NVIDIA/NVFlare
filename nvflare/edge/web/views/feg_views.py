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
from typing import Optional

from flask import Blueprint, request

from nvflare.edge.constants import EdgeApiStatus
from nvflare.edge.web.handlers.edge_task_handler import EdgeTaskHandler
from nvflare.edge.web.handlers.lcp_task_handler import LcpTaskHandler
from nvflare.edge.web.models.api_error import ApiError
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.user_info import UserInfo

feg_bp = Blueprint("feg", __name__)
task_handler: EdgeTaskHandler = LcpTaskHandler()


def process_header(headers) -> (DeviceInfo, Optional[UserInfo]):
    device_id = headers.get("X-Flare-Device-ID", None)
    if not device_id:
        raise ApiError(400, "invalid_request", "Device ID missing")

    device_info = DeviceInfo(device_id)
    device_info_header = headers.get("X-Flare-Device-Info", None)
    if device_info_header:
        device_info.from_query_string(device_info_header)

    user_info_header = headers.get("X-Flare-User-Info", None)
    if user_info_header:
        user_info = UserInfo()
        user_info.from_query_string(user_info_header)
    else:
        user_info = None

    return device_info, user_info


@feg_bp.route("/job", methods=["POST"])
def job_view():
    device_info, user_info = process_header(request.headers)
    data = request.get_json()
    req = JobRequest(device_info, user_info, **data)
    return task_handler.handle_job(req)


@feg_bp.route("/task", methods=["GET"])
def task_view():
    device_info, user_info = process_header(request.headers)
    job_id = request.args.get("job_id")
    cookie = request.args.get("cookie", {})

    req = TaskRequest(device_info, user_info, job_id, cookie)

    return task_handler.handle_task(req)


@feg_bp.route("/result", methods=["POST"])
def result_view():
    device_info, user_info = process_header(request.headers)
    job_id = request.args.get("job_id")
    task_id = request.args.get("task_id")
    task_name = request.args.get("task_name")
    status = request.args.get("status", EdgeApiStatus.OK)
    cookie = request.args.get("cookie")
    data = request.get_json()

    req = ResultReport(
        device_info,
        user_info,
        job_id,
        task_id,
        task_name,
        status=status,
        cookie=cookie,
    )
    req.update(data)

    return task_handler.handle_result(req)
