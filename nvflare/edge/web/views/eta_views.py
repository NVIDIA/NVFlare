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

from nvflare.edge.web.controllers.eta_controller import handle_job_request, handle_task_request, \
    handle_result_report
from nvflare.edge.web.models.api_error import ApiError
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.user_info import UserInfo

eta_bp = Blueprint("eta", __name__)


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


@eta_bp.route("/job", methods=["POST"])
def job_view():
    device_info, user_info = process_header(request.headers)
    data = request.get_json()
    req = JobRequest(**data)
    reply = handle_job_request(device_info, user_info, req)

    return reply


@eta_bp.route("/task", methods=["GET"])
def task_view():
    device_info, user_info = process_header(request.headers)
    session_id = request.args.get("session_id")
    job_id = request.args.get("job_id")

    req = TaskRequest(session_id, job_id)

    return handle_task_request(device_info, user_info, req)


@eta_bp.route("/result", methods=["POST"])
def result_view():
    device_info, user_info = process_header(request.headers)
    session_id = request.args.get("session_id")
    task_id = request.args.get("task_id")
    task_name = request.args.get("task_name")
    data = request.get_json()

    req = ResultReport(session_id, task_id, task_name)
    req.update(data)

    return handle_result_report(device_info, user_info, req)
