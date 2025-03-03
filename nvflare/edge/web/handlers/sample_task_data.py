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
import random
import uuid

from nvflare.edge.web.models.api_error import ApiError
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.edge.web.models.user_info import UserInfo

jobs = [
    JobResponse(
        "OK",
        str(uuid.uuid4()),
        str(uuid.uuid4()),
        "demo_job",
        "ExecuTorch",
        job_data={"executorch_parameters": [1.2, 3.4, 5.6]},
    ),
    JobResponse("OK", str(uuid.uuid4()), str(uuid.uuid4()), "xgb_job", "xgboost"),
    JobResponse("OK", str(uuid.uuid4()), str(uuid.uuid4()), "core_job", "coreML"),
    JobResponse("RETRY", str(uuid.uuid4()), retry_wait=60),
]

job_state = {}

demo_tasks = ["train", "validate", "end_run"]


def handle_job_request(device_info: DeviceInfo, user_info: UserInfo, study_request: JobRequest) -> JobResponse:

    response = random.choice(jobs)
    response.session_id = None

    if response.status == "OK":
        session_id = str(uuid.uuid4())
        response.session_id = session_id
        job_state[session_id] = {
            "next_task": demo_tasks[0],
            "task_id": str(uuid.uuid4()),
            "job": response,
        }

    return response


def handle_task_request(device_info: DeviceInfo, user_info: UserInfo, task_request: TaskRequest) -> TaskResponse:

    session_id = task_request.session_id
    state = job_state.get(session_id)
    if not state:
        raise ApiError(400, "NOT_FOUND", f"Session ID {session_id} not found")

    task_name = state["next_task"]
    task_id = state["task_id"]

    reply = TaskResponse("OK", session_id, None, task_id, task_name, {})

    return reply


def handle_result_report(device_info: DeviceInfo, user_info: UserInfo, result_report: ResultReport) -> ResultResponse:
    session_id = result_report.session_id
    state = job_state.get(session_id)
    if not state:
        raise ApiError(400, "NOT_FOUND", f"Session ID {session_id} not found")

    index = demo_tasks.index(result_report.task_name)
    if index < 0:
        raise ApiError(400, "INVALID_TASK", f"Task {result_report.task_name} not found")
    if index == len(demo_tasks):
        status = "DONE"
    else:
        status = "OK"

    task_id = state["task_id"]
    state["next_task"] = demo_tasks[index + 1]
    state["task_id"] = str(uuid.uuid4())

    reply = ResultResponse(status, None, session_id, task_id, result_report.task_name)

    return reply
