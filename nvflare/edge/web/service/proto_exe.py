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
from typing import Union

from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.selection_request import SelectionRequest
from nvflare.edge.web.models.task_request import TaskRequest


class ReqType:
    JOB = "job"
    TASK = "task"
    REPORT = "report"


class ProtoExecutor:

    def __init__(self, config_file: str):
        with open(config_file, "r") as f:
            config = json.load(f)

        headers = config.get("headers")
        if not headers:
            raise ValueError(f"config error in {config_file}: missing 'headers'")

        if not isinstance(headers, dict):
            raise ValueError(f"config error in {config_file}: 'headers' should be dict but got {type(headers)}")

        user_info = headers.get("user_info")
        if not user_info:
            raise ValueError(f"config error in {config_file}: missing 'user_info' in headers")

        if not isinstance(user_info, dict):
            raise ValueError(f"config error in {config_file}: 'user_info' should be dict but got {type(user_info)}")

        device_info = headers.get("device_info")
        if not device_info:
            raise ValueError(f"config error in {config_file}: missing 'device_info' in headers")

        if not isinstance(device_info, dict):
            raise ValueError(f"config error in {config_file}: 'device_info' should be dict but got {type(device_info)}")

        self.device_info = device_info
        self.user_info = user_info

        steps = config.get("steps")
        if not steps:
            raise ValueError(f"config error in {config_file}: missing 'steps'")

        if not isinstance(steps, list):
            raise ValueError(f"config error in {config_file}: steps should be list but got {type(steps)}")

        self.steps = []
        for i, s in enumerate(steps):
            if not isinstance(s, dict):
                raise ValueError(f"config error in {config_file}: step {i} should be dict but got {type(s)}")

            if len(s) != 2:
                raise ValueError(f"config error in {config_file}: step {i} should have 2 element but got {len(s)}")

            resp = None
            req = None
            req_type = None
            for k, v in s.items():
                if k == "response":
                    resp = v
                else:
                    req_type = k
                    req = v

            if not resp:
                raise ValueError(f"config error in {config_file}: missing 'response' in step {i}")

            if not isinstance(resp, dict):
                raise ValueError(
                    f"config error in {config_file}: 'response' in step {i} should be dict but got {type(resp)}"
                )

            if not isinstance(req, dict):
                raise ValueError(
                    f"config error in {config_file}: '{req_type}' in step {i} should be dict but got {type(req)}"
                )

            valid_req_types = [ReqType.JOB, ReqType.TASK, ReqType.REPORT]
            if req_type not in valid_req_types:
                raise ValueError(
                    f"config error in {config_file}: invalid type '{req_type}' in step {i} (not in {valid_req_types})"
                )

            self.steps.append((req_type, req, resp))

        self.current_step = 0
        print(f"Loaded {config_file}: {len(self.steps)} steps")

    def _compare_dict(self, received: dict, expected: dict):
        for k, v in expected.items():
            if k not in received:
                return f"missing '{k}' in request"

            received_value = received[k]
            if isinstance(v, dict) and isinstance(received_value, dict):
                err = self._compare_dict(received_value, v)
                if err:
                    return err

            if v != received_value:
                return f"value mismatch for '{k}': expect {v}, received {received_value}"

        return ""

    def __call__(self, request: Union[TaskRequest, JobRequest, SelectionRequest, ResultReport]):
        if self.current_step == len(self.steps):
            raise RuntimeError("test is already done")

        i = self.current_step

        if isinstance(request, TaskRequest):
            req_type = ReqType.TASK
            user_info = request.user_info
            device_info = request.device_info
        elif isinstance(request, JobRequest):
            req_type = ReqType.JOB
            user_info = request.user_info
            device_info = request.device_info
        elif isinstance(request, ResultReport):
            req_type = ReqType.REPORT
            user_info = request.user_info
            device_info = request.device_info
        else:
            raise RuntimeError(f"Step {i}: bad request type: {type(request)}")

        expected_type, expected_req, resp = self.steps[self.current_step]
        if req_type != expected_type:
            raise RuntimeError(f"Step {i}: expects request type '{expected_type}' but got {req_type}")

        # make sure that all expected values are available from the received request
        err = self._compare_dict(request, expected_req)
        if err:
            raise RuntimeError(f"Step {i} request error: {err}")

        # make sure headers exist
        err = self._compare_dict(user_info, self.user_info)
        if err:
            raise RuntimeError(f"Step {i} user info: {err}")

        err = self._compare_dict(device_info, self.device_info)
        if err:
            raise RuntimeError(f"Step {i} device info: {err}")

        # everything is okay
        print(f"Step {i} {req_type}: OK")
        self.current_step += 1
        return resp
