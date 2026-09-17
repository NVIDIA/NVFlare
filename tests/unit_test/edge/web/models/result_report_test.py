# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.edge.constants import EdgeApiStatus
from nvflare.edge.web.models.base_model import EdgeProtoKey
from nvflare.edge.web.models.result_report import ResultReport


def _result_report_payload(cookie=None):
    payload = {
        EdgeProtoKey.DEVICE_INFO: {EdgeProtoKey.DEVICE_ID: "device-1"},
        EdgeProtoKey.USER_INFO: {"user_id": "user-1"},
        EdgeProtoKey.JOB_ID: "job-1",
        EdgeProtoKey.TASK_ID: "task-1",
        EdgeProtoKey.TASK_NAME: "train",
        EdgeProtoKey.STATUS: EdgeApiStatus.OK,
        EdgeProtoKey.RESULT: {"accuracy": 0.9},
    }
    if cookie is not None:
        payload[EdgeProtoKey.COOKIE] = cookie
    return payload


def test_result_report_from_dict_accepts_missing_cookie():
    error, report = ResultReport.from_dict(_result_report_payload())

    assert error == ""
    assert report.cookie is None
    assert report.result == {"accuracy": 0.9}


def test_result_report_from_dict_preserves_cookie():
    cookie = {"model_version": 7}

    error, report = ResultReport.from_dict(_result_report_payload(cookie))

    assert error == ""
    assert report.cookie == cookie
