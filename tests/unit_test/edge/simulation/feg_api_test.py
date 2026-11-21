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

from unittest.mock import Mock

import pytest
import requests

from nvflare.edge.simulation import feg_api
from nvflare.edge.web.models.api_error import ApiError
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.user_info import UserInfo


@pytest.fixture
def api():
    return feg_api.FegApi("https://gateway.example/api/", DeviceInfo("device-1"), UserInfo("user-1"))


def get_job(api):
    return api.get_job(JobRequest("test-job", api.device_info, api.user_info))


def test_job_request_has_finite_timeouts_and_preserves_response(api, monkeypatch):
    post = Mock(return_value=Mock(status_code=200, json=lambda: {"status": "OK", "job_id": "job-1"}))
    monkeypatch.setattr(feg_api.requests, "post", post)

    result = get_job(api)

    assert isinstance(result, JobResponse)
    assert result.job_id == "job-1"
    post.assert_called_once_with(
        "https://gateway.example/api/job",
        params={},
        json={"job_name": "test-job", "capabilities": None},
        headers=api.common_headers,
        timeout=(10.0, 60.0),
    )


@pytest.mark.parametrize("error_type", [requests.ConnectTimeout, requests.ReadTimeout])
def test_request_timeouts_propagate_to_simulator(api, monkeypatch, error_type):
    error = error_type("gateway stalled")
    post = Mock(side_effect=error)
    monkeypatch.setattr(feg_api.requests, "post", post)

    with pytest.raises(error_type) as caught:
        get_job(api)

    assert caught.value is error
    assert post.call_count == 1
    assert post.call_args.kwargs["timeout"] == (10.0, 60.0)


def test_http_error_preserves_api_error(api, monkeypatch):
    details = {"reason": "job unavailable"}
    monkeypatch.setattr(feg_api.requests, "post", Mock(return_value=Mock(status_code=503, json=lambda: details)))

    with pytest.raises(ApiError) as caught:
        get_job(api)

    assert caught.value.status_code == 503
    assert caught.value.details == details
