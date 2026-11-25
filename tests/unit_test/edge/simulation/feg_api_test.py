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

from nvflare.edge.constants import EdgeApiStatus
from nvflare.edge.simulation import feg_api
from nvflare.edge.simulation.simulator import Simulator
from nvflare.edge.web.models.api_error import ApiError
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.task_response import TaskResponse
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


@pytest.mark.parametrize(
    "failures, expected_calls",
    [
        ([requests.ConnectTimeout], 2),
        ([requests.ConnectTimeout] * 3, 3),
        ([requests.ReadTimeout], 1),
    ],
)
def test_result_post_retries_only_safe_connect_failures(api, monkeypatch, failures, expected_calls):
    response = Mock(status_code=200, json=lambda: {"status": "OK"})
    post = Mock(side_effect=[*(error("timeout") for error in failures), response])
    monkeypatch.setattr(feg_api.requests, "post", post)
    simulator = Simulator("test-job", Mock(), num_workers=1)
    simulator.set_send_func(lambda request, device: api.report_result(request))
    device = Mock(
        device_id=f"{simulator.device_id_prefix}-1",
        cookie={"selection": "id"},
    )
    device.get_device_info.return_value = api.device_info
    device.get_user_info.return_value = api.user_info
    device.do_task.return_value = {"weights": [1]}
    task = TaskResponse(EdgeApiStatus.OK, job_id="job-1", task_id="task-1", task_name="train")
    try:
        simulator._do_learn(task, device, "selection-1")
        assert post.call_count == expected_calls
        assert all(call == post.call_args_list[0] for call in post.call_args_list)
        assert post.call_args.kwargs["json"]["result"] == {"weights": [1]}
        device.do_task.assert_called_once_with(task)
        assert simulator._determine_my_devices({device.device_id: "selection-1"}) == {}
    finally:
        simulator.worker_pool.shutdown()
