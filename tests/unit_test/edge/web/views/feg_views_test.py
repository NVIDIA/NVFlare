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

from urllib.parse import urlencode

import pytest
from flask import Flask

from nvflare.edge.constants import EdgeApiStatus, HttpHeaderKey
from nvflare.edge.web.models.api_error import ApiError
from nvflare.edge.web.models.base_model import EdgeProtoKey
from nvflare.edge.web.views.feg_views import _process_headers


@pytest.fixture
def app():
    return Flask(__name__)


def _process_headers_with(app, headers):
    with app.test_request_context("/job", headers=headers):
        return _process_headers()


def test_process_headers_uses_device_id_header_if_device_info_is_absent(app):
    headers = {HttpHeaderKey.DEVICE_ID: "device-1"}

    parsed = _process_headers_with(app, headers)

    device_info = parsed[EdgeProtoKey.DEVICE_INFO]
    assert device_info.device_id == "device-1"


def test_process_headers_uses_device_id_header_if_device_info_has_no_device_id(app):
    headers = {
        HttpHeaderKey.DEVICE_ID: "device-1",
        HttpHeaderKey.DEVICE_INFO: urlencode({"platform": "ios"}),
    }

    parsed = _process_headers_with(app, headers)

    device_info = parsed[EdgeProtoKey.DEVICE_INFO]
    assert device_info.device_id == "device-1"
    assert device_info.platform == "ios"


def test_process_headers_accepts_matching_device_info_device_id(app):
    headers = {
        HttpHeaderKey.DEVICE_ID: "device-1",
        HttpHeaderKey.DEVICE_INFO: urlencode({EdgeProtoKey.DEVICE_ID: "device-1", "platform": "ios"}),
    }

    parsed = _process_headers_with(app, headers)

    device_info = parsed[EdgeProtoKey.DEVICE_INFO]
    assert device_info.device_id == "device-1"
    assert device_info.platform == "ios"


def test_process_headers_rejects_mismatched_device_info_device_id(app):
    headers = {
        HttpHeaderKey.DEVICE_ID: "device-1",
        HttpHeaderKey.DEVICE_INFO: urlencode({EdgeProtoKey.DEVICE_ID: "device-2", "platform": "ios"}),
    }

    with pytest.raises(ApiError) as exc_info:
        _process_headers_with(app, headers)

    assert exc_info.value.status_code == 400
    assert exc_info.value.status == EdgeApiStatus.INVALID_REQUEST
    assert "Device ID mismatch" in str(exc_info.value)
