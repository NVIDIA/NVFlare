# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import pytest

from nvflare.dashboard.application.constants import FLARE_DASHBOARD_NAMESPACE
from tests.dashboard_test_utils import TEST_PW, TEST_USER


@pytest.fixture(scope="session")
def app(dashboard_app):
    return dashboard_app


@pytest.fixture(scope="session")
def client(app):
    return app.test_client()


@pytest.fixture(scope="session")
def access_token(client):
    response = client.post(FLARE_DASHBOARD_NAMESPACE + "/api/v1/login", json={"email": TEST_USER, "password": TEST_PW})
    assert response.status_code == 200
    return response.json["access_token"]


@pytest.fixture(scope="session")
def auth_header(access_token):
    return {"Authorization": "Bearer " + access_token}
