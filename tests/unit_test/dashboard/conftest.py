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
import os
import shutil
import tempfile

import pytest

from nvflare.dashboard.application import init_app
from nvflare.dashboard.application.constants import FLARE_DASHBOARD_NAMESPACE

TEST_USER = "admin@test.com"
TEST_PW = "testing1234"


@pytest.fixture(scope="session")
def app():

    web_root = tempfile.mkdtemp(prefix="nvflare-")
    sqlite_file = os.path.join(web_root, "db.sqlite")
    if os.path.exists(sqlite_file):
        os.remove(sqlite_file)
    os.environ["DATABASE_URL"] = f"sqlite:///{sqlite_file}"
    os.environ["NVFL_CREDENTIAL"] = f"{TEST_USER}:{TEST_PW}:nvidia"
    app = init_app()
    app.config.update(
        {
            "TESTING": True,
            "ENV": "prod",  # To get rid of the performance warning
        }
    )

    yield app

    # Cleanup
    shutil.rmtree(web_root, ignore_errors=True)


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
