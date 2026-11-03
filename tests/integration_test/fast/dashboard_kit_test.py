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

"""Exercise real provisioning and system zip with an approved Dashboard client."""

import io
import shutil
from uuid import uuid4
from zipfile import ZipFile

import pytest

from nvflare.dashboard.application import db
from nvflare.dashboard.application.models import Client, Project
from nvflare.dashboard.application.store import Store
from tests.dashboard_test_utils import TEST_PW, TEST_USER


@pytest.mark.skipif(shutil.which("zip") is None, reason="Encrypted startup kits require the system zip executable")
def test_valid_client_names_generate_encrypted_startup_kit(dashboard_app):
    app = dashboard_app
    client = app.test_client()
    api = "/nvflare-dashboard/api/v1"
    response = client.post(api + "/login", json={"email": TEST_USER, "password": TEST_PW})
    headers = {"Authorization": "Bearer " + response.json["access_token"]}
    site_id = None
    with app.app_context():
        project = Project.query.first()
        original = {column.name: getattr(project, column.name) for column in Project.__table__.columns}
    try:
        with app.app_context():
            Store.set_project({"short_name": "kit_test", "server1": "localhost", "frozen": True})
        response = client.post(
            api + "/clients", headers=headers, json={"name": "encrypted-" + uuid4().hex, "organization": "Kit_org"}
        )
        assert response.status_code == 201
        site = response.json["client"]
        site_id = site["id"]
        assert (
            client.patch(api + f"/clients/{site_id}", headers=headers, json={"approval_state": 100}).status_code == 200
        )
        response = client.post(api + f"/clients/{site_id}/blob", json={"pin": "1234"}, headers=headers)
        assert response.status_code == 200
        assert response.headers["Content-Disposition"] == f"attachment; filename=\"{site['name']}.zip\""
        with ZipFile(io.BytesIO(response.data)) as kit:
            assert kit.getinfo("startup/fed_client.json").flag_bits & 1
            assert kit.read("startup/fed_client.json", pwd=b"1234")
        with app.app_context():
            assert db.session.get(Client, site_id).download_count == 1
    finally:
        if site_id is not None:
            client.delete(api + f"/clients/{site_id}", headers=headers)
        with app.app_context():
            project = Project.query.first()
            for name, value in original.items():
                setattr(project, name, value)
            db.session.commit()
