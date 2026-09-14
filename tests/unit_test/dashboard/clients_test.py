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
import io
import json
from uuid import uuid4
from zipfile import ZipFile

import pytest

from nvflare.dashboard.application import db
from nvflare.dashboard.application.constants import FLARE_DASHBOARD_NAMESPACE
from nvflare.dashboard.application.models import Client, Project
from nvflare.dashboard.application.store import Store

CLIENT1 = {"name": "site-1", "organization": "test.com", "capacity": {"num_gpus": 16, "mem_per_gpu_in_GiB": 64}}

CLIENT2 = {"name": "site-2", "organization": "example.com", "capacity": {"num_gpus": 4, "mem_per_gpu_in_GiB": 32}}

NEW_ORG = "company.com"


class TestClients:
    @pytest.fixture(scope="session")
    def client_ids(self, auth_header, client):

        response1 = client.post(FLARE_DASHBOARD_NAMESPACE + "/api/v1/clients", json=CLIENT1, headers=auth_header)
        assert response1.status_code == 201
        response2 = client.post(FLARE_DASHBOARD_NAMESPACE + "/api/v1/clients", json=CLIENT2, headers=auth_header)
        assert response2.status_code == 201

        return [response1.json["client"]["id"], response2.json["client"]["id"]]

    def test_create_clients(self, client_ids):
        # The fixture test the call already
        assert len(client_ids) == 2

    def test_get_all_clients(self, client, client_ids, auth_header):

        response = client.get(FLARE_DASHBOARD_NAMESPACE + "/api/v1/clients", headers=auth_header)

        assert response.status_code == 200
        assert len(response.json["client_list"]) == len(client_ids)

    def test_get_one_client(self, client, client_ids, auth_header):

        client_id = client_ids[0]
        response = client.get(FLARE_DASHBOARD_NAMESPACE + "/api/v1/clients/" + str(client_id), headers=auth_header)

        assert response.status_code == 200
        assert response.json["client"]["id"] == client_id
        assert response.json["client"]["name"] == CLIENT1["name"]

    def test_update_client(self, client, client_ids, auth_header):

        client_id = client_ids[0]
        response = client.patch(
            FLARE_DASHBOARD_NAMESPACE + "/api/v1/clients/" + str(client_id),
            json={"organization": NEW_ORG},
            headers=auth_header,
        )

        assert response.status_code == 200

        # Retrieve through API again
        response = client.get(FLARE_DASHBOARD_NAMESPACE + "/api/v1/clients/" + str(client_id), headers=auth_header)

        assert response.status_code == 200
        assert response.json["client"]["organization"] == NEW_ORG


class TestClientPropsRejection:
    @pytest.fixture(scope="class")
    @classmethod
    def creator_header(cls, client, auth_header):
        email = f"client-creator-{uuid4().hex}@test.com"
        response = client.post(
            FLARE_DASHBOARD_NAMESPACE + "/api/v1/users",
            json={"email": email, "password": "test-password", "organization": "test.com", "role": "member"},
        )
        assert response.status_code == 201
        user_id = response.json["user"]["id"]
        response = client.post(
            FLARE_DASHBOARD_NAMESPACE + "/api/v1/login", json={"email": email, "password": "test-password"}
        )
        assert response.status_code == 200
        yield {"Authorization": "Bearer " + response.json["access_token"]}
        response = client.delete(FLARE_DASHBOARD_NAMESPACE + f"/api/v1/users/{user_id}", headers=auth_header)
        assert response.status_code == 200

    @pytest.mark.parametrize("as_admin", [False, True])
    @pytest.mark.parametrize("props", [{"custom_ca_cert": "/server/private-file"}, {}, None, "invalid", []])
    def test_create_rejects_props(self, client, auth_header, creator_header, as_admin, props):
        url = FLARE_DASHBOARD_NAMESPACE + "/api/v1/clients"
        before = client.get(url, headers=auth_header).json["client_list"]
        response = client.post(
            url,
            json={**CLIENT1, "name": f"props-{uuid4().hex}", "props": props},
            headers=auth_header if as_admin else creator_header,
        )
        try:
            assert response.status_code == 400
            assert response.json["status"] == "error"
            assert "props" in response.json["message"]
            assert client.get(url, headers=auth_header).json["client_list"] == before
        finally:
            if response.status_code == 201:
                client.delete(url + f"/{response.json['client']['id']}", headers=auth_header)

    @pytest.mark.parametrize("as_admin", [False, True])
    @pytest.mark.parametrize("approval_state", [0, 100, 200])
    @pytest.mark.parametrize("props", [{"custom_ca_cert": "/server/private-file"}, {}, None, "invalid", []])
    def test_patch_rejects_props(self, client, auth_header, creator_header, as_admin, approval_state, props):
        response = client.post(
            FLARE_DASHBOARD_NAMESPACE + "/api/v1/clients",
            json={**CLIENT1, "name": f"props-{uuid4().hex}"},
            headers=creator_header,
        )
        assert response.status_code == 201
        url = FLARE_DASHBOARD_NAMESPACE + f"/api/v1/clients/{response.json['client']['id']}"
        try:
            response = client.patch(url, json={"approval_state": approval_state}, headers=auth_header)
            assert response.status_code == 200
            before = client.get(url, headers=auth_header).json["client"]
            response = client.patch(
                url,
                json={"props": props, "name": "changed-name", "approval_state": -1},
                headers=auth_header if as_admin else creator_header,
            )
            assert response.status_code == 400
            assert response.json["status"] == "error"
            assert "props" in response.json["message"]
            assert client.get(url, headers=auth_header).json["client"] == before

            # Ordinary client fields remain editable when props is absent.
            capacity = {"num_of_gpus": 2, "mem_per_gpu_in_GiB": 16}
            response = client.patch(url, json={"capacity": capacity}, headers=creator_header)
            assert response.status_code == 200
            assert response.json["client"]["capacity"] == capacity
        finally:
            response = client.delete(url, headers=auth_header)
            assert response.status_code == 200

    @pytest.mark.parametrize("malformed_props", [False, True])
    def test_blob_ignores_legacy_props(
        self, app, client, auth_header, creator_header, tmp_path, monkeypatch, malformed_props
    ):
        secret = b"legacy-client-props-must-not-disclose-this-file"
        target = tmp_path / "private-file"
        target.write_bytes(secret)
        legacy_props = (
            "invalid JSON"
            if malformed_props
            else json.dumps({"custom_ca_cert": str(target), "connection_security": "clear"})
        )
        # Operator-managed properties and the client's capacity must still reach the kit.
        monkeypatch.setenv("NVFL_WEB_ROOT", str(tmp_path))
        (tmp_path / "properties.json").write_text(json.dumps({"client": {"use_aio": True}}))
        capacity = {"num_of_gpus": 2, "mem_per_gpu_in_GiB": 16}
        response = client.post(
            FLARE_DASHBOARD_NAMESPACE + "/api/v1/clients",
            json={"name": f"legacy-{uuid4().hex}", "organization": "testorg", "capacity": capacity},
            headers=creator_header,
        )
        assert response.status_code == 201
        client_id = response.json["client"]["id"]
        url = FLARE_DASHBOARD_NAMESPACE + f"/api/v1/clients/{client_id}"
        with app.app_context():
            project = Project.query.first()
            original = {key: getattr(project, key) for key in ("short_name", "server1", "root_key", "root_cert")}
            project.short_name = "legacy-test"
            project.server1 = "server.test.com"
            Store.build_project(project)
            legacy_client = db.session.get(Client, client_id)
            legacy_client.props = legacy_props
            legacy_client.approval_state = 200
            db.session.commit()

        try:
            response = client.post(url + "/blob", json={"pin": "1234"}, headers=creator_header)
            assert response.status_code == 200
            assert response.headers["Content-Type"] == "zip"
            with ZipFile(io.BytesIO(response.data)) as kit:
                assert not any(name.endswith("customRootCA.pem") for name in kit.namelist())
                for name in kit.namelist():
                    if not name.endswith("/"):
                        assert secret not in kit.read(name, pwd=b"1234")
                resources = json.loads(kit.read("local/resources.json.default", pwd=b"1234"))
                resource_manager = next(c for c in resources["components"] if c["id"] == "resource_manager")
                for key, value in capacity.items():
                    assert resource_manager["args"][key] == value
                config = json.loads(kit.read("startup/fed_client.json", pwd=b"1234"))
                assert config["client"]["connection_security"] == "mtls"
                assert config["servers"][0]["service"]["scheme"] == "agrpc"

            with app.app_context():
                # Safety comes from ignoring stored props, without depending on a migration.
                assert db.session.get(Client, client_id).props == legacy_props
        finally:
            response = client.delete(url, headers=auth_header)
            assert response.status_code == 200
            with app.app_context():
                project = Project.query.first()
                for key, value in original.items():
                    setattr(project, key, value)
                db.session.commit()
