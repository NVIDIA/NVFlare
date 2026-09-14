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
from uuid import uuid4

import pytest

from nvflare.dashboard.application.constants import FLARE_DASHBOARD_NAMESPACE

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
