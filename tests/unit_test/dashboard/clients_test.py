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
