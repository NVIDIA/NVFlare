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

USER_NAME = "Test User"


class TestUsers:
    @pytest.fixture(scope="session")
    def first_user_id(self, auth_header, client):
        response = client.get(FLARE_DASHBOARD_NAMESPACE + "/api/v1/users", headers=auth_header)

        assert response.status_code == 200
        user_list = response.json["user_list"]
        assert len(user_list) >= 1

        return user_list[0]["id"]

    def test_get_all_users(self, first_user_id):
        # get_all_users is tested by user_id
        assert first_user_id

    def test_get_one_user(self, auth_header, client, first_user_id):

        user = self._get_one_user(auth_header, client, first_user_id)
        assert user["id"] == first_user_id

    def test_update_user(self, auth_header, client, first_user_id):
        user = {"name": USER_NAME}
        response = client.patch(
            FLARE_DASHBOARD_NAMESPACE + "/api/v1/users/" + str(first_user_id), json=user, headers=auth_header
        )
        assert response.status_code == 200

        new_user = self._get_one_user(auth_header, client, first_user_id)
        assert new_user["name"] == USER_NAME

    def test_create_and_delete(self, auth_header, client):
        test_user = {
            "name": USER_NAME,
            "email": "user@test.com",
            "password": "pw123456",
            "organization": "test.com",
            "role": "org_admin",
            "approval_state": 200,
        }

        response = client.post(FLARE_DASHBOARD_NAMESPACE + "/api/v1/users", json=test_user, headers=auth_header)
        assert response.status_code == 201
        new_id = response.json["user"]["id"]

        response = client.delete(FLARE_DASHBOARD_NAMESPACE + "/api/v1/users/" + str(new_id), headers=auth_header)
        assert response.status_code == 200

        # Make sure user is deleted
        response = client.get(FLARE_DASHBOARD_NAMESPACE + "/api/v1/users/" + str(new_id), headers=auth_header)
        assert response.status_code == 200
        # The API returns empty dict for non-existent user
        assert len(response.json["user"]) == 0

    def _get_one_user(self, auth_header, client, user_id):
        response = client.get(FLARE_DASHBOARD_NAMESPACE + "/api/v1/users/" + str(user_id), headers=auth_header)

        assert response.status_code == 200
        return response.json["user"]
