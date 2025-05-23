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


from nvflare.dashboard.application.constants import FLARE_DASHBOARD_NAMESPACE


class TestProject:
    def test_login(self, access_token):
        # login is already tested if access_token is not empty
        assert access_token

    def test_get_project(self, client, auth_header):
        response = client.get(FLARE_DASHBOARD_NAMESPACE + "/api/v1/project", headers=auth_header)

        assert response.status_code == 200
        assert response.json["project"]

    def test_get_orgs(self, client, auth_header):
        response = client.get(FLARE_DASHBOARD_NAMESPACE + "/api/v1/organizations", headers=auth_header)

        assert response.status_code == 200
        assert response.json["client_list"]
