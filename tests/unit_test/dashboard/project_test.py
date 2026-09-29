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
    def test_application_config_route_redirects_to_project_configuration(self, client):
        response = client.get(FLARE_DASHBOARD_NAMESPACE + "/application-config")

        assert response.status_code == 302
        assert response.headers["Location"] == FLARE_DASHBOARD_NAMESPACE + "/project-configuration"

    def test_dashboard_page_routes_resolve(self, app, client):
        page_routes = [
            rule.rule
            for rule in app.url_map.iter_rules()
            if "GET" in rule.methods
            and "<" not in rule.rule
            and "/api/" not in rule.rule
            and (rule.rule == "/" or rule.rule.startswith(FLARE_DASHBOARD_NAMESPACE))
        ]

        for route in page_routes:
            response = client.get(route, follow_redirects=True)
            assert response.status_code == 200, f"Dashboard page route did not resolve: {route}"

    def test_application_config_route_preserves_custom_static_page(self, app, client, monkeypatch, tmp_path):
        custom_static_folder = tmp_path / "static"
        custom_page = custom_static_folder / "nvflare-dashboard" / "application-config.html"
        custom_page.parent.mkdir(parents=True)
        custom_page.write_text("custom application configuration", encoding="utf-8")
        monkeypatch.setattr(app, "static_folder", str(custom_static_folder))

        response = client.get(FLARE_DASHBOARD_NAMESPACE + "/application-config")

        assert response.status_code == 200
        assert response.text == "custom application configuration"

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
