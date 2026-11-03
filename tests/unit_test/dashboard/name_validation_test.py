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

"""Regression coverage for Dashboard names that cannot be provisioned."""

import io
import shutil
import zipfile

import pytest
from flask_jwt_extended import create_access_token

from nvflare.dashboard.application import db
from nvflare.dashboard.application.constants import FLARE_DASHBOARD_NAMESPACE as NS
from nvflare.dashboard.application.models import Client, Organization, Project, Role, User
from nvflare.dashboard.application.store import Store

API = NS + "/api/v1"
INVALID_CLIENT_NAMES = ["Test Holger Client", "site.example", "site/1", "site\n", "", None, 123, []]
INVALID_ORGS = ["Test Holger Client", "test-org", "test.org", "test\n", "", None, 123, {}]


def assert_invalid(response, field):
    assert response.status_code == 400
    assert response.json["status"] == "error"
    assert field in response.json["errors"]
    assert response.json["errors"][field] in response.json["message"]


def counts(app):
    with app.app_context():
        return Client.query.count(), Organization.query.count(), User.query.count(), Role.query.count()


@pytest.fixture
def site(client, auth_header):
    response = client.post(
        API + "/clients", json={"name": "validation-site_1", "organization": "Valid_org"}, headers=auth_header
    )
    assert response.status_code == 201
    record = response.json["client"]
    yield record
    client.delete(API + f"/clients/{record['id']}", headers=auth_header)


@pytest.fixture
def user(client, auth_header):
    response = client.post(
        API + "/users", json={"email": "validation@test.com", "password": "test", "name": "Test User"}
    )
    assert response.status_code == 201
    record = response.json["user"]
    yield record
    client.delete(API + f"/users/{record['id']}", headers=auth_header)


@pytest.fixture
def configured_project(app):
    with app.app_context():
        project = Project.query.first()
        original = {column.name: getattr(project, column.name) for column in Project.__table__.columns}
        Store.set_project({"short_name": "validation", "server1": "localhost", "frozen": True})
    yield
    with app.app_context():
        project = Project.query.first()
        for name, value in original.items():
            setattr(project, name, value)
        db.session.commit()


@pytest.mark.parametrize(
    "field,value", [("name", v) for v in INVALID_CLIENT_NAMES] + [("organization", v) for v in INVALID_ORGS]
)
def test_create_client_rejects_invalid_names_without_writes(app, client, auth_header, field, value):
    before = counts(app)
    data = {"name": "new-site", "organization": "New_org", field: value}
    response = client.post(API + "/clients", json=data, headers=auth_header)
    assert_invalid(response, field)
    assert counts(app) == before


@pytest.mark.parametrize("field", ["name", "organization"])
def test_create_client_requires_names(app, client, auth_header, field):
    data = {"name": "new-site", "organization": "New_org"}
    del data[field]
    before = counts(app)
    assert_invalid(client.post(API + "/clients", json=data, headers=auth_header), field)
    assert counts(app) == before


@pytest.mark.parametrize(
    "field,value", [("name", v) for v in INVALID_CLIENT_NAMES] + [("organization", v) for v in INVALID_ORGS]
)
def test_patch_client_rejects_invalid_names_without_writes(app, client, auth_header, site, field, value):
    before = counts(app)
    # A valid new organization must not be committed when the name is invalid.
    data = {"description": "must not be saved", "organization": "Unsaved_org", field: value}
    response = client.patch(API + f"/clients/{site['id']}", json=data, headers=auth_header)
    assert_invalid(response, field)
    assert counts(app) == before
    saved = client.get(API + f"/clients/{site['id']}", headers=auth_header).json["client"]
    assert saved == site


def test_creator_patch_checks_name_but_preserves_organization_permissions(app, client, site):
    with app.app_context():
        token = create_access_token(
            identity=User.query.get(1).email, additional_claims={"role": "member", "approved": True}
        )
    headers = {"Authorization": f"Bearer {token}"}
    url = API + f"/clients/{site['id']}"
    assert_invalid(client.patch(url, json={"name": "bad name"}, headers=headers), "name")
    response = client.patch(url, json={"name": "renamed-site_2", "organization": "Ignored org"}, headers=headers)
    assert response.status_code == 200
    assert response.json["client"]["name"] == "renamed-site_2"
    assert response.json["client"]["organization"] == site["organization"]


@pytest.mark.parametrize(
    "field,value", [("organization", "Bad org"), ("organization", None), ("email", "bad email"), ("email", None)]
)
def test_registration_rejects_invalid_supplied_names_without_writes(app, client, field, value):
    before = counts(app)
    response = client.post(API + "/users", json={"email": "new@test.com", "organization": "New_org", field: value})
    assert_invalid(response, field)
    assert counts(app) == before


def test_registration_allows_display_names_and_deferred_organization(user):
    assert user["name"] == "Test User"
    assert user["organization"] == ""
    assert user["role"] == ""


@pytest.mark.parametrize("as_admin", [False, True])
def test_user_patch_validates_before_role_or_organization_is_created(app, client, auth_header, user, as_admin):
    with app.app_context():
        token = create_access_token(identity=user["email"], additional_claims={"role": "member"})
    headers = auth_header if as_admin else {"Authorization": f"Bearer {token}"}
    before = counts(app)
    response = client.patch(
        API + f"/users/{user['id']}",
        json={"organization": "Bad org", "role": "lead", "name": "Changed"},
        headers=headers,
    )
    assert_invalid(response, "organization")
    assert counts(app) == before
    saved = client.get(API + f"/users/{user['id']}", headers=auth_header).json["user"]
    assert saved == user


@pytest.mark.parametrize("name", ["bad host", "https://server.test", "host:8002", "host_name", None, 123])
def test_project_rejects_invalid_server_before_saving(app, client, auth_header, name):
    before = client.get(API + "/project").json
    response = client.patch(API + "/project", json={"server1": name, "title": "Unsaved"}, headers=auth_header)
    assert_invalid(response, "server1")
    assert client.get(API + "/project").json == before


@pytest.mark.parametrize(
    "resource,field,value",
    [
        ("clients", "name", "bad name"),
        ("clients", "organization", "Bad org"),
        ("users", "email", "bad email"),
        ("users", "organization", "Bad org"),
        ("users", "role", "bad role"),
        ("servers", "server1", "bad host"),
        ("servers", "organization", "Bad org"),
    ],
)
def test_legacy_names_return_actionable_download_error(
    app, client, auth_header, site, user, configured_project, resource, field, value
):
    with app.app_context():
        if resource == "clients":
            record = Client.query.get(site["id"])
        elif resource == "users":
            record = User.query.get(user["id"])
        else:
            record = User.query.get(1)
        record.approval_state = 100
        target = record
        attribute = field
        if field == "organization":
            target, attribute = record.organization, "name"
        elif field == "role":
            target, attribute = record.role, "name"
        elif field == "server1":
            target = Project.query.first()
        original = getattr(target, attribute)
        setattr(target, attribute, value)
        db.session.commit()
        record_id = record.id
        download_count = record.download_count
        try:
            response = client.post(API + f"/{resource}/{record_id}/blob", json={"pin": "1234"}, headers=auth_header)
            assert_invalid(response, field)
            db.session.refresh(record)
            assert record.download_count == download_count
        finally:
            setattr(target, attribute, original)
            db.session.commit()


@pytest.mark.skipif(shutil.which("zip") is None, reason="Encrypted startup kits require the system zip executable")
def test_valid_client_names_generate_encrypted_startup_kit(app, client, auth_header, site, configured_project):
    client.patch(API + f"/clients/{site['id']}", json={"approval_state": 100}, headers=auth_header)
    response = client.post(API + f"/clients/{site['id']}/blob", json={"pin": "1234"}, headers=auth_header)
    assert response.status_code == 200
    assert response.headers["Content-Disposition"] == f"attachment; filename=\"{site['name']}.zip\""
    with zipfile.ZipFile(io.BytesIO(response.data)) as kit:
        assert "startup/fed_client.json" in kit.namelist()
        assert kit.read("startup/fed_client.json", pwd=b"1234")
    with app.app_context():
        assert Client.query.get(site["id"]).download_count == 1


@pytest.mark.parametrize("organization", ["Bad org", "", None])
def test_seed_user_rejects_invalid_organization_before_writes(app, organization):
    before = counts(app)
    with app.app_context(), pytest.raises(ValueError, match="Organization"):
        Store.seed_user("seed@test.com", "test", organization)
    assert counts(app) == before


def test_project_cannot_freeze_with_empty_server(client, auth_header):
    before = client.get(API + "/project").json
    response = client.patch(API + "/project", json={"frozen": True, "server1": ""}, headers=auth_header)
    assert_invalid(response, "server1")
    assert client.get(API + "/project").json == before
