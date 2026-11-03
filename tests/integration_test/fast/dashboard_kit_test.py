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


@pytest.mark.skipif(shutil.which("zip") is None, reason="Encrypted startup kits require the system zip executable")
@pytest.mark.parametrize("legacy_approved", [False, True])
def test_unreviewed_authorization_details_cannot_reach_certificate(dashboard_app, legacy_approved):
    from cryptography import x509
    from cryptography.x509.oid import NameOID

    from nvflare.dashboard.application.models import User

    app = dashboard_app
    client = app.test_client()
    api = "/nvflare-dashboard/api/v1"
    response = client.post(api + "/login", json={"email": TEST_USER, "password": TEST_PW})
    admin_headers = {"Authorization": "Bearer " + response.json["access_token"]}
    email = f"certificate-review-{uuid4().hex}@test.com"
    user_id = None
    with app.app_context():
        project = Project.query.first()
        original = {column.name: getattr(project, column.name) for column in Project.__table__.columns}
    try:
        with app.app_context():
            Store.set_project({"short_name": "reviewed_kit", "server1": "localhost", "frozen": True})
        response = client.post(api + "/users", json={"email": email, "password": "p"})
        assert response.status_code == 201
        user_id = response.json["user"]["id"]
        url = api + f"/users/{user_id}"
        token = client.post(api + "/login", json={"email": email, "password": "p"}).json["access_token"]
        headers = {"Authorization": "Bearer " + token}
        response = client.patch(
            url,
            headers=admin_headers,
            json={
                "approval_state": 100,
                "reviewed_role": "",
                "reviewed_organization": "",
            },
        )
        assert response.status_code == 400
        assert set(response.json["errors"]) == {"role", "organization"}
        if legacy_approved:
            # A pre-existing incomplete approval must not permit the old exploit either.
            with app.app_context():
                db.session.get(User, user_id).approval_state = 100
                db.session.commit()
        response = client.patch(url, headers=headers, json={"organization": "Unreviewed_org", "role": "org_admin"})
        assert response.status_code == (403 if legacy_approved else 200)
        response = client.post(url + "/blob", headers=headers, json={"pin": "1234"})
        assert response.status_code == (400 if legacy_approved else 403)
        assert response.is_json and response.json["status"] == "error"
        with app.app_context():
            user = db.session.get(User, user_id)
            assert user.role.name == ("" if legacy_approved else "org_admin")
            assert user.organization.name == ("" if legacy_approved else "Unreviewed_org")
            assert user.approval_state == (100 if legacy_approved else 0)
            assert user.download_count == 0
        # Complete the details under administrator control, then explicitly approve that tuple.
        response = client.patch(
            url,
            headers=admin_headers,
            json={
                "approval_state": 0,
                "role": "org_admin",
                "organization": "Reviewed_org",
            },
        )
        assert response.status_code == 200
        response = client.patch(
            url,
            headers=admin_headers,
            json={
                "approval_state": 100,
                "reviewed_role": "org_admin",
                "reviewed_organization": "Reviewed_org",
            },
        )
        assert response.status_code == 200
        for edit in [None, {"organization": "Unreviewed_org"}, {"role": "lead"}, {"role": "project_admin"}]:
            if edit:
                response = client.patch(url, headers=headers, json=edit)
                assert response.status_code == 403
            response = client.post(url + "/blob", headers=headers, json={"pin": "1234"})
            assert response.status_code == 200
            with ZipFile(io.BytesIO(response.data)) as kit:
                assert kit.getinfo("startup/client.crt").flag_bits & 1
                certificate = x509.load_pem_x509_certificate(kit.read("startup/client.crt", pwd=b"1234"))
            assert certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == email
            assert certificate.subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)[0].value == "Reviewed_org"
            assert certificate.subject.get_attributes_for_oid(NameOID.UNSTRUCTURED_NAME)[0].value == "org_admin"
    finally:
        if user_id is not None:
            client.delete(api + f"/users/{user_id}", headers=admin_headers)
        with app.app_context():
            project = Project.query.first()
            for name, value in original.items():
                setattr(project, name, value)
            db.session.commit()


@pytest.mark.skipif(shutil.which("zip") is None, reason="Encrypted startup kits require the system zip executable")
@pytest.mark.parametrize("organization_setup", ["duplicate", "corrected"])
def test_site_owner_organization_identity_allows_approved_kit_download(dashboard_app, organization_setup):
    from cryptography import x509
    from cryptography.x509.oid import NameOID

    from nvflare.dashboard.application.models import Organization, User

    app = dashboard_app
    client = app.test_client()
    api = "/nvflare-dashboard/api/v1"
    response = client.post(api + "/login", json={"email": TEST_USER, "password": TEST_PW})
    admin_headers = {"Authorization": "Bearer " + response.json["access_token"]}
    email = f"org-record-{uuid4().hex}@test.com"
    organization = "Registered_" + uuid4().hex
    user_id = None
    with app.app_context():
        project = Project.query.first()
        original = {column.name: getattr(project, column.name) for column in Project.__table__.columns}
    try:
        response = client.post(
            api + "/users",
            json={
                "email": email,
                "password": "p",
                "role": "org_admin",
                "organization": organization,
            },
        )
        assert response.status_code == 201
        user_id = response.json["user"]["id"]
        if organization_setup == "duplicate":
            with app.app_context():
                duplicate = Organization(name=organization)
                db.session.add(duplicate)
                db.session.flush()
                db.session.get(User, user_id).organization_id = duplicate.id
                db.session.commit()
        token = client.post(api + "/login", json={"email": email, "password": "p"}).json["access_token"]
        headers = {"Authorization": "Bearer " + token}
        site_name = "org-record-" + uuid4().hex
        response = client.post(
            api + "/clients", headers=headers, json={"name": site_name, "organization": organization}
        )
        assert response.status_code == 201
        site = response.json["client"]
        site_url = api + f"/clients/{site['id']}"
        user_url = api + f"/users/{user_id}"
        if organization_setup == "corrected":
            organization = "Corrected_" + uuid4().hex
            assert client.patch(user_url, headers=admin_headers, json={"organization": organization}).status_code == 409
            assert client.patch(site_url, headers=admin_headers, json={"organization": organization}).status_code == 200
            assert client.patch(user_url, headers=admin_headers, json={"organization": organization}).status_code == 200
        with app.app_context():
            owner = db.session.get(User, user_id)
            record = db.session.get(Client, site["id"])
            assert record.organization_id == owner.organization_id
            assert owner.approval_state == record.approval_state == 0
        assert client.get(site_url, headers=headers).status_code == 200
        assert (
            client.patch(site_url, headers=headers, json={"description": "owner still has access"}).status_code == 200
        )
        assert (
            client.patch(
                user_url,
                headers=admin_headers,
                json={
                    "approval_state": 100,
                    "reviewed_role": "org_admin",
                    "reviewed_organization": organization,
                },
            ).status_code
            == 200
        )
        assert client.patch(site_url, headers=admin_headers, json={"approval_state": 100}).status_code == 200
        with app.app_context():
            Store.set_project({"short_name": "org_record_kit", "server1": "localhost", "frozen": True})
        response = client.post(site_url + "/blob", headers=headers, json={"pin": "1234"})
        assert response.status_code == 200
        with ZipFile(io.BytesIO(response.data)) as kit:
            assert kit.getinfo("startup/client.crt").flag_bits & 1
            certificate = x509.load_pem_x509_certificate(kit.read("startup/client.crt", pwd=b"1234"))
        assert certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == site_name
        assert certificate.subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)[0].value == organization
    finally:
        if user_id is not None:
            client.delete(api + f"/users/{user_id}", headers=admin_headers)
        with app.app_context():
            project = Project.query.first()
            for name, value in original.items():
                setattr(project, name, value)
            db.session.commit()
