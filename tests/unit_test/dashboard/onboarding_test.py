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

"""Approval and ownership decisions through the in-process Dashboard API."""

from uuid import uuid4

import pytest

from nvflare.dashboard.application import db
from nvflare.dashboard.application.models import Client, Organization, User
from nvflare.dashboard.application.store import Store
from nvflare.dashboard.application.validation import DashboardError

API = "/nvflare-dashboard/api/v1"


@pytest.fixture
def register(client, auth_header):
    users = []

    def create(role="org_admin", organization="Onboarding_org"):
        email = f"owner-{uuid4().hex}@example.test"
        response = client.post(
            API + "/users", json={"email": email, "password": "p", "role": role, "organization": organization}
        )
        assert response.status_code == 201
        user = response.json["user"]
        users.append(user["id"])
        token = client.post(API + "/login", json={"email": email, "password": "p"}).json["access_token"]
        return user, {"Authorization": f"Bearer {token}"}

    yield create
    for user_id in users:
        client.delete(API + f"/users/{user_id}", headers=auth_header)


def approve(client, auth_header, user):
    response = client.patch(
        API + f"/users/{user['id']}",
        headers=auth_header,
        json={"approval_state": 100, "reviewed_role": user["role"], "reviewed_organization": user["organization"]},
    )
    assert response.status_code == 200, response.json


def test_staged_org_admin_request_is_pending_and_cannot_self_approve(client, register):
    user, headers = register(role="", organization="")
    url = API + f"/users/{user['id']}"
    response = client.patch(
        url, headers=headers, json={"role": "org_admin", "organization": "Workshop_org", "approval_state": 200}
    )
    assert response.status_code == 200
    assert response.json["user"]["role"] == "org_admin"
    assert response.json["user"]["approval_state"] == 0
    assert client.patch(API + "/project", headers=headers, json={"title": "unauthorized"}).status_code == 403
    assert client.get(API + "/users", headers=headers).status_code == 403
    assert client.post(url + "/blob", headers=headers, json={"pin": "1234"}).status_code == 403


@pytest.mark.parametrize("same_organization", [False, True])
def test_pending_site_access_is_owner_scoped(client, auth_header, register, same_organization):
    owner, headers = register()
    other, other_headers = register(organization=owner["organization"] if same_organization else "Other_org")
    response = client.post(
        API + "/clients",
        headers=headers,
        json={"name": "owned-" + uuid4().hex, "organization": owner["organization"], "approval_state": 200},
    )
    assert response.status_code == 201
    site = response.json["client"]
    url = API + f"/clients/{site['id']}"
    assert site["approval_state"] == 0
    assert client.get(API + "/clients", headers=headers).json["client_list"] == [site]
    assert client.get(API + "/clients", headers=other_headers).json["client_list"] == []
    for method in ("get", "patch", "delete", "post"):
        suffix = "/blob" if method == "post" else ""
        response = getattr(client, method)(url + suffix, headers=other_headers, json={"name": "stolen", "pin": "1234"})
        assert response.status_code == 403
    response = client.post(
        API + "/clients", headers=headers, json={"name": "cross-org", "organization": "Someone_else"}
    )
    assert response.status_code == 403
    assert client.patch(url, headers=headers, json={"approval_state": 200}).status_code == 403
    assert client.patch(url, headers=headers, json={"name": "renamed-" + uuid4().hex}).status_code == 200
    assert client.delete(url, headers=headers).status_code == 200


@pytest.mark.parametrize("field,value", [("role", "org_admin"), ("organization", "Different_org")])
def test_approved_blank_fields_cannot_be_filled_without_admin(app, client, auth_header, register, field, value):
    user, headers = register(role="", organization="")
    # Simulate an incomplete account approved by an older Dashboard version.
    with app.app_context():
        record = db.session.get(User, user["id"])
        record.approval_state = 100
        db.session.commit()
    response = client.patch(API + f"/users/{user['id']}", headers=headers, json={field: value, "password": "changed"})
    assert response.status_code == 403
    saved = client.get(API + f"/users/{user['id']}", headers=auth_header).json["user"]
    assert saved["role"] == saved["organization"] == ""
    assert saved["approval_state"] == 100
    assert client.post(API + "/login", json={"email": user["email"], "password": "p"}).status_code == 200


@pytest.mark.parametrize("role", ["member", "lead"])
def test_approved_user_in_org_cannot_self_promote_to_org_admin(client, auth_header, register, role):
    user, headers = register(role=role, organization="OrgA")
    approve(client, auth_header, user)
    response = client.patch(API + f"/users/{user['id']}", headers=headers, json={"role": "org_admin"})
    assert response.status_code == 403
    saved = client.get(API + f"/users/{user['id']}", headers=auth_header).json["user"]
    assert saved["role"] == role
    assert saved["organization"] == "OrgA"
    assert saved["approval_state"] == 100
    response = client.post(API + "/login", json={"email": user["email"], "password": "p"})
    assert response.status_code == 200
    assert response.json["user"]["role"] == role


def test_incomplete_or_changed_registration_cannot_be_approved(client, auth_header, register):
    user, headers = register(role="", organization="")
    url = API + f"/users/{user['id']}"
    assert (
        client.patch(
            url, headers=auth_header, json={"approval_state": 100, "reviewed_role": "", "reviewed_organization": ""}
        ).status_code
        == 400
    )
    assert (
        client.patch(url, headers=headers, json={"role": "org_admin", "organization": "Reviewed_org"}).status_code
        == 200
    )
    response = client.patch(
        url, headers=auth_header, json={"approval_state": 100, "reviewed_role": "", "reviewed_organization": ""}
    )
    assert response.status_code == 409
    assert "Review" in response.json["message"]
    saved = client.get(url, headers=auth_header).json["user"]
    assert saved["approval_state"] == 0
    approve(client, auth_header, saved)
    assert client.get(url, headers=headers).json["user"]["approval_state"] == 100


@pytest.mark.parametrize("race", ["approve", "change_registration"])
def test_approval_and_registration_compare_and_swap(app, register, race):
    user, _ = register(role="", organization="")
    with app.app_context():
        record = db.session.get(User, user["id"])
        expected = record.approval_state
        # A second database transaction commits after the first request's read.
        with db.engine.begin() as connection:
            connection.execute(
                db.update(User).where(User.id == record.id).values(approval_state=100 if race == "approve" else -1)
            )
        assert record.approval_state == expected
        with pytest.raises(DashboardError, match="registration changed"):
            Store._commit_user_patch(record, {"approval_state": 100, "name": "stale update"})
        db.session.rollback()
        db.session.refresh(record)
        assert record.name != "stale update"
        assert record.approval_state == (100 if race == "approve" else -1)


def test_old_token_cannot_be_used_after_denial(client, auth_header, register):
    user, headers = register()
    approve(client, auth_header, user)
    token = client.post(API + "/login", json={"email": user["email"], "password": "p"}).json["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert (
        client.patch(API + f"/users/{user['id']}", headers=auth_header, json={"approval_state": -1}).status_code == 200
    )
    assert client.get(API + "/clients", headers=headers).status_code == 403
    assert (
        client.post(
            API + "/clients", headers=headers, json={"name": "denied", "organization": user["organization"]}
        ).status_code
        == 403
    )
    assert client.post(API + f"/users/{user['id']}/blob", headers=headers, json={"pin": "1234"}).status_code == 403


def test_old_project_admin_token_does_not_retain_privileges(client, auth_header):
    email = f"admin-{uuid4().hex}@example.test"
    response = client.post(
        API + "/users",
        headers=auth_header,
        json={"email": email, "password": "p", "role": "project_admin", "organization": "Admin_org"},
    )
    user = response.json["user"]
    try:
        approve(client, auth_header, user)
        token = client.post(API + "/login", json={"email": email, "password": "p"}).json["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        assert (
            client.patch(API + f"/users/{user['id']}", headers=auth_header, json={"role": "member"}).status_code == 200
        )
        assert client.patch(API + "/project", headers=headers, json={"title": "stale token"}).status_code == 403
        assert (
            client.post(
                API + "/users", headers=headers, json={"email": "not-created@example.test", "role": "project_admin"}
            ).status_code
            == 403
        )
    finally:
        client.delete(API + f"/users/{user['id']}", headers=auth_header)


def test_site_approval_does_not_replace_user_approval(app, client, auth_header, register):
    user, headers = register()
    site = client.post(
        API + "/clients",
        headers=headers,
        json={"name": "approval-" + uuid4().hex, "organization": user["organization"]},
    ).json["client"]
    url = API + f"/clients/{site['id']}"
    assert client.patch(url, headers=auth_header, json={"approval_state": 100}).status_code == 200
    assert client.post(url + "/blob", headers=headers, json={"pin": "1234"}).status_code == 403
    assert client.post(url + "/blob", headers=auth_header, json={"pin": "1234"}).status_code == 403
    with app.app_context():
        assert db.session.get(Client, site["id"]).download_count == 0


def test_project_freeze_publishes_ca_and_frozen_state_together(app, monkeypatch):
    from types import SimpleNamespace

    from nvflare.dashboard.application import store
    from nvflare.dashboard.application.models import Project

    with app.app_context():
        project = Project.query.first()
        original = {column.name: getattr(project, column.name) for column in Project.__table__.columns}
        project.frozen = False
        db.session.commit()

        def create_ca(entity):
            with db.engine.connect() as connection:
                visible = connection.execute(db.select(Project.frozen, Project.root_cert)).one()
            assert visible.frozen is False
            return SimpleNamespace(ser_cert="complete-cert", ser_pri_key="complete-key")

        monkeypatch.setattr(store, "make_root_cert", create_ca)
        try:
            Store.set_project({"short_name": "atomic", "server1": "localhost", "frozen": True})
            with db.engine.connect() as connection:
                visible = connection.execute(db.select(Project.frozen, Project.root_cert, Project.root_key)).one()
            assert tuple(visible) == (True, "complete-cert", "complete-key")
        finally:
            for name, value in original.items():
                setattr(project, name, value)
            db.session.commit()


@pytest.mark.parametrize("first_field", ["role", "organization"])
def test_admin_can_repair_approved_legacy_profile_one_field_at_a_time(app, client, auth_header, register, first_field):
    user, _ = register(role="", organization="")
    with app.app_context():
        record = db.session.get(User, user["id"])
        record.approval_state = 100
        db.session.commit()
    fields = {"role": "org_admin", "organization": "Repaired_org"}
    url = API + f"/users/{user['id']}"
    assert client.patch(url, headers=auth_header, json={first_field: fields.pop(first_field)}).status_code == 200
    response = client.patch(url, headers=auth_header, json=fields)
    assert response.status_code == 200
    assert response.json["user"]["role"] == "org_admin"
    assert response.json["user"]["organization"] == "Repaired_org"


@pytest.mark.parametrize("method", ["patch", "delete"])
def test_site_approval_wins_over_intervening_owner_edit(app, client, auth_header, register, monkeypatch, method):
    user, headers = register()
    site = client.post(
        API + "/clients", headers=headers, json={"name": "race-" + uuid4().hex, "organization": user["organization"]}
    ).json["client"]
    original = Store.patch_client_by_creator if method == "patch" else Store.delete_client

    def approve_then_edit(cls, id, *args, **kwargs):
        with db.engine.begin() as connection:
            connection.execute(db.update(Client).where(Client.id == id).values(approval_state=100))
        return original(id, *args, **kwargs)

    monkeypatch.setattr(
        Store, "patch_client_by_creator" if method == "patch" else "delete_client", classmethod(approve_then_edit)
    )
    response = getattr(client, method)(API + f"/clients/{site['id']}", headers=headers, json={"name": "not-approved"})
    assert response.status_code == 409
    with app.app_context():
        saved = db.session.get(Client, site["id"])
        assert saved.name == site["name"]
        assert saved.approval_state == 100


def test_duplicate_site_rename_returns_conflict_and_allows_retry(client, register):
    user, headers = register()
    sites = []
    for _ in range(2):
        response = client.post(
            API + "/clients",
            headers=headers,
            json={"name": "unique-" + uuid4().hex, "organization": user["organization"], "description": "original"},
        )
        assert response.status_code == 201
        sites.append(response.json["client"])
    url = API + f"/clients/{sites[0]['id']}"
    response = client.patch(url, headers=headers, json={"name": sites[1]["name"], "description": "must not be saved"})
    assert response.status_code == 409
    assert "already exists" in response.json["message"]
    assert "name" in response.json["errors"]
    saved = client.get(url, headers=headers)
    assert saved.status_code == 200
    assert saved.json["client"]["name"] == sites[0]["name"]
    assert saved.json["client"]["description"] == "original"
    response = client.patch(url, headers=headers, json={"name": "retry-" + uuid4().hex})
    assert response.status_code == 200
    assert response.json["client"]["name"].startswith("retry-")


def test_site_creation_reuses_owner_organization_with_existing_duplicate_names(app, client, register):
    organization = "Duplicate_" + uuid4().hex
    first, first_headers = register(organization=organization)
    owner, headers = register(organization=organization)
    with app.app_context():
        first_org_id = db.session.get(User, first["id"]).organization_id
        duplicate = Organization(name=organization)
        db.session.add(duplicate)
        db.session.flush()
        duplicate_id = duplicate.id
        db.session.get(User, owner["id"]).organization_id = duplicate_id
        db.session.commit()
        assert duplicate_id != first_org_id
    response = client.post(
        API + "/clients",
        headers=headers,
        json={"name": "duplicate-org-" + uuid4().hex, "organization": organization},
    )
    assert response.status_code == 201
    site = response.json["client"]
    url = API + f"/clients/{site['id']}"
    with app.app_context():
        assert db.session.get(Client, site["id"]).organization_id == duplicate_id
    assert client.get(url, headers=headers).status_code == 200
    assert client.patch(url, headers=headers, json={"description": "owner edit"}).status_code == 200
    assert [row["id"] for row in client.get(API + "/clients", headers=headers).json["client_list"]] == [site["id"]]
    assert client.get(url, headers=first_headers).status_code == 403
    assert client.get(API + "/clients", headers=first_headers).json["client_list"] == []


def test_admin_can_correct_user_organization_after_all_owned_sites_are_moved(app, client, auth_header, register):
    user, headers = register(organization="Wrong_" + uuid4().hex)
    destination = "Correct_" + uuid4().hex
    sites = []
    for _ in range(2):
        response = client.post(
            API + "/clients",
            headers=headers,
            json={
                "name": "correct-org-" + uuid4().hex,
                "organization": user["organization"],
            },
        )
        assert response.status_code == 201
        sites.append(response.json["client"])
    url = API + f"/users/{user['id']}"
    assert client.patch(url, headers=auth_header, json={"organization": destination}).status_code == 409
    assert (
        client.patch(
            API + f"/clients/{sites[0]['id']}", headers=auth_header, json={"organization": destination}
        ).status_code
        == 200
    )
    assert client.patch(url, headers=auth_header, json={"organization": destination}).status_code == 409
    assert (
        client.patch(
            API + f"/clients/{sites[1]['id']}", headers=auth_header, json={"organization": destination}
        ).status_code
        == 200
    )
    response = client.patch(url, headers=auth_header, json={"organization": destination})
    assert response.status_code == 200
    assert response.json["user"]["organization"] == destination
    assert response.json["user"]["approval_state"] == 0
    with app.app_context():
        saved = db.session.get(User, user["id"])
        for site in sites:
            record = db.session.get(Client, site["id"])
            assert record.organization_id == saved.organization_id
            assert record.approval_state == 0
    for site in sites:
        site_url = API + f"/clients/{site['id']}"
        assert client.get(site_url, headers=headers).status_code == 200
        assert client.patch(site_url, headers=headers, json={"description": "corrected owner edit"}).status_code == 200
