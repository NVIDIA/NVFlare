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

"""Security regression tests for dashboard hardening.

These tests reproduce the exploit paths from the security scan report.
They should FAIL against the pre-fix codebase (exploits succeed) and
PASS after the security hardening (exploits blocked).
"""

import pytest
from werkzeug.security import generate_password_hash

from nvflare.dashboard.application.constants import FLARE_DASHBOARD_NAMESPACE as NS


class TestMassAssignment:
    """Issue 1: setattr mass-assignment allows writing protected fields."""

    def test_role_id_escalation(self, client):
        """PATCH role_id=1 on own user should not grant project_admin."""
        client.post(NS + "/api/v1/users", json={"email": "escalate@test.com", "password": "p", "name": "x"})
        resp = client.post(NS + "/api/v1/login", json={"email": "escalate@test.com", "password": "p"})
        assert resp.status_code == 200
        token = resp.json["access_token"]
        user_id = resp.json["user"]["id"]

        client.patch(
            NS + f"/api/v1/users/{user_id}",
            json={"role_id": 1},
            headers={"Authorization": f"Bearer {token}"},
        )

        resp = client.post(NS + "/api/v1/login", json={"email": "escalate@test.com", "password": "p"})
        assert resp.json["user"]["role"] != "project_admin"

    def test_password_hash_overwrite(self, client):
        """PATCH password_hash directly should be ignored; original password still works."""
        client.post(NS + "/api/v1/users", json={"email": "hashtest@test.com", "password": "original", "name": "x"})
        resp = client.post(NS + "/api/v1/login", json={"email": "hashtest@test.com", "password": "original"})
        assert resp.status_code == 200
        token = resp.json["access_token"]
        user_id = resp.json["user"]["id"]

        client.patch(
            NS + f"/api/v1/users/{user_id}",
            json={"password_hash": generate_password_hash("hacked")},
            headers={"Authorization": f"Bearer {token}"},
        )

        resp = client.post(NS + "/api/v1/login", json={"email": "hashtest@test.com", "password": "original"})
        assert resp.status_code == 200

    def test_approval_state_self_approve(self, client, auth_header):
        """User should not be able to self-approve via PATCH approval_state."""
        client.post(NS + "/api/v1/users", json={"email": "selfapprove@test.com", "password": "p", "name": "x"})
        resp = client.post(NS + "/api/v1/login", json={"email": "selfapprove@test.com", "password": "p"})
        assert resp.status_code == 200
        token = resp.json["access_token"]
        user_id = resp.json["user"]["id"]

        client.patch(
            NS + f"/api/v1/users/{user_id}",
            json={"approval_state": 200},
            headers={"Authorization": f"Bearer {token}"},
        )

        # Admin reads the user to verify approval_state was NOT changed
        resp = client.get(
            NS + f"/api/v1/users/{user_id}",
            headers=auth_header,
        )
        assert resp.json["user"]["approval_state"] == 0


class TestAdminRoleAssignment:
    """Only ordinary roles may be self-assigned; other roles require the project administrator."""

    @pytest.mark.parametrize("role", ["org_admin", "project_admin", "custom_admin"])
    def test_public_registration_cannot_request_admin_role(self, client, role):
        resp = client.post(
            NS + "/api/v1/users",
            json={"email": f"register-{role}@test.com", "password": "p", "name": "x", "role": role},
        )

        assert resp.status_code == 409

    def test_public_registration_rejects_non_string_role(self, client):
        resp = client.post(
            NS + "/api/v1/users",
            json={"email": "malformed-role@test.com", "password": "p", "name": "x", "role": {"name": "member"}},
        )

        assert resp.status_code == 409

    @pytest.mark.parametrize("token", ["", "undefined", "null"])
    def test_public_registration_accepts_malformed_bearer_token(self, client, token):
        resp = client.post(
            NS + "/api/v1/users",
            json={"email": f"malformed-token-{token}@test.com", "password": "p", "name": "x", "role": "member"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert resp.status_code == 201
        assert resp.json["user"]["role"] == "member"

    def test_malformed_bearer_token_cannot_request_admin_role(self, client):
        resp = client.post(
            NS + "/api/v1/users",
            json={"email": "malformed-token-admin@test.com", "password": "p", "name": "x", "role": "org_admin"},
            headers={"Authorization": "Bearer undefined"},
        )

        assert resp.status_code == 409

    @pytest.mark.parametrize("role", ["org_admin", "project_admin", "custom_admin"])
    def test_user_cannot_self_assign_admin_role(self, client, auth_header, role):
        email = f"self-assign-{role}@test.com"
        resp = client.post(NS + "/api/v1/users", json={"email": email, "password": "p", "name": "x"})
        assert resp.status_code == 201
        user_id = resp.json["user"]["id"]

        resp = client.post(NS + "/api/v1/login", json={"email": email, "password": "p"})
        assert resp.status_code == 200
        token = resp.json["access_token"]

        resp = client.patch(
            NS + f"/api/v1/users/{user_id}",
            json={"organization": "target-org", "role": role},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json["status"] == "error"

        resp = client.get(NS + f"/api/v1/users/{user_id}", headers=auth_header)
        assert resp.json["user"]["role"] == ""
        assert resp.json["user"]["organization"] == ""

    @pytest.mark.parametrize("role", ["member", "lead"])
    def test_user_can_self_assign_ordinary_role(self, client, role):
        email = f"self-assign-{role}@test.com"
        resp = client.post(NS + "/api/v1/users", json={"email": email, "password": "p", "name": "x"})
        assert resp.status_code == 201
        user_id = resp.json["user"]["id"]

        resp = client.post(NS + "/api/v1/login", json={"email": email, "password": "p"})
        assert resp.status_code == 200

        resp = client.patch(
            NS + f"/api/v1/users/{user_id}",
            json={"role": role},
            headers={"Authorization": f"Bearer {resp.json['access_token']}"},
        )

        assert resp.status_code == 200
        assert resp.json["status"] == "ok"
        assert resp.json["user"]["role"] == role

    def test_project_admin_can_assign_org_admin_role(self, client, auth_header):
        resp = client.post(
            NS + "/api/v1/users",
            json={"email": "admin-assigned@test.com", "password": "p", "name": "x"},
        )
        assert resp.status_code == 201
        user_id = resp.json["user"]["id"]

        resp = client.patch(
            NS + f"/api/v1/users/{user_id}",
            json={"organization": "approved-org", "role": "org_admin"},
            headers=auth_header,
        )

        assert resp.status_code == 200
        assert resp.json["user"]["role"] == "org_admin"
        assert resp.json["user"]["organization"] == "approved-org"

    def test_unapproved_project_admin_cannot_create_admin_role(self, client, auth_header):
        resp = client.post(
            NS + "/api/v1/users",
            json={"email": "pending-admin@test.com", "password": "p", "name": "x", "role": "project_admin"},
            headers=auth_header,
        )
        assert resp.status_code == 201

        resp = client.post(NS + "/api/v1/login", json={"email": "pending-admin@test.com", "password": "p"})
        assert resp.status_code == 200

        resp = client.post(
            NS + "/api/v1/users",
            json={"email": "pending-created@test.com", "password": "p", "name": "x", "role": "org_admin"},
            headers={"Authorization": f"Bearer {resp.json['access_token']}"},
        )

        assert resp.status_code == 409


class TestApprovalEnforcement:
    """Issue 2: unapproved/denied users should have restricted access."""

    def test_denied_user_cannot_login(self, client, auth_header):
        """Admin denies a user (approval_state=-1); user cannot login."""
        client.post(NS + "/api/v1/users", json={"email": "denied@test.com", "password": "p", "name": "x"})
        resp = client.post(NS + "/api/v1/login", json={"email": "denied@test.com", "password": "p"})
        assert resp.status_code == 200
        user_id = resp.json["user"]["id"]

        client.patch(
            NS + f"/api/v1/users/{user_id}",
            json={"approval_state": -1},
            headers=auth_header,
        )

        resp = client.post(NS + "/api/v1/login", json={"email": "denied@test.com", "password": "p"})
        assert resp.status_code == 401

    def test_pending_user_cannot_list_users(self, client):
        """Unapproved user (approval_state=0) cannot list all users."""
        client.post(NS + "/api/v1/users", json={"email": "pending@test.com", "password": "p", "name": "x"})
        resp = client.post(NS + "/api/v1/login", json={"email": "pending@test.com", "password": "p"})
        if resp.status_code != 200:
            pytest.skip("Login blocked for pending users")
        token = resp.json["access_token"]

        resp = client.get(NS + "/api/v1/users", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 403
