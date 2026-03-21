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
