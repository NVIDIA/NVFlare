# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for Certificate Service HTTP routes (routes.py)."""

import json
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from nvflare.cert_service.store import EnrolledEntity, PendingRequest


class TestBaseRoutes:
    """Base class for route tests with common fixtures."""

    @pytest.fixture
    def app(self):
        """Create CertServiceApp for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from nvflare.cert_service.app import CertServiceApp

            app = CertServiceApp(
                data_dir=tmpdir,
                project_name="TestProject",
                api_key="test-api-key",
            )
            app.flask_app.testing = True
            yield app

    @pytest.fixture
    def client(self, app):
        """Create Flask test client."""
        return app.flask_app.test_client()

    @pytest.fixture
    def auth_headers(self):
        """Return valid authorization headers."""
        return {"Authorization": "Bearer test-api-key"}


class TestHealthRoutes(TestBaseRoutes):
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health check endpoint returns healthy."""
        response = client.get("/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"

    def test_health_no_auth_required(self, client):
        """Test health check does not require authentication."""
        response = client.get("/health")
        assert response.status_code == 200


class TestCARoutes(TestBaseRoutes):
    """Tests for CA certificate endpoints."""

    def test_get_ca_cert(self, client):
        """Test downloading root CA certificate."""
        response = client.get("/api/v1/ca-cert")
        assert response.status_code == 200
        assert response.content_type == "application/x-pem-file"
        assert b"-----BEGIN CERTIFICATE-----" in response.data

    def test_get_ca_info(self, client):
        """Test getting CA info."""
        response = client.get("/api/v1/ca-info")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "subject" in data
        assert "issuer" in data
        assert "not_valid_before" in data
        assert "not_valid_after" in data
        assert "serial_number" in data


class TestEnrolledRoutes(TestBaseRoutes):
    """Tests for enrolled entities endpoint."""

    def test_list_enrolled_empty(self, client, auth_headers):
        """Test listing enrolled entities when empty."""
        response = client.get("/api/v1/enrolled", headers=auth_headers)
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data == []

    def test_list_enrolled_with_entities(self, app, client, auth_headers):
        """Test listing enrolled entities."""
        # Add some entities
        app.enrollment_store.add_enrolled(EnrolledEntity("site-1", "client", datetime.now(timezone.utc), "Org"))
        app.enrollment_store.add_enrolled(EnrolledEntity("site-2", "client", datetime.now(timezone.utc), "Org"))

        response = client.get("/api/v1/enrolled", headers=auth_headers)
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 2

    def test_list_enrolled_filter_by_type(self, app, client, auth_headers):
        """Test listing enrolled entities filtered by type."""
        app.enrollment_store.add_enrolled(EnrolledEntity("site-1", "client", datetime.now(timezone.utc), "Org"))
        app.enrollment_store.add_enrolled(EnrolledEntity("relay-1", "relay", datetime.now(timezone.utc), "Org"))

        response = client.get(
            "/api/v1/enrolled?entity_type=client",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 1
        assert data[0]["name"] == "site-1"

    def test_list_enrolled_requires_auth(self, client):
        """Test listing enrolled entities requires authentication."""
        response = client.get("/api/v1/enrolled")
        assert response.status_code == 401


class TestPendingRoutes(TestBaseRoutes):
    """Tests for pending request management endpoints."""

    @pytest.fixture
    def pending_request(self, app):
        """Add a pending request to the store with a valid CSR."""
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        csr = (
            x509.CertificateSigningRequestBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "site-1")]))
            .sign(private_key, hashes.SHA256())
        )
        csr_pem = csr.public_bytes(serialization.Encoding.PEM).decode()

        now = datetime.now(timezone.utc)
        request = PendingRequest(
            name="site-1",
            entity_type="client",
            org="TestOrg",
            csr_pem=csr_pem,
            submitted_at=now,
            expires_at=now + timedelta(days=7),
            token_subject="site-1",
            source_ip="10.0.0.1",
        )
        app.enrollment_store.add_pending(request)
        return request

    def test_list_pending_empty(self, client, auth_headers):
        """Test listing pending requests when empty."""
        response = client.get("/api/v1/pending", headers=auth_headers)
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data == []

    def test_list_pending_with_requests(self, client, auth_headers, pending_request):
        """Test listing pending requests."""
        response = client.get("/api/v1/pending", headers=auth_headers)
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 1
        assert data[0]["name"] == "site-1"
        assert data[0]["status"] == "pending"

    def test_list_pending_filter_by_type(self, app, client, auth_headers):
        """Test listing pending requests filtered by type."""
        now = datetime.now(timezone.utc)
        app.enrollment_store.add_pending(
            PendingRequest("site-1", "client", "Org", "csr", now, now + timedelta(days=7), "site-1")
        )
        app.enrollment_store.add_pending(
            PendingRequest("relay-1", "relay", "Org", "csr", now, now + timedelta(days=7), "relay-1")
        )

        response = client.get(
            "/api/v1/pending?entity_type=client",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 1

    def test_get_pending_info(self, client, auth_headers, pending_request):
        """Test getting pending request details."""
        response = client.get(
            "/api/v1/pending/site-1?entity_type=client",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["name"] == "site-1"
        assert data["org"] == "TestOrg"
        assert data["source_ip"] == "10.0.0.1"

    def test_get_pending_info_not_found(self, client, auth_headers):
        """Test getting non-existent pending request."""
        response = client.get(
            "/api/v1/pending/nonexistent?entity_type=client",
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_get_pending_info_missing_type(self, client, auth_headers, pending_request):
        """Test getting pending request without entity_type."""
        response = client.get("/api/v1/pending/site-1", headers=auth_headers)
        assert response.status_code == 400
        assert b"entity_type" in response.data

    def test_approve_pending(self, app, client, auth_headers, pending_request):
        """Test approving a pending request."""
        response = client.post(
            "/api/v1/pending/site-1/approve",
            headers=auth_headers,
            json={"entity_type": "client"},
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "approved"
        assert data["certificate_issued"] is True

        # Should be enrolled now
        assert app.enrollment_store.is_enrolled("site-1", "client")

    def test_approve_pending_not_found(self, client, auth_headers):
        """Test approving non-existent pending request."""
        response = client.post(
            "/api/v1/pending/nonexistent/approve",
            headers=auth_headers,
            json={"entity_type": "client"},
        )
        assert response.status_code == 404

    def test_approve_pending_missing_type(self, client, auth_headers, pending_request):
        """Test approving without entity_type."""
        response = client.post(
            "/api/v1/pending/site-1/approve",
            headers=auth_headers,
            json={},
        )
        assert response.status_code == 400

    def test_reject_pending(self, app, client, auth_headers, pending_request):
        """Test rejecting a pending request."""
        response = client.post(
            "/api/v1/pending/site-1/reject",
            headers=auth_headers,
            json={"entity_type": "client", "reason": "Not authorized"},
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "rejected"

        # Should no longer be pending
        assert not app.enrollment_store.is_pending("site-1", "client")

    def test_reject_pending_not_found(self, client, auth_headers):
        """Test rejecting non-existent pending request."""
        response = client.post(
            "/api/v1/pending/nonexistent/reject",
            headers=auth_headers,
            json={"entity_type": "client"},
        )
        assert response.status_code == 404

    def test_approve_batch(self, app, client, auth_headers):
        """Test batch approving pending requests."""
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        def make_csr(cn):
            key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            csr = (
                x509.CertificateSigningRequestBuilder()
                .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)]))
                .sign(key, hashes.SHA256())
            )
            return csr.public_bytes(serialization.Encoding.PEM).decode()

        now = datetime.now(timezone.utc)
        app.enrollment_store.add_pending(
            PendingRequest(
                "hospital-1", "client", "Org", make_csr("hospital-1"), now, now + timedelta(days=7), "hospital-*"
            )
        )
        app.enrollment_store.add_pending(
            PendingRequest(
                "hospital-2", "client", "Org", make_csr("hospital-2"), now, now + timedelta(days=7), "hospital-*"
            )
        )
        app.enrollment_store.add_pending(
            PendingRequest("site-1", "client", "Org", make_csr("site-1"), now, now + timedelta(days=7), "site-1")
        )

        response = client.post(
            "/api/v1/pending/approve_batch",
            headers=auth_headers,
            json={"pattern": "hospital-*", "entity_type": "client"},
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["approved_count"] == 2
        assert "hospital-1" in data["approved_names"]
        assert "hospital-2" in data["approved_names"]

        # site-1 should still be pending
        assert app.enrollment_store.is_pending("site-1", "client")

    def test_reject_batch(self, app, client, auth_headers):
        """Test batch rejecting pending requests."""
        now = datetime.now(timezone.utc)
        app.enrollment_store.add_pending(
            PendingRequest("test-1", "client", "Org", "csr1", now, now + timedelta(days=7), "test-*")
        )
        app.enrollment_store.add_pending(
            PendingRequest("test-2", "client", "Org", "csr2", now, now + timedelta(days=7), "test-*")
        )
        app.enrollment_store.add_pending(
            PendingRequest("site-1", "client", "Org", "csr3", now, now + timedelta(days=7), "site-1")
        )

        response = client.post(
            "/api/v1/pending/reject_batch",
            headers=auth_headers,
            json={
                "pattern": "test-*",
                "entity_type": "client",
                "reason": "Test sites not allowed",
            },
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["rejected_count"] == 2

        # site-1 should still be pending
        assert app.enrollment_store.is_pending("site-1", "client")

    def test_pending_requires_auth(self, client):
        """Test pending endpoints require authentication."""
        response = client.get("/api/v1/pending")
        assert response.status_code == 401


class TestTokenRoutes(TestBaseRoutes):
    """Tests for token generation endpoints."""

    def test_generate_single_token(self, client, auth_headers):
        """Test generating a single enrollment token."""
        response = client.post(
            "/api/v1/token",
            headers=auth_headers,
            json={
                "name": "site-1",
                "entity_type": "client",
                "valid_days": 7,
            },
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "token" in data
        assert data["subject"] == "site-1"
        assert "expires_at" in data

    def test_generate_batch_tokens(self, client, auth_headers):
        """Test generating batch tokens."""
        response = client.post(
            "/api/v1/token",
            headers=auth_headers,
            json={
                "names": ["site-1", "site-2", "site-3"],
                "entity_type": "client",
                "valid_days": 7,
            },
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "tokens" in data
        assert len(data["tokens"]) == 3

    def test_generate_admin_token(self, client, auth_headers):
        """Test generating admin token with role."""
        response = client.post(
            "/api/v1/token",
            headers=auth_headers,
            json={
                "name": "admin-user",
                "entity_type": "admin",
                "role": "lead",
                "valid_days": 7,
            },
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "token" in data

    def test_generate_relay_token(self, client, auth_headers):
        """Test generating relay token."""
        response = client.post(
            "/api/v1/token",
            headers=auth_headers,
            json={
                "name": "relay-1",
                "entity_type": "relay",
                "valid_days": 7,
            },
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "token" in data

    def test_generate_token_missing_name(self, client, auth_headers):
        """Test generating token without name fails."""
        response = client.post(
            "/api/v1/token",
            headers=auth_headers,
            json={"entity_type": "client"},
        )
        assert response.status_code == 400
        assert b"Missing 'name' field" in response.data

    def test_generate_token_empty_batch(self, client, auth_headers):
        """Test generating tokens with empty names list fails."""
        response = client.post(
            "/api/v1/token",
            headers=auth_headers,
            json={"names": [], "entity_type": "client"},
        )
        assert response.status_code == 400
        assert b"non-empty list" in response.data

    def test_generate_token_requires_auth(self, client):
        """Test token generation requires authentication."""
        response = client.post(
            "/api/v1/token",
            json={"name": "site-1", "entity_type": "client"},
        )
        assert response.status_code == 401


class TestEnrollmentRoutes(TestBaseRoutes):
    """Tests for enrollment endpoint."""

    @pytest.fixture
    def valid_token(self, app):
        """Generate a valid enrollment token."""
        return app.token_service.generate_site_token(
            site_name="site-1",
            valid_days=7,
        )

    @pytest.fixture
    def sample_csr(self):
        """Generate a sample CSR."""
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

        csr = (
            x509.CertificateSigningRequestBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "site-1")]))
            .sign(private_key, hashes.SHA256())
        )

        return csr.public_bytes(encoding=serialization.Encoding.PEM).decode()

    def test_enroll_success(self, app, client, valid_token, sample_csr):
        """Test successful enrollment."""
        response = client.post(
            "/api/v1/enroll",
            json={
                "token": valid_token,
                "csr": sample_csr,
                "metadata": {
                    "name": "site-1",
                    "type": "client",
                    "org": "TestOrg",
                },
            },
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "certificate" in data
        assert "ca_cert" in data
        assert "-----BEGIN CERTIFICATE-----" in data["certificate"]

    def test_enroll_already_enrolled(self, app, client, valid_token, sample_csr):
        """Test enrollment fails if already enrolled."""
        # First enrollment
        client.post(
            "/api/v1/enroll",
            json={
                "token": valid_token,
                "csr": sample_csr,
                "metadata": {"name": "site-1", "type": "client"},
            },
        )

        # Second enrollment should fail
        response = client.post(
            "/api/v1/enroll",
            json={
                "token": valid_token,
                "csr": sample_csr,
                "metadata": {"name": "site-1", "type": "client"},
            },
        )
        assert response.status_code == 409
        assert b"already enrolled" in response.data

    def test_enroll_missing_token(self, client, sample_csr):
        """Test enrollment fails without token."""
        response = client.post(
            "/api/v1/enroll",
            json={
                "csr": sample_csr,
                "metadata": {"name": "site-1", "type": "client"},
            },
        )
        assert response.status_code == 400
        assert b"Missing 'token'" in response.data

    def test_enroll_missing_csr(self, client, valid_token):
        """Test enrollment fails without CSR."""
        response = client.post(
            "/api/v1/enroll",
            json={
                "token": valid_token,
                "metadata": {"name": "site-1", "type": "client"},
            },
        )
        assert response.status_code == 400
        assert b"Missing 'csr'" in response.data

    def test_enroll_missing_name(self, client, valid_token, sample_csr):
        """Test enrollment fails without name in metadata."""
        response = client.post(
            "/api/v1/enroll",
            json={
                "token": valid_token,
                "csr": sample_csr,
                "metadata": {"type": "client"},
            },
        )
        assert response.status_code == 400
        assert b"metadata.name" in response.data

    def test_enroll_invalid_type(self, client, valid_token, sample_csr):
        """Test enrollment fails with invalid participant type."""
        response = client.post(
            "/api/v1/enroll",
            json={
                "token": valid_token,
                "csr": sample_csr,
                "metadata": {"name": "site-1", "type": "invalid"},
            },
        )
        assert response.status_code == 400
        assert b"Invalid participant type" in response.data

    def test_enroll_expired_token(self, app, client, sample_csr):
        """Test enrollment fails with expired token."""
        # Generate an expired token by patching datetime
        from datetime import datetime as dt
        from datetime import timedelta

        import jwt

        expired_token = jwt.encode(
            {
                "sub": "site-1",
                "subject_type": "client",
                "iat": dt.utcnow() - timedelta(days=10),
                "exp": dt.utcnow() - timedelta(days=3),
                "policy": {},
            },
            app.token_service.signing_key,
            algorithm="RS256",
        )

        response = client.post(
            "/api/v1/enroll",
            json={
                "token": expired_token,
                "csr": sample_csr,
                "metadata": {"name": "site-1", "type": "client"},
            },
        )
        # Expired tokens return 401 (Unauthorized)
        assert response.status_code in [400, 401]  # Token validation error

    def test_enroll_wrong_name(self, app, client, sample_csr):
        """Test enrollment fails when name doesn't match token."""
        token = app.token_service.generate_site_token(
            site_name="site-1",  # Token is for site-1
            valid_days=7,
        )

        response = client.post(
            "/api/v1/enroll",
            json={
                "token": token,
                "csr": sample_csr,
                "metadata": {"name": "site-2", "type": "client"},  # But enrolling as site-2
            },
        )
        assert response.status_code == 400

    def test_enroll_no_body(self, client):
        """Test enrollment fails without request body."""
        response = client.post(
            "/api/v1/enroll",
            content_type="application/json",
        )
        assert response.status_code == 400


class TestEnrollmentPendingFlow(TestBaseRoutes):
    """Tests for pending enrollment flow."""

    def test_enroll_pending_manual_approval(self, app, client, auth_headers):
        """Test enrollment that requires manual approval."""
        # Create a policy that requires manual approval
        token = app.token_service.generate_site_token(
            site_name="site-1",
            valid_days=7,
            # Policy with pending action would be embedded in token
        )

        # Generate CSR
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        csr = (
            x509.CertificateSigningRequestBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "site-1")]))
            .sign(private_key, hashes.SHA256())
        )
        csr_pem = csr.public_bytes(serialization.Encoding.PEM).decode()

        # Mock the policy to return pending
        from nvflare.cert_service.cert_service import ApprovalAction, ApprovalResult

        original_evaluate = app.cert_service.evaluate_policy
        app.cert_service.evaluate_policy = lambda t, c: ApprovalResult(
            action=ApprovalAction.PENDING,
            rule_name="manual_review",
            message="Requires manual approval",
        )

        try:
            # Submit enrollment
            response = client.post(
                "/api/v1/enroll",
                json={
                    "token": token,
                    "csr": csr_pem,
                    "metadata": {"name": "site-1", "type": "client"},
                },
            )
            assert response.status_code == 202
            data = json.loads(response.data)
            assert data["status"] == "pending"

            # Should be in pending list
            response = client.get("/api/v1/pending", headers=auth_headers)
            data = json.loads(response.data)
            assert len(data) == 1
            assert data[0]["name"] == "site-1"

            # Approve
            response = client.post(
                "/api/v1/pending/site-1/approve",
                headers=auth_headers,
                json={"entity_type": "client"},
            )
            assert response.status_code == 200

            # Re-submit enrollment should get certificate
            response = client.post(
                "/api/v1/enroll",
                json={
                    "token": token,
                    "csr": csr_pem,
                    "metadata": {"name": "site-1", "type": "client"},
                },
            )
            # Already enrolled at this point
            assert response.status_code == 409

        finally:
            app.cert_service.evaluate_policy = original_evaluate
