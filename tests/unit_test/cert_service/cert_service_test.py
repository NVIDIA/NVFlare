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

"""Unit tests for CertService."""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import jwt
import pytest
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from nvflare.cert_service.cert_service import ApprovalAction, CertService, EnrollmentContext
from nvflare.lighter.constants import AdminRole, ParticipantType

# Type aliases for policy structures
PolicyDict = Dict[str, Any]
RuleDict = Dict[str, Any]
RuleList = List[RuleDict]


def generate_root_ca(tmp_dir: str):
    """Generate a root CA certificate and key for testing."""
    # Generate private key
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())

    # Generate self-signed certificate
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, "TestRootCA"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "TestOrg"),
        ]
    )

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        )
        .sign(private_key, hashes.SHA256(), default_backend())
    )

    # Save certificate
    cert_path = os.path.join(tmp_dir, "rootCA.pem")
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    # Save private key
    key_path = os.path.join(tmp_dir, "rootCA.key")
    with open(key_path, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    return private_key


def create_sample_policy() -> PolicyDict:
    """Create a sample policy dictionary for testing."""
    return {
        "metadata": {"project": "test-project", "description": "Test policy", "version": "1.0"},
        "token": {"validity": "7d", "max_uses": 1},
        "site": {"name_pattern": "*"},
        "user": {"allowed_roles": ["project_admin", "org_admin", "lead", "member"], "default_role": "member"},
        "approval": {"method": "policy", "rules": [{"name": "auto-approve-all", "match": {}, "action": "approve"}]},
    }


def make_rules(*rules: RuleDict) -> RuleList:
    """Helper to create typed rule lists for policies."""
    return list(rules)


def create_jwt_token(private_key, subject, subject_type, policy, expires_in=3600, **extra_claims):
    """Create a valid JWT token for testing."""
    import uuid

    now = datetime.now(timezone.utc)
    payload = {
        "jti": str(uuid.uuid4()),
        "sub": subject,
        "subject_type": subject_type,
        "policy": policy,
        "iat": now,
        "exp": now + timedelta(seconds=expires_in),
        "iss": "TestRootCA",
        **extra_claims,
    }
    return jwt.encode(payload, private_key, algorithm="RS256")


def create_csr(subject_name, org_name=None):
    """Create a CSR for testing."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())

    name_attrs = [x509.NameAttribute(NameOID.COMMON_NAME, subject_name)]
    if org_name:
        name_attrs.append(x509.NameAttribute(NameOID.ORGANIZATION_NAME, org_name))

    builder = x509.CertificateSigningRequestBuilder().subject_name(x509.Name(name_attrs))

    csr = builder.sign(private_key, hashes.SHA256(), default_backend())
    return csr.public_bytes(serialization.Encoding.PEM)


@pytest.fixture
def root_ca_dir():
    """Create a temporary directory with root CA."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        generate_root_ca(tmp_dir)
        yield tmp_dir


@pytest.fixture
def root_ca_key(root_ca_dir):
    """Load root CA private key."""
    key_path = os.path.join(root_ca_dir, "rootCA.key")
    with open(key_path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


@pytest.fixture
def cert_service(root_ca_dir):
    """Create a CertService instance for testing."""
    return CertService(
        root_ca_cert_path=os.path.join(root_ca_dir, "rootCA.pem"),
        root_ca_key_path=os.path.join(root_ca_dir, "rootCA.key"),
    )


class TestCertServiceInit:
    """Tests for CertService initialization."""

    def test_init_success(self, root_ca_dir):
        """Test successful initialization with valid CA."""
        service = CertService(
            root_ca_cert_path=os.path.join(root_ca_dir, "rootCA.pem"),
            root_ca_key_path=os.path.join(root_ca_dir, "rootCA.key"),
        )
        assert service.issuer == "TestRootCA"
        assert service.root_cert is not None
        assert service.root_key is not None
        assert service.jwt_public_key is not None

    def test_init_missing_ca(self):
        """Test initialization fails with missing CA."""
        with pytest.raises(FileNotFoundError):
            CertService(
                root_ca_cert_path="/nonexistent/rootCA.pem",
                root_ca_key_path="/nonexistent/rootCA.key",
            )


class TestValidateToken:
    """Tests for token validation."""

    def test_validate_valid_token(self, cert_service, root_ca_key):
        """Test validating a valid token."""
        policy = create_sample_policy()
        token = create_jwt_token(root_ca_key, subject="hospital-01", subject_type=ParticipantType.CLIENT, policy=policy)

        payload = cert_service.validate_token(token)

        assert payload.subject == "hospital-01"
        assert payload.subject_type == ParticipantType.CLIENT
        assert payload.policy == policy
        assert payload.issuer == "TestRootCA"

    def test_validate_expired_token(self, cert_service, root_ca_key):
        """Test that expired token raises error."""
        policy = create_sample_policy()
        token = create_jwt_token(
            root_ca_key,
            subject="hospital-01",
            subject_type=ParticipantType.CLIENT,
            policy=policy,
            expires_in=-3600,  # Expired
        )

        with pytest.raises(ValueError, match="token has expired"):
            cert_service.validate_token(token)

    def test_validate_tampered_token(self, cert_service):
        """Test that tampered token raises error."""
        # Create a token with a different key
        other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
        policy = create_sample_policy()
        token = create_jwt_token(other_key, subject="hospital-01", subject_type=ParticipantType.CLIENT, policy=policy)

        with pytest.raises(ValueError, match="Invalid token signature"):
            cert_service.validate_token(token)

    def test_validate_malformed_token(self, cert_service):
        """Test that malformed token raises error."""
        with pytest.raises(ValueError, match="Malformed token"):
            cert_service.validate_token("not-a-valid-jwt")

    def test_validate_token_with_context_match(self, cert_service, root_ca_key):
        """Test validating token with matching context."""
        policy = create_sample_policy()
        token = create_jwt_token(root_ca_key, subject="hospital-01", subject_type=ParticipantType.CLIENT, policy=policy)

        context = EnrollmentContext(name="hospital-01", participant_type=ParticipantType.CLIENT)

        payload = cert_service.validate_token(token, context)
        assert payload.subject == "hospital-01"

    def test_validate_token_with_context_mismatch(self, cert_service, root_ca_key):
        """Test validating token with mismatched context."""
        policy = create_sample_policy()
        token = create_jwt_token(root_ca_key, subject="hospital-01", subject_type=ParticipantType.CLIENT, policy=policy)

        context = EnrollmentContext(name="different-site", participant_type=ParticipantType.CLIENT)

        with pytest.raises(ValueError, match="does not match token subject"):
            cert_service.validate_token(token, context)

    def test_validate_pattern_token(self, cert_service, root_ca_key):
        """Test validating pattern token with matching name."""
        policy = create_sample_policy()
        token = create_jwt_token(root_ca_key, subject="hospital-*", subject_type="pattern", policy=policy)

        context = EnrollmentContext(name="hospital-01", participant_type=ParticipantType.CLIENT)

        payload = cert_service.validate_token(token, context)
        assert payload.subject == "hospital-*"

    def test_validate_pattern_token_no_match(self, cert_service, root_ca_key):
        """Test validating pattern token with non-matching name."""
        policy = create_sample_policy()
        token = create_jwt_token(root_ca_key, subject="hospital-*", subject_type="pattern", policy=policy)

        context = EnrollmentContext(name="clinic-01", participant_type=ParticipantType.CLIENT)

        with pytest.raises(ValueError, match="does not match token pattern"):
            cert_service.validate_token(token, context)

    def test_validate_token_participant_type_mismatch(self, cert_service, root_ca_key):
        """Test that token subject_type must match context participant_type."""
        policy = create_sample_policy()
        # Token for client
        token = create_jwt_token(root_ca_key, subject="site-01", subject_type=ParticipantType.CLIENT, policy=policy)

        # Context says admin - should fail
        context = EnrollmentContext(name="site-01", participant_type=ParticipantType.ADMIN)

        with pytest.raises(ValueError, match="does not match enrollment type"):
            cert_service.validate_token(token, context)

    def test_validate_admin_token_with_relay_context(self, cert_service, root_ca_key):
        """Test that admin token cannot be used for relay enrollment."""
        policy = create_sample_policy()
        token = create_jwt_token(
            root_ca_key, subject="user@example.com", subject_type=ParticipantType.ADMIN, policy=policy
        )

        context = EnrollmentContext(name="user@example.com", participant_type=ParticipantType.RELAY)

        with pytest.raises(ValueError, match="does not match enrollment type"):
            cert_service.validate_token(token, context)

    def test_validate_relay_token_with_client_context(self, cert_service, root_ca_key):
        """Test that relay token cannot be used for client enrollment."""
        policy = create_sample_policy()
        token = create_jwt_token(root_ca_key, subject="relay-01", subject_type=ParticipantType.RELAY, policy=policy)

        context = EnrollmentContext(name="relay-01", participant_type=ParticipantType.CLIENT)

        with pytest.raises(ValueError, match="does not match enrollment type"):
            cert_service.validate_token(token, context)


class TestEvaluatePolicy:
    """Tests for policy evaluation."""

    def test_evaluate_auto_approve(self, cert_service, root_ca_key):
        """Test policy with auto-approve rule."""
        policy = create_sample_policy()
        token = create_jwt_token(root_ca_key, subject="hospital-01", subject_type=ParticipantType.CLIENT, policy=policy)

        token_payload = cert_service.validate_token(token)
        context = EnrollmentContext(name="hospital-01", participant_type=ParticipantType.CLIENT)

        result = cert_service.evaluate_policy(token_payload, context)

        assert result.action == ApprovalAction.APPROVE
        assert result.rule_name == "auto-approve-all"

    def test_evaluate_reject_rule(self, cert_service, root_ca_key):
        """Test policy with reject rule."""
        policy = create_sample_policy()
        policy["approval"]["rules"] = make_rules(
            {"name": "reject-all", "match": {}, "action": "reject", "message": "All enrollments rejected"}
        )

        token = create_jwt_token(root_ca_key, subject="hospital-01", subject_type=ParticipantType.CLIENT, policy=policy)

        token_payload = cert_service.validate_token(token)
        context = EnrollmentContext(name="hospital-01", participant_type=ParticipantType.CLIENT)

        result = cert_service.evaluate_policy(token_payload, context)

        assert result.action == ApprovalAction.REJECT
        assert result.rule_name == "reject-all"

    def test_evaluate_pending_rule(self, cert_service, root_ca_key):
        """Test policy with pending rule."""
        policy = create_sample_policy()
        policy["approval"]["rules"] = make_rules(
            {"name": "manual-review", "match": {}, "action": "pending", "notify": ["admin@example.com"]}
        )

        token = create_jwt_token(root_ca_key, subject="hospital-01", subject_type=ParticipantType.CLIENT, policy=policy)

        token_payload = cert_service.validate_token(token)
        context = EnrollmentContext(name="hospital-01", participant_type=ParticipantType.CLIENT)

        result = cert_service.evaluate_policy(token_payload, context)

        assert result.action == ApprovalAction.PENDING
        assert result.notify == ["admin@example.com"]

    def test_evaluate_site_name_pattern(self, cert_service, root_ca_key):
        """Test policy with site name pattern matching."""
        policy = create_sample_policy()
        policy["approval"]["rules"] = make_rules(
            {"name": "approve-hospitals", "match": {"site_name_pattern": "hospital-*"}, "action": "approve"},
            {"name": "reject-others", "match": {}, "action": "reject"},
        )

        token = create_jwt_token(root_ca_key, subject="*", subject_type="pattern", policy=policy)

        token_payload = cert_service.validate_token(token)

        # Matching context
        context1 = EnrollmentContext(name="hospital-01", participant_type=ParticipantType.CLIENT)
        result1 = cert_service.evaluate_policy(token_payload, context1)
        assert result1.action == ApprovalAction.APPROVE

        # Non-matching context
        context2 = EnrollmentContext(name="clinic-01", participant_type=ParticipantType.CLIENT)
        result2 = cert_service.evaluate_policy(token_payload, context2)
        assert result2.action == ApprovalAction.REJECT

    def test_evaluate_source_ips(self, cert_service, root_ca_key):
        """Test policy with source IP matching."""
        policy = create_sample_policy()
        policy["approval"]["rules"] = make_rules(
            {"name": "approve-internal", "match": {"source_ips": ["10.0.0.0/8"]}, "action": "approve"},
            {"name": "reject-external", "match": {}, "action": "reject"},
        )

        token = create_jwt_token(root_ca_key, subject="*", subject_type="pattern", policy=policy)

        token_payload = cert_service.validate_token(token)

        # Matching IP
        context1 = EnrollmentContext(name="site-01", participant_type=ParticipantType.CLIENT, source_ip="10.0.0.5")
        result1 = cert_service.evaluate_policy(token_payload, context1)
        assert result1.action == ApprovalAction.APPROVE

        # Non-matching IP
        context2 = EnrollmentContext(name="site-02", participant_type=ParticipantType.CLIENT, source_ip="192.168.1.5")
        result2 = cert_service.evaluate_policy(token_payload, context2)
        assert result2.action == ApprovalAction.REJECT

    def test_evaluate_source_ips_required(self, cert_service, root_ca_key):
        """Test that source IPs rule requires client to provide IP."""
        policy = create_sample_policy()
        policy["approval"]["rules"] = make_rules(
            {"name": "approve-internal", "match": {"source_ips": ["10.0.0.0/8"]}, "action": "approve"},
        )

        token = create_jwt_token(root_ca_key, subject="*", subject_type="pattern", policy=policy)

        token_payload = cert_service.validate_token(token)

        # No IP provided - should fail to match
        context = EnrollmentContext(name="site-01", participant_type=ParticipantType.CLIENT, source_ip=None)
        result = cert_service.evaluate_policy(token_payload, context)
        assert result.action == ApprovalAction.REJECT

    def test_evaluate_admin_role_constraint(self, cert_service, root_ca_key):
        """Test policy with admin role constraints."""
        policy = create_sample_policy()
        policy["user"]["allowed_roles"] = ["project_admin", "org_admin"]

        token = create_jwt_token(
            root_ca_key, subject="user@example.com", subject_type=ParticipantType.ADMIN, policy=policy
        )

        token_payload = cert_service.validate_token(token)

        # Allowed role
        context1 = EnrollmentContext(
            name="user@example.com", participant_type=ParticipantType.ADMIN, role=AdminRole.ORG_ADMIN
        )
        result1 = cert_service.evaluate_policy(token_payload, context1)
        assert result1.action == ApprovalAction.APPROVE

        # Disallowed role
        context2 = EnrollmentContext(
            name="user@example.com", participant_type=ParticipantType.ADMIN, role=AdminRole.MEMBER
        )
        result2 = cert_service.evaluate_policy(token_payload, context2)
        assert result2.action == ApprovalAction.REJECT
        assert "not in allowed roles" in result2.message

    def test_evaluate_no_rules_default_approve(self, cert_service, root_ca_key):
        """Test that no rules defaults to approve."""
        policy = create_sample_policy()
        policy["approval"]["rules"] = []

        token = create_jwt_token(root_ca_key, subject="site-01", subject_type=ParticipantType.CLIENT, policy=policy)

        token_payload = cert_service.validate_token(token)
        context = EnrollmentContext(name="site-01", participant_type=ParticipantType.CLIENT)

        result = cert_service.evaluate_policy(token_payload, context)

        assert result.action == ApprovalAction.APPROVE
        assert result.rule_name == "default"


class TestSignCSR:
    """Tests for CSR signing."""

    def test_sign_csr_approved(self, cert_service, root_ca_key):
        """Test signing CSR for approved enrollment."""
        policy = create_sample_policy()
        token = create_jwt_token(root_ca_key, subject="hospital-01", subject_type=ParticipantType.CLIENT, policy=policy)

        csr_data = create_csr("hospital-01", "Hospital A")
        context = EnrollmentContext(name="hospital-01", participant_type=ParticipantType.CLIENT)

        signed_cert = cert_service.sign_csr(csr_data=csr_data, token=token, context=context)

        assert signed_cert is not None
        assert b"-----BEGIN CERTIFICATE-----" in signed_cert

        # Verify cert can be loaded
        cert = x509.load_pem_x509_certificate(signed_cert, default_backend())
        assert cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == "hospital-01"

    def test_sign_csr_with_org(self, cert_service, root_ca_key):
        """Test signing CSR includes organization in certificate."""
        policy = create_sample_policy()
        token = create_jwt_token(root_ca_key, subject="hospital-01", subject_type=ParticipantType.CLIENT, policy=policy)

        csr_data = create_csr("hospital-01")
        context = EnrollmentContext(name="hospital-01", participant_type=ParticipantType.CLIENT, org="Acme Hospital")

        signed_cert = cert_service.sign_csr(csr_data=csr_data, token=token, context=context)

        # Verify org is in certificate
        cert = x509.load_pem_x509_certificate(signed_cert, default_backend())
        org_attrs = cert.subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)
        assert len(org_attrs) == 1
        assert org_attrs[0].value == "Acme Hospital"

    def test_sign_csr_admin_with_role(self, cert_service, root_ca_key):
        """Test signing CSR for admin includes role in certificate."""
        policy = create_sample_policy()
        token = create_jwt_token(
            root_ca_key, subject="user@example.com", subject_type=ParticipantType.ADMIN, policy=policy
        )

        csr_data = create_csr("user@example.com")
        context = EnrollmentContext(
            name="user@example.com",
            participant_type=ParticipantType.ADMIN,
            org="Research Org",
            role=AdminRole.LEAD,
        )

        signed_cert = cert_service.sign_csr(csr_data=csr_data, token=token, context=context)

        # Verify role is embedded (in unstructured name)
        cert = x509.load_pem_x509_certificate(signed_cert, default_backend())
        cn = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
        assert cn == "user@example.com"

    def test_sign_csr_relay_with_role(self, cert_service, root_ca_key):
        """Test signing CSR for relay includes 'relay' role in certificate."""
        policy = create_sample_policy()
        token = create_jwt_token(root_ca_key, subject="relay-01", subject_type=ParticipantType.RELAY, policy=policy)

        csr_data = create_csr("relay-01")
        context = EnrollmentContext(name="relay-01", participant_type=ParticipantType.RELAY)

        signed_cert = cert_service.sign_csr(csr_data=csr_data, token=token, context=context)

        # Verify cert can be loaded
        cert = x509.load_pem_x509_certificate(signed_cert, default_backend())
        assert cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == "relay-01"

    def test_sign_csr_rejected(self, cert_service, root_ca_key):
        """Test signing CSR for rejected enrollment raises error."""
        policy = create_sample_policy()
        policy["approval"]["rules"] = make_rules(
            {"name": "reject-all", "match": {}, "action": "reject", "message": "Rejected"},
        )

        token = create_jwt_token(root_ca_key, subject="hospital-01", subject_type=ParticipantType.CLIENT, policy=policy)

        csr_data = create_csr("hospital-01")
        context = EnrollmentContext(name="hospital-01", participant_type=ParticipantType.CLIENT)

        with pytest.raises(ValueError, match="Enrollment rejected"):
            cert_service.sign_csr(csr_data=csr_data, token=token, context=context)

    def test_sign_csr_pending(self, cert_service, root_ca_key):
        """Test signing CSR for pending enrollment raises error."""
        policy = create_sample_policy()
        policy["approval"]["rules"] = make_rules(
            {"name": "pending", "match": {}, "action": "pending", "notify": ["admin@test.com"]},
        )

        token = create_jwt_token(root_ca_key, subject="hospital-01", subject_type=ParticipantType.CLIENT, policy=policy)

        csr_data = create_csr("hospital-01")
        context = EnrollmentContext(name="hospital-01", participant_type=ParticipantType.CLIENT)

        with pytest.raises(ValueError, match="manual approval"):
            cert_service.sign_csr(csr_data=csr_data, token=token, context=context)

    def test_sign_csr_invalid_token(self, cert_service):
        """Test signing CSR with invalid token raises error."""
        csr_data = create_csr("hospital-01")
        context = EnrollmentContext(name="hospital-01", participant_type=ParticipantType.CLIENT)

        with pytest.raises(ValueError):
            cert_service.sign_csr(csr_data=csr_data, token="invalid-token", context=context)

    def test_sign_csr_invalid_csr(self, cert_service, root_ca_key):
        """Test signing invalid CSR raises error."""
        policy = create_sample_policy()
        token = create_jwt_token(root_ca_key, subject="hospital-01", subject_type=ParticipantType.CLIENT, policy=policy)

        context = EnrollmentContext(name="hospital-01", participant_type=ParticipantType.CLIENT)

        with pytest.raises(Exception):
            cert_service.sign_csr(csr_data=b"not-a-valid-csr", token=token, context=context)


class TestIPInRanges:
    """Tests for IP range checking."""

    def test_ip_in_single_range(self, cert_service):
        """Test IP matching single CIDR range."""
        assert cert_service._ip_in_ranges("10.0.0.5", ["10.0.0.0/8"]) is True
        assert cert_service._ip_in_ranges("192.168.1.1", ["10.0.0.0/8"]) is False

    def test_ip_in_multiple_ranges(self, cert_service):
        """Test IP matching multiple CIDR ranges."""
        ranges = ["10.0.0.0/8", "192.168.0.0/16", "172.16.0.0/12"]
        assert cert_service._ip_in_ranges("10.0.0.5", ranges) is True
        assert cert_service._ip_in_ranges("192.168.1.1", ranges) is True
        assert cert_service._ip_in_ranges("172.16.0.1", ranges) is True
        assert cert_service._ip_in_ranges("8.8.8.8", ranges) is False

    def test_ip_exact_match(self, cert_service):
        """Test exact IP matching."""
        assert cert_service._ip_in_ranges("10.0.0.5", ["10.0.0.5/32"]) is True
        assert cert_service._ip_in_ranges("10.0.0.6", ["10.0.0.5/32"]) is False

    def test_invalid_ip(self, cert_service):
        """Test invalid IP returns False."""
        assert cert_service._ip_in_ranges("not-an-ip", ["10.0.0.0/8"]) is False

    def test_invalid_range(self, cert_service):
        """Test invalid range is skipped."""
        assert cert_service._ip_in_ranges("10.0.0.5", ["invalid", "10.0.0.0/8"]) is True


class TestPolicyApprovalScenarios:
    """Comprehensive tests for policy approval/rejection scenarios."""

    def test_first_match_wins(self, cert_service, root_ca_key):
        """Test that first matching rule wins (order matters)."""
        policy = create_sample_policy()
        policy["approval"]["rules"] = make_rules(
            {"name": "rule-1", "match": {"site_name_pattern": "hospital-*"}, "action": "reject"},
            {"name": "rule-2", "match": {"site_name_pattern": "hospital-*"}, "action": "approve"},
            {"name": "rule-3", "match": {}, "action": "pending"},
        )

        token = create_jwt_token(root_ca_key, "*", "pattern", policy)
        token_payload = cert_service.validate_token(token)

        # hospital-01 matches rule-1 first -> reject
        context = EnrollmentContext(name="hospital-01", participant_type=ParticipantType.CLIENT)
        result = cert_service.evaluate_policy(token_payload, context)
        assert result.action == ApprovalAction.REJECT
        assert result.rule_name == "rule-1"

    def test_fallthrough_to_later_rule(self, cert_service, root_ca_key):
        """Test fallthrough when first rule doesn't match."""
        policy = create_sample_policy()
        policy["approval"]["rules"] = make_rules(
            {"name": "hospitals", "match": {"site_name_pattern": "hospital-*"}, "action": "approve"},
            {"name": "clinics", "match": {"site_name_pattern": "clinic-*"}, "action": "pending"},
            {"name": "reject-rest", "match": {}, "action": "reject"},
        )

        token = create_jwt_token(root_ca_key, "*", "pattern", policy)
        token_payload = cert_service.validate_token(token)

        # hospital matches first rule
        ctx1 = EnrollmentContext(name="hospital-01", participant_type=ParticipantType.CLIENT)
        assert cert_service.evaluate_policy(token_payload, ctx1).action == ApprovalAction.APPROVE

        # clinic matches second rule
        ctx2 = EnrollmentContext(name="clinic-01", participant_type=ParticipantType.CLIENT)
        assert cert_service.evaluate_policy(token_payload, ctx2).action == ApprovalAction.PENDING

        # unknown matches catch-all
        ctx3 = EnrollmentContext(name="lab-01", participant_type=ParticipantType.CLIENT)
        assert cert_service.evaluate_policy(token_payload, ctx3).action == ApprovalAction.REJECT

    def test_no_matching_rule_default_deny(self, cert_service, root_ca_key):
        """Test default deny when no rule matches."""
        policy = create_sample_policy()
        policy["approval"]["rules"] = make_rules(
            {"name": "hospitals-only", "match": {"site_name_pattern": "hospital-*"}, "action": "approve"},
        )

        token = create_jwt_token(root_ca_key, "*", "pattern", policy)
        token_payload = cert_service.validate_token(token)

        # No match -> default deny
        context = EnrollmentContext(name="clinic-01", participant_type=ParticipantType.CLIENT)
        result = cert_service.evaluate_policy(token_payload, context)
        assert result.action == ApprovalAction.REJECT
        assert result.rule_name == "default-deny"

    def test_combined_site_and_ip_match(self, cert_service, root_ca_key):
        """Test matching both site name and source IP."""
        policy = create_sample_policy()
        policy["approval"]["rules"] = make_rules(
            {
                "name": "internal-hospitals",
                "match": {"site_name_pattern": "hospital-*", "source_ips": ["10.0.0.0/8"]},
                "action": "approve",
            },
            {"name": "reject-rest", "match": {}, "action": "reject"},
        )

        token = create_jwt_token(root_ca_key, "*", "pattern", policy)
        token_payload = cert_service.validate_token(token)

        # Both conditions match
        ctx1 = EnrollmentContext(name="hospital-01", participant_type=ParticipantType.CLIENT, source_ip="10.0.0.5")
        assert cert_service.evaluate_policy(token_payload, ctx1).action == ApprovalAction.APPROVE

        # Name matches but IP doesn't
        ctx2 = EnrollmentContext(name="hospital-01", participant_type=ParticipantType.CLIENT, source_ip="192.168.1.1")
        assert cert_service.evaluate_policy(token_payload, ctx2).action == ApprovalAction.REJECT

        # IP matches but name doesn't
        ctx3 = EnrollmentContext(name="clinic-01", participant_type=ParticipantType.CLIENT, source_ip="10.0.0.5")
        assert cert_service.evaluate_policy(token_payload, ctx3).action == ApprovalAction.REJECT

    def test_ip_whitelist_no_site_restriction(self, cert_service, root_ca_key):
        """Test IP whitelist without site name restriction."""
        policy = create_sample_policy()
        policy["approval"]["rules"] = make_rules(
            {"name": "internal-network", "match": {"source_ips": ["10.0.0.0/8", "172.16.0.0/12"]}, "action": "approve"},
            {"name": "reject-external", "match": {}, "action": "reject"},
        )

        token = create_jwt_token(root_ca_key, "*", "pattern", policy)
        token_payload = cert_service.validate_token(token)

        # Any site from internal network
        ctx1 = EnrollmentContext(name="any-site-01", participant_type=ParticipantType.CLIENT, source_ip="10.0.0.5")
        assert cert_service.evaluate_policy(token_payload, ctx1).action == ApprovalAction.APPROVE

        ctx2 = EnrollmentContext(name="any-site-02", participant_type=ParticipantType.CLIENT, source_ip="172.16.0.1")
        assert cert_service.evaluate_policy(token_payload, ctx2).action == ApprovalAction.APPROVE

        # External IP rejected
        ctx3 = EnrollmentContext(name="any-site-03", participant_type=ParticipantType.CLIENT, source_ip="8.8.8.8")
        assert cert_service.evaluate_policy(token_payload, ctx3).action == ApprovalAction.REJECT

    def test_wildcard_patterns(self, cert_service, root_ca_key):
        """Test various wildcard patterns."""
        policy = create_sample_policy()
        policy["approval"]["rules"] = make_rules(
            {"name": "prefix", "match": {"site_name_pattern": "prod-*"}, "action": "approve"},
            {"name": "suffix", "match": {"site_name_pattern": "*-prod"}, "action": "approve"},
            {"name": "contains", "match": {"site_name_pattern": "*staging*"}, "action": "pending"},
            {"name": "exact", "match": {"site_name_pattern": "special"}, "action": "approve"},
            {"name": "reject-rest", "match": {}, "action": "reject"},
        )

        token = create_jwt_token(root_ca_key, "*", "pattern", policy)
        token_payload = cert_service.validate_token(token)

        # Prefix match
        ctx1 = EnrollmentContext(name="prod-server-01", participant_type=ParticipantType.CLIENT)
        assert cert_service.evaluate_policy(token_payload, ctx1).action == ApprovalAction.APPROVE

        # Suffix match
        ctx2 = EnrollmentContext(name="server-prod", participant_type=ParticipantType.CLIENT)
        assert cert_service.evaluate_policy(token_payload, ctx2).action == ApprovalAction.APPROVE

        # Contains match
        ctx3 = EnrollmentContext(name="server-staging-01", participant_type=ParticipantType.CLIENT)
        assert cert_service.evaluate_policy(token_payload, ctx3).action == ApprovalAction.PENDING

        # Exact match
        ctx4 = EnrollmentContext(name="special", participant_type=ParticipantType.CLIENT)
        assert cert_service.evaluate_policy(token_payload, ctx4).action == ApprovalAction.APPROVE

        # No match
        ctx5 = EnrollmentContext(name="dev-server", participant_type=ParticipantType.CLIENT)
        assert cert_service.evaluate_policy(token_payload, ctx5).action == ApprovalAction.REJECT

    def test_admin_role_enforcement(self, cert_service, root_ca_key):
        """Test admin role enforcement with different roles."""
        policy = create_sample_policy()
        policy["user"]["allowed_roles"] = ["project_admin", "org_admin"]
        policy["approval"]["rules"] = make_rules(
            {"name": "approve-all", "match": {}, "action": "approve"},
        )

        token = create_jwt_token(root_ca_key, "user@example.com", ParticipantType.ADMIN, policy)
        token_payload = cert_service.validate_token(token)

        # Allowed role
        ctx1 = EnrollmentContext(
            name="user@example.com", participant_type=ParticipantType.ADMIN, role=AdminRole.PROJECT_ADMIN
        )
        assert cert_service.evaluate_policy(token_payload, ctx1).action == ApprovalAction.APPROVE

        # Another allowed role
        ctx2 = EnrollmentContext(
            name="user@example.com", participant_type=ParticipantType.ADMIN, role=AdminRole.ORG_ADMIN
        )
        assert cert_service.evaluate_policy(token_payload, ctx2).action == ApprovalAction.APPROVE

        # Disallowed role
        ctx3 = EnrollmentContext(name="user@example.com", participant_type=ParticipantType.ADMIN, role=AdminRole.MEMBER)
        result = cert_service.evaluate_policy(token_payload, ctx3)
        assert result.action == ApprovalAction.REJECT
        assert result.rule_name == "role-constraint"

        # Disallowed role
        ctx4 = EnrollmentContext(name="user@example.com", participant_type=ParticipantType.ADMIN, role=AdminRole.LEAD)
        assert cert_service.evaluate_policy(token_payload, ctx4).action == ApprovalAction.REJECT

    def test_role_bypass_for_clients(self, cert_service, root_ca_key):
        """Test that role constraints don't affect client enrollments."""
        policy = create_sample_policy()
        policy["user"]["allowed_roles"] = ["project_admin"]  # Restrictive
        policy["approval"]["rules"] = make_rules(
            {"name": "approve-all", "match": {}, "action": "approve"},
        )

        token = create_jwt_token(root_ca_key, "site-01", ParticipantType.CLIENT, policy)
        token_payload = cert_service.validate_token(token)

        # Client enrollment should not be affected by role constraints
        context = EnrollmentContext(name="site-01", participant_type=ParticipantType.CLIENT)
        assert cert_service.evaluate_policy(token_payload, context).action == ApprovalAction.APPROVE

    def test_empty_match_matches_all(self, cert_service, root_ca_key):
        """Test that empty match condition matches everything."""
        policy = create_sample_policy()
        policy["approval"]["rules"] = make_rules(
            {"name": "approve-all", "match": {}, "action": "approve"},
        )

        token = create_jwt_token(root_ca_key, "*", "pattern", policy)
        token_payload = cert_service.validate_token(token)

        # Should match any site
        for name in ["site-01", "hospital-01", "clinic-01", "x", "long-name-with-dashes"]:
            context = EnrollmentContext(name=name, participant_type=ParticipantType.CLIENT)
            assert cert_service.evaluate_policy(token_payload, context).action == ApprovalAction.APPROVE

    def test_pending_with_notification(self, cert_service, root_ca_key):
        """Test pending action with notification list."""
        policy = create_sample_policy()
        policy["approval"]["rules"] = make_rules(
            {
                "name": "manual-review",
                "match": {"site_name_pattern": "external-*"},
                "action": "pending",
                "notify": ["admin@company.com", "security@company.com"],
                "message": "External site requires manual review",
            },
            {"name": "approve-rest", "match": {}, "action": "approve"},
        )

        token = create_jwt_token(root_ca_key, "*", "pattern", policy)
        token_payload = cert_service.validate_token(token)

        context = EnrollmentContext(name="external-partner-01", participant_type=ParticipantType.CLIENT)
        result = cert_service.evaluate_policy(token_payload, context)

        assert result.action == ApprovalAction.PENDING
        assert result.notify == ["admin@company.com", "security@company.com"]
        assert result.message == "External site requires manual review"

    def test_reject_with_message(self, cert_service, root_ca_key):
        """Test reject action with custom message."""
        policy = create_sample_policy()
        policy["approval"]["rules"] = make_rules(
            {
                "name": "block-test",
                "match": {"site_name_pattern": "test-*"},
                "action": "reject",
                "message": "Test environments not allowed in production",
            },
            {"name": "approve-rest", "match": {}, "action": "approve"},
        )

        token = create_jwt_token(root_ca_key, "*", "pattern", policy)
        token_payload = cert_service.validate_token(token)

        context = EnrollmentContext(name="test-server-01", participant_type=ParticipantType.CLIENT)
        result = cert_service.evaluate_policy(token_payload, context)

        assert result.action == ApprovalAction.REJECT
        assert result.message == "Test environments not allowed in production"

    def test_multiple_ip_ranges(self, cert_service, root_ca_key):
        """Test matching against multiple IP ranges (AWS, GCP, Azure style)."""
        policy = create_sample_policy()
        cloud_ips = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10"]
        policy["approval"]["rules"] = make_rules(
            {"name": "cloud-providers", "match": {"source_ips": cloud_ips}, "action": "approve"},
            {"name": "reject-public", "match": {}, "action": "reject"},
        )

        token = create_jwt_token(root_ca_key, "*", "pattern", policy)
        token_payload = cert_service.validate_token(token)

        # AWS internal
        ctx1 = EnrollmentContext(name="aws-01", participant_type=ParticipantType.CLIENT, source_ip="10.128.0.5")
        assert cert_service.evaluate_policy(token_payload, ctx1).action == ApprovalAction.APPROVE

        # GCP internal
        ctx2 = EnrollmentContext(name="gcp-01", participant_type=ParticipantType.CLIENT, source_ip="172.20.0.5")
        assert cert_service.evaluate_policy(token_payload, ctx2).action == ApprovalAction.APPROVE

        # Azure internal
        ctx3 = EnrollmentContext(name="azure-01", participant_type=ParticipantType.CLIENT, source_ip="192.168.5.10")
        assert cert_service.evaluate_policy(token_payload, ctx3).action == ApprovalAction.APPROVE

        # CGNAT
        ctx4 = EnrollmentContext(name="cgnat-01", participant_type=ParticipantType.CLIENT, source_ip="100.64.1.5")
        assert cert_service.evaluate_policy(token_payload, ctx4).action == ApprovalAction.APPROVE

        # Public IP rejected
        ctx5 = EnrollmentContext(name="public-01", participant_type=ParticipantType.CLIENT, source_ip="203.0.113.5")
        assert cert_service.evaluate_policy(token_payload, ctx5).action == ApprovalAction.REJECT

    def test_relay_enrollment(self, cert_service, root_ca_key):
        """Test relay node enrollment policies."""
        policy = create_sample_policy()
        policy["approval"]["rules"] = make_rules(
            {"name": "approve-relays", "match": {"site_name_pattern": "relay-*"}, "action": "approve"},
            {"name": "reject-rest", "match": {}, "action": "reject"},
        )

        token = create_jwt_token(root_ca_key, "relay-*", "pattern", policy)
        token_payload = cert_service.validate_token(token)

        # Relay enrollment
        context = EnrollmentContext(name="relay-01", participant_type=ParticipantType.RELAY)
        assert cert_service.evaluate_policy(token_payload, context).action == ApprovalAction.APPROVE

    def test_tiered_approval(self, cert_service, root_ca_key):
        """Test tiered approval based on site naming convention."""
        policy = create_sample_policy()
        policy["approval"]["rules"] = make_rules(
            {"name": "trusted-partners", "match": {"site_name_pattern": "partner-trusted-*"}, "action": "approve"},
            {
                "name": "new-partners",
                "match": {"site_name_pattern": "partner-*"},
                "action": "pending",
                "notify": ["partnerships@company.com"],
            },
            {"name": "internal", "match": {"site_name_pattern": "internal-*"}, "action": "approve"},
            {"name": "default-reject", "match": {}, "action": "reject"},
        )

        token = create_jwt_token(root_ca_key, "*", "pattern", policy)
        token_payload = cert_service.validate_token(token)

        # Trusted partner - auto approve
        ctx1 = EnrollmentContext(name="partner-trusted-acme", participant_type=ParticipantType.CLIENT)
        assert cert_service.evaluate_policy(token_payload, ctx1).action == ApprovalAction.APPROVE

        # New partner - pending
        ctx2 = EnrollmentContext(name="partner-newcorp", participant_type=ParticipantType.CLIENT)
        result = cert_service.evaluate_policy(token_payload, ctx2)
        assert result.action == ApprovalAction.PENDING
        assert result.notify == ["partnerships@company.com"]

        # Internal - approved
        ctx3 = EnrollmentContext(name="internal-server-01", participant_type=ParticipantType.CLIENT)
        assert cert_service.evaluate_policy(token_payload, ctx3).action == ApprovalAction.APPROVE

        # Unknown - rejected
        ctx4 = EnrollmentContext(name="unknown-site", participant_type=ParticipantType.CLIENT)
        assert cert_service.evaluate_policy(token_payload, ctx4).action == ApprovalAction.REJECT


class TestParticipantTypeValidation:
    """Tests for participant type validation between token and enrollment context."""

    def test_client_token_for_client_enrollment(self, cert_service, root_ca_key):
        """Test client token works for client enrollment."""
        policy = create_sample_policy()
        token = create_jwt_token(root_ca_key, "site-01", ParticipantType.CLIENT, policy)

        context = EnrollmentContext(name="site-01", participant_type=ParticipantType.CLIENT)
        payload = cert_service.validate_token(token, context)
        assert payload.subject == "site-01"

    def test_admin_token_for_admin_enrollment(self, cert_service, root_ca_key):
        """Test admin token works for admin enrollment."""
        policy = create_sample_policy()
        token = create_jwt_token(root_ca_key, "user@example.com", ParticipantType.ADMIN, policy)

        context = EnrollmentContext(
            name="user@example.com", participant_type=ParticipantType.ADMIN, role=AdminRole.LEAD
        )
        payload = cert_service.validate_token(token, context)
        assert payload.subject == "user@example.com"

    def test_relay_token_for_relay_enrollment(self, cert_service, root_ca_key):
        """Test relay token works for relay enrollment."""
        policy = create_sample_policy()
        token = create_jwt_token(root_ca_key, "relay-01", ParticipantType.RELAY, policy)

        context = EnrollmentContext(name="relay-01", participant_type=ParticipantType.RELAY)
        payload = cert_service.validate_token(token, context)
        assert payload.subject == "relay-01"

    def test_pattern_token_allows_any_participant_type(self, cert_service, root_ca_key):
        """Test pattern token can be used for any participant type."""
        policy = create_sample_policy()
        token = create_jwt_token(root_ca_key, "*-01", "pattern", policy)

        # Pattern token allows client
        ctx1 = EnrollmentContext(name="site-01", participant_type=ParticipantType.CLIENT)
        payload1 = cert_service.validate_token(token, ctx1)
        assert payload1.subject == "*-01"

        # Pattern token allows relay
        ctx2 = EnrollmentContext(name="relay-01", participant_type=ParticipantType.RELAY)
        payload2 = cert_service.validate_token(token, ctx2)
        assert payload2.subject == "*-01"

    def test_cross_type_enrollment_fails(self, cert_service, root_ca_key):
        """Test that using wrong token type for enrollment fails."""
        policy = create_sample_policy()

        # Client token
        client_token = create_jwt_token(root_ca_key, "site-01", ParticipantType.CLIENT, policy)

        # Cannot use for admin
        with pytest.raises(ValueError, match="does not match enrollment type"):
            cert_service.validate_token(
                client_token, EnrollmentContext(name="site-01", participant_type=ParticipantType.ADMIN)
            )

        # Cannot use for relay
        with pytest.raises(ValueError, match="does not match enrollment type"):
            cert_service.validate_token(
                client_token, EnrollmentContext(name="site-01", participant_type=ParticipantType.RELAY)
            )


class TestEnrollmentContextOrg:
    """Tests for organization handling in enrollment context."""

    def test_context_with_org(self, cert_service, root_ca_key):
        """Test enrollment context with organization."""
        policy = create_sample_policy()
        token = create_jwt_token(root_ca_key, "site-01", ParticipantType.CLIENT, policy)

        context = EnrollmentContext(name="site-01", participant_type=ParticipantType.CLIENT, org="Acme Corp")
        payload = cert_service.validate_token(token, context)
        assert payload.subject == "site-01"

    def test_context_without_org(self, cert_service, root_ca_key):
        """Test enrollment context without organization."""
        policy = create_sample_policy()
        token = create_jwt_token(root_ca_key, "site-01", ParticipantType.CLIENT, policy)

        context = EnrollmentContext(name="site-01", participant_type=ParticipantType.CLIENT)
        payload = cert_service.validate_token(token, context)
        assert payload.subject == "site-01"

    def test_admin_context_with_org_and_role(self, cert_service, root_ca_key):
        """Test admin enrollment context with org and role."""
        policy = create_sample_policy()
        token = create_jwt_token(root_ca_key, "user@example.com", ParticipantType.ADMIN, policy)

        context = EnrollmentContext(
            name="user@example.com",
            participant_type=ParticipantType.ADMIN,
            org="Research Institute",
            role=AdminRole.LEAD,
        )
        payload = cert_service.validate_token(token, context)
        assert payload.subject == "user@example.com"
