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

"""Unit tests for TokenService."""

import os
import tempfile
from datetime import datetime, timedelta, timezone

import jwt
import pytest
import yaml
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from nvflare.lighter.constants import AdminRole, ParticipantType
from nvflare.tool.enrollment.token_service import TokenService


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

    return tmp_dir


def create_sample_policy():
    """Create a sample policy dictionary for testing."""
    return {
        "metadata": {"project": "test-project", "description": "Test policy", "version": "1.0"},
        "token": {"validity": "7d", "max_uses": 1},
        "site": {"name_pattern": "*"},
        "user": {"allowed_roles": ["project_admin", "org_admin", "lead", "member"], "default_role": "member"},
        "approval": {"method": "policy", "rules": [{"name": "auto-approve-all", "match": {}, "action": "approve"}]},
    }


@pytest.fixture
def root_ca_dir():
    """Create a temporary directory with root CA."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        generate_root_ca(tmp_dir)
        yield tmp_dir


@pytest.fixture
def token_service(root_ca_dir):
    """Create a TokenService instance for testing."""
    return TokenService(root_ca_path=root_ca_dir)


@pytest.fixture
def policy_file(root_ca_dir):
    """Create a policy YAML file for testing."""
    policy = create_sample_policy()
    policy_path = os.path.join(root_ca_dir, "policy.yaml")
    with open(policy_path, "w") as f:
        yaml.dump(policy, f)
    return policy_path


class TestTokenServiceInit:
    """Tests for TokenService initialization."""

    def test_init_success(self, root_ca_dir):
        """Test successful initialization with valid CA."""
        service = TokenService(root_ca_path=root_ca_dir)
        assert service.issuer == "TestRootCA"
        assert service.signing_key is not None
        assert service.root_cert is not None

    def test_init_missing_ca(self):
        """Test initialization fails with missing CA."""
        with pytest.raises(FileNotFoundError):
            TokenService(root_ca_path="/nonexistent/path")


class TestGenerateToken:
    """Tests for token generation."""

    def test_generate_client_token(self, token_service):
        """Test generating a client enrollment token."""
        policy = create_sample_policy()
        token = token_service.generate_token(policy=policy, subject="hospital-01", subject_type=ParticipantType.CLIENT)

        # Verify token is a valid JWT
        assert token is not None
        assert len(token.split(".")) == 3  # JWT has 3 parts

        # Decode and verify claims
        payload = jwt.decode(token, options={"verify_signature": False})
        assert payload["sub"] == "hospital-01"
        assert payload["subject_type"] == ParticipantType.CLIENT
        assert payload["max_uses"] == 1
        assert "policy" in payload
        assert "jti" in payload
        assert "iat" in payload
        assert "exp" in payload

    def test_generate_admin_token(self, token_service):
        """Test generating an admin enrollment token."""
        policy = create_sample_policy()
        token = token_service.generate_token(
            policy=policy, subject="admin@example.com", subject_type=ParticipantType.ADMIN, roles=["org_admin"]
        )

        payload = jwt.decode(token, options={"verify_signature": False})
        assert payload["sub"] == "admin@example.com"
        assert payload["subject_type"] == ParticipantType.ADMIN
        assert payload["roles"] == ["org_admin"]

    def test_generate_relay_token(self, token_service):
        """Test generating a relay enrollment token."""
        policy = create_sample_policy()
        token = token_service.generate_token(policy=policy, subject="relay-01", subject_type=ParticipantType.RELAY)

        payload = jwt.decode(token, options={"verify_signature": False})
        assert payload["sub"] == "relay-01"
        assert payload["subject_type"] == ParticipantType.RELAY

    def test_generate_pattern_token(self, token_service):
        """Test generating a pattern token for wildcard matching."""
        policy = create_sample_policy()
        token = token_service.generate_token(
            policy=policy, subject="hospital-*", subject_type=TokenService.SUBJECT_TYPE_PATTERN
        )

        payload = jwt.decode(token, options={"verify_signature": False})
        assert payload["sub"] == "hospital-*"
        assert payload["subject_type"] == "pattern"

    def test_generate_token_with_source_ips(self, token_service):
        """Test generating a token with source IP restrictions."""
        policy = create_sample_policy()
        token = token_service.generate_token(
            policy=policy,
            subject="dc-server-01",
            subject_type=ParticipantType.CLIENT,
            source_ips=["10.0.0.0/8", "192.168.0.0/16"],
        )

        payload = jwt.decode(token, options={"verify_signature": False})
        assert payload["source_ips"] == ["10.0.0.0/8", "192.168.0.0/16"]

    def test_generate_token_with_metadata(self, token_service):
        """Test generating a token with custom metadata."""
        policy = create_sample_policy()
        token = token_service.generate_token(
            policy=policy,
            subject="site-01",
            subject_type=ParticipantType.CLIENT,
            metadata={"region": "us-west", "env": "production"},
        )

        payload = jwt.decode(token, options={"verify_signature": False})
        assert payload["metadata"] == {"region": "us-west", "env": "production"}

    def test_generate_token_custom_validity(self, token_service):
        """Test generating a token with custom validity."""
        policy = create_sample_policy()
        token = token_service.generate_token(
            policy=policy, subject="site-01", subject_type=ParticipantType.CLIENT, validity="30d"
        )

        payload = jwt.decode(token, options={"verify_signature": False})
        iat = datetime.fromtimestamp(payload["iat"], tz=timezone.utc)
        exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        delta = exp - iat
        assert delta.days == 30

    def test_generate_token_empty_subject(self, token_service):
        """Test that empty subject raises error."""
        policy = create_sample_policy()
        with pytest.raises(ValueError, match="subject is required"):
            token_service.generate_token(policy=policy, subject="", subject_type=ParticipantType.CLIENT)

    def test_generate_token_invalid_subject_type(self, token_service):
        """Test that invalid subject type raises error."""
        policy = create_sample_policy()
        with pytest.raises(ValueError, match="subject_type must be one of"):
            token_service.generate_token(policy=policy, subject="site-01", subject_type="invalid")


class TestConvenienceMethods:
    """Tests for convenience token generation methods."""

    def test_generate_client_token_method(self, token_service, policy_file):
        """Test generate_client_token convenience method."""
        token = token_service.generate_client_token(policy_file=policy_file, client_name="hospital-01")

        payload = jwt.decode(token, options={"verify_signature": False})
        assert payload["sub"] == "hospital-01"
        assert payload["subject_type"] == ParticipantType.CLIENT

    def test_generate_admin_token_method(self, token_service, policy_file):
        """Test generate_admin_token convenience method."""
        token = token_service.generate_admin_token(policy_file=policy_file, user_id="user@example.com", roles=["lead"])

        payload = jwt.decode(token, options={"verify_signature": False})
        assert payload["sub"] == "user@example.com"
        assert payload["subject_type"] == ParticipantType.ADMIN
        assert payload["roles"] == ["lead"]

    def test_generate_admin_token_default_role(self, token_service, policy_file):
        """Test generate_admin_token uses default LEAD role when no roles specified (has job submission permissions)."""
        token = token_service.generate_admin_token(policy_file=policy_file, user_id="user@example.com")

        payload = jwt.decode(token, options={"verify_signature": False})
        assert payload["sub"] == "user@example.com"
        assert payload["subject_type"] == ParticipantType.ADMIN
        assert payload["roles"] == [AdminRole.LEAD]

    def test_generate_relay_token_method(self, token_service, policy_file):
        """Test generate_relay_token convenience method."""
        token = token_service.generate_relay_token(policy_file=policy_file, relay_name="relay-01")

        payload = jwt.decode(token, options={"verify_signature": False})
        assert payload["sub"] == "relay-01"
        assert payload["subject_type"] == ParticipantType.RELAY

    def test_generate_site_token_method(self, token_service, policy_file):
        """Test generate_site_token convenience method (alias for client token)."""
        token = token_service.generate_site_token(policy_file=policy_file, site_name="hospital-01")

        payload = jwt.decode(token, options={"verify_signature": False})
        assert payload["sub"] == "hospital-01"
        assert payload["subject_type"] == ParticipantType.CLIENT


class TestBatchGeneration:
    """Tests for batch token generation."""

    def test_batch_generate_with_count(self, token_service, policy_file):
        """Test batch generation with count and prefix."""
        tokens = token_service.batch_generate_tokens(policy_file=policy_file, count=5, name_prefix="hospital")

        assert len(tokens) == 5
        assert tokens[0]["name"] == "hospital-001"
        assert tokens[4]["name"] == "hospital-005"

        # Verify each token is valid
        for t in tokens:
            payload = jwt.decode(t["token"], options={"verify_signature": False})
            assert payload["sub"] == t["name"]

    def test_batch_generate_with_names(self, token_service, policy_file):
        """Test batch generation with explicit names."""
        names = ["site-a", "site-b", "site-c"]
        tokens = token_service.batch_generate_tokens(policy_file=policy_file, names=names)

        assert len(tokens) == 3
        assert [t["name"] for t in tokens] == names

    def test_batch_generate_to_file(self, token_service, policy_file, root_ca_dir):
        """Test batch generation with file output."""
        output_file = os.path.join(root_ca_dir, "tokens.csv")
        tokens = token_service.batch_generate_tokens(
            policy_file=policy_file, count=3, name_prefix="site", output_file=output_file
        )

        assert os.path.exists(output_file)
        with open(output_file, "r") as f:
            content = f.read()
            assert "name,token" in content
            assert "site-001" in content


class TestParseDuration:
    """Tests for duration parsing."""

    def test_parse_days(self, token_service):
        """Test parsing days."""
        delta = token_service._parse_duration("7d")
        assert delta == timedelta(days=7)

    def test_parse_hours(self, token_service):
        """Test parsing hours."""
        delta = token_service._parse_duration("24h")
        assert delta == timedelta(hours=24)

    def test_parse_minutes(self, token_service):
        """Test parsing minutes."""
        delta = token_service._parse_duration("30m")
        assert delta == timedelta(minutes=30)

    def test_parse_seconds(self, token_service):
        """Test parsing seconds."""
        delta = token_service._parse_duration("60s")
        assert delta == timedelta(seconds=60)

    def test_parse_invalid_format(self, token_service):
        """Test parsing invalid format raises error."""
        with pytest.raises(ValueError, match="Invalid duration format"):
            token_service._parse_duration("invalid")


class TestGetTokenInfo:
    """Tests for token inspection."""

    def test_get_token_info(self, token_service):
        """Test getting token information."""
        policy = create_sample_policy()
        token = token_service.generate_token(policy=policy, subject="site-01", subject_type=ParticipantType.CLIENT)

        info = token_service.get_token_info(token)

        assert info["subject"] == "site-01"
        assert info["subject_type"] == ParticipantType.CLIENT
        assert info["issuer"] == "TestRootCA"
        assert info["max_uses"] == 1
        assert info["token_id"] is not None
        assert info["issued_at"] is not None
        assert info["expires_at"] is not None

    def test_get_token_info_invalid(self, token_service):
        """Test getting info from invalid token raises error."""
        with pytest.raises(ValueError, match="Failed to decode token"):
            token_service.get_token_info("invalid-token")


class TestGetPublicKey:
    """Tests for public key export."""

    def test_get_public_key_pem(self, token_service):
        """Test getting public key in PEM format."""
        pem = token_service.get_public_key_pem()

        assert pem is not None
        assert b"-----BEGIN PUBLIC KEY-----" in pem
        assert b"-----END PUBLIC KEY-----" in pem
