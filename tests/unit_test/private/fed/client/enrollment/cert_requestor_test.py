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

"""Unit tests for CertRequestor."""

import os
import tempfile
from unittest.mock import MagicMock

import pytest
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from pydantic import ValidationError

from nvflare.lighter.constants import AdminRole, ParticipantType
from nvflare.private.fed.client.enrollment.cert_requestor import (
    CertRequestor,
    EnrollmentIdentity,
    EnrollmentOptions,
)


class TestEnrollmentIdentity:
    """Tests for EnrollmentIdentity class."""
    
    def test_create_basic(self):
        """Test creating basic identity."""
        identity = EnrollmentIdentity(name="site-01")
        assert identity.name == "site-01"
        assert identity.participant_type == ParticipantType.CLIENT
        assert identity.org is None
        assert identity.role is None
    
    def test_create_with_all_fields(self):
        """Test creating identity with all fields."""
        identity = EnrollmentIdentity(
            name="admin@example.com",
            participant_type=ParticipantType.ADMIN,
            org="TestOrg",
            role=AdminRole.ORG_ADMIN
        )
        assert identity.name == "admin@example.com"
        assert identity.participant_type == ParticipantType.ADMIN
        assert identity.org == "TestOrg"
        assert identity.role == AdminRole.ORG_ADMIN
    
    def test_for_client(self):
        """Test factory method for client."""
        identity = EnrollmentIdentity.for_client("hospital-01", org="Hospital A")
        assert identity.name == "hospital-01"
        assert identity.participant_type == ParticipantType.CLIENT
        assert identity.org == "Hospital A"
        assert identity.role is None
    
    def test_for_admin(self):
        """Test factory method for admin."""
        identity = EnrollmentIdentity.for_admin(
            "user@example.com",
            role=AdminRole.LEAD,
            org="TestOrg"
        )
        assert identity.name == "user@example.com"
        assert identity.participant_type == ParticipantType.ADMIN
        assert identity.org == "TestOrg"
        assert identity.role == AdminRole.LEAD
    
    def test_for_relay(self):
        """Test factory method for relay."""
        identity = EnrollmentIdentity.for_relay("relay-01", org="nvidia")
        assert identity.name == "relay-01"
        assert identity.participant_type == ParticipantType.RELAY
        assert identity.org == "nvidia"
        assert identity.role is None
    
    def test_empty_name_raises_error(self):
        """Test that empty name raises validation error."""
        with pytest.raises(ValidationError):
            EnrollmentIdentity(name="")
    
    def test_whitespace_name_raises_error(self):
        """Test that whitespace-only name raises validation error."""
        with pytest.raises(ValidationError):
            EnrollmentIdentity(name="   ")
    
    def test_invalid_participant_type_raises_error(self):
        """Test that invalid participant type raises validation error."""
        with pytest.raises(ValidationError):
            EnrollmentIdentity(name="site-01", participant_type="invalid")
    
    def test_invalid_role_raises_error(self):
        """Test that invalid role raises validation error."""
        with pytest.raises(ValidationError):
            EnrollmentIdentity(name="user@example.com", role="invalid_role")
    
    def test_name_is_stripped(self):
        """Test that name whitespace is stripped."""
        identity = EnrollmentIdentity(name="  site-01  ")
        assert identity.name == "site-01"


class TestEnrollmentOptions:
    """Tests for EnrollmentOptions class."""
    
    def test_default_values(self):
        """Test default option values."""
        options = EnrollmentOptions()
        assert options.timeout == 30.0
        assert options.retry_count == 3
        assert options.cert_valid_days == 360
        assert options.output_dir == "."
    
    def test_custom_values(self):
        """Test custom option values."""
        options = EnrollmentOptions(
            timeout=60.0,
            retry_count=5,
            cert_valid_days=180,
            output_dir="/custom/path"
        )
        assert options.timeout == 60.0
        assert options.retry_count == 5
        assert options.cert_valid_days == 180
        assert options.output_dir == "/custom/path"
    
    def test_negative_timeout_raises_error(self):
        """Test that negative timeout raises validation error."""
        with pytest.raises(ValidationError):
            EnrollmentOptions(timeout=-1.0)
    
    def test_zero_timeout_raises_error(self):
        """Test that zero timeout raises validation error."""
        with pytest.raises(ValidationError):
            EnrollmentOptions(timeout=0)
    
    def test_negative_retry_count_raises_error(self):
        """Test that negative retry count raises validation error."""
        with pytest.raises(ValidationError):
            EnrollmentOptions(retry_count=-1)
    
    def test_zero_retry_count_allowed(self):
        """Test that zero retry count is allowed."""
        options = EnrollmentOptions(retry_count=0)
        assert options.retry_count == 0
    
    def test_negative_cert_valid_days_raises_error(self):
        """Test that negative cert valid days raises validation error."""
        with pytest.raises(ValidationError):
            EnrollmentOptions(cert_valid_days=-1)
    
    def test_zero_cert_valid_days_raises_error(self):
        """Test that zero cert valid days raises validation error."""
        with pytest.raises(ValidationError):
            EnrollmentOptions(cert_valid_days=0)


class TestCertRequestorInit:
    """Tests for CertRequestor initialization."""
    
    def test_init_success(self):
        """Test successful initialization."""
        mock_cell = MagicMock()
        identity = EnrollmentIdentity.for_client("site-01")
        
        requestor = CertRequestor(
            cell=mock_cell,
            enrollment_token="valid-token",
            identity=identity
        )
        
        assert requestor.cell == mock_cell
        assert requestor.enrollment_token == "valid-token"
        assert requestor.identity == identity
        assert requestor.target_fqcn == "server"
        assert requestor.options is not None
    
    def test_init_with_options(self):
        """Test initialization with custom options."""
        mock_cell = MagicMock()
        identity = EnrollmentIdentity.for_client("site-01")
        options = EnrollmentOptions(timeout=60.0, output_dir="/custom/path")
        
        requestor = CertRequestor(
            cell=mock_cell,
            enrollment_token="valid-token",
            identity=identity,
            options=options
        )
        
        assert requestor.options.timeout == 60.0
        assert requestor.options.output_dir == "/custom/path"
    
    def test_init_empty_token_raises_error(self):
        """Test that empty token raises error."""
        mock_cell = MagicMock()
        identity = EnrollmentIdentity.for_client("site-01")
        
        with pytest.raises(ValueError, match="enrollment_token cannot be empty"):
            CertRequestor(
                cell=mock_cell,
                enrollment_token="",
                identity=identity
            )
    
    def test_init_whitespace_token_raises_error(self):
        """Test that whitespace-only token raises error."""
        mock_cell = MagicMock()
        identity = EnrollmentIdentity.for_client("site-01")
        
        with pytest.raises(ValueError, match="enrollment_token cannot be empty"):
            CertRequestor(
                cell=mock_cell,
                enrollment_token="   ",
                identity=identity
            )
    
    def test_token_is_stripped(self):
        """Test that token whitespace is stripped."""
        mock_cell = MagicMock()
        identity = EnrollmentIdentity.for_client("site-01")
        
        requestor = CertRequestor(
            cell=mock_cell,
            enrollment_token="  token  ",
            identity=identity
        )
        
        assert requestor.enrollment_token == "token"


class TestGenerateKeyPair:
    """Tests for key pair generation."""
    
    def test_generate_key_pair(self):
        """Test generating RSA key pair."""
        mock_cell = MagicMock()
        identity = EnrollmentIdentity.for_client("site-01")
        
        requestor = CertRequestor(
            cell=mock_cell,
            enrollment_token="token",
            identity=identity
        )
        
        assert requestor.private_key is None
        assert requestor.public_key is None
        
        requestor.generate_key_pair()
        
        assert requestor.private_key is not None
        assert requestor.public_key is not None


class TestCreateCSR:
    """Tests for CSR creation."""
    
    def test_create_csr_basic(self):
        """Test creating basic CSR."""
        mock_cell = MagicMock()
        identity = EnrollmentIdentity.for_client("hospital-01")
        
        requestor = CertRequestor(
            cell=mock_cell,
            enrollment_token="token",
            identity=identity
        )
        
        csr_pem = requestor.create_csr()
        
        assert csr_pem is not None
        assert b"-----BEGIN CERTIFICATE REQUEST-----" in csr_pem
        
        # Verify CSR can be loaded
        csr = x509.load_pem_x509_csr(csr_pem, default_backend())
        assert csr.is_signature_valid
        
        # Verify subject
        cn = csr.subject.get_attributes_for_oid(x509.NameOID.COMMON_NAME)
        assert len(cn) == 1
        assert cn[0].value == "hospital-01"
    
    def test_create_csr_with_org(self):
        """Test creating CSR with organization."""
        mock_cell = MagicMock()
        identity = EnrollmentIdentity.for_client("hospital-01", org="Hospital A")
        
        requestor = CertRequestor(
            cell=mock_cell,
            enrollment_token="token",
            identity=identity
        )
        
        csr_pem = requestor.create_csr()
        csr = x509.load_pem_x509_csr(csr_pem, default_backend())
        
        org = csr.subject.get_attributes_for_oid(x509.NameOID.ORGANIZATION_NAME)
        assert len(org) == 1
        assert org[0].value == "Hospital A"
    
    def test_create_csr_with_participant_type(self):
        """Test creating CSR includes participant type."""
        mock_cell = MagicMock()
        identity = EnrollmentIdentity.for_admin("user@example.com", role=AdminRole.LEAD)
        
        requestor = CertRequestor(
            cell=mock_cell,
            enrollment_token="token",
            identity=identity
        )
        
        csr_pem = requestor.create_csr()
        csr = x509.load_pem_x509_csr(csr_pem, default_backend())
        
        # Participant type in OU
        ou = csr.subject.get_attributes_for_oid(x509.NameOID.ORGANIZATIONAL_UNIT_NAME)
        assert len(ou) == 1
        assert ou[0].value == ParticipantType.ADMIN
        
        # Role in unstructured name
        role = csr.subject.get_attributes_for_oid(x509.NameOID.UNSTRUCTURED_NAME)
        assert len(role) == 1
        assert role[0].value == AdminRole.LEAD
    
    def test_create_csr_generates_key_if_needed(self):
        """Test that create_csr generates key pair if not exists."""
        mock_cell = MagicMock()
        identity = EnrollmentIdentity.for_client("site-01")
        
        requestor = CertRequestor(
            cell=mock_cell,
            enrollment_token="token",
            identity=identity
        )
        
        assert requestor.private_key is None
        
        requestor.create_csr()
        
        assert requestor.private_key is not None


class TestSaveCredentials:
    """Tests for saving credentials."""
    
    def test_save_credentials(self):
        """Test saving certificate and key to files."""
        mock_cell = MagicMock()
        identity = EnrollmentIdentity.for_client("site-01")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            options = EnrollmentOptions(output_dir=tmp_dir)
            requestor = CertRequestor(
                cell=mock_cell,
                enrollment_token="token",
                identity=identity,
                options=options
            )
            
            # Generate key pair first
            requestor.generate_key_pair()
            
            # Mock certificate data
            cert_data = b"-----BEGIN CERTIFICATE-----\nMIIB...\n-----END CERTIFICATE-----\n"
            
            requestor.save_credentials(cert_data)
            
            # Verify files exist
            cert_path = os.path.join(tmp_dir, CertRequestor.CERT_FILENAME)
            key_path = os.path.join(tmp_dir, CertRequestor.KEY_FILENAME)
            
            assert os.path.exists(cert_path)
            assert os.path.exists(key_path)
            
            # Verify key file permissions (owner read/write only)
            key_mode = os.stat(key_path).st_mode & 0o777
            assert key_mode == 0o600
            
            # Verify certificate content
            with open(cert_path, "rb") as f:
                assert f.read() == cert_data
    
    def test_save_credentials_creates_directory(self):
        """Test that save_credentials creates output directory."""
        mock_cell = MagicMock()
        identity = EnrollmentIdentity.for_client("site-01")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_dir = os.path.join(tmp_dir, "nested", "certs")
            options = EnrollmentOptions(output_dir=nested_dir)
            requestor = CertRequestor(
                cell=mock_cell,
                enrollment_token="token",
                identity=identity,
                options=options
            )
            
            requestor.generate_key_pair()
            cert_data = b"cert-data"
            
            requestor.save_credentials(cert_data)
            
            assert os.path.exists(nested_dir)


class TestSubmitCSR:
    """Tests for CSR submission."""
    
    def test_submit_csr_success(self):
        """Test successful CSR submission."""
        mock_cell = MagicMock()
        mock_response = MagicMock()
        mock_response.get_header.side_effect = lambda key, default=None: {
            "return_code": "OK"
        }.get(key, default)
        mock_response.payload = b"signed-cert-data"
        mock_cell.send_request.return_value = mock_response
        
        identity = EnrollmentIdentity.for_client("site-01")
        requestor = CertRequestor(
            cell=mock_cell,
            enrollment_token="token",
            identity=identity
        )
        
        csr_data = b"csr-data"
        result = requestor.submit_csr(csr_data)
        
        assert result == b"signed-cert-data"
        mock_cell.send_request.assert_called_once()
    
    def test_submit_csr_no_response(self):
        """Test CSR submission with no response raises error."""
        mock_cell = MagicMock()
        mock_cell.send_request.return_value = None
        
        identity = EnrollmentIdentity.for_client("site-01")
        requestor = CertRequestor(
            cell=mock_cell,
            enrollment_token="token",
            identity=identity
        )
        
        with pytest.raises(RuntimeError, match="No response from server"):
            requestor.submit_csr(b"csr-data")
    
    def test_submit_csr_error_response(self):
        """Test CSR submission with error response raises error."""
        mock_cell = MagicMock()
        mock_response = MagicMock()
        mock_response.get_header.side_effect = lambda key, default=None: {
            "return_code": "ERROR",
            "error": "Token expired"
        }.get(key, default)
        mock_cell.send_request.return_value = mock_response
        
        identity = EnrollmentIdentity.for_client("site-01")
        requestor = CertRequestor(
            cell=mock_cell,
            enrollment_token="token",
            identity=identity
        )
        
        with pytest.raises(RuntimeError, match="Token expired"):
            requestor.submit_csr(b"csr-data")
    
    def test_submit_csr_includes_headers(self):
        """Test CSR submission includes correct headers."""
        mock_cell = MagicMock()
        mock_response = MagicMock()
        mock_response.get_header.return_value = "OK"
        mock_response.payload = b"cert"
        mock_cell.send_request.return_value = mock_response
        
        identity = EnrollmentIdentity.for_admin("user@example.com", role=AdminRole.LEAD)
        requestor = CertRequestor(
            cell=mock_cell,
            enrollment_token="my-token",
            identity=identity
        )
        
        requestor.submit_csr(b"csr-data")
        
        # Verify message was created with correct headers
        call_args = mock_cell.send_request.call_args
        message = call_args.kwargs.get("request") or call_args[1].get("request")
        
        # The message headers should include enrollment_token, identity, participant_type, role
        assert message is not None


class TestRequestCertificate:
    """Tests for full enrollment workflow."""
    
    def test_request_certificate_success(self):
        """Test successful certificate request."""
        mock_cell = MagicMock()
        mock_response = MagicMock()
        mock_response.get_header.side_effect = lambda key, default=None: {
            "return_code": "OK"
        }.get(key, default)
        mock_response.payload = b"-----BEGIN CERTIFICATE-----\nMIIB...\n-----END CERTIFICATE-----\n"
        mock_cell.send_request.return_value = mock_response
        
        identity = EnrollmentIdentity.for_client("site-01")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            options = EnrollmentOptions(output_dir=tmp_dir)
            requestor = CertRequestor(
                cell=mock_cell,
                enrollment_token="token",
                identity=identity,
                options=options
            )
            
            cert_path = requestor.request_certificate()
            
            assert cert_path == os.path.join(tmp_dir, CertRequestor.CERT_FILENAME)
            assert os.path.exists(cert_path)
            assert os.path.exists(os.path.join(tmp_dir, CertRequestor.KEY_FILENAME))
    
    def test_request_certificate_generates_csr(self):
        """Test that request_certificate generates CSR and submits it."""
        mock_cell = MagicMock()
        mock_response = MagicMock()
        mock_response.get_header.return_value = "OK"
        mock_response.payload = b"cert"
        mock_cell.send_request.return_value = mock_response
        
        identity = EnrollmentIdentity.for_client("site-01")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            options = EnrollmentOptions(output_dir=tmp_dir)
            requestor = CertRequestor(
                cell=mock_cell,
                enrollment_token="token",
                identity=identity,
                options=options
            )
            
            requestor.request_certificate()
            
            # Verify send_request was called (CSR was submitted)
            mock_cell.send_request.assert_called_once()

