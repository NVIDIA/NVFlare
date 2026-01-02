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

"""Unit tests for FederatedServer enrollment integration."""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.message import Message
from nvflare.private.defs import CellChannel, CellChannelTopic


def generate_test_ca(tmp_dir: str):
    """Generate a test root CA certificate and key."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "TestRootCA"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "TestOrg"),
    ])
    
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
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))
    
    return cert_path, key_path


@pytest.fixture
def temp_ca_dir():
    """Create a temporary directory with CA files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cert_path, key_path = generate_test_ca(tmp_dir)
        yield tmp_dir, cert_path, key_path


class TestInitCertService:
    """Tests for FederatedServer._init_cert_service()."""
    
    def test_init_cert_service_success(self, temp_ca_dir):
        """Test successful CertService initialization."""
        tmp_dir, cert_path, key_path = temp_ca_dir
        
        # Create a mock FederatedServer
        mock_server = MagicMock()
        mock_server.logger = MagicMock()
        mock_server.cert_service = None
        
        # Import and call the method
        from nvflare.private.fed.server.fed_server import FederatedServer
        
        grpc_args = {"ssl_root_cert": cert_path}
        
        # Call the method directly on mock
        FederatedServer._init_cert_service(mock_server, grpc_args)
        
        # Verify CertService was created
        assert mock_server.cert_service is not None
        mock_server.logger.info.assert_called()
    
    def test_init_cert_service_no_root_cert(self):
        """Test CertService disabled when no ssl_root_cert configured."""
        mock_server = MagicMock()
        mock_server.logger = MagicMock()
        mock_server.cert_service = None
        
        from nvflare.private.fed.server.fed_server import FederatedServer
        
        grpc_args = {}  # No ssl_root_cert
        
        FederatedServer._init_cert_service(mock_server, grpc_args)
        
        # Verify CertService was not created
        assert mock_server.cert_service is None
        mock_server.logger.info.assert_called_with(
            "CSR enrollment disabled: ssl_root_cert not configured"
        )
    
    def test_init_cert_service_no_root_key(self, temp_ca_dir):
        """Test CertService disabled when rootCA.key is missing."""
        tmp_dir, cert_path, key_path = temp_ca_dir
        
        # Remove the key file
        os.remove(key_path)
        
        mock_server = MagicMock()
        mock_server.logger = MagicMock()
        mock_server.cert_service = None
        
        from nvflare.private.fed.server.fed_server import FederatedServer
        
        grpc_args = {"ssl_root_cert": cert_path}
        
        FederatedServer._init_cert_service(mock_server, grpc_args)
        
        # Verify CertService was not created
        assert mock_server.cert_service is None
        # Check that appropriate message was logged
        assert any(
            "rootCA.key not found" in str(call)
            for call in mock_server.logger.info.call_args_list
        )
    
    def test_init_cert_service_exception_handling(self, temp_ca_dir):
        """Test graceful handling of initialization errors."""
        tmp_dir, cert_path, key_path = temp_ca_dir
        
        # Corrupt the certificate file
        with open(cert_path, "w") as f:
            f.write("not a valid certificate")
        
        mock_server = MagicMock()
        mock_server.logger = MagicMock()
        mock_server.cert_service = None
        
        from nvflare.private.fed.server.fed_server import FederatedServer
        
        grpc_args = {"ssl_root_cert": cert_path}
        
        # Should not raise, but log warning
        FederatedServer._init_cert_service(mock_server, grpc_args)
        
        assert mock_server.cert_service is None
        mock_server.logger.warning.assert_called()


class TestValidateAuthHeadersBypass:
    """Tests for authentication bypass for CSR enrollment."""
    
    def test_csr_enrollment_bypasses_auth(self):
        """Test that CSR_ENROLLMENT requests bypass authentication."""
        mock_server = MagicMock()
        mock_server.logger = MagicMock()
        
        from nvflare.private.fed.server.fed_server import FederatedServer
        
        # Create a message with CSR_ENROLLMENT topic
        message = Message()
        message.add_headers({
            MessageHeaderKey.TOPIC: CellChannelTopic.CSR_ENROLLMENT,
            MessageHeaderKey.CHANNEL: CellChannel.SERVER_MAIN,
        })
        
        # Call _validate_auth_headers
        result = FederatedServer._validate_auth_headers(mock_server, message)
        
        # Should return None (bypassed)
        assert result is None
        mock_server.logger.debug.assert_called()
    
    def test_other_requests_require_auth(self):
        """Test that non-CSR requests still require authentication."""
        mock_server = MagicMock()
        mock_server.logger = MagicMock()
        mock_server._get_id_asserter = MagicMock(return_value=None)
        
        from nvflare.private.fed.server.fed_server import FederatedServer
        
        # Create a message with a different topic
        message = Message()
        message.add_headers({
            MessageHeaderKey.TOPIC: CellChannelTopic.Register,
            MessageHeaderKey.CHANNEL: CellChannel.SERVER_MAIN,
        })
        
        # Call _validate_auth_headers
        result = FederatedServer._validate_auth_headers(mock_server, message)
        
        # Should not bypass (id_asserter returns None, so returns None anyway)
        # but the important thing is it didn't bypass due to topic
        mock_server._get_id_asserter.assert_called()
    
    def test_csr_enrollment_wrong_channel_not_bypassed(self):
        """Test that CSR_ENROLLMENT on wrong channel is not bypassed."""
        mock_server = MagicMock()
        mock_server.logger = MagicMock()
        mock_server._get_id_asserter = MagicMock(return_value=None)
        
        from nvflare.private.fed.server.fed_server import FederatedServer
        
        # Create a message with CSR_ENROLLMENT but wrong channel
        message = Message()
        message.add_headers({
            MessageHeaderKey.TOPIC: CellChannelTopic.CSR_ENROLLMENT,
            MessageHeaderKey.CHANNEL: CellChannel.CLIENT_MAIN,  # Wrong channel
        })
        
        # Call _validate_auth_headers
        result = FederatedServer._validate_auth_headers(mock_server, message)
        
        # Should not bypass - check that id_asserter was called
        mock_server._get_id_asserter.assert_called()


class TestRegisterCellnetCbs:
    """Tests for CSR handler registration."""
    
    def test_csr_handler_registered_when_cert_service_available(self):
        """Test CSR handler is registered when CertService is available."""
        mock_server = MagicMock()
        mock_server.cell = MagicMock()
        mock_server.cert_service = MagicMock()
        mock_server.max_reg_duration = 60.0
        mock_server.logger = MagicMock()
        
        from nvflare.private.fed.server.fed_server import FederatedServer
        from nvflare.fuel.utils.config_service import ConfigService
        
        # Mock ConfigService
        with patch.object(ConfigService, 'get_float_var', return_value=60.0):
            with patch('threading.Thread'):
                FederatedServer._register_cellnet_cbs(mock_server)
        
        # Verify CSR handler was registered
        csr_registration_calls = [
            call for call in mock_server.cell.register_request_cb.call_args_list
            if call.kwargs.get('topic') == CellChannelTopic.CSR_ENROLLMENT
        ]
        assert len(csr_registration_calls) == 1
    
    def test_csr_handler_not_registered_when_cert_service_none(self):
        """Test CSR handler is NOT registered when CertService is None."""
        mock_server = MagicMock()
        mock_server.cell = MagicMock()
        mock_server.cert_service = None  # No CertService
        mock_server.max_reg_duration = 60.0
        mock_server.logger = MagicMock()
        
        from nvflare.private.fed.server.fed_server import FederatedServer
        from nvflare.fuel.utils.config_service import ConfigService
        
        with patch.object(ConfigService, 'get_float_var', return_value=60.0):
            with patch('threading.Thread'):
                FederatedServer._register_cellnet_cbs(mock_server)
        
        # Verify CSR handler was NOT registered
        csr_registration_calls = [
            call for call in mock_server.cell.register_request_cb.call_args_list
            if call.kwargs.get('topic') == CellChannelTopic.CSR_ENROLLMENT
        ]
        assert len(csr_registration_calls) == 0

