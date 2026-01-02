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

"""Backward compatibility tests for token enrollment feature.

These tests ensure that existing customers using traditional startup kit
provisioning (with pre-existing certificates) are not affected by the
new token-based enrollment feature.
"""

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

from nvflare.apis.fl_constant import SecureTrainConst
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.message import Message
from nvflare.private.defs import CellChannel, CellChannelTopic


def generate_test_cert(tmp_dir: str, name: str = "client"):
    """Generate a test certificate and key."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, name),
    ])
    
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
        .sign(private_key, hashes.SHA256(), default_backend())
    )
    
    cert_path = os.path.join(tmp_dir, f"{name}.crt")
    key_path = os.path.join(tmp_dir, f"{name}.key")
    
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    
    with open(key_path, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))
    
    return cert_path, key_path


def generate_root_ca(tmp_dir: str, include_key: bool = True):
    """Generate a root CA certificate and optionally the key."""
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
    
    cert_path = os.path.join(tmp_dir, "rootCA.pem")
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    
    key_path = None
    if include_key:
        key_path = os.path.join(tmp_dir, "rootCA.key")
        with open(key_path, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))
    
    return cert_path, key_path


class TestServerBackwardCompatibility:
    """Tests ensuring server backward compatibility for existing customers."""
    
    def test_server_works_without_root_ca_key(self):
        """Test that server starts normally without rootCA.key (enrollment disabled)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create rootCA.pem but NOT rootCA.key (typical existing deployment)
            cert_path, _ = generate_root_ca(tmp_dir, include_key=False)
            
            mock_server = MagicMock()
            mock_server.logger = MagicMock()
            mock_server.cert_service = None
            
            from nvflare.private.fed.server.fed_server import FederatedServer
            
            grpc_args = {"ssl_root_cert": cert_path}
            
            # Should not raise any exception
            FederatedServer._init_cert_service(mock_server, grpc_args)
            
            # cert_service should remain None
            assert mock_server.cert_service is None
            
            # Should log info message about enrollment being disabled
            log_calls = [str(call) for call in mock_server.logger.info.call_args_list]
            assert any("rootCA.key not found" in call for call in log_calls)
    
    def test_normal_auth_works_when_enrollment_disabled(self):
        """Test that normal authentication still works when enrollment is disabled."""
        mock_server = MagicMock()
        mock_server.logger = MagicMock()
        mock_server.cert_service = None  # Enrollment disabled
        mock_server._get_id_asserter = MagicMock(return_value=None)
        
        from nvflare.private.fed.server.fed_server import FederatedServer
        
        # Create a normal registration request (not CSR_ENROLLMENT)
        message = Message()
        message.add_headers({
            MessageHeaderKey.TOPIC: CellChannelTopic.Register,
            MessageHeaderKey.CHANNEL: CellChannel.SERVER_MAIN,
        })
        
        # Should process normally through existing auth path
        result = FederatedServer._validate_auth_headers(mock_server, message)
        
        # _get_id_asserter should be called (normal auth flow)
        mock_server._get_id_asserter.assert_called()
    
    def test_csr_handler_not_registered_when_enrollment_disabled(self):
        """Test that CSR handler is NOT registered when enrollment is disabled."""
        mock_server = MagicMock()
        mock_server.cell = MagicMock()
        mock_server.cert_service = None  # Enrollment disabled
        mock_server.max_reg_duration = 60.0
        mock_server.logger = MagicMock()
        
        from nvflare.private.fed.server.fed_server import FederatedServer
        from nvflare.fuel.utils.config_service import ConfigService
        
        with patch.object(ConfigService, 'get_float_var', return_value=60.0):
            with patch('threading.Thread'):
                FederatedServer._register_cellnet_cbs(mock_server)
        
        # Verify no CSR_ENROLLMENT handler was registered
        for call in mock_server.cell.register_request_cb.call_args_list:
            if hasattr(call, 'kwargs') and call.kwargs.get('topic'):
                assert call.kwargs['topic'] != CellChannelTopic.CSR_ENROLLMENT
    
    def test_server_handles_unexpected_csr_request_gracefully(self):
        """Test that unexpected CSR requests don't crash the server."""
        mock_server = MagicMock()
        mock_server.logger = MagicMock()
        mock_server.cert_service = None  # No enrollment service
        
        from nvflare.private.fed.server.fed_server import FederatedServer
        
        # CSR enrollment request when enrollment is disabled
        message = Message()
        message.add_headers({
            MessageHeaderKey.TOPIC: CellChannelTopic.CSR_ENROLLMENT,
            MessageHeaderKey.CHANNEL: CellChannel.SERVER_MAIN,
        })
        
        # Auth bypass still works (returns None, not an error)
        result = FederatedServer._validate_auth_headers(mock_server, message)
        assert result is None
        
        # The request would be unhandled (no registered handler), 
        # but this is expected behavior - client would timeout


class TestClientBackwardCompatibility:
    """Tests ensuring client backward compatibility for existing customers."""
    
    def test_existing_certificates_not_affected(self):
        """Test that clients with existing certificates don't trigger enrollment."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create existing certificates (traditional startup kit)
            cert_path, key_path = generate_test_cert(tmp_dir, "client")
            root_ca_path, _ = generate_root_ca(tmp_dir)
            
            mock_client = MagicMock()
            mock_client.secure_train = True
            mock_client.client_args = {
                SecureTrainConst.SSL_CERT: cert_path,
                SecureTrainConst.SSL_ROOT_CERT: root_ca_path,
                SecureTrainConst.PRIVATE_KEY: key_path,
            }
            mock_client.logger = MagicMock()
            mock_client._perform_enrollment = MagicMock()
            
            from nvflare.private.fed.client.fed_client_base import FederatedClientBase
            
            result = FederatedClientBase._auto_enroll_if_needed(
                mock_client, "localhost:8002", "grpc"
            )
            
            # Should return False (no enrollment needed)
            assert result is False
            
            # _perform_enrollment should NOT be called
            mock_client._perform_enrollment.assert_not_called()
    
    def test_existing_certificates_take_priority_over_token(self):
        """Test that existing certificates are used even if token is set."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create existing certificates
            cert_path, key_path = generate_test_cert(tmp_dir, "client")
            root_ca_path, _ = generate_root_ca(tmp_dir)
            
            # Also create a token file (simulating accidental token presence)
            token_file = os.path.join(tmp_dir, "enrollment_token")
            with open(token_file, "w") as f:
                f.write("some-enrollment-token")
            
            mock_client = MagicMock()
            mock_client.secure_train = True
            mock_client.client_args = {
                SecureTrainConst.SSL_CERT: cert_path,
                SecureTrainConst.SSL_ROOT_CERT: root_ca_path,
                SecureTrainConst.PRIVATE_KEY: key_path,
            }
            mock_client.logger = MagicMock()
            mock_client._perform_enrollment = MagicMock()
            
            from nvflare.private.fed.client.fed_client_base import FederatedClientBase
            
            # Even with token file present, enrollment should NOT happen
            result = FederatedClientBase._auto_enroll_if_needed(
                mock_client, "localhost:8002", "grpc"
            )
            
            assert result is False
            mock_client._perform_enrollment.assert_not_called()
    
    def test_existing_certificates_take_priority_over_env_token(self):
        """Test that existing certificates are used even if env token is set."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create existing certificates
            cert_path, key_path = generate_test_cert(tmp_dir, "client")
            root_ca_path, _ = generate_root_ca(tmp_dir)
            
            mock_client = MagicMock()
            mock_client.secure_train = True
            mock_client.client_args = {
                SecureTrainConst.SSL_CERT: cert_path,
                SecureTrainConst.SSL_ROOT_CERT: root_ca_path,
                SecureTrainConst.PRIVATE_KEY: key_path,
            }
            mock_client.logger = MagicMock()
            mock_client._perform_enrollment = MagicMock()
            
            from nvflare.private.fed.client.fed_client_base import FederatedClientBase
            
            # Set env token (simulating accidental env var)
            with patch.dict(os.environ, {"NVFLARE_ENROLLMENT_TOKEN": "env-token"}):
                result = FederatedClientBase._auto_enroll_if_needed(
                    mock_client, "localhost:8002", "grpc"
                )
            
            # Should still use existing certs, not enroll
            assert result is False
            mock_client._perform_enrollment.assert_not_called()
    
    def test_no_cert_no_token_fails_gracefully(self):
        """Test that misconfigured client (no cert, no token) fails as before."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_ca_path, _ = generate_root_ca(tmp_dir)
            nonexistent_cert = os.path.join(tmp_dir, "nonexistent.crt")
            
            mock_client = MagicMock()
            mock_client.secure_train = True
            mock_client.client_args = {
                SecureTrainConst.SSL_CERT: nonexistent_cert,
                SecureTrainConst.SSL_ROOT_CERT: root_ca_path,
            }
            mock_client.logger = MagicMock()
            
            from nvflare.private.fed.client.fed_client_base import FederatedClientBase
            
            # No token set
            with patch.dict(os.environ, {}, clear=True):
                result = FederatedClientBase._auto_enroll_if_needed(
                    mock_client, "localhost:8002", "grpc"
                )
            
            # Should return False (no enrollment attempted)
            assert result is False
            
            # Debug log should indicate no token found
            mock_client.logger.debug.assert_called()
    
    def test_non_secure_train_not_affected(self):
        """Test that non-secure training mode is not affected."""
        mock_client = MagicMock()
        mock_client.secure_train = False  # Not secure mode
        mock_client.client_args = {}
        mock_client.logger = MagicMock()
        
        from nvflare.private.fed.client.fed_client_base import FederatedClientBase
        
        # Should return immediately without any checks
        result = FederatedClientBase._auto_enroll_if_needed(
            mock_client, "localhost:8002", "grpc"
        )
        
        assert result is False
        # Logger should not even be called
        mock_client.logger.debug.assert_not_called()


class TestEnrollmentIntegrationSafety:
    """Tests for safe integration with existing code paths."""
    
    def test_create_cell_continues_after_enrollment_skip(self):
        """Test that _create_cell continues normally when enrollment is skipped."""
        # This verifies the flow: enrollment check → skip → normal cell creation
        from nvflare.private.fed.client.fed_client_base import FederatedClientBase
        import inspect
        
        source = inspect.getsource(FederatedClientBase._create_cell)
        
        # Verify _auto_enroll_if_needed is called
        assert "_auto_enroll_if_needed" in source
        
        # Verify it's followed by normal cell creation logic
        lines = source.split('\n')
        found_enroll = False
        found_cell_creation = False
        
        for line in lines:
            if "_auto_enroll_if_needed" in line:
                found_enroll = True
            if found_enroll and "self.cell = Cell(" in line:
                found_cell_creation = True
                break
        
        assert found_enroll, "_auto_enroll_if_needed should be called in _create_cell"
        assert found_cell_creation, "Cell creation should follow enrollment check"
    
    def test_enrollment_error_does_not_silently_continue(self):
        """Test that enrollment errors are raised, not silently ignored."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_ca_path, _ = generate_root_ca(tmp_dir)
            
            mock_client = MagicMock()
            mock_client.secure_train = True
            mock_client.client_name = "test-client"
            mock_client.client_args = {
                SecureTrainConst.SSL_CERT: os.path.join(tmp_dir, "nonexistent.crt"),
                SecureTrainConst.SSL_ROOT_CERT: root_ca_path,
            }
            mock_client.logger = MagicMock()
            mock_client._perform_enrollment = MagicMock(
                side_effect=Exception("Enrollment failed")
            )
            
            from nvflare.private.fed.client.fed_client_base import FederatedClientBase
            
            with patch.dict(os.environ, {"NVFLARE_ENROLLMENT_TOKEN": "test-token"}):
                # Should raise RuntimeError, not silently continue
                with pytest.raises(RuntimeError, match="Auto-enrollment failed"):
                    FederatedClientBase._auto_enroll_if_needed(
                        mock_client, "localhost:8002", "grpc"
                    )

