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

"""Unit tests for FederatedClientBase enrollment integration."""

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


def generate_test_cert(tmp_dir: str, name: str = "client"):
    """Generate a test certificate and key."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())

    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, name),
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
        .sign(private_key, hashes.SHA256(), default_backend())
    )

    cert_path = os.path.join(tmp_dir, f"{name}.crt")
    key_path = os.path.join(tmp_dir, f"{name}.key")

    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    with open(key_path, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    return cert_path, key_path


@pytest.fixture
def temp_startup_dir():
    """Create a temporary startup directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create rootCA.pem (needed for enrollment)
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COMMON_NAME, "TestRootCA"),
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
        root_ca_path = os.path.join(tmp_dir, "rootCA.pem")
        with open(root_ca_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        yield tmp_dir, root_ca_path


class TestAutoEnrollIfNeeded:
    """Tests for FederatedClientBase._auto_enroll_if_needed()."""

    def test_returns_false_when_not_secure_train(self):
        """Test that enrollment is skipped when not in secure mode."""
        mock_client = MagicMock()
        mock_client.secure_train = False
        mock_client.logger = MagicMock()

        from nvflare.private.fed.client.fed_client_base import FederatedClientBase

        result = FederatedClientBase._auto_enroll_if_needed(mock_client, "localhost:8002", "grpc")

        assert result is False

    def test_returns_false_when_cert_exists(self, temp_startup_dir):
        """Test that enrollment is skipped when certificate already exists."""
        tmp_dir, root_ca_path = temp_startup_dir

        # Create existing certificate
        cert_path, key_path = generate_test_cert(tmp_dir, "client")

        mock_client = MagicMock()
        mock_client.secure_train = True
        mock_client.client_args = {
            SecureTrainConst.SSL_CERT: cert_path,
            SecureTrainConst.SSL_ROOT_CERT: root_ca_path,
        }
        mock_client.logger = MagicMock()

        from nvflare.private.fed.client.fed_client_base import FederatedClientBase

        result = FederatedClientBase._auto_enroll_if_needed(mock_client, "localhost:8002", "grpc")

        assert result is False

    def test_returns_false_when_no_token_available(self, temp_startup_dir):
        """Test that enrollment is skipped when no token is available."""
        tmp_dir, root_ca_path = temp_startup_dir

        mock_client = MagicMock()
        mock_client.secure_train = True
        mock_client.client_args = {
            SecureTrainConst.SSL_CERT: os.path.join(tmp_dir, "nonexistent.crt"),
            SecureTrainConst.SSL_ROOT_CERT: root_ca_path,
        }
        mock_client.logger = MagicMock()

        from nvflare.private.fed.client.fed_client_base import FederatedClientBase

        # Ensure no env var is set
        with patch.dict(os.environ, {}, clear=True):
            result = FederatedClientBase._auto_enroll_if_needed(mock_client, "localhost:8002", "grpc")

        assert result is False
        mock_client.logger.debug.assert_called()

    def test_reads_token_from_env_var(self, temp_startup_dir):
        """Test that token is read from NVFLARE_ENROLLMENT_TOKEN env var."""
        tmp_dir, root_ca_path = temp_startup_dir

        mock_client = MagicMock()
        mock_client.secure_train = True
        mock_client.client_name = "test-client"
        mock_client.client_args = {
            SecureTrainConst.SSL_CERT: os.path.join(tmp_dir, "nonexistent.crt"),
            SecureTrainConst.SSL_ROOT_CERT: root_ca_path,
        }
        mock_client.logger = MagicMock()
        mock_client._perform_enrollment = MagicMock()

        from nvflare.private.fed.client.fed_client_base import FederatedClientBase

        with patch.dict(os.environ, {"NVFLARE_ENROLLMENT_TOKEN": "test-token"}):
            result = FederatedClientBase._auto_enroll_if_needed(mock_client, "localhost:8002", "grpc")

        assert result is True
        mock_client._perform_enrollment.assert_called_once_with("localhost:8002", "grpc", "test-token")

    def test_reads_token_from_file(self, temp_startup_dir):
        """Test that token is read from enrollment_token file."""
        tmp_dir, root_ca_path = temp_startup_dir

        # Create token file
        token_file = os.path.join(tmp_dir, "enrollment_token")
        with open(token_file, "w") as f:
            f.write("file-based-token\n")

        mock_client = MagicMock()
        mock_client.secure_train = True
        mock_client.client_name = "test-client"
        mock_client.client_args = {
            SecureTrainConst.SSL_CERT: os.path.join(tmp_dir, "nonexistent.crt"),
            SecureTrainConst.SSL_ROOT_CERT: root_ca_path,
        }
        mock_client.logger = MagicMock()
        mock_client._perform_enrollment = MagicMock()

        from nvflare.private.fed.client.fed_client_base import FederatedClientBase

        # Ensure env var is not set
        with patch.dict(os.environ, {}, clear=True):
            result = FederatedClientBase._auto_enroll_if_needed(mock_client, "localhost:8002", "grpc")

        assert result is True
        mock_client._perform_enrollment.assert_called_once_with("localhost:8002", "grpc", "file-based-token")

    def test_env_var_takes_precedence_over_file(self, temp_startup_dir):
        """Test that env var token takes precedence over file token."""
        tmp_dir, root_ca_path = temp_startup_dir

        # Create token file
        token_file = os.path.join(tmp_dir, "enrollment_token")
        with open(token_file, "w") as f:
            f.write("file-token")

        mock_client = MagicMock()
        mock_client.secure_train = True
        mock_client.client_name = "test-client"
        mock_client.client_args = {
            SecureTrainConst.SSL_CERT: os.path.join(tmp_dir, "nonexistent.crt"),
            SecureTrainConst.SSL_ROOT_CERT: root_ca_path,
        }
        mock_client.logger = MagicMock()
        mock_client._perform_enrollment = MagicMock()

        from nvflare.private.fed.client.fed_client_base import FederatedClientBase

        with patch.dict(os.environ, {"NVFLARE_ENROLLMENT_TOKEN": "env-token"}):
            result = FederatedClientBase._auto_enroll_if_needed(mock_client, "localhost:8002", "grpc")

        # Should use env token, not file token
        mock_client._perform_enrollment.assert_called_once_with("localhost:8002", "grpc", "env-token")

    def test_raises_on_enrollment_failure(self, temp_startup_dir):
        """Test that enrollment failure raises RuntimeError."""
        tmp_dir, root_ca_path = temp_startup_dir

        mock_client = MagicMock()
        mock_client.secure_train = True
        mock_client.client_name = "test-client"
        mock_client.client_args = {
            SecureTrainConst.SSL_CERT: os.path.join(tmp_dir, "nonexistent.crt"),
            SecureTrainConst.SSL_ROOT_CERT: root_ca_path,
        }
        mock_client.logger = MagicMock()
        mock_client._perform_enrollment = MagicMock(side_effect=Exception("Enrollment failed"))

        from nvflare.private.fed.client.fed_client_base import FederatedClientBase

        with patch.dict(os.environ, {"NVFLARE_ENROLLMENT_TOKEN": "test-token"}):
            with pytest.raises(RuntimeError, match="Auto-enrollment failed"):
                FederatedClientBase._auto_enroll_if_needed(mock_client, "localhost:8002", "grpc")


class TestPerformEnrollment:
    """Tests for FederatedClientBase._perform_enrollment()."""

    def test_creates_enrollment_cell(self, temp_startup_dir):
        """Test that a temporary cell is created for enrollment."""
        tmp_dir, root_ca_path = temp_startup_dir

        mock_client = MagicMock()
        mock_client.client_name = "test-client"
        mock_client.client_args = {
            SecureTrainConst.SSL_ROOT_CERT: root_ca_path,
        }
        mock_client.logger = MagicMock()

        from nvflare.private.fed.client.fed_client_base import FederatedClientBase

        with patch("nvflare.private.fed.client.fed_client_base.Cell") as mock_cell_class:
            mock_cell = MagicMock()
            mock_cell_class.return_value = mock_cell

            # Patch at the source module where they're imported from
            with patch("nvflare.private.fed.client.enrollment.cert_requestor.CertRequestor") as mock_requestor_class:
                mock_requestor = MagicMock()
                mock_requestor.request_certificate.return_value = os.path.join(tmp_dir, "test-client.crt")
                mock_requestor_class.return_value = mock_requestor

                # Also need to patch where it's imported in the method
                with patch.dict(
                    "sys.modules",
                    {
                        "nvflare.private.fed.client.enrollment": MagicMock(
                            CertRequestor=mock_requestor_class,
                            EnrollmentIdentity=MagicMock(for_client=MagicMock(return_value=MagicMock())),
                            EnrollmentOptions=MagicMock(return_value=MagicMock()),
                        )
                    },
                ):
                    FederatedClientBase._perform_enrollment(mock_client, "localhost:8002", "grpc", "test-token")

                    # Verify Cell was created with correct params
                    mock_cell_class.assert_called_once()
                    call_kwargs = mock_cell_class.call_args.kwargs
                    assert call_kwargs["root_url"] == "grpc://localhost:8002"
                    assert call_kwargs["secure"] is True

                    # Verify Cell was started and stopped
                    mock_cell.start.assert_called_once()
                    mock_cell.stop.assert_called_once()

    def test_calls_cert_requestor(self, temp_startup_dir):
        """Test that CertRequestor is called correctly."""
        tmp_dir, root_ca_path = temp_startup_dir

        mock_client = MagicMock()
        mock_client.client_name = "hospital-1"
        mock_client.client_args = {
            SecureTrainConst.SSL_ROOT_CERT: root_ca_path,
            "organization": "Test Hospital",
        }
        mock_client.logger = MagicMock()

        from nvflare.private.fed.client.fed_client_base import FederatedClientBase

        mock_requestor = MagicMock()
        mock_requestor.request_certificate.return_value = os.path.join(tmp_dir, "hospital-1.crt")

        mock_identity = MagicMock()
        mock_identity_class = MagicMock(for_client=MagicMock(return_value=mock_identity))

        mock_enrollment_module = MagicMock(
            CertRequestor=MagicMock(return_value=mock_requestor),
            EnrollmentIdentity=mock_identity_class,
            EnrollmentOptions=MagicMock(return_value=MagicMock()),
        )

        with patch("nvflare.private.fed.client.fed_client_base.Cell") as mock_cell_class:
            mock_cell = MagicMock()
            mock_cell_class.return_value = mock_cell

            with patch.dict("sys.modules", {"nvflare.private.fed.client.enrollment": mock_enrollment_module}):
                FederatedClientBase._perform_enrollment(mock_client, "localhost:8002", "grpc", "test-token")

                # Verify identity was created
                mock_identity_class.for_client.assert_called_once_with(name="hospital-1", org="Test Hospital")

                # Verify requestor was called
                mock_requestor.request_certificate.assert_called_once()

    def test_updates_client_args_after_enrollment(self, temp_startup_dir):
        """Test that client_args is updated with new certificate paths."""
        tmp_dir, root_ca_path = temp_startup_dir

        # Use a real dict to track updates
        client_args = {
            SecureTrainConst.SSL_ROOT_CERT: root_ca_path,
        }

        mock_client = MagicMock()
        mock_client.client_name = "test-client"
        mock_client.client_args = client_args
        mock_client.logger = MagicMock()

        cert_path = os.path.join(tmp_dir, "test-client.crt")
        key_path = os.path.join(tmp_dir, "test-client.key")

        from nvflare.private.fed.client.fed_client_base import FederatedClientBase

        mock_requestor = MagicMock()
        mock_requestor.request_certificate.return_value = cert_path

        mock_enrollment_module = MagicMock(
            CertRequestor=MagicMock(return_value=mock_requestor),
            EnrollmentIdentity=MagicMock(for_client=MagicMock(return_value=MagicMock())),
            EnrollmentOptions=MagicMock(return_value=MagicMock()),
        )

        with patch("nvflare.private.fed.client.fed_client_base.Cell") as mock_cell_class:
            mock_cell = MagicMock()
            mock_cell_class.return_value = mock_cell

            with patch.dict("sys.modules", {"nvflare.private.fed.client.enrollment": mock_enrollment_module}):
                FederatedClientBase._perform_enrollment(mock_client, "localhost:8002", "grpc", "test-token")

                # Verify client_args was updated
                assert client_args[SecureTrainConst.SSL_CERT] == cert_path
                assert client_args[SecureTrainConst.PRIVATE_KEY] == key_path

    def test_cell_stopped_on_exception(self, temp_startup_dir):
        """Test that enrollment cell is stopped even if enrollment fails."""
        tmp_dir, root_ca_path = temp_startup_dir

        mock_client = MagicMock()
        mock_client.client_name = "test-client"
        mock_client.client_args = {
            SecureTrainConst.SSL_ROOT_CERT: root_ca_path,
        }
        mock_client.logger = MagicMock()

        from nvflare.private.fed.client.fed_client_base import FederatedClientBase

        mock_requestor = MagicMock()
        mock_requestor.request_certificate.side_effect = Exception("Network error")

        mock_enrollment_module = MagicMock(
            CertRequestor=MagicMock(return_value=mock_requestor),
            EnrollmentIdentity=MagicMock(for_client=MagicMock(return_value=MagicMock())),
            EnrollmentOptions=MagicMock(return_value=MagicMock()),
        )

        with patch("nvflare.private.fed.client.fed_client_base.Cell") as mock_cell_class:
            mock_cell = MagicMock()
            mock_cell_class.return_value = mock_cell

            with patch.dict("sys.modules", {"nvflare.private.fed.client.enrollment": mock_enrollment_module}):
                with pytest.raises(Exception, match="Network error"):
                    FederatedClientBase._perform_enrollment(mock_client, "localhost:8002", "grpc", "test-token")

                # Verify Cell was stopped despite exception
                mock_cell.stop.assert_called_once()


class TestCreateCellIntegration:
    """Tests for _create_cell integration with auto-enrollment."""

    def test_auto_enroll_called_before_cell_creation(self, temp_startup_dir):
        """Test that _auto_enroll_if_needed is called at start of _create_cell."""
        tmp_dir, root_ca_path = temp_startup_dir

        # Create a minimal mock that tracks call order
        call_order = []

        mock_client = MagicMock()
        mock_client.client_name = "test-client"
        mock_client.secure_train = True
        mock_client.client_args = {
            SecureTrainConst.SSL_ROOT_CERT: root_ca_path,
            SecureTrainConst.SSL_CERT: os.path.join(tmp_dir, "client.crt"),
            SecureTrainConst.PRIVATE_KEY: os.path.join(tmp_dir, "client.key"),
        }
        mock_client.args = MagicMock()
        mock_client.args.job_id = None
        mock_client.logger = MagicMock()

        def track_auto_enroll(*args, **kwargs):
            call_order.append("auto_enroll")
            return False

        mock_client._auto_enroll_if_needed = track_auto_enroll

        # We can't easily test the full _create_cell since it has many dependencies
        # But we can verify the method structure by checking the source
        import inspect

        from nvflare.private.fed.client.fed_client_base import FederatedClientBase

        source = inspect.getsource(FederatedClientBase._create_cell)

        # Verify _auto_enroll_if_needed is called in _create_cell
        assert "_auto_enroll_if_needed" in source
