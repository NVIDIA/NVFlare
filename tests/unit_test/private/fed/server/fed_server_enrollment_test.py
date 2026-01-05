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

"""Unit tests for FederatedServer enrollment integration (HTTP-based)."""

import json
import os
import tempfile
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.fl_constant import SecureTrainConst


@dataclass
class MockEnrollmentResult:
    """Mock enrollment result."""

    cert_path: str
    key_path: str
    ca_path: str
    private_key: object = None
    certificate_pem: str = "cert-pem"
    ca_cert_pem: str = "ca-pem"


@pytest.fixture
def temp_startup_dir():
    """Create a temporary startup directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


class TestServerAutoEnrollIfNeeded:
    """Tests for FederatedServer._auto_enroll_if_needed()."""

    def test_returns_false_when_cert_exists(self, temp_startup_dir):
        """Test that enrollment is skipped when certificate already exists."""
        # Create existing certificate
        cert_path = os.path.join(temp_startup_dir, "server.crt")
        with open(cert_path, "w") as f:
            f.write("existing-cert")

        grpc_args = {
            SecureTrainConst.SSL_CERT: cert_path,
        }

        mock_server = MagicMock()
        mock_server.logger = MagicMock()

        from nvflare.private.fed.server.fed_server import FederatedServer

        result = FederatedServer._auto_enroll_if_needed(mock_server, grpc_args, temp_startup_dir)

        assert result is False

    def test_returns_false_when_no_token_available(self, temp_startup_dir):
        """Test that enrollment is skipped when no token is available."""
        grpc_args = {
            SecureTrainConst.SSL_CERT: os.path.join(temp_startup_dir, "nonexistent.crt"),
        }

        mock_server = MagicMock()
        mock_server.logger = MagicMock()

        from nvflare.private.fed.server.fed_server import FederatedServer

        # Ensure no env var is set
        with patch.dict(os.environ, {}, clear=True):
            result = FederatedServer._auto_enroll_if_needed(mock_server, grpc_args, temp_startup_dir)

        assert result is False
        mock_server.logger.debug.assert_called()

    def test_reads_token_from_env_var(self, temp_startup_dir):
        """Test that token is read from NVFLARE_ENROLLMENT_TOKEN env var."""
        grpc_args = {
            SecureTrainConst.SSL_CERT: os.path.join(temp_startup_dir, "nonexistent.crt"),
            "name": "test-server",
            "organization": "TestOrg",
        }

        mock_server = MagicMock()
        mock_server.logger = MagicMock()
        mock_server._perform_enrollment = MagicMock()

        from nvflare.private.fed.server.fed_server import FederatedServer

        with patch.dict(
            os.environ,
            {
                "NVFLARE_ENROLLMENT_TOKEN": "test-token",
                "NVFLARE_CERT_SERVICE_URL": "https://cert-svc:8443",
            },
        ):
            result = FederatedServer._auto_enroll_if_needed(mock_server, grpc_args, temp_startup_dir)

        assert result is True
        mock_server._perform_enrollment.assert_called_once_with(
            "https://cert-svc:8443",
            "test-token",
            temp_startup_dir,
            "test-server",
            "TestOrg",
            grpc_args,
        )

    def test_reads_token_from_file(self, temp_startup_dir):
        """Test that token is read from enrollment_token file."""
        # Create token file
        token_file = os.path.join(temp_startup_dir, "enrollment_token")
        with open(token_file, "w") as f:
            f.write("file-based-token\n")

        # Create enrollment.json with cert service URL
        config_file = os.path.join(temp_startup_dir, "enrollment.json")
        with open(config_file, "w") as f:
            json.dump({"cert_service_url": "https://cert-svc:8443"}, f)

        grpc_args = {
            SecureTrainConst.SSL_CERT: os.path.join(temp_startup_dir, "nonexistent.crt"),
            "name": "test-server",
        }

        mock_server = MagicMock()
        mock_server.logger = MagicMock()
        mock_server._perform_enrollment = MagicMock()

        from nvflare.private.fed.server.fed_server import FederatedServer

        with patch.dict(os.environ, {}, clear=True):
            result = FederatedServer._auto_enroll_if_needed(mock_server, grpc_args, temp_startup_dir)

        assert result is True
        mock_server._perform_enrollment.assert_called_once()

    def test_raises_when_cert_service_url_missing(self, temp_startup_dir):
        """Test that error is raised when token exists but cert service URL is missing."""
        grpc_args = {
            SecureTrainConst.SSL_CERT: os.path.join(temp_startup_dir, "nonexistent.crt"),
            "name": "test-server",
        }

        mock_server = MagicMock()
        mock_server.logger = MagicMock()

        from nvflare.private.fed.server.fed_server import FederatedServer

        with patch.dict(os.environ, {"NVFLARE_ENROLLMENT_TOKEN": "test-token"}, clear=True):
            with pytest.raises(RuntimeError, match="Certificate Service URL required"):
                FederatedServer._auto_enroll_if_needed(mock_server, grpc_args, temp_startup_dir)

    def test_uses_default_server_name(self, temp_startup_dir):
        """Test that default server name is used when not specified."""
        grpc_args = {
            SecureTrainConst.SSL_CERT: os.path.join(temp_startup_dir, "nonexistent.crt"),
            # No "name" key
        }

        mock_server = MagicMock()
        mock_server.logger = MagicMock()
        mock_server._perform_enrollment = MagicMock()

        from nvflare.private.fed.server.fed_server import FederatedServer

        with patch.dict(
            os.environ,
            {
                "NVFLARE_ENROLLMENT_TOKEN": "test-token",
                "NVFLARE_CERT_SERVICE_URL": "https://cert-svc:8443",
            },
        ):
            FederatedServer._auto_enroll_if_needed(mock_server, grpc_args, temp_startup_dir)

        # Should use default "server" name
        call_args = mock_server._perform_enrollment.call_args
        assert call_args[0][3] == "server"  # server_name arg

    def test_raises_on_enrollment_failure(self, temp_startup_dir):
        """Test that enrollment failure raises RuntimeError."""
        grpc_args = {
            SecureTrainConst.SSL_CERT: os.path.join(temp_startup_dir, "nonexistent.crt"),
            "name": "test-server",
        }

        mock_server = MagicMock()
        mock_server.logger = MagicMock()
        mock_server._perform_enrollment = MagicMock(side_effect=Exception("Enrollment failed"))

        from nvflare.private.fed.server.fed_server import FederatedServer

        with patch.dict(
            os.environ,
            {
                "NVFLARE_ENROLLMENT_TOKEN": "test-token",
                "NVFLARE_CERT_SERVICE_URL": "https://cert-svc:8443",
            },
        ):
            with pytest.raises(RuntimeError, match="Auto-enrollment failed"):
                FederatedServer._auto_enroll_if_needed(mock_server, grpc_args, temp_startup_dir)


class TestServerPerformEnrollment:
    """Tests for FederatedServer._perform_enrollment()."""

    def test_calls_enroll_function(self, temp_startup_dir):
        """Test that enroll function is called with correct parameters."""
        grpc_args = {}

        mock_server = MagicMock()
        mock_server.logger = MagicMock()

        mock_result = MockEnrollmentResult(
            cert_path=os.path.join(temp_startup_dir, "server.crt"),
            key_path=os.path.join(temp_startup_dir, "server.key"),
            ca_path=os.path.join(temp_startup_dir, "rootCA.pem"),
        )

        from nvflare.private.fed.server.fed_server import FederatedServer

        with patch("nvflare.security.enrollment.enroll", return_value=mock_result) as mock_enroll:
            with patch("nvflare.security.enrollment.EnrollmentIdentity") as mock_identity_class:
                mock_identity = MagicMock()
                mock_identity_class.for_server.return_value = mock_identity

                FederatedServer._perform_enrollment(
                    mock_server,
                    "https://cert-svc:8443",
                    "test-token",
                    temp_startup_dir,
                    "my-server",
                    "TestOrg",
                    grpc_args,
                )

                # Verify identity was created correctly
                mock_identity_class.for_server.assert_called_once_with(name="my-server", org="TestOrg")

                # Verify enroll was called
                mock_enroll.assert_called_once_with(
                    "https://cert-svc:8443", "test-token", mock_identity, temp_startup_dir
                )

    def test_updates_grpc_args_after_enrollment(self, temp_startup_dir):
        """Test that grpc_args is updated with new certificate paths."""
        grpc_args = {}

        mock_server = MagicMock()
        mock_server.logger = MagicMock()

        mock_result = MockEnrollmentResult(
            cert_path=os.path.join(temp_startup_dir, "server.crt"),
            key_path=os.path.join(temp_startup_dir, "server.key"),
            ca_path=os.path.join(temp_startup_dir, "rootCA.pem"),
        )

        from nvflare.private.fed.server.fed_server import FederatedServer

        with patch("nvflare.security.enrollment.enroll", return_value=mock_result):
            with patch("nvflare.security.enrollment.EnrollmentIdentity"):
                FederatedServer._perform_enrollment(
                    mock_server,
                    "https://cert-svc:8443",
                    "test-token",
                    temp_startup_dir,
                    "my-server",
                    "TestOrg",
                    grpc_args,
                )

                # Verify grpc_args was updated
                assert grpc_args[SecureTrainConst.SSL_CERT] == mock_result.cert_path
                assert grpc_args[SecureTrainConst.PRIVATE_KEY] == mock_result.key_path
                assert grpc_args[SecureTrainConst.SSL_ROOT_CERT] == mock_result.ca_path


class TestDeployIntegration:
    """Tests for deploy() integration with auto-enrollment."""

    def test_auto_enroll_called_in_deploy(self):
        """Test that _auto_enroll_if_needed is called in BaseServer.deploy() for secure mode."""
        import inspect

        from nvflare.private.fed.server.fed_server import BaseServer

        source = inspect.getsource(BaseServer.deploy)

        # Verify _auto_enroll_if_needed is called in deploy
        assert "_auto_enroll_if_needed" in source
