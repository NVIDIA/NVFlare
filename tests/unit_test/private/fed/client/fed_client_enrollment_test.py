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

"""Unit tests for FederatedClientBase enrollment integration (HTTP-based)."""

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
    """Create a temporary startup directory with rootCA.pem."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_ca_path = os.path.join(tmp_dir, "rootCA.pem")
        with open(root_ca_path, "w") as f:
            f.write("-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----\n")
        yield tmp_dir, root_ca_path


class TestAutoEnrollIfNeeded:
    """Tests for FederatedClientBase._auto_enroll_if_needed()."""

    def test_returns_false_when_not_secure_train(self):
        """Test that enrollment is skipped when not in secure mode."""
        mock_client = MagicMock()
        mock_client.secure_train = False
        mock_client.logger = MagicMock()

        from nvflare.private.fed.client.fed_client_base import FederatedClientBase

        result = FederatedClientBase._auto_enroll_if_needed(mock_client)

        assert result is False

    def test_returns_false_when_cert_exists(self, temp_startup_dir):
        """Test that enrollment is skipped when certificate already exists."""
        tmp_dir, root_ca_path = temp_startup_dir

        # Create existing certificate
        cert_path = os.path.join(tmp_dir, "client.crt")
        with open(cert_path, "w") as f:
            f.write("existing-cert")

        mock_client = MagicMock()
        mock_client.secure_train = True
        mock_client.client_args = {
            SecureTrainConst.SSL_CERT: cert_path,
            SecureTrainConst.SSL_ROOT_CERT: root_ca_path,
        }
        mock_client.logger = MagicMock()

        from nvflare.private.fed.client.fed_client_base import FederatedClientBase

        result = FederatedClientBase._auto_enroll_if_needed(mock_client)

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
            result = FederatedClientBase._auto_enroll_if_needed(mock_client)

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

        with patch.dict(
            os.environ,
            {
                "NVFLARE_ENROLLMENT_TOKEN": "test-token",
                "NVFLARE_CERT_SERVICE_URL": "https://cert-svc:8443",
            },
        ):
            result = FederatedClientBase._auto_enroll_if_needed(mock_client)

        assert result is True
        mock_client._perform_enrollment.assert_called_once_with("https://cert-svc:8443", "test-token", tmp_dir)

    def test_reads_token_from_file(self, temp_startup_dir):
        """Test that token is read from enrollment_token file."""
        tmp_dir, root_ca_path = temp_startup_dir

        # Create token file
        token_file = os.path.join(tmp_dir, "enrollment_token")
        with open(token_file, "w") as f:
            f.write("file-based-token\n")

        # Create enrollment.json with cert service URL
        config_file = os.path.join(tmp_dir, "enrollment.json")
        with open(config_file, "w") as f:
            json.dump({"cert_service_url": "https://cert-svc:8443"}, f)

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
            result = FederatedClientBase._auto_enroll_if_needed(mock_client)

        assert result is True
        mock_client._perform_enrollment.assert_called_once_with("https://cert-svc:8443", "file-based-token", tmp_dir)

    def test_raises_when_cert_service_url_missing(self, temp_startup_dir):
        """Test that error is raised when token exists but cert service URL is missing."""
        tmp_dir, root_ca_path = temp_startup_dir

        mock_client = MagicMock()
        mock_client.secure_train = True
        mock_client.client_name = "test-client"
        mock_client.client_args = {
            SecureTrainConst.SSL_CERT: os.path.join(tmp_dir, "nonexistent.crt"),
            SecureTrainConst.SSL_ROOT_CERT: root_ca_path,
        }
        mock_client.logger = MagicMock()

        from nvflare.private.fed.client.fed_client_base import FederatedClientBase

        # Token from env but no URL
        with patch.dict(os.environ, {"NVFLARE_ENROLLMENT_TOKEN": "test-token"}, clear=True):
            with pytest.raises(RuntimeError, match="Certificate Service URL required"):
                FederatedClientBase._auto_enroll_if_needed(mock_client)

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

        with patch.dict(
            os.environ,
            {
                "NVFLARE_ENROLLMENT_TOKEN": "test-token",
                "NVFLARE_CERT_SERVICE_URL": "https://cert-svc:8443",
            },
        ):
            with pytest.raises(RuntimeError, match="Auto-enrollment failed"):
                FederatedClientBase._auto_enroll_if_needed(mock_client)


class TestPerformEnrollment:
    """Tests for FederatedClientBase._perform_enrollment()."""

    def test_calls_enroll_function(self, temp_startup_dir):
        """Test that enroll function is called with correct parameters."""
        tmp_dir, root_ca_path = temp_startup_dir

        mock_client = MagicMock()
        mock_client.client_name = "hospital-1"
        mock_client.client_args = {
            "organization": "Test Hospital",
        }
        mock_client.logger = MagicMock()

        mock_result = MockEnrollmentResult(
            cert_path=os.path.join(tmp_dir, "client.crt"),
            key_path=os.path.join(tmp_dir, "client.key"),
            ca_path=os.path.join(tmp_dir, "rootCA.pem"),
        )

        from nvflare.private.fed.client.fed_client_base import FederatedClientBase

        with patch("nvflare.security.enrollment.enroll", return_value=mock_result) as mock_enroll:
            with patch("nvflare.security.enrollment.EnrollmentIdentity") as mock_identity_class:
                mock_identity = MagicMock()
                mock_identity_class.for_client.return_value = mock_identity

                FederatedClientBase._perform_enrollment(mock_client, "https://cert-svc:8443", "test-token", tmp_dir)

                # Verify identity was created correctly
                mock_identity_class.for_client.assert_called_once_with(name="hospital-1", org="Test Hospital")

                # Verify enroll was called
                mock_enroll.assert_called_once_with("https://cert-svc:8443", "test-token", mock_identity, tmp_dir)

    def test_updates_client_args_after_enrollment(self, temp_startup_dir):
        """Test that client_args is updated with new certificate paths."""
        tmp_dir, root_ca_path = temp_startup_dir

        client_args = {}

        mock_client = MagicMock()
        mock_client.client_name = "test-client"
        mock_client.client_args = client_args
        mock_client.logger = MagicMock()

        mock_result = MockEnrollmentResult(
            cert_path=os.path.join(tmp_dir, "client.crt"),
            key_path=os.path.join(tmp_dir, "client.key"),
            ca_path=os.path.join(tmp_dir, "rootCA.pem"),
        )

        from nvflare.private.fed.client.fed_client_base import FederatedClientBase

        with patch("nvflare.security.enrollment.enroll", return_value=mock_result):
            with patch("nvflare.security.enrollment.EnrollmentIdentity"):
                FederatedClientBase._perform_enrollment(mock_client, "https://cert-svc:8443", "test-token", tmp_dir)

                # Verify client_args was updated
                assert client_args[SecureTrainConst.SSL_CERT] == mock_result.cert_path
                assert client_args[SecureTrainConst.PRIVATE_KEY] == mock_result.key_path
                assert client_args[SecureTrainConst.SSL_ROOT_CERT] == mock_result.ca_path


class TestCreateCellIntegration:
    """Tests for _create_cell integration with auto-enrollment."""

    def test_auto_enroll_called_in_create_cell(self):
        """Test that _auto_enroll_if_needed is called in _create_cell."""
        import inspect

        from nvflare.private.fed.client.fed_client_base import FederatedClientBase

        source = inspect.getsource(FederatedClientBase._create_cell)

        # Verify _auto_enroll_if_needed is called in _create_cell
        assert "_auto_enroll_if_needed" in source


class TestHelperFunctions:
    """Tests for helper functions in nvflare.security.enrollment."""

    def test_get_enrollment_token_from_env(self):
        """Test getting token from environment variable."""
        from nvflare.security.enrollment import get_enrollment_token

        with patch.dict(os.environ, {"NVFLARE_ENROLLMENT_TOKEN": "  env-token  "}):
            token = get_enrollment_token()
            assert token == "env-token"

    def test_get_enrollment_token_from_file(self, temp_startup_dir):
        """Test getting token from file."""
        tmp_dir, _ = temp_startup_dir

        token_file = os.path.join(tmp_dir, "enrollment_token")
        with open(token_file, "w") as f:
            f.write("file-token\n")

        from nvflare.security.enrollment import get_enrollment_token

        with patch.dict(os.environ, {}, clear=True):
            token = get_enrollment_token(tmp_dir)
            assert token == "file-token"

    def test_get_enrollment_token_returns_none(self):
        """Test that None is returned when no token is available."""
        from nvflare.security.enrollment import get_enrollment_token

        with patch.dict(os.environ, {}, clear=True):
            token = get_enrollment_token()
            assert token is None

    def test_get_cert_service_url_from_env(self):
        """Test getting URL from environment variable."""
        from nvflare.security.enrollment import get_cert_service_url

        with patch.dict(os.environ, {"NVFLARE_CERT_SERVICE_URL": "  https://cert-svc:8443  "}):
            url = get_cert_service_url()
            assert url == "https://cert-svc:8443"

    def test_get_cert_service_url_from_file(self, temp_startup_dir):
        """Test getting URL from enrollment.json file."""
        tmp_dir, _ = temp_startup_dir

        config_file = os.path.join(tmp_dir, "enrollment.json")
        with open(config_file, "w") as f:
            json.dump({"cert_service_url": "https://cert-svc:8443"}, f)

        from nvflare.security.enrollment import get_cert_service_url

        with patch.dict(os.environ, {}, clear=True):
            url = get_cert_service_url(tmp_dir)
            assert url == "https://cert-svc:8443"

    def test_get_cert_service_url_returns_none(self):
        """Test that None is returned when no URL is available."""
        from nvflare.security.enrollment import get_cert_service_url

        with patch.dict(os.environ, {}, clear=True):
            url = get_cert_service_url()
            assert url is None

    def test_enroll_function(self, temp_startup_dir):
        """Test the enroll convenience function."""
        tmp_dir, _ = temp_startup_dir

        mock_result = MockEnrollmentResult(
            cert_path=os.path.join(tmp_dir, "client.crt"),
            key_path=os.path.join(tmp_dir, "client.key"),
            ca_path=os.path.join(tmp_dir, "rootCA.pem"),
        )

        from nvflare.security.enrollment import EnrollmentIdentity, enroll

        with patch("nvflare.security.enrollment.CertRequestor") as mock_requestor_class:
            mock_requestor = MagicMock()
            mock_requestor.request_certificate.return_value = mock_result
            mock_requestor_class.return_value = mock_requestor

            identity = EnrollmentIdentity.for_client("site-1")
            result = enroll("https://cert-svc:8443", "token", identity, tmp_dir)

            assert result == mock_result
            mock_requestor_class.assert_called_once()
            mock_requestor.request_certificate.assert_called_once()
