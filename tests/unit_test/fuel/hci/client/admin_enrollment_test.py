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

"""Unit tests for AdminAPI enrollment integration (HTTP-based)."""

import json
import os
import tempfile
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.hci.client.api_spec import AdminConfigKey


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


class TestAdminAutoEnrollIfNeeded:
    """Tests for AdminAPI._auto_enroll_if_needed()."""

    def test_returns_when_cert_exists(self, temp_startup_dir):
        """Test that enrollment is skipped when certificate already exists."""
        # Create existing certificate
        cert_path = os.path.join(temp_startup_dir, "client.crt")
        with open(cert_path, "w") as f:
            f.write("existing-cert")

        admin_config = {
            AdminConfigKey.CLIENT_CERT: cert_path,
            AdminConfigKey.STARTUP_DIR: temp_startup_dir,
        }

        mock_admin = MagicMock()
        mock_admin.logger = MagicMock()

        from nvflare.fuel.hci.client.api import AdminAPI

        # Should return without calling _perform_enrollment
        AdminAPI._auto_enroll_if_needed(mock_admin, admin_config)

        # Verify _perform_enrollment was not called
        mock_admin._perform_enrollment.assert_not_called()

    def test_returns_when_no_token_available(self, temp_startup_dir):
        """Test that enrollment is skipped when no token is available."""
        admin_config = {
            AdminConfigKey.CLIENT_CERT: os.path.join(temp_startup_dir, "nonexistent.crt"),
            AdminConfigKey.STARTUP_DIR: temp_startup_dir,
        }

        mock_admin = MagicMock()
        mock_admin.logger = MagicMock()

        from nvflare.fuel.hci.client.api import AdminAPI

        # Ensure no env var is set
        with patch.dict(os.environ, {}, clear=True):
            AdminAPI._auto_enroll_if_needed(mock_admin, admin_config)

        mock_admin.logger.debug.assert_called()
        mock_admin._perform_enrollment.assert_not_called()

    def test_reads_token_from_env_var(self, temp_startup_dir):
        """Test that token is read from NVFLARE_ENROLLMENT_TOKEN env var."""
        admin_config = {
            AdminConfigKey.CLIENT_CERT: os.path.join(temp_startup_dir, "nonexistent.crt"),
            AdminConfigKey.STARTUP_DIR: temp_startup_dir,
        }

        mock_admin = MagicMock()
        mock_admin.user_name = "admin@example.com"
        mock_admin.logger = MagicMock()
        mock_admin._perform_enrollment = MagicMock()

        from nvflare.fuel.hci.client.api import AdminAPI

        with patch.dict(
            os.environ,
            {
                "NVFLARE_ENROLLMENT_TOKEN": "test-token",
                "NVFLARE_CERT_SERVICE_URL": "https://cert-svc:8443",
            },
        ):
            AdminAPI._auto_enroll_if_needed(mock_admin, admin_config)

        mock_admin._perform_enrollment.assert_called_once_with(
            "https://cert-svc:8443", "test-token", temp_startup_dir, admin_config
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

        admin_config = {
            AdminConfigKey.CLIENT_CERT: os.path.join(temp_startup_dir, "nonexistent.crt"),
            AdminConfigKey.STARTUP_DIR: temp_startup_dir,
        }

        mock_admin = MagicMock()
        mock_admin.user_name = "admin@example.com"
        mock_admin.logger = MagicMock()
        mock_admin._perform_enrollment = MagicMock()

        from nvflare.fuel.hci.client.api import AdminAPI

        with patch.dict(os.environ, {}, clear=True):
            AdminAPI._auto_enroll_if_needed(mock_admin, admin_config)

        mock_admin._perform_enrollment.assert_called_once()

    def test_raises_when_cert_service_url_missing(self, temp_startup_dir):
        """Test that error is raised when token exists but cert service URL is missing."""
        admin_config = {
            AdminConfigKey.CLIENT_CERT: os.path.join(temp_startup_dir, "nonexistent.crt"),
            AdminConfigKey.STARTUP_DIR: temp_startup_dir,
        }

        mock_admin = MagicMock()
        mock_admin.user_name = "admin@example.com"
        mock_admin.logger = MagicMock()

        from nvflare.fuel.hci.client.api import AdminAPI, ConfigError

        with patch.dict(os.environ, {"NVFLARE_ENROLLMENT_TOKEN": "test-token"}, clear=True):
            with pytest.raises(ConfigError, match="Certificate Service URL required"):
                AdminAPI._auto_enroll_if_needed(mock_admin, admin_config)

    def test_raises_on_enrollment_failure(self, temp_startup_dir):
        """Test that enrollment failure raises ConfigError."""
        admin_config = {
            AdminConfigKey.CLIENT_CERT: os.path.join(temp_startup_dir, "nonexistent.crt"),
            AdminConfigKey.STARTUP_DIR: temp_startup_dir,
        }

        mock_admin = MagicMock()
        mock_admin.user_name = "admin@example.com"
        mock_admin.logger = MagicMock()
        mock_admin._perform_enrollment = MagicMock(side_effect=Exception("Enrollment failed"))

        from nvflare.fuel.hci.client.api import AdminAPI, ConfigError

        with patch.dict(
            os.environ,
            {
                "NVFLARE_ENROLLMENT_TOKEN": "test-token",
                "NVFLARE_CERT_SERVICE_URL": "https://cert-svc:8443",
            },
        ):
            with pytest.raises(ConfigError, match="Auto-enrollment failed"):
                AdminAPI._auto_enroll_if_needed(mock_admin, admin_config)


class TestAdminPerformEnrollment:
    """Tests for AdminAPI._perform_enrollment()."""

    def test_calls_enroll_function(self, temp_startup_dir):
        """Test that enroll function is called with correct parameters."""
        admin_config = {"role": "org_admin"}

        mock_admin = MagicMock()
        mock_admin.user_name = "admin@example.com"
        mock_admin.logger = MagicMock()

        mock_result = MockEnrollmentResult(
            cert_path=os.path.join(temp_startup_dir, "client.crt"),
            key_path=os.path.join(temp_startup_dir, "client.key"),
            ca_path=os.path.join(temp_startup_dir, "rootCA.pem"),
        )

        from nvflare.fuel.hci.client.api import AdminAPI

        with patch("nvflare.security.enrollment.enroll", return_value=mock_result) as mock_enroll:
            with patch("nvflare.security.enrollment.EnrollmentIdentity") as mock_identity_class:
                mock_identity = MagicMock()
                mock_identity_class.for_admin.return_value = mock_identity

                AdminAPI._perform_enrollment(
                    mock_admin,
                    "https://cert-svc:8443",
                    "test-token",
                    temp_startup_dir,
                    admin_config,
                )

                # Verify identity was created correctly
                mock_identity_class.for_admin.assert_called_once_with(email="admin@example.com", role="org_admin")

                # Verify enroll was called
                mock_enroll.assert_called_once_with(
                    "https://cert-svc:8443", "test-token", mock_identity, temp_startup_dir
                )

    def test_uses_default_role_member(self, temp_startup_dir):
        """Test that default role 'member' is used when not specified."""
        admin_config = {}  # No role specified

        mock_admin = MagicMock()
        mock_admin.user_name = "admin@example.com"
        mock_admin.logger = MagicMock()

        mock_result = MockEnrollmentResult(
            cert_path=os.path.join(temp_startup_dir, "client.crt"),
            key_path=os.path.join(temp_startup_dir, "client.key"),
            ca_path=os.path.join(temp_startup_dir, "rootCA.pem"),
        )

        from nvflare.fuel.hci.client.api import AdminAPI

        with patch("nvflare.security.enrollment.enroll", return_value=mock_result):
            with patch("nvflare.security.enrollment.EnrollmentIdentity") as mock_identity_class:
                mock_identity = MagicMock()
                mock_identity_class.for_admin.return_value = mock_identity

                AdminAPI._perform_enrollment(
                    mock_admin,
                    "https://cert-svc:8443",
                    "test-token",
                    temp_startup_dir,
                    admin_config,
                )

                # Verify default role "member" is used
                mock_identity_class.for_admin.assert_called_once_with(email="admin@example.com", role="member")

    def test_updates_admin_config_after_enrollment(self, temp_startup_dir):
        """Test that admin_config and self attributes are updated."""
        admin_config = {}

        mock_admin = MagicMock()
        mock_admin.user_name = "admin@example.com"
        mock_admin.logger = MagicMock()

        mock_result = MockEnrollmentResult(
            cert_path=os.path.join(temp_startup_dir, "client.crt"),
            key_path=os.path.join(temp_startup_dir, "client.key"),
            ca_path=os.path.join(temp_startup_dir, "rootCA.pem"),
        )

        from nvflare.fuel.hci.client.api import AdminAPI

        with patch("nvflare.security.enrollment.enroll", return_value=mock_result):
            with patch("nvflare.security.enrollment.EnrollmentIdentity"):
                AdminAPI._perform_enrollment(
                    mock_admin,
                    "https://cert-svc:8443",
                    "test-token",
                    temp_startup_dir,
                    admin_config,
                )

                # Verify admin_config was updated
                assert admin_config[AdminConfigKey.CA_CERT] == mock_result.ca_path
                assert admin_config[AdminConfigKey.CLIENT_CERT] == mock_result.cert_path
                assert admin_config[AdminConfigKey.CLIENT_KEY] == mock_result.key_path

                # Verify self attributes were updated
                assert mock_admin.ca_cert == mock_result.ca_path
                assert mock_admin.client_cert == mock_result.cert_path
                assert mock_admin.client_key == mock_result.key_path


class TestInitIntegration:
    """Tests for __init__ integration with auto-enrollment."""

    def test_auto_enroll_called_in_init(self):
        """Test that _auto_enroll_if_needed is called in __init__."""
        import inspect

        from nvflare.fuel.hci.client.api import AdminAPI

        source = inspect.getsource(AdminAPI.__init__)

        # Verify _auto_enroll_if_needed is called in __init__
        assert "_auto_enroll_if_needed" in source

    def test_startup_dir_config_key_exists(self):
        """Test that STARTUP_DIR key is defined in AdminConfigKey."""
        assert hasattr(AdminConfigKey, "STARTUP_DIR")
        assert AdminConfigKey.STARTUP_DIR == "startup_dir"
