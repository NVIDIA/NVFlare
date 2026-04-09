# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import sys
from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.sec.security_content_service import SecurityContentService
from nvflare.private.fed.utils.fed_utils import security_init, security_init_for_job


@pytest.fixture(autouse=True)
def reset_security_service():
    """Reset SecurityContentService class state between tests."""
    SecurityContentService.security_content_manager = None
    SecurityContentService.content_folder = None
    yield
    SecurityContentService.security_content_manager = None
    SecurityContentService.content_folder = None


def _make_workspace(startup_dir):
    ws = MagicMock()
    ws.get_startup_kit_dir.return_value = startup_dir
    ws.get_audit_file_path.return_value = startup_dir + "/audit.log"
    ws.get_authorization_file_path.return_value = None
    ws.get_file_path_in_site_config.return_value = startup_dir + "/study_registry.json"
    return ws


def _install_mock_manager(valid_config: bool):
    """Set SecurityContentService.security_content_manager to a mock with the given valid_config."""
    mock_manager = MagicMock()
    mock_manager.valid_config = valid_config
    SecurityContentService.security_content_manager = mock_manager


# ---------------------------------------------------------------------------
# security_init tests
# ---------------------------------------------------------------------------


class TestSecurityInit:
    def test_security_init_valid_config_false_no_check(self, tmp_path):
        """secure_train=True, valid_config=False -> _check_secure_content NOT called."""
        startup_dir = str(tmp_path)
        workspace = _make_workspace(startup_dir)

        with (
            patch("nvflare.private.fed.utils.fed_utils.SecurityContentService") as mock_scs,
            patch("nvflare.private.fed.utils.fed_utils._check_secure_content") as mock_check,
            patch("nvflare.private.fed.utils.fed_utils.AuditService"),
            patch("nvflare.private.fed.utils.fed_utils.AuthorizationService") as mock_authz,
        ):
            mock_authz.initialize.return_value = (None, None)
            mock_mgr = MagicMock()
            mock_mgr.valid_config = False
            mock_scs.security_content_manager = mock_mgr

            security_init(
                secure_train=True,
                site_org="test-org",
                workspace=workspace,
                app_validator=None,
                site_type="client",
            )

        mock_check.assert_not_called()

    def test_security_init_valid_config_true_clean_kit(self, tmp_path):
        """secure_train=True, valid_config=True, clean kit -> check called, no sys.exit."""
        startup_dir = str(tmp_path)
        workspace = _make_workspace(startup_dir)

        with (
            patch("nvflare.private.fed.utils.fed_utils.SecurityContentService") as mock_scs,
            patch("nvflare.private.fed.utils.fed_utils._check_secure_content", return_value=[]) as mock_check,
            patch("nvflare.private.fed.utils.fed_utils.AuditService"),
            patch("nvflare.private.fed.utils.fed_utils.AuthorizationService") as mock_authz,
            patch.object(sys, "exit") as mock_exit,
        ):
            mock_authz.initialize.return_value = (None, None)
            mock_mgr = MagicMock()
            mock_mgr.valid_config = True
            mock_scs.security_content_manager = mock_mgr

            security_init(
                secure_train=True,
                site_org="test-org",
                workspace=workspace,
                app_validator=None,
                site_type="client",
            )

        mock_check.assert_called_once()
        mock_exit.assert_not_called()

    def test_security_init_valid_config_true_tampered_file(self, tmp_path):
        """secure_train=True, valid_config=True, tampered file -> sys.exit(1)."""
        startup_dir = str(tmp_path)
        workspace = _make_workspace(startup_dir)

        with (
            patch("nvflare.private.fed.utils.fed_utils.SecurityContentService") as mock_scs,
            patch(
                "nvflare.private.fed.utils.fed_utils._check_secure_content", return_value=["fed_client.json"]
            ) as mock_check,
            patch("nvflare.private.fed.utils.fed_utils.AuditService"),
            patch.object(sys, "exit") as mock_exit,
        ):
            mock_mgr = MagicMock()
            mock_mgr.valid_config = True
            mock_scs.security_content_manager = mock_mgr

            security_init(
                secure_train=True,
                site_org="test-org",
                workspace=workspace,
                app_validator=None,
                site_type="client",
            )

        mock_check.assert_called_once()
        mock_exit.assert_called_once_with(1)

    def test_security_init_secure_false_no_check(self, tmp_path):
        """secure_train=False -> _check_secure_content NOT called regardless of valid_config."""
        startup_dir = str(tmp_path)
        workspace = _make_workspace(startup_dir)

        with (
            patch("nvflare.private.fed.utils.fed_utils.SecurityContentService") as mock_scs,
            patch("nvflare.private.fed.utils.fed_utils._check_secure_content") as mock_check,
            patch("nvflare.private.fed.utils.fed_utils.AuditService"),
            patch("nvflare.private.fed.utils.fed_utils.AuthorizationService") as mock_authz,
        ):
            mock_authz.initialize.return_value = (None, None)
            mock_mgr = MagicMock()
            mock_mgr.valid_config = True  # valid_config=True but secure_train=False
            mock_scs.security_content_manager = mock_mgr

            security_init(
                secure_train=False,
                site_org="test-org",
                workspace=workspace,
                app_validator=None,
                site_type="client",
            )

        mock_check.assert_not_called()


# ---------------------------------------------------------------------------
# security_init_for_job tests
# ---------------------------------------------------------------------------


class TestSecurityInitForJob:
    def test_security_init_for_job_valid_config_false_no_check(self, tmp_path):
        """secure_train=True, valid_config=False -> _check_secure_content NOT called."""
        startup_dir = str(tmp_path)
        workspace = _make_workspace(startup_dir)
        workspace.get_audit_file_path.return_value = startup_dir + "/audit.log"

        with (
            patch("nvflare.private.fed.utils.fed_utils.SecurityContentService") as mock_scs,
            patch("nvflare.private.fed.utils.fed_utils._check_secure_content") as mock_check,
            patch("nvflare.private.fed.utils.fed_utils.AuditService"),
        ):
            mock_mgr = MagicMock()
            mock_mgr.valid_config = False
            mock_scs.security_content_manager = mock_mgr

            security_init_for_job(
                secure_train=True,
                workspace=workspace,
                site_type="client",
                job_id="job123",
            )

        mock_check.assert_not_called()

    def test_security_init_for_job_valid_config_true_clean_kit(self, tmp_path):
        """secure_train=True, valid_config=True, clean kit -> check called, no sys.exit."""
        startup_dir = str(tmp_path)
        workspace = _make_workspace(startup_dir)

        with (
            patch("nvflare.private.fed.utils.fed_utils.SecurityContentService") as mock_scs,
            patch("nvflare.private.fed.utils.fed_utils._check_secure_content", return_value=[]) as mock_check,
            patch("nvflare.private.fed.utils.fed_utils.AuditService"),
            patch.object(sys, "exit") as mock_exit,
        ):
            mock_mgr = MagicMock()
            mock_mgr.valid_config = True
            mock_scs.security_content_manager = mock_mgr

            security_init_for_job(
                secure_train=True,
                workspace=workspace,
                site_type="client",
                job_id="job123",
            )

        mock_check.assert_called_once()
        mock_exit.assert_not_called()

    def test_security_init_for_job_valid_config_true_tampered(self, tmp_path):
        """secure_train=True, valid_config=True, tampered content -> sys.exit(1)."""
        startup_dir = str(tmp_path)
        workspace = _make_workspace(startup_dir)

        with (
            patch("nvflare.private.fed.utils.fed_utils.SecurityContentService") as mock_scs,
            patch(
                "nvflare.private.fed.utils.fed_utils._check_secure_content", return_value=["fed_client.json"]
            ) as mock_check,
            patch("nvflare.private.fed.utils.fed_utils.AuditService"),
            patch.object(sys, "exit") as mock_exit,
        ):
            mock_mgr = MagicMock()
            mock_mgr.valid_config = True
            mock_scs.security_content_manager = mock_mgr

            security_init_for_job(
                secure_train=True,
                workspace=workspace,
                site_type="client",
                job_id="job123",
            )

        mock_check.assert_called_once()
        mock_exit.assert_called_once_with(1)

    def test_security_init_for_job_secure_false_no_check(self, tmp_path):
        """secure_train=False -> _check_secure_content NOT called regardless of valid_config."""
        startup_dir = str(tmp_path)
        workspace = _make_workspace(startup_dir)

        with (
            patch("nvflare.private.fed.utils.fed_utils.SecurityContentService") as mock_scs,
            patch("nvflare.private.fed.utils.fed_utils._check_secure_content") as mock_check,
            patch("nvflare.private.fed.utils.fed_utils.AuditService"),
        ):
            mock_mgr = MagicMock()
            mock_mgr.valid_config = True  # valid_config=True but secure_train=False
            mock_scs.security_content_manager = mock_mgr

            security_init_for_job(
                secure_train=False,
                workspace=workspace,
                site_type="client",
                job_id="job123",
            )

        mock_check.assert_not_called()
