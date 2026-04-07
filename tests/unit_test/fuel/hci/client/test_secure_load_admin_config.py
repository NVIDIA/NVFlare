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

"""Unit tests for secure_load_admin_config in nvflare/fuel/hci/client/config.py.

Tests verify that:
- valid_config=False (no signature.json) skips the tamper check and returns conf
- valid_config=True (signature.json present) enforces the tamper check strictly
- INVALID_SIGNATURE is still rejected when valid_config=True (security regression)
"""

from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.hci.client.config import secure_load_admin_config
from nvflare.fuel.sec.security_content_service import LoadResult


def _make_mgr(valid_config: bool, load_result: LoadResult):
    """Build a mock SecurityContentManager with given valid_config and load_json return."""
    mgr = MagicMock()
    mgr.valid_config = valid_config
    mgr.load_json.return_value = ({}, load_result)
    return mgr


def _make_conf():
    """Build a mock FLAdminClientStarterConfigurator."""
    conf = MagicMock()
    conf.configure.return_value = None
    return conf


class TestSecureLoadAdminConfig:
    """Tests for secure_load_admin_config."""

    def _run(self, mgr_mock, conf_mock, workspace=None):
        """Patch both dependencies and call the function under test."""
        if workspace is None:
            workspace = MagicMock()
        with (
            patch("nvflare.fuel.hci.client.config.SecurityContentManager", return_value=mgr_mock),
            patch("nvflare.fuel.hci.client.config.FLAdminClientStarterConfigurator", return_value=conf_mock),
        ):
            return secure_load_admin_config(workspace)

    # ------------------------------------------------------------------
    # valid_config=False scenarios (no signature.json)
    # ------------------------------------------------------------------

    def test_valid_config_false_no_error(self):
        """No signature.json → NOT_SIGNED → function returns conf without error.

        Simulates a non-CC, non-HE centralized kit after Change 1, or any kit
        provisioned without signature.json.
        """
        mgr = _make_mgr(valid_config=False, load_result=LoadResult.NOT_SIGNED)
        conf = _make_conf()
        result = self._run(mgr, conf)
        assert result is conf
        conf.configure.assert_called_once()

    def test_valid_config_false_manual_workflow(self):
        """No signature.json, no rootCA.pem (Manual Workflow) → NOT_SIGNED → no error.

        valid_config=False because neither signature.json nor rootCA.pem exist.
        mTLS is the trust anchor; tamper check is not applicable.
        """
        mgr = _make_mgr(valid_config=False, load_result=LoadResult.NOT_SIGNED)
        conf = _make_conf()
        result = self._run(mgr, conf)
        assert result is conf

    # ------------------------------------------------------------------
    # valid_config=True scenarios (signature.json present — CC or HE mode)
    # ------------------------------------------------------------------

    def test_valid_config_true_ok(self):
        """signature.json present, file matches signature → OK → returns conf."""
        mgr = _make_mgr(valid_config=True, load_result=LoadResult.OK)
        conf = _make_conf()
        result = self._run(mgr, conf)
        assert result is conf
        conf.configure.assert_called_once()

    def test_valid_config_true_tampered(self):
        """signature.json present, file content modified → INVALID_SIGNATURE → ConfigError."""
        mgr = _make_mgr(valid_config=True, load_result=LoadResult.INVALID_SIGNATURE)
        conf = _make_conf()
        with pytest.raises(ConfigError) as exc_info:
            self._run(mgr, conf)
        assert "tampered" in str(exc_info.value)
        conf.configure.assert_not_called()

    def test_valid_config_true_not_signed(self):
        """signature.json present but fed_admin.json not listed in it → NOT_SIGNED → ConfigError.

        This catches the case where a file was added to the kit after provisioning
        (not in the signature dict despite signature.json being present).
        """
        mgr = _make_mgr(valid_config=True, load_result=LoadResult.NOT_SIGNED)
        conf = _make_conf()
        with pytest.raises(ConfigError) as exc_info:
            self._run(mgr, conf)
        assert "tampered" in str(exc_info.value)
        conf.configure.assert_not_called()

    # ------------------------------------------------------------------
    # Security regression checks
    # ------------------------------------------------------------------

    def test_security_regression_invalid_signature_still_rejected(self):
        """valid_config=True + INVALID_SIGNATURE → ConfigError; the guard must not soften this.

        Regression: the new `if mgr.valid_config:` guard must NOT allow an active tamper
        (INVALID_SIGNATURE) through in CC/HE mode. Explicit check to prevent regression.
        """
        mgr = _make_mgr(valid_config=True, load_result=LoadResult.INVALID_SIGNATURE)
        conf = _make_conf()
        with pytest.raises(ConfigError) as exc_info:
            self._run(mgr, conf)
        error_msg = str(exc_info.value)
        assert "tampered" in error_msg
        # The LoadResult enum repr appears in the message so operators can diagnose
        assert "INVALID_SIGNATURE" in error_msg
        conf.configure.assert_not_called()

    def test_security_regression_not_signed_with_valid_config_rejected(self):
        """valid_config=True + NOT_SIGNED → ConfigError; NOT_SIGNED is not allowed in CC/HE mode.

        In CC/HE mode (valid_config=True), every managed file must appear in signature.json.
        NOT_SIGNED means the file is not in the sig dict, which is anomalous and must be rejected.
        """
        mgr = _make_mgr(valid_config=True, load_result=LoadResult.NOT_SIGNED)
        conf = _make_conf()
        with pytest.raises(ConfigError):
            self._run(mgr, conf)
        conf.configure.assert_not_called()
