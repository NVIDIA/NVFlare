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

"""Unit tests for enrollment CLI commands."""

import argparse
import os
from unittest.mock import MagicMock, patch

import pytest


class TestEnrollmentCLIImports:
    """Test CLI module imports and constants."""

    def test_import_enrollment_cli(self):
        """Test that enrollment_cli module can be imported."""
        from nvflare.tool.enrollment import enrollment_cli

        assert enrollment_cli is not None

    def test_constants_defined(self):
        """Test that CLI constants are defined."""
        from nvflare.tool.enrollment.enrollment_cli import (
            CMD_ENROLLMENT,
            ENV_API_KEY,
            ENV_CERT_SERVICE_URL,
            SUBCMD_APPROVE,
            SUBCMD_ENROLLED,
            SUBCMD_INFO,
            SUBCMD_LIST,
            SUBCMD_REJECT,
        )

        assert CMD_ENROLLMENT == "enrollment"
        assert SUBCMD_LIST == "list"
        assert SUBCMD_INFO == "info"
        assert SUBCMD_APPROVE == "approve"
        assert SUBCMD_REJECT == "reject"
        assert SUBCMD_ENROLLED == "enrolled"
        assert ENV_CERT_SERVICE_URL == "NVFLARE_CERT_SERVICE_URL"
        assert ENV_API_KEY == "NVFLARE_API_KEY"


class TestGetCertServiceUrl:
    """Test Certificate Service URL resolution."""

    def test_url_from_args(self):
        """Test URL from CLI argument takes priority."""
        from nvflare.tool.enrollment.enrollment_cli import _get_cert_service_url

        args = argparse.Namespace(cert_service="https://cert-svc:8443")
        result = _get_cert_service_url(args)
        assert result == "https://cert-svc:8443"

    def test_url_from_env_var(self):
        """Test URL from environment variable."""
        from nvflare.tool.enrollment.enrollment_cli import ENV_CERT_SERVICE_URL, _get_cert_service_url

        args = argparse.Namespace(cert_service=None)
        with patch.dict(os.environ, {ENV_CERT_SERVICE_URL: "https://from-env:8443"}):
            result = _get_cert_service_url(args)
            assert result == "https://from-env:8443"

    def test_url_args_overrides_env(self):
        """Test CLI argument overrides environment variable."""
        from nvflare.tool.enrollment.enrollment_cli import ENV_CERT_SERVICE_URL, _get_cert_service_url

        args = argparse.Namespace(cert_service="https://from-args:8443")
        with patch.dict(os.environ, {ENV_CERT_SERVICE_URL: "https://from-env:8443"}):
            result = _get_cert_service_url(args)
            assert result == "https://from-args:8443"

    def test_url_trailing_slash_stripped(self):
        """Test trailing slash is stripped from URL."""
        from nvflare.tool.enrollment.enrollment_cli import _get_cert_service_url

        args = argparse.Namespace(cert_service="https://cert-svc:8443/")
        result = _get_cert_service_url(args)
        assert result == "https://cert-svc:8443"

    def test_url_missing_exits(self):
        """Test that missing URL causes exit."""
        from nvflare.tool.enrollment.enrollment_cli import _get_cert_service_url

        args = argparse.Namespace(cert_service=None)
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NVFLARE_CERT_SERVICE_URL", None)
            with pytest.raises(SystemExit) as exc_info:
                _get_cert_service_url(args)
            assert exc_info.value.code == 1


class TestGetAdminToken:
    """Test admin token resolution."""

    def test_token_from_args(self):
        """Test token from CLI argument takes priority."""
        from nvflare.tool.enrollment.enrollment_cli import _get_api_key

        args = argparse.Namespace(api_key="my-admin-token")
        result = _get_api_key(args)
        assert result == "my-admin-token"

    def test_token_from_env_var(self):
        """Test token from environment variable."""
        from nvflare.tool.enrollment.enrollment_cli import ENV_API_KEY, _get_api_key

        args = argparse.Namespace(api_key=None)
        with patch.dict(os.environ, {ENV_API_KEY: "env-token"}):
            result = _get_api_key(args)
            assert result == "env-token"

    def test_token_missing_exits(self):
        """Test that missing token causes exit."""
        from nvflare.tool.enrollment.enrollment_cli import _get_api_key

        args = argparse.Namespace(api_key=None)
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NVFLARE_API_KEY", None)
            with pytest.raises(SystemExit) as exc_info:
                _get_api_key(args)
            assert exc_info.value.code == 1


class TestHandleList:
    """Test 'enrollment list' command."""

    @patch("requests.get")
    def test_list_pending_empty(self, mock_get, capsys):
        """Test listing with no pending requests."""
        from nvflare.tool.enrollment.enrollment_cli import _handle_list

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"pending": []}
        mock_get.return_value = mock_response

        args = argparse.Namespace(
            cert_service="https://cert-svc:8443",
            api_key="token",
            entity_type=None,
        )

        _handle_list(args)

        captured = capsys.readouterr()
        assert "No pending enrollment requests" in captured.out

    @patch("requests.get")
    def test_list_pending_with_results(self, mock_get, capsys):
        """Test listing with pending requests."""
        from nvflare.tool.enrollment.enrollment_cli import _handle_list

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "pending": [
                {
                    "name": "hospital-1",
                    "type": "client",
                    "org": "Hospital A",
                    "submitted": "2025-01-04",
                    "status": "pending",
                },
                {
                    "name": "hospital-2",
                    "type": "client",
                    "org": "Hospital B",
                    "submitted": "2025-01-04",
                    "status": "pending",
                },
            ]
        }
        mock_get.return_value = mock_response

        args = argparse.Namespace(
            cert_service="https://cert-svc:8443",
            api_key="token",
            entity_type=None,
        )

        _handle_list(args)

        captured = capsys.readouterr()
        assert "hospital-1" in captured.out
        assert "hospital-2" in captured.out
        assert "Hospital A" in captured.out

    @patch("requests.get")
    def test_list_with_type_filter(self, mock_get):
        """Test listing with entity type filter."""
        from nvflare.tool.enrollment.enrollment_cli import _handle_list

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"pending": []}
        mock_get.return_value = mock_response

        args = argparse.Namespace(
            cert_service="https://cert-svc:8443",
            api_key="token",
            entity_type="client",
        )

        _handle_list(args)

        # Verify URL includes type filter
        call_args = mock_get.call_args
        assert "?type=client" in call_args[0][0]


class TestHandleInfo:
    """Test 'enrollment info' command."""

    @patch("requests.get")
    def test_info_success(self, mock_get, capsys):
        """Test viewing info of a pending request."""
        from nvflare.tool.enrollment.enrollment_cli import _handle_info

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "hospital-1",
            "type": "client",
            "org": "Hospital A",
            "status": "pending",
            "submitted": "2025-01-04 10:00:00",
            "token_subject": "hospital-*",
            "source_ip": "10.2.3.4",
            "csr_subject": "CN=hospital-1, O=Hospital A",
        }
        mock_get.return_value = mock_response

        args = argparse.Namespace(
            cert_service="https://cert-svc:8443",
            api_key="token",
            name="hospital-1",
            entity_type="client",
        )

        _handle_info(args)

        captured = capsys.readouterr()
        assert "hospital-1" in captured.out
        assert "Hospital A" in captured.out
        assert "pending" in captured.out
        assert "10.2.3.4" in captured.out

    @patch("requests.get")
    def test_info_not_found(self, mock_get, capsys):
        """Test info for non-existent request."""
        from nvflare.tool.enrollment.enrollment_cli import _handle_info

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"error": "Not found"}
        mock_get.return_value = mock_response

        args = argparse.Namespace(
            cert_service="https://cert-svc:8443",
            api_key="token",
            name="unknown-site",
            entity_type=None,
        )

        _handle_info(args)

        captured = capsys.readouterr()
        assert "Not found" in captured.out

    def test_info_missing_name_exits(self, capsys):
        """Test that info without name exits."""
        from nvflare.tool.enrollment.enrollment_cli import _handle_info

        args = argparse.Namespace(
            cert_service="https://cert-svc:8443",
            api_key="token",
            name=None,
            entity_type=None,
        )

        with pytest.raises(SystemExit):
            _handle_info(args)


class TestHandleApprove:
    """Test 'enrollment approve' command."""

    @patch("requests.post")
    def test_approve_single(self, mock_post, capsys):
        """Test approving single request."""
        from nvflare.tool.enrollment.enrollment_cli import _handle_approve

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"approved": 1}
        mock_post.return_value = mock_response

        args = argparse.Namespace(
            cert_service="https://cert-svc:8443",
            api_key="token",
            name="hospital-1",
            pattern=None,
            entity_type="client",
        )

        _handle_approve(args)

        captured = capsys.readouterr()
        assert "Approved 1 enrollment request" in captured.out

        # Verify correct endpoint
        call_args = mock_post.call_args
        assert "/api/v1/pending/hospital-1/approve" in call_args[0][0]

    @patch("requests.post")
    def test_approve_bulk_pattern(self, mock_post, capsys):
        """Test bulk approve with pattern."""
        from nvflare.tool.enrollment.enrollment_cli import _handle_approve

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"approved": ["hospital-1", "hospital-2", "hospital-3"]}
        mock_post.return_value = mock_response

        args = argparse.Namespace(
            cert_service="https://cert-svc:8443",
            api_key="token",
            name=None,
            pattern="hospital-*",
            entity_type="client",
        )

        _handle_approve(args)

        captured = capsys.readouterr()
        assert "Approved 3 enrollment request" in captured.out

        # Verify bulk endpoint and pattern in payload
        call_args = mock_post.call_args
        assert "/api/v1/pending/approve_batch" in call_args[0][0]
        assert call_args[1]["json"]["pattern"] == "hospital-*"

    def test_approve_missing_name_and_pattern_exits(self, capsys):
        """Test that approve without name or pattern exits."""
        from nvflare.tool.enrollment.enrollment_cli import _handle_approve

        args = argparse.Namespace(
            cert_service="https://cert-svc:8443",
            api_key="token",
            name=None,
            pattern=None,
            entity_type=None,
        )

        with pytest.raises(SystemExit):
            _handle_approve(args)


class TestHandleReject:
    """Test 'enrollment reject' command."""

    @patch("requests.post")
    def test_reject_with_reason(self, mock_post, capsys):
        """Test rejecting with a reason."""
        from nvflare.tool.enrollment.enrollment_cli import _handle_reject

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"rejected": True}
        mock_post.return_value = mock_response

        args = argparse.Namespace(
            cert_service="https://cert-svc:8443",
            api_key="token",
            name="hospital-2",
            reason="Not authorized",
            entity_type="client",
        )

        _handle_reject(args)

        captured = capsys.readouterr()
        assert "Rejected enrollment request" in captured.out

        # Verify reason in payload
        call_args = mock_post.call_args
        assert call_args[1]["json"]["reason"] == "Not authorized"

    @patch("requests.post")
    def test_reject_default_reason(self, mock_post, capsys):
        """Test rejecting with default reason."""
        from nvflare.tool.enrollment.enrollment_cli import _handle_reject

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"rejected": True}
        mock_post.return_value = mock_response

        args = argparse.Namespace(
            cert_service="https://cert-svc:8443",
            api_key="token",
            name="hospital-2",
            reason=None,
            entity_type="client",
        )

        _handle_reject(args)

        # Verify default reason
        call_args = mock_post.call_args
        assert "Rejected by admin" in call_args[1]["json"]["reason"]


class TestHandleEnrolled:
    """Test 'enrollment enrolled' command."""

    @patch("requests.get")
    def test_enrolled_empty(self, mock_get, capsys):
        """Test listing with no enrolled entities."""
        from nvflare.tool.enrollment.enrollment_cli import _handle_enrolled

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"enrolled": []}
        mock_get.return_value = mock_response

        args = argparse.Namespace(
            cert_service="https://cert-svc:8443",
            api_key="token",
            entity_type=None,
        )

        _handle_enrolled(args)

        captured = capsys.readouterr()
        assert "No enrolled entities" in captured.out

    @patch("requests.get")
    def test_enrolled_with_results(self, mock_get, capsys):
        """Test listing enrolled entities."""
        from nvflare.tool.enrollment.enrollment_cli import _handle_enrolled

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "enrolled": [
                {"name": "hospital-1", "type": "client", "org": "Hospital A", "enrolled_at": "2025-01-04"},
                {"name": "admin@org.com", "type": "user", "org": "Org Inc", "enrolled_at": "2025-01-04"},
            ]
        }
        mock_get.return_value = mock_response

        args = argparse.Namespace(
            cert_service="https://cert-svc:8443",
            api_key="token",
            entity_type=None,
        )

        _handle_enrolled(args)

        captured = capsys.readouterr()
        assert "hospital-1" in captured.out
        assert "admin@org.com" in captured.out


class TestDefineEnrollmentParser:
    """Test parser definition."""

    def test_define_parser(self):
        """Test that parser is correctly defined."""
        from nvflare.tool.enrollment.enrollment_cli import CMD_ENROLLMENT, define_enrollment_parser

        main_parser = argparse.ArgumentParser()
        sub_cmd = main_parser.add_subparsers()

        result = define_enrollment_parser(sub_cmd)

        assert CMD_ENROLLMENT in result
        assert result[CMD_ENROLLMENT] is not None


class TestHandleEnrollmentCmd:
    """Test main command handler."""

    @patch("nvflare.tool.enrollment.enrollment_cli._handle_list")
    def test_dispatch_list(self, mock_handle_list):
        """Test dispatching to list handler."""
        from nvflare.tool.enrollment.enrollment_cli import handle_enrollment_cmd

        args = argparse.Namespace(enrollment_cmd="list")
        handle_enrollment_cmd(args)
        mock_handle_list.assert_called_once_with(args)

    @patch("nvflare.tool.enrollment.enrollment_cli._handle_info")
    def test_dispatch_info(self, mock_handle_info):
        """Test dispatching to info handler."""
        from nvflare.tool.enrollment.enrollment_cli import handle_enrollment_cmd

        args = argparse.Namespace(enrollment_cmd="info")
        handle_enrollment_cmd(args)
        mock_handle_info.assert_called_once_with(args)

    @patch("nvflare.tool.enrollment.enrollment_cli._handle_approve")
    def test_dispatch_approve(self, mock_handle_approve):
        """Test dispatching to approve handler."""
        from nvflare.tool.enrollment.enrollment_cli import handle_enrollment_cmd

        args = argparse.Namespace(enrollment_cmd="approve")
        handle_enrollment_cmd(args)
        mock_handle_approve.assert_called_once_with(args)

    @patch("nvflare.tool.enrollment.enrollment_cli._handle_reject")
    def test_dispatch_reject(self, mock_handle_reject):
        """Test dispatching to reject handler."""
        from nvflare.tool.enrollment.enrollment_cli import handle_enrollment_cmd

        args = argparse.Namespace(enrollment_cmd="reject")
        handle_enrollment_cmd(args)
        mock_handle_reject.assert_called_once_with(args)

    @patch("nvflare.tool.enrollment.enrollment_cli._handle_enrolled")
    def test_dispatch_enrolled(self, mock_handle_enrolled):
        """Test dispatching to enrolled handler."""
        from nvflare.tool.enrollment.enrollment_cli import handle_enrollment_cmd

        args = argparse.Namespace(enrollment_cmd="enrolled")
        handle_enrollment_cmd(args)
        mock_handle_enrolled.assert_called_once_with(args)

    def test_no_subcommand_exits(self):
        """Test that no subcommand causes exit."""
        from nvflare.tool.enrollment.enrollment_cli import handle_enrollment_cmd

        args = argparse.Namespace(enrollment_cmd=None)
        with pytest.raises(SystemExit):
            handle_enrollment_cmd(args)
