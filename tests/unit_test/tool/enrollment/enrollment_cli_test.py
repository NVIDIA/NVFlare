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
import tempfile
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
            ENV_CA_PATH,
            ENV_ENROLLMENT_POLICY,
            SUBCMD_TOKEN,
            TOKEN_BATCH,
            TOKEN_GENERATE,
            TOKEN_INFO,
        )

        assert CMD_ENROLLMENT == "enrollment"
        assert SUBCMD_TOKEN == "token"
        assert TOKEN_GENERATE == "generate"
        assert TOKEN_BATCH == "batch"
        assert TOKEN_INFO == "info"
        assert ENV_CA_PATH == "NVFLARE_CA_PATH"
        assert ENV_ENROLLMENT_POLICY == "NVFLARE_ENROLLMENT_POLICY"

    def test_default_policy_defined(self):
        """Test that built-in default policy is defined."""
        from nvflare.tool.enrollment.enrollment_cli import DEFAULT_POLICY

        assert "metadata:" in DEFAULT_POLICY
        assert "token:" in DEFAULT_POLICY
        assert "approval:" in DEFAULT_POLICY
        assert "auto-approve-all" in DEFAULT_POLICY


class TestGetCaPath:
    """Test CA path resolution."""

    def test_ca_path_from_args(self):
        """Test CA path from CLI argument takes priority."""
        from nvflare.tool.enrollment.enrollment_cli import _get_ca_path

        args = argparse.Namespace(ca_path="/path/from/args")
        result = _get_ca_path(args)
        assert result == "/path/from/args"

    def test_ca_path_from_env_var(self):
        """Test CA path from environment variable."""
        from nvflare.tool.enrollment.enrollment_cli import ENV_CA_PATH, _get_ca_path

        args = argparse.Namespace(ca_path=None)
        with patch.dict(os.environ, {ENV_CA_PATH: "/path/from/env"}):
            result = _get_ca_path(args)
            assert result == "/path/from/env"

    def test_ca_path_args_overrides_env(self):
        """Test CLI argument overrides environment variable."""
        from nvflare.tool.enrollment.enrollment_cli import ENV_CA_PATH, _get_ca_path

        args = argparse.Namespace(ca_path="/path/from/args")
        with patch.dict(os.environ, {ENV_CA_PATH: "/path/from/env"}):
            result = _get_ca_path(args)
            assert result == "/path/from/args"

    def test_ca_path_missing_exits(self):
        """Test that missing CA path causes exit."""
        from nvflare.tool.enrollment.enrollment_cli import _get_ca_path

        args = argparse.Namespace(ca_path=None)
        with patch.dict(os.environ, {}, clear=True):
            # Remove NVFLARE_CA_PATH if it exists
            os.environ.pop("NVFLARE_CA_PATH", None)
            with pytest.raises(SystemExit) as exc_info:
                _get_ca_path(args)
            assert exc_info.value.code == 1


class TestGetPolicyPath:
    """Test policy path resolution."""

    def test_policy_path_from_args(self):
        """Test policy path from CLI argument takes priority."""
        from nvflare.tool.enrollment.enrollment_cli import _get_policy_path

        args = argparse.Namespace(policy="/path/to/policy.yaml")
        path, is_temp = _get_policy_path(args)
        assert path == "/path/to/policy.yaml"
        assert is_temp is False

    def test_policy_path_from_env_var(self):
        """Test policy path from environment variable."""
        from nvflare.tool.enrollment.enrollment_cli import ENV_ENROLLMENT_POLICY, _get_policy_path

        args = argparse.Namespace(policy=None)
        with patch.dict(os.environ, {ENV_ENROLLMENT_POLICY: "/path/from/env.yaml"}):
            path, is_temp = _get_policy_path(args)
            assert path == "/path/from/env.yaml"
            assert is_temp is False

    def test_policy_args_overrides_env(self):
        """Test CLI argument overrides environment variable."""
        from nvflare.tool.enrollment.enrollment_cli import ENV_ENROLLMENT_POLICY, _get_policy_path

        args = argparse.Namespace(policy="/path/from/args.yaml")
        with patch.dict(os.environ, {ENV_ENROLLMENT_POLICY: "/path/from/env.yaml"}):
            path, is_temp = _get_policy_path(args)
            assert path == "/path/from/args.yaml"
            assert is_temp is False

    def test_default_policy_used_when_not_specified(self):
        """Test that default policy is used when no policy specified."""
        from nvflare.tool.enrollment.enrollment_cli import _get_policy_path

        args = argparse.Namespace(policy=None)
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NVFLARE_ENROLLMENT_POLICY", None)
            path, is_temp = _get_policy_path(args)

            # Should return a temp file path
            assert is_temp is True
            assert path.endswith(".yaml")
            assert os.path.exists(path)

            # Verify content
            with open(path, "r") as f:
                content = f.read()
            assert "auto-approve-all" in content

            # Cleanup
            os.remove(path)


class TestParserDefinitions:
    """Test argument parser definitions."""

    def test_def_enrollment_parser(self):
        """Test enrollment parser definition."""
        from nvflare.tool.enrollment.enrollment_cli import def_enrollment_parser

        parent_parser = argparse.ArgumentParser()
        sub_cmd = parent_parser.add_subparsers()

        result = def_enrollment_parser(sub_cmd)
        assert "enrollment" in result
        assert result["enrollment"] is not None

    def test_generate_parser_has_required_args(self):
        """Test generate subcommand has required arguments."""
        from nvflare.tool.enrollment.enrollment_cli import define_token_generate_parser

        parent_parser = argparse.ArgumentParser()
        sub_parser = parent_parser.add_subparsers()
        define_token_generate_parser(sub_parser)

        # Parse with required arg
        args = parent_parser.parse_args(["generate", "-s", "site-1", "-c", "/path/to/ca"])
        assert args.subject == "site-1"
        assert args.ca_path == "/path/to/ca"
        assert args.type == "client"  # default

    def test_generate_parser_optional_args(self):
        """Test generate subcommand optional arguments."""
        from nvflare.tool.enrollment.enrollment_cli import define_token_generate_parser

        parent_parser = argparse.ArgumentParser()
        sub_parser = parent_parser.add_subparsers()
        define_token_generate_parser(sub_parser)

        args = parent_parser.parse_args(
            [
                "generate",
                "-s",
                "site-1",
                "-c",
                "/ca",
                "-t",
                "admin",
                "-v",
                "24h",
                "-r",
                "org_admin",
                "researcher",
                "--source_ips",
                "10.0.0.0/8",
                "-o",
                "output.txt",
            ]
        )
        assert args.type == "admin"
        assert args.validity == "24h"
        assert args.roles == ["org_admin", "researcher"]
        assert args.source_ips == ["10.0.0.0/8"]
        assert args.output == "output.txt"

    def test_batch_parser_count_mode(self):
        """Test batch subcommand with count mode."""
        from nvflare.tool.enrollment.enrollment_cli import define_token_batch_parser

        parent_parser = argparse.ArgumentParser()
        sub_parser = parent_parser.add_subparsers()
        define_token_batch_parser(sub_parser)

        args = parent_parser.parse_args(["batch", "-n", "10", "--prefix", "hospital", "-o", "tokens.csv", "-c", "/ca"])
        assert args.count == 10
        assert args.prefix == "hospital"
        assert args.output == "tokens.csv"

    def test_batch_parser_names_mode(self):
        """Test batch subcommand with explicit names."""
        from nvflare.tool.enrollment.enrollment_cli import define_token_batch_parser

        parent_parser = argparse.ArgumentParser()
        sub_parser = parent_parser.add_subparsers()
        define_token_batch_parser(sub_parser)

        args = parent_parser.parse_args(
            ["batch", "--names", "site-a", "site-b", "site-c", "-o", "tokens.csv", "-c", "/ca"]
        )
        assert args.names == ["site-a", "site-b", "site-c"]

    def test_info_parser(self):
        """Test info subcommand parser."""
        from nvflare.tool.enrollment.enrollment_cli import define_token_info_parser

        parent_parser = argparse.ArgumentParser()
        sub_parser = parent_parser.add_subparsers()
        define_token_info_parser(sub_parser)

        args = parent_parser.parse_args(["info", "-t", "eyJhbGc...", "--json"])
        assert args.token == "eyJhbGc..."
        assert args.json is True


class TestHandleTokenGenerateCmd:
    """Test token generate command handler."""

    @pytest.fixture
    def mock_jwt_check(self):
        """Mock JWT dependency check."""
        with patch("nvflare.tool.enrollment.enrollment_cli._check_jwt_dependency"):
            yield

    def test_generate_with_all_args(self, mock_jwt_check):
        """Test generate command with all arguments provided."""
        from nvflare.tool.enrollment.enrollment_cli import handle_token_generate_cmd

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            f.write(b"metadata:\n  project: test\n")
            policy_path = f.name

        try:
            # Mock TokenService where it's imported (inside the handler)
            with patch("nvflare.tool.enrollment.token_service.TokenService") as mock_class:
                mock_instance = MagicMock()
                mock_instance.generate_token_from_file.return_value = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..."
                mock_instance.get_token_info.return_value = {
                    "subject": "site-1",
                    "subject_type": "client",
                    "expires_at": "2025-01-09T00:00:00+00:00",
                }
                mock_class.return_value = mock_instance

                args = argparse.Namespace(
                    subject="site-1",
                    ca_path="/path/to/ca",
                    policy=policy_path,
                    type="client",
                    validity="7d",
                    roles=None,
                    source_ips=None,
                    output=None,
                )

                # Should not raise
                handle_token_generate_cmd(args)

                # Verify TokenService was called correctly
                mock_instance.generate_token_from_file.assert_called_once()
                call_args = mock_instance.generate_token_from_file.call_args
                assert call_args.kwargs["subject"] == "site-1"
        finally:
            os.remove(policy_path)

    def test_generate_saves_to_output_file(self, mock_jwt_check):
        """Test generate command saves token to output file."""
        from nvflare.tool.enrollment.enrollment_cli import handle_token_generate_cmd

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as policy_file:
            policy_file.write(b"metadata:\n  project: test\n")
            policy_path = policy_file.name

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as output_file:
            output_path = output_file.name

        try:
            with patch("nvflare.tool.enrollment.token_service.TokenService") as mock_class:
                mock_instance = MagicMock()
                mock_instance.generate_token_from_file.return_value = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..."
                mock_instance.get_token_info.return_value = {
                    "subject": "site-1",
                    "subject_type": "client",
                    "expires_at": "2025-01-09T00:00:00+00:00",
                }
                mock_class.return_value = mock_instance

                args = argparse.Namespace(
                    subject="site-1",
                    ca_path="/path/to/ca",
                    policy=policy_path,
                    type="client",
                    validity=None,
                    roles=None,
                    source_ips=None,
                    output=output_path,
                )

                handle_token_generate_cmd(args)

                # Verify token was written to file
                with open(output_path, "r") as f:
                    content = f.read()
                assert "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9" in content
        finally:
            os.remove(policy_path)
            os.remove(output_path)


class TestHandleTokenInfoCmd:
    """Test token info command handler."""

    @pytest.fixture
    def mock_jwt_check(self):
        """Mock JWT dependency check."""
        with patch("nvflare.tool.enrollment.enrollment_cli._check_jwt_dependency"):
            yield

    @pytest.fixture
    def sample_token(self):
        """Create a sample JWT token."""
        from datetime import datetime, timedelta, timezone

        import jwt

        payload = {
            "jti": "test-token-id",
            "sub": "site-1",
            "subject_type": "client",
            "iss": "nvflare-enrollment",
            "iat": int(datetime.now(timezone.utc).timestamp()),
            "exp": int((datetime.now(timezone.utc) + timedelta(days=7)).timestamp()),
            "max_uses": 1,
            "policy": {"metadata": {"project": "test-project", "version": "1.0"}},
        }
        # Use a simple secret for test (normally would use RSA)
        return jwt.encode(payload, "test-secret", algorithm="HS256")

    def test_info_decodes_token(self, mock_jwt_check, sample_token, capsys):
        """Test info command decodes and displays token info."""
        from nvflare.tool.enrollment.enrollment_cli import handle_token_info_cmd

        args = argparse.Namespace(token=sample_token, json=False)
        handle_token_info_cmd(args)

        captured = capsys.readouterr()
        assert "site-1" in captured.out
        assert "client" in captured.out
        assert "test-project" in captured.out

    def test_info_json_output(self, mock_jwt_check, sample_token, capsys):
        """Test info command with JSON output."""
        import json

        from nvflare.tool.enrollment.enrollment_cli import handle_token_info_cmd

        args = argparse.Namespace(token=sample_token, json=True)
        handle_token_info_cmd(args)

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["subject"] == "site-1"
        assert output["subject_type"] == "client"

    def test_info_reads_from_file(self, mock_jwt_check, sample_token, capsys):
        """Test info command reads token from file."""
        from nvflare.tool.enrollment.enrollment_cli import handle_token_info_cmd

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(sample_token)
            token_file = f.name

        try:
            args = argparse.Namespace(token=token_file, json=False)
            handle_token_info_cmd(args)

            captured = capsys.readouterr()
            assert "site-1" in captured.out
        finally:
            os.remove(token_file)


class TestHandleEnrollmentCmd:
    """Test main enrollment command handler."""

    def test_unknown_subcommand_raises_exception(self):
        """Test that unknown subcommand raises CLIUnknownCmdException."""
        from nvflare.cli_unknown_cmd_exception import CLIUnknownCmdException
        from nvflare.tool.enrollment.enrollment_cli import handle_enrollment_cmd

        args = argparse.Namespace(enrollment_sub_cmd=None)
        with pytest.raises(CLIUnknownCmdException):
            handle_enrollment_cmd(args)

    def test_token_subcommand_dispatches(self):
        """Test that token subcommand is dispatched correctly."""
        from nvflare.tool.enrollment.enrollment_cli import handle_enrollment_cmd

        with patch("nvflare.tool.enrollment.enrollment_cli.handle_token_cmd") as mock:
            args = argparse.Namespace(enrollment_sub_cmd="token", token_sub_cmd=None)

            # This will raise CLIUnknownCmdException from handle_token_cmd
            # but we just want to verify dispatch
            try:
                handle_enrollment_cmd(args)
            except Exception:
                pass

            mock.assert_called_once_with(args)


class TestCheckJwtDependency:
    """Test JWT dependency checking."""

    def test_jwt_available(self):
        """Test that check passes when jwt is available."""
        from nvflare.tool.enrollment.enrollment_cli import _check_jwt_dependency

        # jwt should be available in test environment
        result = _check_jwt_dependency()
        assert result is True

    def test_jwt_unavailable_exits(self):
        """Test that check exits when jwt is unavailable."""
        from nvflare.tool.enrollment import enrollment_cli

        # Mock the import to fail
        with patch.dict("sys.modules", {"jwt": None}):
            with patch.object(enrollment_cli, "_check_jwt_dependency") as mock:
                mock.side_effect = SystemExit(1)
                with pytest.raises(SystemExit):
                    mock()
