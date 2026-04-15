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

import json
from unittest.mock import MagicMock, patch

import pytest

from nvflare.tool import cli_output


class TestPreflightOutput:
    """Tests for nvflare preflight_check output format."""

    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def test_invalid_package_path_exits_4(self, tmp_path):
        """Non-existent package_path should exit 4."""
        from nvflare.tool.preflight_check import check_packages

        args = MagicMock()
        args.package_path = str(tmp_path / "nonexistent")
        args.output = "json"

        with pytest.raises(SystemExit) as exc_info:
            check_packages(args)
        assert exc_info.value.code == 4

    def test_missing_startup_dir_exits_4(self, tmp_path):
        """Package dir without startup/ subdir should exit 4."""
        from nvflare.tool.preflight_check import check_packages

        pkg_path = tmp_path / "package"
        pkg_path.mkdir()

        args = MagicMock()
        args.package_path = str(pkg_path)
        args.output = "json"

        with pytest.raises(SystemExit) as exc_info:
            check_packages(args)
        assert exc_info.value.code == 4

    def test_all_pass_json_envelope(self, capsys, tmp_path):
        """When all checks pass: stdout is exactly one JSON line, stderr has checker report."""
        from nvflare.tool.preflight_check import check_packages

        pkg_path = tmp_path / "package"
        pkg_path.mkdir()
        (pkg_path / "startup").mkdir()

        import sys

        mock_checker = MagicMock()
        mock_checker.should_be_checked.return_value = True
        mock_checker.check.return_value = 0
        mock_checker.__class__.__name__ = "ServerPackageChecker"
        mock_checker.report = {str(pkg_path): []}
        # Make print_report() write a sentinel to stderr so we can assert routing
        mock_checker.print_report.side_effect = lambda: sys.stderr.write("CHECKER_REPORT\n")

        args = MagicMock()
        args.package_path = str(pkg_path)
        args.output = "json"

        with (
            patch("nvflare.tool.preflight_check.ServerPackageChecker", return_value=mock_checker),
            patch("nvflare.tool.preflight_check.ClientPackageChecker", return_value=mock_checker),
            patch("nvflare.tool.preflight_check.NVFlareConsolePackageChecker", return_value=mock_checker),
        ):
            check_packages(args)

        captured = capsys.readouterr()

        # stdout: exactly one JSON line, nothing else
        stdout_lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert len(stdout_lines) == 1, f"Expected one JSON line on stdout, got: {captured.out!r}"
        data = json.loads(stdout_lines[0])
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert data["data"]["overall"] == "pass"
        assert "checks" in data["data"]
        assert "package" in data["data"]

        # checker report lands on stderr, not stdout
        assert "CHECKER_REPORT" in captured.err
        assert "CHECKER_REPORT" not in captured.out

    def test_fail_exits_1(self, capsys, tmp_path):
        """When a check fails, exits with code 1."""
        from nvflare.tool.preflight_check import check_packages

        pkg_path = tmp_path / "package2"
        pkg_path.mkdir()
        (pkg_path / "startup").mkdir()

        mock_checker = MagicMock()
        mock_checker.should_be_checked.return_value = True
        mock_checker.check.return_value = 1  # fail
        mock_checker.__class__.__name__ = "ServerPackageChecker"

        args = MagicMock()
        args.package_path = str(pkg_path)
        args.output = "json"

        with (
            patch("nvflare.tool.preflight_check.ServerPackageChecker", return_value=mock_checker),
            patch("nvflare.tool.preflight_check.ClientPackageChecker", return_value=mock_checker),
            patch("nvflare.tool.preflight_check.NVFlareConsolePackageChecker", return_value=mock_checker),
        ):
            with pytest.raises(SystemExit) as exc_info:
                check_packages(args)
        assert exc_info.value.code == 1

    def test_per_component_checks(self, capsys, tmp_path):
        """Each checker appears as a separate entry; stdout has only one JSON line."""
        from nvflare.tool.preflight_check import check_packages

        pkg_path = tmp_path / "package3"
        pkg_path.mkdir()
        (pkg_path / "startup").mkdir()

        mock_checker = MagicMock()
        mock_checker.should_be_checked.return_value = True
        mock_checker.check.return_value = 0
        mock_checker.report = {str(pkg_path): []}

        args = MagicMock()
        args.package_path = str(pkg_path)
        args.output = "json"

        with (
            patch("nvflare.tool.preflight_check.ServerPackageChecker", return_value=mock_checker),
            patch("nvflare.tool.preflight_check.ClientPackageChecker", return_value=mock_checker),
            patch("nvflare.tool.preflight_check.NVFlareConsolePackageChecker", return_value=mock_checker),
        ):
            check_packages(args)

        captured = capsys.readouterr()

        # stdout: exactly one JSON line
        stdout_lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert len(stdout_lines) == 1, f"Expected one JSON line on stdout, got: {captured.out!r}"
        data = json.loads(stdout_lines[0])
        checks = data["data"]["checks"]
        # 3 checkers → 3 entries
        assert len(checks) == 3
        for check in checks:
            assert "component" in check
            assert "status" in check
            assert check["status"] in ("pass", "fail")

        # print_report() was called (report is not swallowed)
        assert mock_checker.print_report.called
        # report text must not appear on stdout — stdout is JSON only
        assert "Checking Package" not in captured.out

    def test_preflight_parser_has_schema_flag(self):
        """preflight_check parser should have --schema flag."""
        import argparse

        from nvflare.tool.preflight_check import define_preflight_check_parser

        parser = argparse.ArgumentParser()
        define_preflight_check_parser(parser)
        args = parser.parse_args(["-p", "/some/path", "--schema"])
        assert args.schema is True
