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


class TestPreflightOutput:
    """Tests for nvflare preflight_check output format."""

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
        """When all checks pass, JSON envelope with overall=pass."""
        from nvflare.tool.preflight_check import check_packages

        pkg_path = tmp_path / "package"
        pkg_path.mkdir()
        (pkg_path / "startup").mkdir()

        mock_checker = MagicMock()
        mock_checker.should_be_checked.return_value = True
        mock_checker.check.return_value = 0
        mock_checker.__class__.__name__ = "ServerPackageChecker"

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
        data = json.loads(captured.out)
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"
        assert data["data"]["overall"] == "pass"
        assert "checks" in data["data"]
        assert "package" in data["data"]

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
        """Each checker should appear as a separate entry in checks list."""
        from nvflare.tool.preflight_check import check_packages

        pkg_path = tmp_path / "package3"
        pkg_path.mkdir()
        (pkg_path / "startup").mkdir()

        mock_checker = MagicMock()
        mock_checker.should_be_checked.return_value = True
        mock_checker.check.return_value = 0

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
        data = json.loads(captured.out)
        checks = data["data"]["checks"]
        # 3 checkers → 3 entries
        assert len(checks) == 3
        for check in checks:
            assert "component" in check
            assert "status" in check
            assert check["status"] in ("pass", "fail")

    def test_preflight_parser_has_schema_flag(self):
        """preflight_check parser should have --schema flag."""
        import argparse

        from nvflare.tool.preflight_check import define_preflight_check_parser

        parser = argparse.ArgumentParser()
        define_preflight_check_parser(parser)
        args = parser.parse_args(["-p", "/some/path", "--schema"])
        assert args.schema is True
