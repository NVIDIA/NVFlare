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

"""Tests for submit_job's _resolve_job_folder auto-discovery logic."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest


def _make_valid_job(root: str) -> str:
    """Create a minimal valid job structure at root and return root."""
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "meta.json"), "w") as f:
        f.write("{}")
    config_dir = os.path.join(root, "app", "config")
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, "config_fed_server.json"), "w") as f:
        f.write("{}")
    return root


class TestResolveJobFolder:
    """_resolve_job_folder inside submit_job selects the correct folder."""

    def _invoke_submit(self, job_folder, capsys, monkeypatch):
        """Run submit_job up to (but not including) the session connection step."""
        from unittest.mock import MagicMock, patch

        from nvflare.tool import cli_output

        monkeypatch.setattr(cli_output, "_output_format", "json")

        args = MagicMock()
        args.job_folder = job_folder
        args.study = "default"
        args.debug = False
        args.config_file = None
        args.target = None
        args.startup_kit = None

        with patch(
            "nvflare.tool.job.job_cli.find_admin_user_and_dir", return_value=("admin@nvidia.com", "/tmp/startup")
        ):
            with patch("nvflare.tool.job.job_cli.get_app_dirs_from_job_folder", return_value=[]):
                with patch("nvflare.tool.job.job_cli.prepare_job_config"):
                    with patch("nvflare.tool.job.job_cli.internal_submit_job"):
                        with patch("sys.argv", ["nvflare", "job", "submit", "-j", job_folder]):
                            from nvflare.tool.job.job_cli import submit_job

                            submit_job(args)

        return capsys.readouterr()

    def test_valid_job_at_root_uses_root(self, capsys, monkeypatch, tmp_path):
        """Folder with meta.json + app/config/config_fed_server.json → used directly."""
        job_dir = str(tmp_path / "myjob")
        _make_valid_job(job_dir)

        captured = self._invoke_submit(job_dir, capsys, monkeypatch)
        # Should NOT print "Using job folder: <subdir>" — no auto-discovery
        assert "Using job folder" not in captured.out + captured.err

    def test_single_valid_subdir_is_auto_discovered(self, capsys, monkeypatch, tmp_path):
        """Parent with one valid subdir → subdir selected silently in JSON mode."""
        parent = str(tmp_path / "parent")
        os.makedirs(parent, exist_ok=True)
        child = os.path.join(parent, "myjob")
        _make_valid_job(child)

        captured = self._invoke_submit(parent, capsys, monkeypatch)
        assert "Using job folder" not in captured.out + captured.err

    def test_single_invalid_subdir_falls_back_to_root(self, capsys, monkeypatch, tmp_path):
        """Parent with one subdir that lacks meta → falls back to parent (no print)."""
        parent = str(tmp_path / "parent")
        child = os.path.join(parent, "notajob")
        os.makedirs(child, exist_ok=True)

        captured = self._invoke_submit(parent, capsys, monkeypatch)
        assert "Using job folder" not in captured.out + captured.err

    def test_multiple_subdirs_falls_back_to_root(self, capsys, monkeypatch, tmp_path):
        """Parent with multiple subdirs → no auto-discovery (ambiguous)."""
        parent = str(tmp_path / "parent")
        for name in ("job_a", "job_b"):
            _make_valid_job(os.path.join(parent, name))

        captured = self._invoke_submit(parent, capsys, monkeypatch)
        assert "Using job folder" not in captured.out + captured.err

    def test_hidden_subdirs_ignored(self, capsys, monkeypatch, tmp_path):
        """Hidden dirs (starting with '.') are ignored during subdir scan."""
        parent = str(tmp_path / "parent")
        # One hidden dir + one valid job dir → auto-discover the valid one
        _make_valid_job(os.path.join(parent, "myjob"))
        os.makedirs(os.path.join(parent, ".git"), exist_ok=True)

        captured = self._invoke_submit(parent, capsys, monkeypatch)
        assert "Using job folder" not in captured.out + captured.err

    def test_yaml_meta_is_recognized(self, capsys, monkeypatch, tmp_path):
        """meta.yaml is also a valid job meta file."""
        job_dir = str(tmp_path / "myjob")
        os.makedirs(job_dir)
        with open(os.path.join(job_dir, "meta.yaml"), "w") as f:
            f.write("{}")
        config_dir = os.path.join(job_dir, "app", "config")
        os.makedirs(config_dir)
        with open(os.path.join(config_dir, "config_fed_server.yml"), "w") as f:
            f.write("{}")

        captured = self._invoke_submit(job_dir, capsys, monkeypatch)
        assert "Using job folder" not in captured.out + captured.err

    def test_invalid_job_folder_returns_structured_error_without_help(self, capsys, monkeypatch):
        """Invalid job folder should emit INVALID_ARGS and usage help in human mode."""
        from nvflare.tool import cli_output
        from nvflare.tool.job.job_cli import submit_job

        monkeypatch.setattr(cli_output, "_output_format", "txt")

        args = MagicMock()
        args.job_folder = "/no/such/job"
        args.study = "default"
        args.debug = False
        args.config_file = None
        args.target = None
        args.startup_kit = None

        with patch("sys.argv", ["nvflare", "job", "submit", "-j", "/no/such/job"]):
            with pytest.raises(SystemExit) as exc_info:
                submit_job(args)

        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        assert "INVALID_ARGS" in captured.err
        assert "invalid job folder" in captured.err
        assert "usage:" in captured.err

    def test_submit_resolves_target_and_startup_kit_args(self, monkeypatch, tmp_path):
        from nvflare.tool.job.job_cli import submit_job

        job_dir = str(tmp_path / "myjob")
        _make_valid_job(job_dir)

        args = MagicMock()
        args.job_folder = job_dir
        args.study = "default"
        args.debug = False
        args.config_file = None
        args.startup_target = "prod"
        args.startup_kit = None

        with patch(
            "nvflare.tool.job.job_cli.find_admin_user_and_dir", return_value=("admin@nvidia.com", "/tmp/startup")
        ) as mock_find:
            with patch("nvflare.tool.job.job_cli.get_app_dirs_from_job_folder", return_value=[]):
                with patch("nvflare.tool.job.job_cli.prepare_job_config"):
                    with patch("nvflare.tool.job.job_cli.internal_submit_job"):
                        with patch("sys.argv", ["nvflare", "job", "submit", "-j", job_dir, "--startup-target", "prod"]):
                            submit_job(args)

        _, kwargs = mock_find.call_args
        assert kwargs["target"] == "prod"
        assert kwargs["startup_kit_dir"] is None

    def test_find_admin_user_and_dir_requires_admin_startup_kit_dir(self, tmp_path):
        from nvflare.tool.job.job_cli import find_admin_user_and_dir

        prod_root = tmp_path / "prod_00"
        prod_root.mkdir()

        with pytest.raises(ValueError) as exc_info:
            find_admin_user_and_dir(startup_kit_dir=str(prod_root))

        assert "admin startup kit directory" in str(exc_info.value)

    def test_find_admin_user_and_dir_accepts_admin_dir(self, tmp_path):
        from nvflare.tool.job.job_cli import find_admin_user_and_dir

        admin_dir = tmp_path / "admin@nvidia.com"
        startup_dir = admin_dir / "startup"
        startup_dir.mkdir(parents=True)
        (startup_dir / "fed_admin.json").write_text('{"admin": {"username": "admin@nvidia.com"}}')

        with patch("nvflare.tool.job.job_cli.get_startup_kit_dir_for_target", return_value=str(admin_dir)):
            username, resolved = find_admin_user_and_dir(startup_kit_dir=str(admin_dir))

        assert username == "admin@nvidia.com"
        assert resolved == str(admin_dir)

    def test_find_admin_user_and_dir_accepts_startup_subdir(self, tmp_path):
        from nvflare.tool.job.job_cli import find_admin_user_and_dir

        admin_dir = tmp_path / "admin@nvidia.com"
        startup_dir = admin_dir / "startup"
        startup_dir.mkdir(parents=True)
        (startup_dir / "fed_admin.json").write_text('{"admin": {"username": "admin@nvidia.com"}}')

        with patch("nvflare.tool.job.job_cli.get_startup_kit_dir_for_target", return_value=str(startup_dir)):
            username, resolved = find_admin_user_and_dir(startup_kit_dir=str(startup_dir))

        assert username == "admin@nvidia.com"
        assert resolved == str(admin_dir)


def test_get_session_missing_startup_kit_emits_startup_kit_missing(capsys, monkeypatch):
    from nvflare.tool import cli_output
    from nvflare.tool.job.job_cli import _get_session

    monkeypatch.setattr(cli_output, "_output_format", "json")

    with patch(
        "nvflare.tool.job.job_cli.find_admin_user_and_dir",
        side_effect=ValueError("no startup kit configured"),
    ):
        with pytest.raises(SystemExit) as exc_info:
            _get_session()

    assert exc_info.value.code == 2
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["error_code"] == "STARTUP_KIT_MISSING"


def test_get_session_missing_startup_kit_still_exits_when_output_error_is_mocked():
    from nvflare.tool.job.job_cli import _get_session

    with patch(
        "nvflare.tool.job.job_cli.find_admin_user_and_dir",
        side_effect=ValueError("startup kit directory /tmp/missing must be an admin startup kit directory"),
    ):
        with patch("nvflare.tool.cli_output.output_error") as output_error:
            with pytest.raises(SystemExit) as exc_info:
                _get_session()

    assert exc_info.value.code == 2
    output_error.assert_called_once()


def test_get_session_forwards_startup_args():
    from nvflare.tool.job.job_cli import _get_session

    args = MagicMock()
    args.startup_kit = "/tmp/custom-startup"
    args.startup_target = "prod"

    with patch(
        "nvflare.tool.job.job_cli.find_admin_user_and_dir",
        return_value=("admin@nvidia.com", "/tmp/custom-startup"),
    ) as find_admin:
        with patch("nvflare.tool.job.job_cli.new_cli_session") as new_session:
            _get_session(args=args, study="default")

    _, kwargs = find_admin.call_args
    assert kwargs["startup_kit_dir"] == "/tmp/custom-startup"
    assert kwargs["target"] == "prod"
    assert new_session.called
