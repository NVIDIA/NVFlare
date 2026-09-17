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
from argparse import Namespace
from contextlib import ExitStack, contextmanager
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


def _configure_active_startup_kit(tmp_path, monkeypatch):
    home = tmp_path / "home"
    admin_dir = tmp_path / "active-admin"
    startup_dir = admin_dir / "startup"
    startup_dir.mkdir(parents=True)
    (startup_dir / "fed_admin.json").write_text('{"admin": {"username": "admin@nvidia.com"}}', encoding="utf-8")
    (startup_dir / "client.crt").write_text("cert", encoding="utf-8")
    (startup_dir / "rootCA.pem").write_text("root", encoding="utf-8")

    config_dir = home / ".nvflare"
    config_dir.mkdir(parents=True)
    (config_dir / "config.conf").write_text(
        f"""
        version = 2
        startup_kits {{
          active = "admin@nvidia.com"
          entries {{
            "admin@nvidia.com" = "{admin_dir}"
          }}
        }}
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)
    return admin_dir


@contextmanager
def _assert_no_active_session_created():
    with ExitStack() as stack:
        stack.enter_context(patch("nvflare.tool.cli_session.new_secure_session", side_effect=AssertionError))
        stack.enter_context(
            patch("nvflare.tool.cli_session.new_active_cli_session", side_effect=AssertionError, create=True)
        )
        stack.enter_context(
            patch("nvflare.tool.job.job_cli.new_active_cli_session", side_effect=AssertionError, create=True)
        )
        stack.enter_context(
            patch("nvflare.tool.job.job_cli.find_admin_user_and_dir", side_effect=AssertionError, create=True)
        )
        yield


class TestResolveJobFolder:
    """_resolve_job_folder inside submit_job selects the correct folder."""

    def _invoke_submit(self, job_folder, capsys, monkeypatch, tmp_path):
        """Run submit_job up to (but not including) the session connection step."""
        from unittest.mock import patch

        from nvflare.tool import cli_output

        monkeypatch.setattr(cli_output, "_output_format", "json")

        args = Namespace(job_folder=job_folder, study="default", debug=False, config_file=None)

        with patch("nvflare.tool.job.job_cli.get_app_dirs_from_job_folder", return_value=[]):
            with patch("nvflare.tool.job.job_cli.prepare_job_config"):
                with patch("nvflare.tool.job.job_cli.internal_submit_job") as submit:
                    with patch("sys.argv", ["nvflare", "job", "submit", "-j", job_folder]):
                        from nvflare.tool.job.job_cli import submit_job

                        submit_job(args)

        submit.assert_called_once()
        assert submit.call_args.args[0] is None
        assert submit.call_args.args[1] is None

        return capsys.readouterr()

    def test_valid_job_at_root_uses_root(self, capsys, monkeypatch, tmp_path):
        """Folder with meta.json + app/config/config_fed_server.json → used directly."""
        job_dir = str(tmp_path / "myjob")
        _make_valid_job(job_dir)

        captured = self._invoke_submit(job_dir, capsys, monkeypatch, tmp_path)
        # Should NOT print "Using job folder: <subdir>" — no auto-discovery
        assert "Using job folder" not in captured.out + captured.err

    def test_single_valid_subdir_is_auto_discovered(self, capsys, monkeypatch, tmp_path):
        """Parent with one valid subdir → subdir selected silently in JSON mode."""
        parent = str(tmp_path / "parent")
        os.makedirs(parent, exist_ok=True)
        child = os.path.join(parent, "myjob")
        _make_valid_job(child)

        captured = self._invoke_submit(parent, capsys, monkeypatch, tmp_path)
        assert "Using job folder" not in captured.out + captured.err

    def test_single_invalid_subdir_falls_back_to_root(self, capsys, monkeypatch, tmp_path):
        """Parent with one subdir that lacks meta → falls back to parent (no print)."""
        parent = str(tmp_path / "parent")
        child = os.path.join(parent, "notajob")
        os.makedirs(child, exist_ok=True)

        captured = self._invoke_submit(parent, capsys, monkeypatch, tmp_path)
        assert "Using job folder" not in captured.out + captured.err

    def test_multiple_subdirs_falls_back_to_root(self, capsys, monkeypatch, tmp_path):
        """Parent with multiple subdirs → no auto-discovery (ambiguous)."""
        parent = str(tmp_path / "parent")
        for name in ("job_a", "job_b"):
            _make_valid_job(os.path.join(parent, name))

        captured = self._invoke_submit(parent, capsys, monkeypatch, tmp_path)
        assert "Using job folder" not in captured.out + captured.err

    def test_hidden_subdirs_ignored(self, capsys, monkeypatch, tmp_path):
        """Hidden dirs (starting with '.') are ignored during subdir scan."""
        parent = str(tmp_path / "parent")
        # One hidden dir + one valid job dir → auto-discover the valid one
        _make_valid_job(os.path.join(parent, "myjob"))
        os.makedirs(os.path.join(parent, ".git"), exist_ok=True)

        captured = self._invoke_submit(parent, capsys, monkeypatch, tmp_path)
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

        captured = self._invoke_submit(job_dir, capsys, monkeypatch, tmp_path)
        assert "Using job folder" not in captured.out + captured.err

    def test_invalid_job_folder_returns_structured_error_without_help(self, capsys, monkeypatch):
        """Invalid job folder should emit INVALID_ARGS and usage help in human mode."""
        from nvflare.tool import cli_output
        from nvflare.tool.job.job_cli import submit_job

        monkeypatch.setattr(cli_output, "_output_format", "txt")

        args = Namespace(job_folder="/no/such/job", study="default", debug=False, config_file=None)

        with patch("sys.argv", ["nvflare", "job", "submit", "-j", "/no/such/job"]):
            with pytest.raises(SystemExit) as exc_info:
                submit_job(args)

        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        assert "INVALID_ARGS" in captured.err
        assert "invalid job folder" in captured.err
        assert "usage:" in captured.err

    def test_submit_uses_active_startup_kit_for_server_connection(self, monkeypatch, tmp_path):
        from nvflare.tool.job.job_cli import submit_job

        job_dir = str(tmp_path / "myjob")
        _make_valid_job(job_dir)
        active_admin_dir = _configure_active_startup_kit(tmp_path, monkeypatch)

        args = Namespace(job_folder=job_dir, study="default", debug=False, config_file=None)

        with patch("nvflare.tool.job.job_cli.get_app_dirs_from_job_folder", return_value=[]):
            with patch("nvflare.tool.job.job_cli.prepare_job_config"):
                fake_session = MagicMock()
                fake_session.submit_job.return_value = "job-123"
                with patch("nvflare.tool.cli_session.new_secure_session", return_value=fake_session) as new_secure:
                    with patch("sys.argv", ["nvflare", "job", "submit", "-j", job_dir]):
                        submit_job(args)

        assert new_secure.call_args.kwargs["username"] == "admin@nvidia.com"
        assert new_secure.call_args.kwargs["startup_kit_location"] == str(active_admin_dir)
        fake_session.submit_job.assert_called_once()


def test_get_session_uses_active_startup_kit_config(tmp_path, monkeypatch):
    from nvflare.tool.job.job_cli import _get_session

    active_admin_dir = _configure_active_startup_kit(tmp_path, monkeypatch)
    fake_session = MagicMock()

    with patch("nvflare.tool.cli_session.new_secure_session", return_value=fake_session) as new_secure:
        sess = _get_session(study="default")

    assert sess is fake_session
    assert new_secure.call_args.kwargs["username"] == "admin@nvidia.com"
    assert new_secure.call_args.kwargs["startup_kit_location"] == str(active_admin_dir)


def test_get_session_missing_startup_kit_emits_startup_kit_missing(capsys, monkeypatch, tmp_path):
    from nvflare.tool import cli_output
    from nvflare.tool.job.job_cli import _get_session

    monkeypatch.setattr(cli_output, "_output_format", "json")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        _get_session()

    assert exc_info.value.code == 2
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["error_code"] == "STARTUP_KIT_MISSING"


def test_get_session_missing_startup_kit_still_exits_when_output_error_is_mocked(monkeypatch, tmp_path):
    from nvflare.tool.job.job_cli import _get_session

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)

    with patch("nvflare.tool.cli_output.output_error") as output_error:
        with pytest.raises(SystemExit) as exc_info:
            _get_session()

    assert exc_info.value.code == 2
    output_error.assert_called_once()


def test_list_templates_does_not_require_active_startup_kit(monkeypatch):
    from nvflare.tool.job import job_cli

    args = MagicMock()
    args.job_sub_cmd = "list-templates"
    args.job_templates_dir = None
    args.debug = False

    with _assert_no_active_session_created():
        monkeypatch.setattr(job_cli, "find_job_templates_location", lambda _path=None: "/tmp/templates")
        monkeypatch.setattr(job_cli, "build_job_template_indices", lambda _path: MagicMock())
        monkeypatch.setattr(job_cli, "display_available_templates", lambda _conf: None)
        monkeypatch.setattr(job_cli, "update_job_templates_dir", lambda _path: None)
        job_cli.list_templates(args)


def test_show_variables_does_not_require_active_startup_kit(monkeypatch, tmp_path):
    from nvflare.tool.job import job_cli

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    args = MagicMock()
    args.job_sub_cmd = "show-variables"
    args.job_folder = str(job_dir)
    args.debug = False

    with _assert_no_active_session_created():
        monkeypatch.setattr(job_cli, "get_app_dirs_from_job_folder", lambda _path: [])
        monkeypatch.setattr(job_cli, "build_config_file_indices", lambda _path, _apps: {})
        monkeypatch.setattr(job_cli, "filter_indices", lambda app_indices_configs=None, **_kwargs: {})
        monkeypatch.setattr(job_cli, "display_template_variables", lambda _job_folder, _values: None)
        job_cli.show_variables(args)


def test_create_job_does_not_require_active_startup_kit(monkeypatch, tmp_path):
    from nvflare.tool.job import job_cli

    args = MagicMock()
    args.job_sub_cmd = "create"
    args.job_folder = str(tmp_path / "job")
    args.template = str(tmp_path / "template")
    args.script_dir = None
    args.force = False
    args.debug = False

    with _assert_no_active_session_created():
        monkeypatch.setattr(job_cli, "get_src_template", lambda _args: args.template)
        monkeypatch.setattr(job_cli, "get_app_dirs_from_template", lambda _template: [])
        monkeypatch.setattr(job_cli, "prepare_job_folder", lambda _args: None)
        monkeypatch.setattr(job_cli, "prepare_app_dirs", lambda _folder, _apps: [])
        monkeypatch.setattr(job_cli, "prepare_app_scripts", lambda _folder, _dirs, _args: None)
        monkeypatch.setattr(job_cli, "get_config_dirs", lambda _folder, _apps: [])
        monkeypatch.setattr(job_cli.ConfigFactory, "search_config_format", lambda *_args, **_kwargs: (None, None))
        monkeypatch.setattr(job_cli.shutil, "copytree", lambda *args, **kwargs: None)
        monkeypatch.setattr(job_cli, "remove_extra_files", lambda _path: None)
        monkeypatch.setattr(job_cli, "prepare_meta_config", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(job_cli, "prepare_job_config", lambda *_args, **_kwargs: {})
        monkeypatch.setattr(job_cli, "display_template_variables", lambda _job_folder, _values: None)
        job_cli.create_job(args)
