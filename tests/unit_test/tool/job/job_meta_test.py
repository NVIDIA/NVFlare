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

from nvflare.fuel.flare_api.api_spec import AuthenticationError, JobNotFound
from nvflare.tool import cli_output


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


class TestJobMeta:
    """Tests for nvflare job meta command."""

    @pytest.fixture(autouse=True)
    def agent_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _make_args(self, job_id="abc123", output="json"):
        args = MagicMock()
        args.job_id = job_id
        args.output = output
        return args

    def test_meta_success_json(self, capsys):
        """job meta success: returns job metadata in JSON envelope."""
        from nvflare.tool.job.job_cli import cmd_job_meta

        args = self._make_args()
        meta = {"job_id": "abc123", "name": "my_job", "status": "FINISHED_OK"}
        mock_sess = MagicMock()
        mock_sess.get_job_meta.return_value = meta

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            cmd_job_meta(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert data["data"]["job_id"] == "abc123"

    def test_meta_not_found_exits_1(self):
        """job meta JOB_NOT_FOUND exits with code 1."""
        from nvflare.tool.job.job_cli import cmd_job_meta

        args = self._make_args(job_id="notfound")
        mock_sess = MagicMock()
        mock_sess.get_job_meta.side_effect = JobNotFound("job not found")

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_meta(args)
        assert exc_info.value.code == 1

    def test_meta_returns_none_exits_1(self):
        """When get_job_meta returns None, exits with code 1."""
        from nvflare.tool.job.job_cli import cmd_job_meta

        args = self._make_args(job_id="missing")
        mock_sess = MagicMock()
        mock_sess.get_job_meta.return_value = None

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_meta(args)
        assert exc_info.value.code == 1

    def test_meta_authentication_error_propagates(self):
        from nvflare.tool.job.job_cli import cmd_job_meta

        args = self._make_args()
        mock_sess = MagicMock()
        mock_sess.get_job_meta.side_effect = AuthenticationError("bad cert")

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with pytest.raises(AuthenticationError):
                cmd_job_meta(args)

    def test_meta_uses_active_startup_kit_session(self, tmp_path, monkeypatch):
        from nvflare.tool.job.job_cli import cmd_job_meta

        active_admin_dir = _configure_active_startup_kit(tmp_path, monkeypatch)
        args = self._make_args()
        mock_sess = MagicMock()
        mock_sess.get_job_meta.return_value = {"job_id": "abc123", "status": "RUNNING"}

        with patch("nvflare.tool.cli_session.new_secure_session", return_value=mock_sess) as new_secure:
            cmd_job_meta(args)

        assert new_secure.call_args.kwargs["username"] == "admin@nvidia.com"
        assert new_secure.call_args.kwargs["startup_kit_location"] == str(active_admin_dir)

    def test_meta_parser_positional_job_id(self):
        """meta parser should accept positional job_id."""
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_job_cli_parser(subs)

        parser = job_sub_cmd_parser["meta"]
        assert parser is not None
        args = parser.parse_args(["abc123"])
        assert args.job_id == "abc123"

    @pytest.mark.parametrize(
        ("selector", "value"),
        [
            ("--startup-target", "prod"),
            ("--startup_target", "prod"),
            ("--startup_kit", "/tmp/startup"),
        ],
    )
    def test_meta_parser_rejects_old_startup_selectors(self, selector, value):
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_job_cli_parser(subs)

        parser = job_sub_cmd_parser["meta"]
        with pytest.raises(SystemExit):
            parser.parse_args(["abc123", selector, value])

    @pytest.mark.parametrize(
        ("selector", "value", "dest"),
        [
            ("--startup-kit", "/tmp/startup", "startup_kit"),
            ("--kit-id", "prod_admin", "kit_id"),
        ],
    )
    def test_meta_parser_accepts_scoped_startup_selectors(self, selector, value, dest):
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_job_cli_parser(subs)

        parser = job_sub_cmd_parser["meta"]
        args = parser.parse_args(["abc123", selector, value])

        assert getattr(args, dest) == value

    def test_meta_help_and_schema_include_scoped_startup_selectors(self, capsys):
        import argparse

        from nvflare.tool.job.job_cli import cmd_job_meta, def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        def_job_cli_parser(root.add_subparsers())

        help_text = job_sub_cmd_parser["meta"].format_help()
        for token in ("--startup-target", "--startup_target", "--startup_kit"):
            assert token not in help_text
        assert "--startup-kit" in help_text
        assert "--kit-id" in help_text

        with patch("sys.argv", ["nvflare", "job", "meta", "--schema"]):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_meta(MagicMock())

        assert exc_info.value.code == 0
        schema_text = capsys.readouterr().out
        for token in ("--startup-target", "--startup_target", "--startup_kit"):
            assert token not in schema_text
        assert "--startup-kit" in schema_text
        assert "--kit-id" in schema_text
        schema = json.loads(schema_text)
        assert schema["output_modes"] == ["json"]
        assert schema["streaming"] is False
        assert schema["mutating"] is False
        assert schema["idempotent"] is True
        assert schema["retry_token"] == {"supported": False}
