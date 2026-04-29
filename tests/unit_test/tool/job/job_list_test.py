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

from nvflare.fuel.flare_api.api_spec import AuthenticationError
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


def _make_admin_startup_kit(parent, username):
    admin_dir = parent / username
    startup_dir = admin_dir / "startup"
    startup_dir.mkdir(parents=True)
    (startup_dir / "fed_admin.json").write_text(f'{{"admin": {{"username": "{username}"}}}}', encoding="utf-8")
    (startup_dir / "client.crt").write_text("cert", encoding="utf-8")
    (startup_dir / "rootCA.pem").write_text("root", encoding="utf-8")
    return admin_dir


class TestJobList:
    """Tests for nvflare job list command."""

    @pytest.fixture(autouse=True)
    def agent_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _make_args(self, **kwargs):
        args = MagicMock()
        args.name = kwargs.get("name", None)
        args.id = kwargs.get("id", None)
        args.reverse = kwargs.get("reverse", False)
        args.max = kwargs.get("max", None)
        args.study = kwargs.get("study", "default")
        args.submit_token = kwargs.get("submit_token", None)
        return args

    def test_list_json_envelope(self, capsys):
        """job list success: JSON envelope wraps job list."""
        from nvflare.tool.cli_output import output_ok

        jobs = [{"job_id": "abc123", "name": "test_job", "status": "FINISHED_OK"}]
        output_ok(jobs)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert isinstance(data["data"], list)
        assert data["data"][0]["job_id"] == "abc123"

    def test_list_with_name_filter(self):
        """name filter is passed to list_jobs."""
        from nvflare.tool.job.job_cli import cmd_job_list

        args = self._make_args(name="cifar")
        mock_sess = MagicMock()
        mock_sess.list_jobs.return_value = []

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            cmd_job_list(args)

        mock_sess.list_jobs.assert_called_once_with(
            name_prefix="cifar", id_prefix=None, reverse=False, limit=None, submit_token=None
        )

    def test_list_with_reverse_flag(self):
        """reverse flag is passed to list_jobs."""
        from nvflare.tool.job.job_cli import cmd_job_list

        args = self._make_args(reverse=True)
        mock_sess = MagicMock()
        mock_sess.list_jobs.return_value = []

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            cmd_job_list(args)

        mock_sess.list_jobs.assert_called_once_with(
            name_prefix=None, id_prefix=None, reverse=True, limit=None, submit_token=None
        )

    def _init_parsers(self):
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_job_cli_parser(subs)

    def test_list_parser_study_field(self):
        """list parser should have --study argument."""
        self._init_parsers()
        from nvflare.tool.job.job_cli import job_sub_cmd_parser

        parser = job_sub_cmd_parser["list"]
        assert parser is not None
        args = parser.parse_args(["--study", "my_study"])
        assert args.study == "my_study"

    def test_list_parser_accepts_submit_token(self):
        self._init_parsers()
        from nvflare.tool.job.job_cli import job_sub_cmd_parser

        parser = job_sub_cmd_parser["list"]
        args = parser.parse_args(["--submit-token", "retry.01:A_b-c"])

        assert args.submit_token == "retry.01:A_b-c"

    @pytest.mark.parametrize("token", ["", "bad token", "bad/token", "x" * 129])
    def test_list_parser_rejects_invalid_submit_token(self, token):
        self._init_parsers()
        from nvflare.tool.job.job_cli import job_sub_cmd_parser

        parser = job_sub_cmd_parser["list"]
        with pytest.raises(SystemExit):
            parser.parse_args(["--submit-token", token])

    @pytest.mark.parametrize(
        ("selector", "value"),
        [
            ("--startup-target", "prod"),
            ("--startup_target", "prod"),
            ("--startup_kit", "/tmp/startup"),
        ],
    )
    def test_list_parser_rejects_old_startup_selectors(self, selector, value):
        self._init_parsers()
        from nvflare.tool.job.job_cli import job_sub_cmd_parser

        parser = job_sub_cmd_parser["list"]
        with pytest.raises(SystemExit):
            parser.parse_args([selector, value])

    @pytest.mark.parametrize(
        ("selector", "value", "dest"),
        [
            ("--startup-kit", "/tmp/startup", "startup_kit"),
            ("--kit-id", "prod_admin", "kit_id"),
        ],
    )
    def test_list_parser_accepts_scoped_startup_selectors(self, selector, value, dest):
        self._init_parsers()
        from nvflare.tool.job.job_cli import job_sub_cmd_parser

        parser = job_sub_cmd_parser["list"]
        args = parser.parse_args([selector, value])

        assert getattr(args, dest) == value

    def test_list_help_and_schema_include_scoped_startup_selectors(self, capsys):
        self._init_parsers()
        from nvflare.tool.job.job_cli import cmd_job_list, job_sub_cmd_parser

        help_text = job_sub_cmd_parser["list"].format_help()
        for token in ("--startup-target", "--startup_target", "--startup_kit"):
            assert token not in help_text
        assert "--startup-kit" in help_text
        assert "--kit-id" in help_text

        with patch("sys.argv", ["nvflare", "job", "list", "--schema"]):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_list(MagicMock())

        assert exc_info.value.code == 0
        schema_text = capsys.readouterr().out
        for token in ("--startup-target", "--startup_target", "--startup_kit"):
            assert token not in schema_text
        assert "--startup-kit" in schema_text
        assert "--kit-id" in schema_text

    def test_list_forwards_named_study_to_session(self):
        from nvflare.tool.job.job_cli import cmd_job_list

        args = self._make_args(study="cancer_research")
        mock_sess = MagicMock()
        mock_sess.list_jobs.return_value = []

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess) as get_session:
            cmd_job_list(args)

        assert get_session.call_args.kwargs["study"] == "cancer_research"

    def test_list_forwards_submit_token_to_session(self):
        from nvflare.tool.job.job_cli import cmd_job_list

        args = self._make_args(submit_token="retry-1")
        mock_sess = MagicMock()
        mock_sess.list_jobs.return_value = []

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            cmd_job_list(args)

        mock_sess.list_jobs.assert_called_once_with(
            name_prefix=None, id_prefix=None, reverse=False, limit=None, submit_token="retry-1"
        )

    def test_list_uses_active_startup_kit_session(self, tmp_path, monkeypatch):
        from nvflare.tool.job.job_cli import cmd_job_list

        active_admin_dir = _configure_active_startup_kit(tmp_path, monkeypatch)
        args = self._make_args()
        mock_sess = MagicMock()
        mock_sess.list_jobs.return_value = []

        with patch("nvflare.tool.cli_session.new_secure_session", return_value=mock_sess) as new_secure:
            cmd_job_list(args)

        assert new_secure.call_args.kwargs["username"] == "admin@nvidia.com"
        assert new_secure.call_args.kwargs["startup_kit_location"] == str(active_admin_dir)

    def test_list_uses_scoped_startup_kit_session(self, tmp_path, monkeypatch):
        from nvflare.tool.job.job_cli import cmd_job_list

        scoped_admin_dir = _make_admin_startup_kit(tmp_path, "scoped@nvidia.com")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)
        args = self._make_args()
        args.startup_kit = str(scoped_admin_dir)
        args.kit_id = None
        mock_sess = MagicMock()
        mock_sess.list_jobs.return_value = []

        with patch("nvflare.tool.cli_session.new_secure_session", return_value=mock_sess) as new_secure:
            cmd_job_list(args)

        assert new_secure.call_args.kwargs["username"] == "scoped@nvidia.com"
        assert new_secure.call_args.kwargs["startup_kit_location"] == str(scoped_admin_dir)

    def test_study_field_injected_in_each_job(self, capsys):
        """cmd_job_list injects study field into each job entry when missing."""
        import json

        from nvflare.tool.job.job_cli import cmd_job_list

        args = self._make_args(study="my_study")
        mock_sess = MagicMock()
        mock_sess.list_jobs.return_value = [
            {"job_id": "id1", "name": "job1"},
            {"job_id": "id2", "name": "job2", "study": "existing_study"},
        ]

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            cmd_job_list(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        jobs = data["data"]
        # First job should have study injected
        assert jobs[0]["study"] == "my_study"
        # Second job already had a study field — should be preserved
        assert jobs[1]["study"] == "existing_study"

    def test_study_field_present_in_all_jobs(self, capsys):
        """All jobs returned from list must contain a study field."""
        import json

        from nvflare.tool.job.job_cli import cmd_job_list

        args = self._make_args(study="default")
        mock_sess = MagicMock()
        mock_sess.list_jobs.return_value = [
            {"job_id": "id1"},
            {"job_id": "id2"},
            {"job_id": "id3"},
        ]

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            cmd_job_list(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        for job in data["data"]:
            assert "study" in job

    def test_authentication_error_propagates(self):
        from nvflare.tool.job.job_cli import cmd_job_list

        args = self._make_args()
        mock_sess = MagicMock()
        mock_sess.list_jobs.side_effect = AuthenticationError("bad cert")

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with pytest.raises(AuthenticationError):
                cmd_job_list(args)

    def test_schema_flag_prints_json_and_exits_0(self, capsys):
        """--schema prints JSON schema to stdout and exits 0."""
        import argparse

        from nvflare.tool.cli_schema import handle_schema_flag

        parser = argparse.ArgumentParser()
        parser.add_argument("job_id", nargs="?", default=None)
        parser.add_argument("--schema", action="store_true")

        import pytest

        with pytest.raises(SystemExit) as exc_info:
            handle_schema_flag(parser, "nvflare job list", ["nvflare job list"], ["--schema"])
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["schema_version"] == "1"
        assert result["command"] == "nvflare job list"
        assert isinstance(result["args"], list)
        assert isinstance(result["examples"], list)

    def test_schema_flag_deprecated_fields_present(self, capsys):
        """--schema for deprecated commands includes deprecated and deprecated_message fields."""
        import argparse

        import pytest

        from nvflare.tool.cli_schema import handle_schema_flag

        parser = argparse.ArgumentParser()
        parser.add_argument("--schema", action="store_true")

        with pytest.raises(SystemExit) as exc_info:
            handle_schema_flag(
                parser,
                "nvflare simulator",
                ["nvflare simulator job/"],
                ["--schema"],
                deprecated=True,
                deprecated_message="Use nvflare job submit instead",
            )
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result.get("deprecated") is True
        assert "deprecated_message" in result
        assert result["deprecated_message"] == "Use nvflare job submit instead"
