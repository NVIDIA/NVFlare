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
from argparse import Namespace
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.flare_api.api_spec import CommandError
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


class TestStudyCli:
    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def test_study_list_outputs_json_envelope(self, capsys):
        from nvflare.tool.study.study_cli import cmd_list

        args = MagicMock()
        mock_sess = MagicMock()
        mock_sess.list_studies.return_value = {
            "identity": {"name": "lead@nvidia.com", "org": "nvidia", "role": "lead"},
            "studies": ["cancer-research"],
            "study_details": [
                {
                    "name": "cancer-research",
                    "role": "lead",
                    "capabilities": {"submit_job": True},
                    "can_submit_job": True,
                }
            ],
        }

        with (
            patch(
                "nvflare.tool.study.study_cli._study_session",
                return_value=nullcontext(mock_sess),
            ),
            patch(
                "nvflare.tool.study.study_cli.resolve_startup_kit_info_for_args",
                return_value={"source": "active", "id": "lead", "path": "/kits/lead"},
            ),
        ):
            cmd_list(args)

        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "ok"
        assert payload["data"]["studies"] == ["cancer-research"]
        assert payload["data"]["identity"] == {"name": "lead@nvidia.com", "org": "nvidia", "role": "lead"}
        assert payload["data"]["study_details"][0]["can_submit_job"] is True
        assert payload["data"]["startup_kit"] == {"source": "active", "id": "lead", "path": "/kits/lead"}

    def test_study_show_maps_command_error(self, capsys):
        from nvflare.tool.study.study_cli import cmd_show

        args = MagicMock()
        args.name = "ghost-study"
        mock_sess = MagicMock()
        mock_sess.show_study.side_effect = CommandError(
            error_code="STUDY_NOT_FOUND",
            message="Study 'ghost-study' not found.",
            hint="Verify the study name.",
            exit_code=1,
        )

        with patch(
            "nvflare.tool.study.study_cli._study_session",
            return_value=nullcontext(mock_sess),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cmd_show(args)

        assert exc_info.value.code == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["error_code"] == "STUDY_NOT_FOUND"

    def test_missing_study_subcommand_prints_help_without_usage_error(self, capsys, monkeypatch):
        from nvflare.tool.study.study_cli import handle_study_cmd

        monkeypatch.setattr(cli_output, "_output_format", "txt")

        handle_study_cmd(Namespace(study_sub_cmd=None))

        captured = capsys.readouterr()
        assert "usage: nvflare study" in captured.out
        assert "study subcommands" in captured.out
        assert "Invalid arguments" not in captured.out
        assert captured.err == ""

    def test_study_session_uses_active_startup_kit(self, tmp_path, monkeypatch):
        from nvflare.tool.study.study_cli import _study_session

        active_admin_dir = _configure_active_startup_kit(tmp_path, monkeypatch)
        fake_session = MagicMock()

        with patch("nvflare.tool.cli_session.new_secure_session", return_value=fake_session) as new_secure:
            with _study_session(Namespace()) as sess:
                assert sess is fake_session

        assert new_secure.call_args.kwargs["username"] == "admin@nvidia.com"
        assert new_secure.call_args.kwargs["startup_kit_location"] == str(active_admin_dir)

    def test_study_session_uses_scoped_startup_kit(self, tmp_path, monkeypatch):
        from nvflare.tool.study.study_cli import _study_session

        scoped_admin_dir = _make_admin_startup_kit(tmp_path, "scoped@nvidia.com")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)
        fake_session = MagicMock()

        with patch("nvflare.tool.cli_session.new_secure_session", return_value=fake_session) as new_secure:
            with _study_session(Namespace(startup_kit=str(scoped_admin_dir), kit_id=None)) as sess:
                assert sess is fake_session

        assert new_secure.call_args.kwargs["username"] == "scoped@nvidia.com"
        assert new_secure.call_args.kwargs["startup_kit_location"] == str(scoped_admin_dir)

    def test_resolve_startup_kit_info_uses_active_startup_kit(self, tmp_path, monkeypatch):
        from nvflare.tool.cli_session import resolve_startup_kit_info_for_args

        active_admin_dir = _configure_active_startup_kit(tmp_path, monkeypatch)

        info = resolve_startup_kit_info_for_args(Namespace())

        assert info == {
            "source": "active",
            "id": "admin@nvidia.com",
            "path": str(active_admin_dir.resolve()),
        }

    def test_resolve_startup_kit_info_uses_explicit_startup_kit(self, tmp_path, monkeypatch):
        from nvflare.tool.cli_session import resolve_startup_kit_info_for_args

        scoped_admin_dir = _make_admin_startup_kit(tmp_path, "scoped@nvidia.com")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)

        info = resolve_startup_kit_info_for_args(Namespace(startup_kit=str(scoped_admin_dir), kit_id=None))

        assert info == {
            "source": "startup_kit",
            "id": None,
            "path": str(scoped_admin_dir.resolve()),
        }

    def test_try_get_caller_role_uses_active_startup_kit(self, tmp_path, monkeypatch):
        from nvflare.tool.study.study_cli import _try_get_caller_role

        active_admin_dir = _configure_active_startup_kit(tmp_path, monkeypatch)

        with patch(
            "nvflare.tool.study.study_cli._get_caller_role_from_startup_kit",
            return_value="project_admin",
        ) as get_role:
            assert _try_get_caller_role(Namespace()) == "project_admin"

        get_role.assert_called_once_with(str(active_admin_dir))

    def test_try_get_caller_role_uses_scoped_startup_kit(self, tmp_path, monkeypatch):
        from nvflare.tool.study.study_cli import _try_get_caller_role

        scoped_admin_dir = _make_admin_startup_kit(tmp_path, "scoped@nvidia.com")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)

        with patch(
            "nvflare.tool.study.study_cli._get_caller_role_from_startup_kit",
            return_value="org_admin",
        ) as get_role:
            assert _try_get_caller_role(Namespace(startup_kit=str(scoped_admin_dir), kit_id=None)) == "org_admin"

        get_role.assert_called_once_with(str(scoped_admin_dir))

    def test_get_caller_role_from_startup_kit_reads_cert_role(self, tmp_path):
        from nvflare.fuel.hci.client.api_spec import AdminConfigKey
        from nvflare.tool.study.study_cli import _get_caller_role_from_startup_kit

        cert_path = tmp_path / "client.crt"
        cert_path.write_text("cert", encoding="utf-8")
        (tmp_path / "startup").mkdir()
        (tmp_path / "local").mkdir()
        fake_conf = MagicMock()
        fake_conf.get_admin_config.return_value = {AdminConfigKey.CLIENT_CERT: str(cert_path)}

        with patch(
            "nvflare.fuel.hci.client.config.secure_load_admin_config",
            return_value=fake_conf,
        ):
            with patch(
                "nvflare.private.fed.utils.identity_utils.load_cert_file",
                return_value=object(),
            ):
                with patch(
                    "nvflare.lighter.utils.cert_to_dict",
                    return_value={"subject": {"unstructuredName": "project_admin"}},
                ):
                    assert _get_caller_role_from_startup_kit(str(tmp_path)) == "project_admin"

    def test_study_session_missing_startup_kit_when_no_source_resolves(self, capsys, monkeypatch, tmp_path):
        from nvflare.tool.study.study_cli import _study_session

        args = Namespace()
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            with _study_session(args):
                pass

        assert exc_info.value.code == 4
        payload = json.loads(capsys.readouterr().out)
        assert payload["error_code"] == "STARTUP_KIT_MISSING"

    def test_add_user_uses_active_startup_kit_and_preserves_study_args(self, tmp_path, monkeypatch):
        from nvflare.tool.study.study_cli import cmd_add_user

        active_admin_dir = _configure_active_startup_kit(tmp_path, monkeypatch)
        args = Namespace(study="cancer-research", user="trainer@org_a.com")
        fake_session = MagicMock()
        fake_session.add_study_user.return_value = {
            "study": args.study,
            "user": args.user,
        }

        with patch("sys.argv", ["nvflare", "study", "add-user", args.study, args.user]):
            with patch("nvflare.tool.cli_session.new_secure_session", return_value=fake_session) as new_secure:
                cmd_add_user(args)

        assert new_secure.call_args.kwargs["startup_kit_location"] == str(active_admin_dir)
        fake_session.add_study_user.assert_called_once_with("cancer-research", "trainer@org_a.com")

    @pytest.mark.parametrize(
        ("argv_prefix"),
        [
            ["register", "cancer-research", "--site-org", "nvidia:site-1"],
            ["add-site", "cancer-research", "--site-org", "nvidia:site-2"],
            ["remove-site", "cancer-research", "--site-org", "nvidia:site-2"],
            ["remove", "cancer-research"],
            ["list"],
            ["show", "cancer-research"],
            ["add-user", "cancer-research", "trainer@org_a.com"],
            ["remove-user", "cancer-research", "trainer@org_a.com"],
        ],
    )
    @pytest.mark.parametrize(
        ("selector", "value"),
        [
            ("--startup-target", "prod"),
            ("--startup_kit", "/tmp/startup"),
            ("--startup_target", "prod"),
        ],
    )
    def test_study_parser_rejects_old_startup_selectors(self, argv_prefix, selector, value):
        import argparse

        from nvflare.tool.study.study_cli import def_study_cli_parser

        root = argparse.ArgumentParser(prog="nvflare")
        parser = def_study_cli_parser(root.add_subparsers(dest="sub_command"))["study"]

        with pytest.raises(SystemExit):
            parser.parse_args([*argv_prefix, selector, value])

    @pytest.mark.parametrize(
        ("argv_prefix"),
        [
            ["register", "cancer-research", "--site-org", "nvidia:site-1"],
            ["add-site", "cancer-research", "--site-org", "nvidia:site-2"],
            ["remove-site", "cancer-research", "--site-org", "nvidia:site-2"],
            ["remove", "cancer-research"],
            ["list"],
            ["show", "cancer-research"],
            ["add-user", "cancer-research", "trainer@org_a.com"],
            ["remove-user", "cancer-research", "trainer@org_a.com"],
        ],
    )
    @pytest.mark.parametrize(
        ("selector", "value", "dest"),
        [
            ("--startup-kit", "/tmp/startup", "startup_kit"),
            ("--kit-id", "prod_admin", "kit_id"),
        ],
    )
    def test_study_parser_accepts_scoped_startup_selectors(self, argv_prefix, selector, value, dest):
        import argparse

        from nvflare.tool.study.study_cli import def_study_cli_parser

        root = argparse.ArgumentParser(prog="nvflare")
        parser = def_study_cli_parser(root.add_subparsers(dest="sub_command"))["study"]

        args = parser.parse_args([*argv_prefix, selector, value])

        assert getattr(args, dest) == value

    def test_study_parser_accepts_multiple_space_delimited_sites(self):
        import argparse

        from nvflare.tool.study.study_cli import def_study_cli_parser

        root = argparse.ArgumentParser(prog="nvflare")
        parser = def_study_cli_parser(root.add_subparsers(dest="sub_command"))["study"]

        args = parser.parse_args(["register", "cancer-research", "--sites", "site-1", "site-2"])

        assert args.sites == ["site-1", "site-2"]

    def test_study_help_includes_scoped_startup_selectors(self):
        import argparse

        from nvflare.tool.study import study_cli

        root = argparse.ArgumentParser(prog="nvflare")
        study_cli.def_study_cli_parser(root.add_subparsers(dest="sub_command"))

        for parser in study_cli._study_sub_cmd_parsers.values():
            help_text = parser.format_help()
            for token in ("--startup-target", "--startup_target", "--startup_kit"):
                assert token not in help_text
            assert "--startup-kit" in help_text
            assert "--kit-id" in help_text

    @pytest.mark.parametrize(
        ("cmd_name", "handler_name"),
        [
            ("register", "cmd_register"),
            ("add-site", "cmd_add_site"),
            ("remove-site", "cmd_remove_site"),
            ("remove", "cmd_remove"),
            ("list", "cmd_list"),
            ("show", "cmd_show"),
            ("add-user", "cmd_add_user"),
            ("remove-user", "cmd_remove_user"),
        ],
    )
    def test_study_schema_includes_scoped_startup_selectors(self, capsys, cmd_name, handler_name):
        import nvflare.tool.study.study_cli as study_cli_mod

        with patch("sys.argv", ["nvflare", "study", cmd_name, "--schema"]):
            with pytest.raises(SystemExit) as exc_info:
                getattr(study_cli_mod, handler_name)(MagicMock())

        assert exc_info.value.code == 0
        schema_text = capsys.readouterr().out
        for token in ("--startup-target", "--startup_target", "--startup_kit"):
            assert token not in schema_text
        assert "--startup-kit" in schema_text
        assert "--kit-id" in schema_text

    def test_register_missing_sites_is_structured_usage_error(self, capsys):
        from nvflare.tool.study.study_cli import cmd_register

        args = MagicMock()
        args.name = "cancer-research"
        args.sites = None
        args.site_org = []

        with patch("nvflare.tool.study.study_cli._try_get_caller_role", return_value=None):
            with pytest.raises(SystemExit) as exc_info:
                cmd_register(args)

        assert exc_info.value.code == 4
        payload = json.loads(capsys.readouterr().out)
        assert payload["error_code"] == "INVALID_ARGS"
        assert payload["message"].endswith("provide --sites for org_admin or --site-org for project_admin")

    def test_register_project_admin_missing_input_requires_site_org_before_connecting(self, capsys):
        from nvflare.tool.study.study_cli import cmd_register

        args = MagicMock()
        args.name = "cancer-research"
        args.sites = None
        args.site_org = []

        with patch(
            "nvflare.tool.study.study_cli._try_get_caller_role",
            return_value="project_admin",
        ):
            with patch("nvflare.tool.study.study_cli._study_session") as study_session:
                with pytest.raises(SystemExit) as exc_info:
                    cmd_register(args)

        assert exc_info.value.code == 4
        study_session.assert_not_called()
        payload = json.loads(capsys.readouterr().out)
        assert payload["error_code"] == "INVALID_ARGS"
        assert "project_admin must provide --site-org" in payload["message"]
        assert "nvidia:site-1,site-2" in payload["hint"]
        assert "POC default org: nvidia" in payload["hint"]
        assert "--sites is required" not in payload["message"]

    # ------------------------------------------------------------------
    # INVALID_ARGS — CLI-side fast-fail (before any server connection)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "cmd_fn_name,cmd_name",
        [
            ("cmd_register", "register"),
            ("cmd_add_site", "add-site"),
            ("cmd_remove_site", "remove-site"),
        ],
    )
    def test_mixed_sites_and_site_org_rejected_before_connecting(self, capsys, cmd_fn_name, cmd_name):
        import nvflare.tool.study.study_cli as study_cli_mod

        cmd_fn = getattr(study_cli_mod, cmd_fn_name)

        args = MagicMock()
        args.name = "cancer-research"
        args.sites = "hospital-a"
        args.site_org = ["org_a:hospital-b"]

        with patch("nvflare.tool.study.study_cli._try_get_caller_role", return_value=""):
            with patch("nvflare.tool.study.study_cli._study_session") as study_session:
                with pytest.raises(SystemExit) as exc_info:
                    cmd_fn(args)

        assert exc_info.value.code == 4
        study_session.assert_not_called()
        payload = json.loads(capsys.readouterr().out)
        assert payload["error_code"] == "INVALID_ARGS"

    @pytest.mark.parametrize("cmd_fn_name", ["cmd_register", "cmd_add_site", "cmd_remove_site"])
    def test_org_admin_with_site_org_rejected_before_connecting(self, capsys, cmd_fn_name):
        import nvflare.tool.study.study_cli as study_cli_mod

        cmd_fn = getattr(study_cli_mod, cmd_fn_name)

        args = MagicMock()
        args.name = "cancer-research"
        args.sites = None
        args.site_org = ["org_a:hospital-a"]

        with patch(
            "nvflare.tool.study.study_cli._try_get_caller_role",
            return_value="org_admin",
        ):
            with patch("nvflare.tool.study.study_cli._study_session") as study_session:
                with pytest.raises(SystemExit) as exc_info:
                    cmd_fn(args)

        assert exc_info.value.code == 4
        study_session.assert_not_called()
        payload = json.loads(capsys.readouterr().out)
        assert payload["error_code"] == "INVALID_ARGS"

    @pytest.mark.parametrize("cmd_fn_name", ["cmd_register", "cmd_add_site", "cmd_remove_site"])
    def test_project_admin_with_sites_rejected_before_connecting(self, capsys, cmd_fn_name):
        import nvflare.tool.study.study_cli as study_cli_mod

        cmd_fn = getattr(study_cli_mod, cmd_fn_name)

        args = MagicMock()
        args.name = "cancer-research"
        args.sites = "hospital-a"
        args.site_org = []

        with patch(
            "nvflare.tool.study.study_cli._try_get_caller_role",
            return_value="project_admin",
        ):
            with patch("nvflare.tool.study.study_cli._study_session") as study_session:
                with pytest.raises(SystemExit) as exc_info:
                    cmd_fn(args)

        assert exc_info.value.code == 4
        study_session.assert_not_called()
        payload = json.loads(capsys.readouterr().out)
        assert payload["error_code"] == "INVALID_ARGS"
        assert "nvidia:site-1,site-2" in payload["hint"]
        assert "POC default org: nvidia" in payload["hint"]

    # ------------------------------------------------------------------
    # Remaining command golden-path and error-mapping coverage
    # ------------------------------------------------------------------

    def test_add_site_outputs_ok_envelope(self, capsys):
        from nvflare.tool.study.study_cli import cmd_add_site

        args = MagicMock()
        args.name = "cancer-research"
        args.sites = "hospital-b"
        args.site_org = []
        mock_sess = MagicMock()
        mock_sess.add_study_site.return_value = {
            "study": "cancer-research",
            "added": ["hospital-b"],
        }

        with patch(
            "nvflare.tool.study.study_cli._study_session",
            return_value=nullcontext(mock_sess),
        ):
            with patch(
                "nvflare.tool.study.study_cli._try_get_caller_role",
                return_value="org_admin",
            ):
                cmd_add_site(args)

        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "ok"
        assert payload["data"]["added"] == ["hospital-b"]

    def test_register_passes_multiple_sites_to_session(self, capsys):
        from nvflare.tool.study.study_cli import cmd_register

        args = MagicMock()
        args.name = "cancer-research"
        args.sites = ["site-1", "site-2"]
        args.site_org = []
        mock_sess = MagicMock()
        mock_sess.register_study.return_value = {
            "name": "cancer-research",
            "sites": ["site-1", "site-2"],
        }

        with patch(
            "nvflare.tool.study.study_cli._study_session",
            return_value=nullcontext(mock_sess),
        ):
            with patch(
                "nvflare.tool.study.study_cli._try_get_caller_role",
                return_value="org_admin",
            ):
                cmd_register(args)

        mock_sess.register_study.assert_called_once_with("cancer-research", sites=["site-1", "site-2"], site_orgs=None)
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "ok"
        assert payload["data"]["sites"] == ["site-1", "site-2"]

    def test_remove_site_outputs_ok_envelope(self, capsys):
        from nvflare.tool.study.study_cli import cmd_remove_site

        args = MagicMock()
        args.name = "cancer-research"
        args.sites = "hospital-b"
        args.site_org = []
        mock_sess = MagicMock()
        mock_sess.remove_study_site.return_value = {
            "study": "cancer-research",
            "removed": ["hospital-b"],
        }

        with patch(
            "nvflare.tool.study.study_cli._study_session",
            return_value=nullcontext(mock_sess),
        ):
            with patch(
                "nvflare.tool.study.study_cli._try_get_caller_role",
                return_value="org_admin",
            ):
                cmd_remove_site(args)

        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "ok"
        assert payload["data"]["removed"] == ["hospital-b"]

    def test_remove_outputs_ok_envelope(self, capsys):
        from nvflare.tool.study.study_cli import cmd_remove

        args = MagicMock()
        args.name = "cancer-research"
        mock_sess = MagicMock()
        mock_sess.remove_study.return_value = {
            "name": "cancer-research",
            "removed": True,
        }

        with patch(
            "nvflare.tool.study.study_cli._study_session",
            return_value=nullcontext(mock_sess),
        ):
            cmd_remove(args)

        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "ok"
        assert payload["data"]["removed"] is True

    def test_show_success_outputs_study_data(self, capsys):
        from nvflare.tool.study.study_cli import cmd_show

        args = MagicMock()
        args.name = "cancer-research"
        mock_sess = MagicMock()
        mock_sess.show_study.return_value = {
            "name": "cancer-research",
            "site_orgs": {"org_a": ["site-a"]},
            "users": ["admin@x.com"],
        }

        with patch(
            "nvflare.tool.study.study_cli._study_session",
            return_value=nullcontext(mock_sess),
        ):
            cmd_show(args)

        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "ok"
        assert payload["data"]["name"] == "cancer-research"

    def test_add_user_outputs_ok_envelope(self, capsys):
        from nvflare.tool.study.study_cli import cmd_add_user

        args = MagicMock()
        args.study = "cancer-research"
        args.user = "trainer@org_a.com"
        mock_sess = MagicMock()
        mock_sess.add_study_user.return_value = {
            "study": "cancer-research",
            "user": "trainer@org_a.com",
        }

        with patch(
            "nvflare.tool.study.study_cli._study_session",
            return_value=nullcontext(mock_sess),
        ):
            cmd_add_user(args)

        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "ok"
        assert payload["data"]["user"] == "trainer@org_a.com"

    def test_remove_user_outputs_ok_envelope(self, capsys):
        from nvflare.tool.study.study_cli import cmd_remove_user

        args = MagicMock()
        args.study = "cancer-research"
        args.user = "trainer@org_a.com"
        mock_sess = MagicMock()
        mock_sess.remove_study_user.return_value = {
            "study": "cancer-research",
            "user": "trainer@org_a.com",
            "removed": True,
        }

        with patch(
            "nvflare.tool.study.study_cli._study_session",
            return_value=nullcontext(mock_sess),
        ):
            cmd_remove_user(args)

        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "ok"
        assert payload["data"]["removed"] is True

    def test_register_project_admin_site_org_outputs_ok_envelope(self, capsys):
        from nvflare.tool.study.study_cli import cmd_register

        args = MagicMock()
        args.name = "cancer-research"
        args.sites = None
        args.site_org = ["org_a:hospital-a"]
        mock_sess = MagicMock()
        mock_sess.register_study.return_value = {
            "name": "cancer-research",
            "site_orgs": {"org_a": ["hospital-a"]},
        }

        with patch(
            "nvflare.tool.study.study_cli._study_session",
            return_value=nullcontext(mock_sess),
        ):
            with patch(
                "nvflare.tool.study.study_cli._try_get_caller_role",
                return_value="project_admin",
            ):
                cmd_register(args)

        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "ok"
        assert "hospital-a" in payload["data"]["site_orgs"]["org_a"]
