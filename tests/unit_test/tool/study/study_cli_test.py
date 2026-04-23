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
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.flare_api.api_spec import CommandError
from nvflare.tool import cli_output


class TestStudyCli:
    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def test_study_list_outputs_json_envelope(self, capsys):
        from nvflare.tool.study.study_cli import cmd_list

        args = MagicMock()
        mock_sess = MagicMock()
        mock_sess.list_studies.return_value = {"studies": ["cancer-research"]}

        with patch("nvflare.tool.study.study_cli._study_session", return_value=nullcontext(mock_sess)):
            cmd_list(args)

        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "ok"
        assert payload["data"]["studies"] == ["cancer-research"]

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

        with patch("nvflare.tool.study.study_cli._study_session", return_value=nullcontext(mock_sess)):
            with pytest.raises(SystemExit) as exc_info:
                cmd_show(args)

        assert exc_info.value.code == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["error_code"] == "STUDY_NOT_FOUND"

    def test_study_session_requires_explicit_resolution_source(self, capsys, monkeypatch):
        from nvflare.tool.study.study_cli import _study_session

        args = MagicMock()
        args.startup_target = None
        args.startup_kit = None
        monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            with _study_session(args):
                pass

        assert exc_info.value.code == 4
        payload = json.loads(capsys.readouterr().out)
        assert payload["error_code"] == "STARTUP_KIT_MISSING"

    def test_register_missing_sites_is_structured_usage_error(self, capsys):
        from nvflare.tool.study.study_cli import cmd_register

        args = MagicMock()
        args.name = "cancer-research"
        args.sites = None

        with pytest.raises(SystemExit) as exc_info:
            cmd_register(args)

        assert exc_info.value.code == 4
        payload = json.loads(capsys.readouterr().out)
        assert payload["error_code"] == "INVALID_ARGS"

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
            with pytest.raises(SystemExit) as exc_info:
                cmd_fn(args)

        assert exc_info.value.code == 4
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

        with patch("nvflare.tool.study.study_cli._try_get_caller_role", return_value="org_admin"):
            with pytest.raises(SystemExit) as exc_info:
                cmd_fn(args)

        assert exc_info.value.code == 4
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

        with patch("nvflare.tool.study.study_cli._try_get_caller_role", return_value="project_admin"):
            with pytest.raises(SystemExit) as exc_info:
                cmd_fn(args)

        assert exc_info.value.code == 4
        payload = json.loads(capsys.readouterr().out)
        assert payload["error_code"] == "INVALID_ARGS"

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
        mock_sess.add_study_site.return_value = {"study": "cancer-research", "added": ["hospital-b"]}

        with patch("nvflare.tool.study.study_cli._study_session", return_value=nullcontext(mock_sess)):
            with patch("nvflare.tool.study.study_cli._try_get_caller_role", return_value="org_admin"):
                cmd_add_site(args)

        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "ok"
        assert payload["data"]["added"] == ["hospital-b"]

    def test_remove_site_outputs_ok_envelope(self, capsys):
        from nvflare.tool.study.study_cli import cmd_remove_site

        args = MagicMock()
        args.name = "cancer-research"
        args.sites = "hospital-b"
        args.site_org = []
        mock_sess = MagicMock()
        mock_sess.remove_study_site.return_value = {"study": "cancer-research", "removed": ["hospital-b"]}

        with patch("nvflare.tool.study.study_cli._study_session", return_value=nullcontext(mock_sess)):
            with patch("nvflare.tool.study.study_cli._try_get_caller_role", return_value="org_admin"):
                cmd_remove_site(args)

        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "ok"
        assert payload["data"]["removed"] == ["hospital-b"]

    def test_remove_outputs_ok_envelope(self, capsys):
        from nvflare.tool.study.study_cli import cmd_remove

        args = MagicMock()
        args.name = "cancer-research"
        mock_sess = MagicMock()
        mock_sess.remove_study.return_value = {"name": "cancer-research", "removed": True}

        with patch("nvflare.tool.study.study_cli._study_session", return_value=nullcontext(mock_sess)):
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

        with patch("nvflare.tool.study.study_cli._study_session", return_value=nullcontext(mock_sess)):
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
        mock_sess.add_study_user.return_value = {"study": "cancer-research", "user": "trainer@org_a.com"}

        with patch("nvflare.tool.study.study_cli._study_session", return_value=nullcontext(mock_sess)):
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

        with patch("nvflare.tool.study.study_cli._study_session", return_value=nullcontext(mock_sess)):
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
        mock_sess.register_study.return_value = {"name": "cancer-research", "site_orgs": {"org_a": ["hospital-a"]}}

        with patch("nvflare.tool.study.study_cli._study_session", return_value=nullcontext(mock_sess)):
            with patch("nvflare.tool.study.study_cli._try_get_caller_role", return_value="project_admin"):
                cmd_register(args)

        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "ok"
        assert "hospital-a" in payload["data"]["site_orgs"]["org_a"]
