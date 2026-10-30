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

import argparse
from contextlib import nullcontext

import pytest

from nvflare.tool.resources import resources_cli


class TestResourcesCLI:
    def test_resources_parser_supports_job_study_and_site_without_all_jobs(self):
        parser = argparse.ArgumentParser(prog="nvflare")
        subparsers = parser.add_subparsers(dest="command")
        resources_cli.def_resources_cli_parser(subparsers)

        args = parser.parse_args(["resources", "--job", "job-1", "--study", "cancer-research", "--site", "site-1"])

        assert args.command == "resources"
        assert args.job_id == "job-1"
        assert args.study == "cancer-research"
        assert args.site == "site-1"

    def test_resources_cli_displays_metadata_name_without_copying_it_into_json(self, monkeypatch):
        from nvflare.tool import cli_output, cli_schema
        from nvflare.tool.resources import resources_output

        parser = argparse.ArgumentParser(prog="nvflare")
        resources_cli.def_resources_cli_parser(parser.add_subparsers(dest="command"))
        summary = {"job_id": "job-1", "participants": []}
        result = {"resource_summary": summary, "job_name": "hello-pt"}
        session = type("Session", (), {"get_job_resources": lambda self, job_id, site=None: result})()
        monkeypatch.setattr(resources_cli, "_resources_session", lambda *args, **kwargs: nullcontext(session))
        monkeypatch.setattr(cli_schema, "handle_schema_flag", lambda *args, **kwargs: None)
        args = argparse.Namespace(job_id="job-1", study=None, site=None)

        json_outputs = []
        monkeypatch.setattr(cli_output, "is_json_mode", lambda: True)
        monkeypatch.setattr(cli_output, "output_ok", json_outputs.append)
        resources_cli.handle_resources_cmd(args)
        assert json_outputs == [{"selection": {"job_id": "job-1", "site": "all"}, "summary": summary}]
        assert "job_name" not in json_outputs[0]["summary"]

        displayed = []
        monkeypatch.setattr(cli_output, "is_json_mode", lambda: False)
        monkeypatch.setattr(cli_output, "print_human", displayed.append)
        monkeypatch.setattr(
            resources_output,
            "render_job_resources",
            lambda report, participant=None, *, job_name=None: f"{report['job_id']} {job_name}",
        )
        resources_cli.handle_resources_cmd(args)
        assert displayed == ["job-1 hello-pt"]

    @pytest.mark.parametrize("job_id, study", [("job-1", None), (None, "study-1")])
    def test_resources_cli_preserves_authorization_error_for_global_auth_handler(self, monkeypatch, job_id, study):
        from nvflare.fuel.flare_api.api_spec import AuthorizationError
        from nvflare.tool import cli_output, cli_schema

        parser = argparse.ArgumentParser(prog="nvflare")
        resources_cli.def_resources_cli_parser(parser.add_subparsers(dest="command"))
        error = AuthorizationError("not authorized for resource statistics")

        class Session:
            def get_job_resources(self, job_id, site=None):
                raise error

            def get_study_resources(self):
                raise error

        monkeypatch.setattr(resources_cli, "_resources_session", lambda *args, **kwargs: nullcontext(Session()))
        monkeypatch.setattr(cli_schema, "handle_schema_flag", lambda *args, **kwargs: None)
        monkeypatch.setattr(cli_output, "output_error", lambda *args, **kwargs: pytest.fail("must use auth handler"))

        with pytest.raises(AuthorizationError) as raised:
            resources_cli.handle_resources_cmd(argparse.Namespace(job_id=job_id, study=study, site=None))

        assert raised.value is error


@pytest.mark.parametrize("prefix", [[], ["--format", "json"]])
def test_top_level_schema_needs_no_session(monkeypatch, capsys, prefix):
    import json
    import sys

    from nvflare import cli

    monkeypatch.setattr(sys, "argv", ["nvflare", *prefix, "resources", "--schema"])
    monkeypatch.setattr(resources_cli, "_resources_session", lambda *a, **kw: pytest.fail("schema must be offline"))
    with pytest.raises(SystemExit) as result:
        cli.run("nvflare")
    assert result.value.code == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["command"] == "nvflare resources"
    assert schema["mutating"] is False
    assert schema["idempotent"] is True
    assert schema["streaming"] is False
    assert schema["output_modes"] == ["json"]
    assert {"--job", "--study", "--site", "--kit-id", "--startup-kit"} <= {a["name"] for a in schema["args"]}


@pytest.mark.parametrize("selection", [[], ["--site", "site-1"], ["--study", "s", "--site", "site-1"]])
def test_top_level_json_selection_errors_are_envelopes(monkeypatch, capsys, selection):
    import json
    import sys

    from nvflare import cli
    from nvflare.tool import cli_output

    monkeypatch.setattr(cli_output, "_output_format", "txt")
    monkeypatch.setattr(sys, "argv", ["nvflare", "resources", *selection, "--format", "json"])
    monkeypatch.setattr(
        resources_cli, "_resources_session", lambda *a, **kw: pytest.fail("invalid selector must be offline")
    )
    with pytest.raises(SystemExit) as result:
        cli.run("nvflare")
    assert result.value.code == 4
    assert json.loads(capsys.readouterr().out)["error_code"] == "INVALID_ARGUMENT"


@pytest.mark.parametrize("selection", [["--job", "job-1", "--site", "site-1"], ["--study", "study-1"]])
def test_top_level_json_result_keeps_precision_and_selection(monkeypatch, capsys, selection):
    import json
    import sys

    from nvflare import cli
    from nvflare.tool import cli_output

    summary = {"resource_utilization": {"cpu_consumed": {"cpu_seconds": "0.000123456789"}}}
    participant = {"duration_seconds": "1.23456789"}

    class Session:
        def get_job_resources(self, job_id, site=None):
            assert (job_id, site) == ("job-1", "site-1")
            return {"resource_summary": summary, "participant_summary": participant}

        def get_study_resources(self):
            return summary

    monkeypatch.setattr(cli_output, "_output_format", "txt")
    monkeypatch.setattr(sys, "argv", ["nvflare", "resources", *selection, "--format", "json"])
    monkeypatch.setattr(resources_cli, "_resources_session", lambda *a, **kw: nullcontext(Session()))
    cli.run("nvflare")
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok"
    assert result["data"]["summary"] == summary
    if selection[0] == "--job":
        assert result["data"]["participant"] == participant
        assert result["data"]["selection"] == {"job_id": "job-1", "site": "site-1"}
    else:
        assert result["data"]["selection"] == {"study": "study-1"}


def test_session_uses_shared_kit_timeout_study_and_closes(monkeypatch):
    from unittest.mock import Mock

    from nvflare.tool import cli_output, cli_session

    args = argparse.Namespace(kit_id="admin", startup_kit=None)
    session = Mock()
    factory = Mock(return_value=session)
    monkeypatch.setattr(cli_session, "new_cli_session_for_args", factory)
    monkeypatch.setattr(cli_output, "get_validated_connect_timeout", lambda: 7.0)
    with pytest.raises(RuntimeError):
        with resources_cli._resources_session(args, study="research"):
            raise RuntimeError("query failed")
    factory.assert_called_once_with(args=args, timeout=7.0, study="research")
    session.close.assert_called_once()


def test_bare_human_command_shows_help_and_old_job_path_is_removed(monkeypatch, capsys):
    import sys

    from nvflare import cli
    from nvflare.tool import cli_output

    monkeypatch.setattr(cli_output, "_output_format", "txt")
    monkeypatch.setattr(sys, "argv", ["nvflare", "resources"])
    cli.run("nvflare")
    assert "--job" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["nvflare", "job", "resources"])
    with pytest.raises(SystemExit):
        cli.run("nvflare")
