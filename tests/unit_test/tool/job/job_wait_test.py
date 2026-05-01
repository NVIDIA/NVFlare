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
import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.flare_api.api_spec import AuthorizationError, JobNotFound, JobTimeout
from nvflare.tool import cli_output


def _make_args(job_id="abc123", timeout=0, interval=2, study="default", startup_kit=None, kit_id=None):
    args = MagicMock()
    args.job_id = job_id
    args.timeout = timeout
    args.interval = interval
    args.study = study
    args.startup_kit = startup_kit
    args.kit_id = kit_id
    return args


def _make_meta(status="FINISHED_OK", job_name="test-job", duration="0:01:30"):
    from nvflare.apis.job_def import JobMetaKey

    return {
        JobMetaKey.STATUS.value: status,
        JobMetaKey.JOB_NAME.value: job_name,
        JobMetaKey.DURATION.value: duration,
    }


def _session_ctx(mock_sess):
    @contextmanager
    def _fake_session(*args, **kwargs):
        yield mock_sess

    return patch("nvflare.tool.job.job_cli._job_session_for_args", side_effect=_fake_session)


def _init_parser():
    from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

    root = argparse.ArgumentParser()
    def_job_cli_parser(root.add_subparsers())
    return job_sub_cmd_parser["wait"]


class TestJobWait:
    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def test_parser_args(self):
        parser = _init_parser()

        args = parser.parse_args(["abc123", "--timeout", "300", "--interval", "5", "--study", "cancer"])

        assert args.job_id == "abc123"
        assert args.timeout == 300
        assert args.interval == 5
        assert args.study == "cancer"

    @pytest.mark.parametrize(
        ("selector", "value", "dest"),
        [
            ("--startup-kit", "/tmp/startup", "startup_kit"),
            ("--kit-id", "prod_admin", "kit_id"),
        ],
    )
    def test_parser_accepts_scoped_startup_selectors(self, selector, value, dest):
        parser = _init_parser()

        args = parser.parse_args(["abc123", selector, value])

        assert getattr(args, dest) == value

    def test_help_and_schema_include_study_and_startup_selectors(self, capsys):
        from nvflare.tool.job.job_cli import cmd_job_wait

        parser = _init_parser()
        help_text = parser.format_help()
        assert "--study" in help_text
        assert "--startup-kit" in help_text
        assert "--kit-id" in help_text

        with patch("sys.argv", ["nvflare", "job", "wait", "--schema"]):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_wait(MagicMock())

        assert exc_info.value.code == 0
        schema_text = capsys.readouterr().out
        assert "--study" in schema_text
        assert "--startup-kit" in schema_text
        assert "--kit-id" in schema_text
        schema = json.loads(schema_text)
        assert schema["output_modes"] == ["json"]
        assert schema["streaming"] is False
        assert schema["mutating"] is False
        assert schema["idempotent"] is True
        assert schema["retry_token"] == {"supported": False}

    def test_success_outputs_ok_envelope(self, capsys):
        from nvflare.tool.job.job_cli import cmd_job_wait

        mock_sess = MagicMock()
        mock_sess.wait_for_job.return_value = _make_meta()

        with _session_ctx(mock_sess):
            cmd_job_wait(_make_args())

        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert data["data"]["job_id"] == "abc123"
        assert data["data"]["status"] == "FINISHED_OK"

    def test_human_success_prints_terminal_summary_without_json_payload(self, capsys, monkeypatch):
        from nvflare.tool.job.job_cli import cmd_job_wait

        monkeypatch.setattr(cli_output, "_output_format", "txt")
        mock_sess = MagicMock()
        mock_sess.wait_for_job.return_value = _make_meta(job_name="hello-pt")

        with _session_ctx(mock_sess):
            cmd_job_wait(_make_args())

        captured = capsys.readouterr()
        assert "Waiting for job abc123 in study default ..." in captured.out
        assert "Job abc123 status: COMPLETED (FINISHED_OK)" in captured.out
        assert "Name: hello-pt" in captured.out
        assert "job_meta:" not in captured.out
        assert "last_stats:" not in captured.out
        assert captured.err == ""

    def test_human_wait_message_includes_named_study(self, capsys, monkeypatch):
        from nvflare.tool.job.job_cli import cmd_job_wait

        monkeypatch.setattr(cli_output, "_output_format", "txt")
        mock_sess = MagicMock()
        mock_sess.wait_for_job.return_value = _make_meta(job_name="hello-pt")

        with _session_ctx(mock_sess):
            cmd_job_wait(_make_args(study="cancer"))

        captured = capsys.readouterr()
        assert "Waiting for job abc123 in study cancer ..." in captured.out
        assert "Study: cancer" in captured.out

    def test_timeout_exits_3(self, capsys):
        from nvflare.tool.job.job_cli import cmd_job_wait

        mock_sess = MagicMock()
        mock_sess.wait_for_job.side_effect = JobTimeout("job abc123 did not finish within 10s")

        with _session_ctx(mock_sess):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_wait(_make_args(timeout=10))

        assert exc_info.value.code == 3
        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "error"
        assert data["error_code"] == "TIMEOUT"

    @pytest.mark.parametrize(
        ("kwargs", "detail"),
        [
            ({"timeout": -1}, "--timeout must be >= 0"),
            ({"interval": 0}, "--interval must be > 0"),
            ({"interval": -1}, "--interval must be > 0"),
        ],
    )
    def test_invalid_wait_arguments_exit_before_session_creation(self, capsys, kwargs, detail):
        from nvflare.tool.job.job_cli import cmd_job_wait

        with patch("nvflare.tool.job.job_cli._job_session_for_args") as session_factory:
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_wait(_make_args(**kwargs))

        assert exc_info.value.code == 4
        session_factory.assert_not_called()
        data = json.loads(capsys.readouterr().out)
        assert data["error_code"] == "INVALID_ARGS"
        assert detail in data["message"]

    def test_job_not_found_exits_1(self, capsys):
        from nvflare.tool.job.job_cli import cmd_job_wait

        mock_sess = MagicMock()
        mock_sess.wait_for_job.side_effect = JobNotFound("job does not exist")

        with _session_ctx(mock_sess):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_wait(_make_args())

        assert exc_info.value.code == 1
        data = json.loads(capsys.readouterr().out)
        assert data["error_code"] == "JOB_NOT_FOUND"
        assert "searched study 'default'" in data["message"]
        assert "nvflare job list --study <study_name>" in data["hint"]

    def test_authorization_error_exits_2(self, capsys):
        from nvflare.tool.job.job_cli import cmd_job_wait

        mock_sess = MagicMock()
        mock_sess.wait_for_job.side_effect = AuthorizationError("not authorized for study")

        with _session_ctx(mock_sess):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_wait(_make_args())

        assert exc_info.value.code == 2
        data = json.loads(capsys.readouterr().out)
        assert data["error_code"] == "AUTH_FAILED"

    def test_failed_terminal_status_exits_1(self, capsys):
        from nvflare.tool.job.job_cli import cmd_job_wait

        mock_sess = MagicMock()
        mock_sess.wait_for_job.return_value = _make_meta("FAILED")

        with _session_ctx(mock_sess):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_wait(_make_args())

        assert exc_info.value.code == 1
        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "error"
        assert data["error_code"] == "JOB_FAILED"
        assert data["data"]["status"] == "FAILED"

    def test_forwards_study_and_wait_options_to_session(self):
        from nvflare.tool.job.job_cli import cmd_job_wait

        mock_sess = MagicMock()
        mock_sess.wait_for_job.return_value = _make_meta()
        args = _make_args(timeout=60, interval=7, study="cancer")

        with _session_ctx(mock_sess) as session_factory:
            cmd_job_wait(args)

        assert session_factory.call_args.args[0] is args
        assert session_factory.call_args.kwargs["study"] == "cancer"
        mock_sess.wait_for_job.assert_called_once_with(
            "abc123",
            timeout=60,
            poll_interval=7,
        )

    def test_forwards_startup_selector_args_to_session(self):
        from nvflare.tool.job.job_cli import cmd_job_wait

        mock_sess = MagicMock()
        mock_sess.wait_for_job.return_value = _make_meta()
        args = _make_args(startup_kit="/tmp/startup")

        with _session_ctx(mock_sess) as session_factory:
            cmd_job_wait(args)

        assert session_factory.call_args.args[0] is args
