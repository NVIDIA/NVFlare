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

import datetime
import json
import os
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.flare_api.api_spec import (
    AuthenticationError,
    AuthorizationError,
    JobNotFound,
    MonitorReturnCode,
    NoConnection,
)
from nvflare.tool import cli_output
from nvflare.tool.job.job_cli import (
    _calculate_monitor_elapsed,
    _format_job_meta_timestamp,
    _parse_monitor_duration_seconds,
    _parse_monitor_start_ts,
)


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


def _make_args(
    job_id="abc123",
    timeout=0,
    interval=2,
    study="default",
    stats_target="server",
    metrics=None,
):
    args = MagicMock()
    args.job_id = job_id
    args.timeout = timeout
    args.interval = interval
    args.study = study
    args.stats_target = stats_target
    args.metrics = metrics or []
    return args


def _make_meta(status="FINISHED_OK", job_name="test-job", duration="0:01:30"):
    from nvflare.apis.job_def import JobMetaKey

    return {
        JobMetaKey.STATUS.value: status,
        JobMetaKey.JOB_NAME.value: job_name,
        JobMetaKey.DURATION.value: duration,
    }


@contextmanager
def _process_timezone(zone):
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset() is unavailable")
    previous = os.environ.get("TZ")
    try:
        os.environ["TZ"] = zone
        time.tzset()
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


@pytest.mark.parametrize(
    ("zone", "expected_time", "expected_zone"),
    [
        ("UTC", "2026-08-03 21:30:05", "UTC"),
        ("America/Los_Angeles", "2026-08-03 14:30:05", "PDT"),
    ],
)
def test_job_meta_timestamps_display_in_process_timezone(zone, expected_time, expected_zone):
    with _process_timezone(zone):
        submitted = _format_job_meta_timestamp("2026-08-03T14:30:05-07:00", include_zone=True)
        started = _format_job_meta_timestamp("2026-08-03T21:30:05+00:00")

    assert submitted == f"{expected_time} {expected_zone}"
    assert started == expected_time


@pytest.mark.parametrize(
    ("producer_zone", "consumer_zone"),
    [
        ("UTC", "America/Los_Angeles"),
        ("America/Los_Angeles", "UTC"),
        ("Asia/Kathmandu", "America/Los_Angeles"),
    ],
)
def test_canonical_start_is_timezone_independent(tmp_path, producer_zone, consumer_zone):
    from nvflare.apis.fl_context import FLContext
    from nvflare.apis.impl.job_def_manager import SimpleJobDefManager
    from nvflare.apis.job_def import JobMetaKey, RunStatus

    manager = SimpleJobDefManager(uri_root=str(tmp_path / "jobs"))
    store = MagicMock()
    with patch.object(manager, "_get_job_store", return_value=store):
        with _process_timezone(producer_zone):
            manager.set_status("job-1", RunStatus.RUNNING, FLContext())

    produced_meta = store.update_meta.call_args.kwargs["meta"]
    canonical_value = produced_meta[JobMetaKey.START_TIME.value]
    expected = datetime.datetime.fromisoformat(canonical_value).timestamp()

    with _process_timezone(consumer_zone):
        now = time.time()
        start_ts = _parse_monitor_start_ts(produced_meta, JobMetaKey.START_TIME.value, JobMetaKey.SUBMIT_TIME_ISO.value)

    assert start_ts == pytest.approx(expected)
    assert _calculate_monitor_elapsed(now, now - 1, start_ts) >= 0


@pytest.mark.parametrize(
    "canonical_value",
    [
        "2026-04-16T12:34:56+00:00",
        "2026-04-16T12:34:56-07:00",
        "2026-04-16T12:34:56+05:45",
    ],
)
def test_parse_monitor_start_accepts_explicit_offsets(canonical_value):
    from nvflare.apis.job_def import JobMetaKey

    expected = datetime.datetime.fromisoformat(canonical_value).timestamp()
    start_ts = _parse_monitor_start_ts(
        {JobMetaKey.START_TIME.value: canonical_value},
        JobMetaKey.START_TIME.value,
        JobMetaKey.SUBMIT_TIME_ISO.value,
    )

    assert start_ts == pytest.approx(expected)


def test_legacy_naive_start_clamps_negative_elapsed_to_zero():
    from nvflare.apis.job_def import JobMetaKey

    now = datetime.datetime.fromisoformat("2026-04-16T12:35:00+00:00").timestamp()
    meta = {JobMetaKey.START_TIME.value: "2026-04-16 12:34:56.000000"}

    with _process_timezone("America/Los_Angeles"):
        start_ts = _parse_monitor_start_ts(meta, JobMetaKey.START_TIME.value, JobMetaKey.SUBMIT_TIME_ISO.value)

    assert start_ts > now
    assert _calculate_monitor_elapsed(now, now - 1, start_ts) == 0


def test_malformed_start_uses_submit_time():
    from nvflare.apis.job_def import JobMetaKey

    submit_time = "2026-04-16T12:34:00+00:00"
    meta = {
        JobMetaKey.START_TIME.value: "not-a-timestamp",
        JobMetaKey.SUBMIT_TIME_ISO.value: submit_time,
    }

    start_ts = _parse_monitor_start_ts(meta, JobMetaKey.START_TIME.value, JobMetaKey.SUBMIT_TIME_ISO.value)

    assert start_ts == datetime.datetime.fromisoformat(submit_time).timestamp()


def test_future_start_clamps_elapsed_to_zero():
    now = 100.0
    assert _calculate_monitor_elapsed(now, 95.0, now + 60) == 0


def test_missing_start_uses_monitor_start():
    now = 100.0
    start_ts = _parse_monitor_start_ts({}, "start_time", "submit_time_iso")
    assert start_ts is None
    assert _calculate_monitor_elapsed(now, 95.0, start_ts) == 5.0


def test_human_jsonl_timeout_and_terminal_elapsed_share_calculation(capsys, monkeypatch):
    from nvflare.apis.job_def import JobMetaKey
    from nvflare.tool.job.job_cli import (
        _build_monitor_output_data,
        _build_monitor_terminal_event,
        _build_monitor_timeout_event,
        _emit_monitor_jsonl_progress,
        _emit_monitor_progress,
        _make_monitor_state,
    )

    now = 1_000.0
    monitor_start = 995.0
    job_start = 910.0
    running_meta = {
        JobMetaKey.STATUS.value: "RUNNING",
    }
    monitor_state = _make_monitor_state()

    monkeypatch.setattr(cli_output, "_output_format", "txt")
    _emit_monitor_progress("job-1", running_meta, monitor_state, now, monitor_start, job_start)
    human_output = capsys.readouterr().out

    _emit_monitor_jsonl_progress("job-1", running_meta, monitor_state, now, monitor_start, job_start)
    jsonl_event = json.loads(capsys.readouterr().out)

    with patch("nvflare.tool.job.job_cli.time.time", return_value=now):
        timeout_event = _build_monitor_timeout_event("job-1", 120, monitor_start, job_start, monitor_state)
        live_terminal = _build_monitor_output_data(
            "job-1",
            {JobMetaKey.STATUS.value: "RUNNING", JobMetaKey.DURATION.value: "N/A"},
            monitor_start,
            job_start,
            monitor_state,
            json_mode=True,
        )
        final_terminal = _build_monitor_output_data(
            "job-1",
            {JobMetaKey.STATUS.value: "FINISHED:COMPLETED", JobMetaKey.DURATION.value: "0:01:30"},
            monitor_start,
            job_start,
            monitor_state,
            json_mode=True,
        )
        terminal_event = _build_monitor_terminal_event(final_terminal)

    assert "elapsed_s: 90.0" in human_output
    assert jsonl_event["elapsed_s"] == 90.0
    assert timeout_event["elapsed_s"] == 90.0
    assert live_terminal["duration_s"] == 90.0
    assert final_terminal["duration_s"] == 90.0
    assert terminal_event["duration_s"] == 90.0


def test_parse_monitor_duration_seconds_preserves_zero():
    assert _parse_monitor_duration_seconds(0) == 0.0


def _mock_session(rc, meta):
    """Return a patch context for _session() that yields a mock session."""
    mock_sess = MagicMock()
    mock_sess.monitor_job_and_return_job_meta.return_value = (rc, meta)
    mock_sess.show_stats.return_value = {}

    @contextmanager
    def _fake_session(*args, **kwargs):
        yield mock_sess

    return (
        patch("nvflare.tool.job.job_cli._session", side_effect=_fake_session),
        mock_sess,
    )


class TestJobMonitorOutput:
    """Tests for nvflare job monitor output format and exit codes."""

    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    # ------------------------------------------------------------------
    # Terminal states — exit codes and envelope shape
    # ------------------------------------------------------------------

    def test_finished_ok_exits_0(self, capsys):
        meta = _make_meta("FINISHED_OK")
        ctx, _ = _mock_session(MonitorReturnCode.JOB_FINISHED, meta)
        with ctx:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            cmd_job_monitor(_make_args())

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert captured.err == ""
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert data["data"]["job_id"] == "abc123"
        assert data["data"]["status"] == "FINISHED_OK"

    @pytest.mark.parametrize(
        ("kwargs", "detail"),
        [
            ({"timeout": -1}, "--timeout must be >= 0"),
            ({"interval": 0}, "--interval must be > 0"),
            ({"interval": -1}, "--interval must be > 0"),
        ],
    )
    def test_invalid_monitor_arguments_exit_before_session_creation(self, capsys, kwargs, detail):
        from nvflare.tool.job.job_cli import cmd_job_monitor

        with patch("nvflare.tool.job.job_cli._job_session_for_args") as session_factory:
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args(**kwargs))

        assert exc_info.value.code == 4
        session_factory.assert_not_called()
        data = json.loads(capsys.readouterr().out)
        assert data["error_code"] == "INVALID_ARGS"
        assert detail in data["message"]

    @pytest.mark.parametrize(
        ("selector", "value"),
        [
            ("--startup-target", "prod"),
            ("--startup_target", "prod"),
            ("--startup_kit", "/tmp/startup"),
        ],
    )
    def test_monitor_parser_rejects_old_startup_selectors(self, selector, value):
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_job_cli_parser(subs)

        parser = job_sub_cmd_parser["monitor"]
        with pytest.raises(SystemExit):
            parser.parse_args(["abc123", selector, value])

    @pytest.mark.parametrize(
        ("selector", "value", "dest"),
        [
            ("--startup-kit", "/tmp/startup", "startup_kit"),
            ("--kit-id", "prod_admin", "kit_id"),
        ],
    )
    def test_monitor_parser_accepts_scoped_startup_selectors(self, selector, value, dest):
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_job_cli_parser(subs)

        parser = job_sub_cmd_parser["monitor"]
        args = parser.parse_args(["abc123", selector, value])

        assert getattr(args, dest) == value

    def test_monitor_help_and_schema_include_scoped_startup_selectors(self, capsys):
        import argparse

        from nvflare.tool.job.job_cli import cmd_job_monitor, def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        def_job_cli_parser(root.add_subparsers())

        help_text = job_sub_cmd_parser["monitor"].format_help()
        for token in ("--startup-target", "--startup_target", "--startup_kit"):
            assert token not in help_text
        assert "--study" in help_text
        assert "--startup-kit" in help_text
        assert "--kit-id" in help_text

        with patch("sys.argv", ["nvflare", "job", "monitor", "--schema"]):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(MagicMock())

        assert exc_info.value.code == 0
        schema_text = capsys.readouterr().out
        for token in ("--startup-target", "--startup_target", "--startup_kit"):
            assert token not in schema_text
        assert "--study" in schema_text
        assert "--startup-kit" in schema_text
        assert "--kit-id" in schema_text
        schema = json.loads(schema_text)
        assert schema["streaming"] is True
        assert schema["output_modes"] == ["json", "jsonl"]
        assert schema["mutating"] is False
        assert schema["idempotent"] is True
        assert schema["retry_token"] == {"supported": False}

    def test_failed_outputs_error_envelope_exits_1(self, capsys):
        meta = _make_meta("FAILED")
        ctx, _ = _mock_session(MonitorReturnCode.JOB_FINISHED, meta)
        with ctx:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args())
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "error"
        assert data["exit_code"] == 1
        assert data["error_code"] == "JOB_FAILED"
        assert "job logs" in data["hint"]
        assert "message" in data
        assert data["data"]["status"] == "FAILED"

    def test_aborted_outputs_error_envelope_exits_1(self, capsys):
        meta = _make_meta("ABORTED")
        ctx, _ = _mock_session(MonitorReturnCode.JOB_FINISHED, meta)
        with ctx:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args())
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "error"
        assert data["error_code"] == "JOB_ABORTED"
        assert "job meta" in data["hint"]
        assert data["data"]["status"] == "ABORTED"

    def test_abandoned_exits_1(self, capsys):
        meta = _make_meta("ABANDONED")
        ctx, _ = _mock_session(MonitorReturnCode.JOB_FINISHED, meta)
        with ctx:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args())
        assert exc_info.value.code == 1

        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "error"
        assert data["error_code"] == "JOB_ABANDONED"
        assert data["data"]["status"] == "ABANDONED"

    def test_finished_exception_exits_1(self, capsys):
        meta = _make_meta("FINISHED_EXCEPTION")
        ctx, _ = _mock_session(MonitorReturnCode.JOB_FINISHED, meta)
        with ctx:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args())
        assert exc_info.value.code == 1

        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "error"
        assert data["error_code"] == "JOB_FINISHED_EXCEPTION"
        assert data["data"]["status"] == "FINISHED_EXCEPTION"

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_timeout_exits_3_with_error_code(self, capsys):
        ctx, _ = _mock_session(MonitorReturnCode.TIMEOUT, None)
        with ctx:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args(timeout=10))
        assert exc_info.value.code == 3

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "error"
        assert data["error_code"] == "TIMEOUT"
        assert data["exit_code"] == 3

    def test_jsonl_timeout_emits_terminal_event(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "jsonl")
        ctx, _ = _mock_session(MonitorReturnCode.TIMEOUT, None)
        with ctx:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args(timeout=10))
        assert exc_info.value.code == 3

        lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert lines[-1]["status"] == "TIMEOUT"
        assert lines[-1]["terminal"] is True
        assert lines[-1]["timeout_seconds"] == 10

    def test_job_not_found_exits_1(self, capsys):
        @contextmanager
        def _fake_session(*args, **kwargs):
            sess = MagicMock()
            sess.monitor_job_and_return_job_meta.side_effect = JobNotFound("job does not exist")
            yield sess

        with patch("nvflare.tool.job.job_cli._session", side_effect=_fake_session):
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args())
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["error_code"] == "JOB_NOT_FOUND"
        assert "searched study 'default'" in data["message"]
        assert "nvflare job list --study <study_name>" in data["hint"]

    def test_job_not_found_human_hint_includes_default_study(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")

        @contextmanager
        def _fake_session(*args, **kwargs):
            sess = MagicMock()
            sess.monitor_job_and_return_job_meta.side_effect = JobNotFound("job does not exist")
            yield sess

        with patch("nvflare.tool.job.job_cli._session", side_effect=_fake_session):
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args())
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "Job 'abc123' does not exist. \u2014 searched study 'default'" in captured.err
        assert "This command searched study 'default'." in captured.err
        assert "nvflare job list --study <study_name>" in captured.err

    def test_job_not_found_human_hint_includes_named_study(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")

        @contextmanager
        def _fake_session(*args, **kwargs):
            sess = MagicMock()
            sess.monitor_job_and_return_job_meta.side_effect = JobNotFound("job does not exist")
            yield sess

        with patch("nvflare.tool.job.job_cli._session", side_effect=_fake_session):
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args(study="cancer"))
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "Job 'abc123' does not exist. \u2014 searched study 'cancer'" in captured.err
        assert "nvflare job list --study cancer" in captured.err

    def test_connection_error_emits_structured_envelope(self, capsys):
        @contextmanager
        def _fake_session(*args, **kwargs):
            sess = MagicMock()
            sess.monitor_job_and_return_job_meta.side_effect = NoConnection("connection refused")
            yield sess

        with patch("nvflare.tool.job.job_cli._session", side_effect=_fake_session):
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args())
        assert exc_info.value.code == 2

        envelope = json.loads(capsys.readouterr().out)
        assert envelope["status"] == "error"
        assert envelope["error_code"] == "CONNECTION_FAILED"
        assert envelope["exit_code"] == 2

    def test_jsonl_connection_error_emits_terminal_event(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "jsonl")

        @contextmanager
        def _fake_session(*args, **kwargs):
            sess = MagicMock()
            sess.monitor_job_and_return_job_meta.side_effect = NoConnection("connection refused")
            yield sess

        with patch("nvflare.tool.job.job_cli._session", side_effect=_fake_session):
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args())
        assert exc_info.value.code == 2

        event = json.loads(capsys.readouterr().out)
        assert event["event"] == "terminal"
        assert event["status"] == "error"
        assert event["terminal"] is True
        assert event["error_code"] == "CONNECTION_FAILED"

    def test_authentication_error_emits_structured_envelope(self, capsys):
        @contextmanager
        def _fake_session(*args, **kwargs):
            sess = MagicMock()
            sess.monitor_job_and_return_job_meta.side_effect = AuthenticationError("bad cert")
            yield sess

        with patch("nvflare.tool.job.job_cli._session", side_effect=_fake_session):
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args())
        assert exc_info.value.code == 2

        envelope = json.loads(capsys.readouterr().out)
        assert envelope["status"] == "error"
        assert envelope["error_code"] == "AUTH_FAILED"
        assert envelope["exit_code"] == 2

    def test_jsonl_authentication_error_emits_terminal_event(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "jsonl")

        @contextmanager
        def _fake_session(*args, **kwargs):
            sess = MagicMock()
            sess.monitor_job_and_return_job_meta.side_effect = AuthenticationError("bad cert")
            yield sess

        with patch("nvflare.tool.job.job_cli._session", side_effect=_fake_session):
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args())
        assert exc_info.value.code == 2

        event = json.loads(capsys.readouterr().out)
        assert event["event"] == "terminal"
        assert event["status"] == "error"
        assert event["terminal"] is True
        assert event["error_code"] == "AUTH_FAILED"

    def test_authorization_error_emits_structured_envelope(self, capsys):
        @contextmanager
        def _fake_session(*args, **kwargs):
            sess = MagicMock()
            sess.monitor_job_and_return_job_meta.side_effect = AuthorizationError("not authorized")
            yield sess

        with patch("nvflare.tool.job.job_cli._session", side_effect=_fake_session):
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args())
        assert exc_info.value.code == 2

        envelope = json.loads(capsys.readouterr().out)
        assert envelope["status"] == "error"
        assert envelope["error_code"] == "AUTH_FAILED"
        assert envelope["exit_code"] == 2

    def test_jsonl_authorization_error_emits_terminal_event(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "jsonl")

        @contextmanager
        def _fake_session(*args, **kwargs):
            sess = MagicMock()
            sess.monitor_job_and_return_job_meta.side_effect = AuthorizationError("not authorized")
            yield sess

        with patch("nvflare.tool.job.job_cli._session", side_effect=_fake_session):
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args())
        assert exc_info.value.code == 2

        event = json.loads(capsys.readouterr().out)
        assert event["event"] == "terminal"
        assert event["status"] == "error"
        assert event["terminal"] is True
        assert event["error_code"] == "AUTH_FAILED"

    def test_missing_meta_exits_internal_error(self, capsys):
        ctx, _ = _mock_session(MonitorReturnCode.JOB_FINISHED, None)
        with ctx:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args())
        assert exc_info.value.code == 5

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["error_code"] == "INTERNAL_ERROR"

    def test_unexpected_monitor_exception_exits_internal_error(self, capsys):
        @contextmanager
        def _fake_session(*args, **kwargs):
            sess = MagicMock()
            sess.monitor_job_and_return_job_meta.side_effect = KeyError("stats blew up")
            yield sess

        with patch("nvflare.tool.job.job_cli._session", side_effect=_fake_session):
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args())
        assert exc_info.value.code == 5

        data = json.loads(capsys.readouterr().out)
        assert data["error_code"] == "INTERNAL_ERROR"

    # ------------------------------------------------------------------
    # Envelope contents
    # ------------------------------------------------------------------

    def test_envelope_contains_duration_s(self, capsys):
        meta = _make_meta("FINISHED_OK", duration="0:01:30")
        ctx, _ = _mock_session(MonitorReturnCode.JOB_FINISHED, meta)
        with ctx:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            cmd_job_monitor(_make_args())

        data = json.loads(capsys.readouterr().out)
        assert "duration_s" in data["data"]
        assert data["data"]["duration_s"] == 90.0

    def test_envelope_contains_job_meta_summary(self, capsys):
        meta = _make_meta("FINISHED_OK", job_name="hello-pt")
        ctx, _ = _mock_session(MonitorReturnCode.JOB_FINISHED, meta)
        with ctx:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            cmd_job_monitor(_make_args())

        data = json.loads(capsys.readouterr().out)
        assert data["data"]["job_meta"]["job_name"] == "hello-pt"
        assert data["data"]["job_meta"]["status"] == "FINISHED_OK"

    def test_stats_raw_included_in_json_mode(self, capsys):
        """In json mode, stats_raw is included in the envelope data."""
        meta = _make_meta("FINISHED_OK")
        mock_sess = MagicMock()
        mock_sess.monitor_job_and_return_job_meta.return_value = (
            MonitorReturnCode.JOB_FINISHED,
            meta,
        )
        mock_sess.show_stats.return_value = {"server": {"round": 10, "loss": 0.05}}

        @contextmanager
        def _fake_session(*args, **kwargs):
            yield mock_sess

        with patch("nvflare.tool.job.job_cli._session", side_effect=_fake_session):
            from nvflare.tool.job.job_cli import cmd_job_monitor

            cmd_job_monitor(_make_args())

        data = json.loads(capsys.readouterr().out)
        assert "stats_raw" in data["data"]

    def test_monitor_uses_active_startup_kit_session(self, tmp_path, monkeypatch):
        active_admin_dir = _configure_active_startup_kit(tmp_path, monkeypatch)
        meta = _make_meta("FINISHED_OK")
        mock_sess = MagicMock()
        mock_sess.monitor_job_and_return_job_meta.return_value = (
            MonitorReturnCode.JOB_FINISHED,
            meta,
        )
        mock_sess.show_stats.return_value = {}

        with patch("nvflare.tool.cli_session.new_secure_session", return_value=mock_sess) as new_secure:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            cmd_job_monitor(_make_args())

        assert new_secure.call_args.kwargs["username"] == "admin@nvidia.com"
        assert new_secure.call_args.kwargs["startup_kit_location"] == str(active_admin_dir)

    def test_monitor_uses_named_study_session(self, capsys):
        meta = _make_meta("FINISHED_OK")
        mock_sess = MagicMock()
        mock_sess.monitor_job_and_return_job_meta.return_value = (
            MonitorReturnCode.JOB_FINISHED,
            meta,
        )
        mock_sess.show_stats.return_value = {}

        @contextmanager
        def _fake_session(*args, **kwargs):
            yield mock_sess

        with patch("nvflare.tool.job.job_cli._session", side_effect=_fake_session) as session_factory:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            cmd_job_monitor(_make_args(study="cancer"))

        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "ok"
        assert session_factory.call_args.kwargs["study"] == "cancer"

    def test_no_human_text_on_stdout(self, capsys):
        """In json mode, stdout contains only one JSON line."""
        meta = _make_meta("FINISHED_OK")
        ctx, _ = _mock_session(MonitorReturnCode.JOB_FINISHED, meta)
        with ctx:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            cmd_job_monitor(_make_args())

        captured = capsys.readouterr()
        stdout_lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert len(stdout_lines) == 1
        json.loads(stdout_lines[0])  # must be valid JSON

    def test_human_mode_prints_terminal_summary_without_json_payload(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        meta = _make_meta("FINISHED_OK", job_name="hello-pt")
        ctx, _ = _mock_session(MonitorReturnCode.JOB_FINISHED, meta)
        with ctx:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            cmd_job_monitor(_make_args())

        captured = capsys.readouterr()
        assert "Job abc123 status: COMPLETED (FINISHED_OK)" in captured.out
        assert "Name: hello-pt" in captured.out
        assert "job_meta:" not in captured.out
        assert "last_stats:" not in captured.out
        assert captured.err == ""

    def test_jsonl_finished_ok_emits_terminal_event(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "jsonl")
        meta = _make_meta("FINISHED_OK")
        ctx, _ = _mock_session(MonitorReturnCode.JOB_FINISHED, meta)
        with ctx:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            cmd_job_monitor(_make_args())

        stdout_lines = [json.loads(ln) for ln in capsys.readouterr().out.splitlines()]
        assert len(stdout_lines) == 1
        event = stdout_lines[0]
        assert event["schema_version"] == "1"
        assert event["event"] == "terminal"
        assert event["status"] == "COMPLETED"
        assert event["job_status"] == "FINISHED_OK"
        assert event["terminal"] is True

    def test_jsonl_failed_job_emits_terminal_event_and_exits_1(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "jsonl")
        meta = _make_meta("FAILED")
        ctx, _ = _mock_session(MonitorReturnCode.JOB_FINISHED, meta)
        with ctx:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args())
        assert exc_info.value.code == 1

        event = json.loads(capsys.readouterr().out.splitlines()[-1])
        assert event["event"] == "terminal"
        assert event["status"] == "FAILED"
        assert event["job_status"] == "FAILED"
        assert event["terminal"] is True
        assert event["error_code"] == "JOB_FAILED"

    def test_monitor_terminal_event_keeps_null_protocol_keys(self):
        from nvflare.tool.job.job_cli import _build_monitor_terminal_event

        event = _build_monitor_terminal_event({"status": "FINISHED_OK", "job_meta": {}})

        assert event["event"] == "terminal"
        assert event["job_id"] is None
        assert event["duration_s"] is None
        assert event["metrics"] is None

    # ------------------------------------------------------------------
    # ENDED_BY_CB path (callback stopped monitoring)
    # ------------------------------------------------------------------

    def test_ended_by_cb_uses_last_meta(self, capsys):
        """When rc is ENDED_BY_CB, result comes from cb_state last_meta."""
        terminal_meta = _make_meta("FINISHED_OK")

        def _side_effect(job_id, timeout, poll_interval, cb, state):
            # Simulate callback receiving terminal status and stopping
            state["last_meta"] = terminal_meta
            cb(MagicMock(), job_id, terminal_meta, state)
            return MonitorReturnCode.ENDED_BY_CB, None

        mock_sess = MagicMock()
        mock_sess.monitor_job_and_return_job_meta.side_effect = _side_effect
        mock_sess.show_stats.return_value = {}

        @contextmanager
        def _fake_session(*args, **kwargs):
            yield mock_sess

        with patch("nvflare.tool.job.job_cli._session", side_effect=_fake_session):
            from nvflare.tool.job.job_cli import cmd_job_monitor

            cmd_job_monitor(_make_args())

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert captured.err == ""
        assert data["status"] == "ok"
        assert data["data"]["status"] == "FINISHED_OK"

    def test_jsonl_progress_and_terminal_events(self, capsys, monkeypatch):
        """JSONL monitor emits progress events plus one terminal event."""
        monkeypatch.setattr(cli_output, "_output_format", "jsonl")
        running_meta = _make_meta("RUNNING")
        terminal_meta = _make_meta("FINISHED_OK")

        def _side_effect(job_id, timeout, poll_interval, cb, state):
            cb(mock_sess, job_id, running_meta, state)
            state["last_meta"] = terminal_meta
            cb(mock_sess, job_id, terminal_meta, state)
            return MonitorReturnCode.ENDED_BY_CB, None

        mock_sess = MagicMock()
        mock_sess.monitor_job_and_return_job_meta.side_effect = _side_effect
        mock_sess.show_stats.return_value = {"server": {"round": 1, "loss": 0.5}}

        @contextmanager
        def _fake_session(*args, **kwargs):
            yield mock_sess

        with patch("nvflare.tool.job.job_cli._session", side_effect=_fake_session):
            from nvflare.tool.job.job_cli import cmd_job_monitor

            cmd_job_monitor(_make_args())

        events = [json.loads(ln) for ln in capsys.readouterr().out.splitlines()]
        assert [event["event"] for event in events] == ["progress", "progress", "terminal"]
        assert events[0]["status"] == "RUNNING"
        assert events[0]["terminal"] is False
        assert events[-1]["status"] == "COMPLETED"
        assert events[-1]["terminal"] is True

    def test_jsonl_status_callback_stops_on_colon_finished_status(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "jsonl")

        from nvflare.tool.job.job_cli import _build_monitor_status_callback, _make_monitor_state

        state = _make_monitor_state()
        cb = _build_monitor_status_callback(
            start=0.0,
            start_ts_holder={"value": None},
            emit_interval=1,
            stats_interval=10,
            stats_target="server",
            key_aliases={},
            jsonl_mode=True,
        )

        should_continue = cb(MagicMock(), "abc123", _make_meta("FINISHED:COMPLETED"), state)

        assert should_continue is False
        event = json.loads(capsys.readouterr().out)
        assert event["event"] == "progress"
        assert event["status"] == "COMPLETED"
        assert event["job_status"] == "FINISHED:COMPLETED"
        assert event["terminal"] is False

    # ------------------------------------------------------------------
    # Parser
    # ------------------------------------------------------------------

    def test_parser_args(self):
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        def_job_cli_parser(root.add_subparsers())

        parser = job_sub_cmd_parser["monitor"]
        args = parser.parse_args(["abc123", "--timeout", "300", "--interval", "5", "--study", "cancer"])
        assert args.job_id == "abc123"
        assert args.timeout == 300
        assert args.interval == 5
        assert args.study == "cancer"

    def test_parser_defaults(self):
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        def_job_cli_parser(root.add_subparsers())

        parser = job_sub_cmd_parser["monitor"]
        args = parser.parse_args(["abc123"])
        assert args.timeout == 0
        assert args.interval == 2
        assert args.study == "default"
        assert args.stats_target == "server"
        assert args.metrics is None

    def test_parser_stats_target_and_metric(self):
        """--stats-target and --metric are parsed correctly."""
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        def_job_cli_parser(root.add_subparsers())

        parser = job_sub_cmd_parser["monitor"]
        args = parser.parse_args(
            [
                "abc123",
                "--stats-target",
                "client",
                "--metric",
                "loss",
                "--metric",
                "accuracy",
            ]
        )
        assert args.stats_target == "client"
        assert args.metrics == ["loss", "accuracy"]
