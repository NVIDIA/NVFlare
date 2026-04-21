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

from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.fl_constant import AdminCommandNames
from nvflare.fuel.flare_api.api_spec import MonitorReturnCode, TargetType
from nvflare.fuel.hci.client.api import ResultKey
from nvflare.fuel.hci.cmd_arg_utils import split_to_args
from nvflare.fuel.hci.proto import MetaKey, MetaStatusValue


def _make_session():
    """Return a Session with __init__ bypassed; we only test the methods."""
    from nvflare.fuel.flare_api.flare_api import Session

    session = object.__new__(Session)
    session.api = MagicMock()
    session.api.closed = False
    session.upload_dir = "/tmp/upload"
    session.download_dir = "/tmp/download"
    return session


def _ok_meta_result(meta_dict=None):
    """Build a minimal _do_command-style result with OK status."""
    from nvflare.fuel.hci.client.api import APIStatus

    return {
        ResultKey.STATUS: APIStatus.SUCCESS,
        ResultKey.META: {MetaKey.STATUS: MetaStatusValue.OK, **(meta_dict or {})},
        "data": [],
    }


class TestListJobs:
    def test_sends_list_jobs_command(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result({MetaKey.JOBS: []})) as mock_cmd:
            session.list_jobs()
        assert mock_cmd.call_count == 1
        cmd = mock_cmd.call_args[0][0]
        assert AdminCommandNames.LIST_JOBS in cmd

    def test_returns_jobs_list(self):
        session = _make_session()
        jobs = [{"id": "job1"}, {"id": "job2"}]
        with patch.object(session, "_do_command", return_value=_ok_meta_result({MetaKey.JOBS: jobs})):
            result = session.list_jobs()
        assert result == jobs

    def test_quotes_name_and_id_prefixes(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result({MetaKey.JOBS: []})) as mock_cmd:
            session.list_jobs(name_prefix="hello world", id_prefix="job 1")
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [AdminCommandNames.LIST_JOBS, "-n", "hello world", "job 1"]

    def test_accepts_legacy_kwargs(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result({MetaKey.JOBS: []})) as mock_cmd:
            session.list_jobs(max_num=5, job_id_prefix="job", job_name_prefix="hello")
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [AdminCommandNames.LIST_JOBS, "-m", "5", "-n", "hello", "job"]

    def test_rejects_unknown_legacy_kwargs(self):
        session = _make_session()
        with pytest.raises(TypeError, match="unsupported list_jobs kwargs"):
            session.list_jobs(unknown_filter="x")


class TestGetJobMeta:
    def test_sends_get_job_meta_command(self):
        session = _make_session()
        job_meta = {"id": "abc123", "status": "RUNNING"}
        with patch.object(
            session, "_do_command", return_value=_ok_meta_result({MetaKey.JOB_META: job_meta})
        ) as mock_cmd:
            session.get_job_meta("abc123")
        cmd = mock_cmd.call_args[0][0]
        assert AdminCommandNames.GET_JOB_META in cmd
        assert "abc123" in cmd

    def test_returns_job_meta(self):
        session = _make_session()
        job_meta = {"id": "abc123", "status": "RUNNING"}
        with patch.object(session, "_do_command", return_value=_ok_meta_result({MetaKey.JOB_META: job_meta})):
            result = session.get_job_meta("abc123")
        assert result == job_meta


class TestDeleteJob:
    def test_sends_delete_job_command(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.delete_job("abc123")
        cmd = mock_cmd.call_args[0][0]
        assert AdminCommandNames.DELETE_JOB in cmd
        assert "abc123" in cmd


class TestCheckStatus:
    def test_sends_check_status_server(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.check_status(TargetType.SERVER)
        cmd = mock_cmd.call_args[0][0]
        assert AdminCommandNames.CHECK_STATUS in cmd
        assert TargetType.SERVER in cmd

    def test_sends_check_status_client(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.check_status(TargetType.CLIENT)
        cmd = mock_cmd.call_args[0][0]
        assert AdminCommandNames.CHECK_STATUS in cmd

    def test_returns_meta(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result({"server_status": "running"})):
            result = session.check_status(TargetType.SERVER)
        assert "server_status" in result

    def test_quotes_client_targets(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.check_status(TargetType.CLIENT, ["site-1"])
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [AdminCommandNames.CHECK_STATUS, TargetType.CLIENT, "site-1"]

    def test_check_status_preserves_multiple_client_targets(self):
        # Regression: prior code joined multiple names into a single
        # whitespace-separated string and handed it to join_args, which wrapped
        # it in double quotes; server-side shlex.split then collapsed the names
        # back into one token, causing INVALID_CLIENT. Each name must round-trip
        # as its own command arg.
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.check_status(TargetType.CLIENT, ["site-1", "site-2"])
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [
            AdminCommandNames.CHECK_STATUS,
            TargetType.CLIENT,
            "site-1",
            "site-2",
        ]

    def test_get_client_job_status_quotes_client_names(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.get_client_job_status(["site-1"])
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [AdminCommandNames.CHECK_STATUS, TargetType.CLIENT, "site-1"]

    def test_get_client_job_status_preserves_multiple_client_names(self):
        # Regression: see test_check_status_preserves_multiple_client_targets.
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.get_client_job_status(["site-1", "site-2", "site-3"])
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [
            AdminCommandNames.CHECK_STATUS,
            TargetType.CLIENT,
            "site-1",
            "site-2",
            "site-3",
        ]


class TestReportResources:
    def test_sends_report_resources_command(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.report_resources(TargetType.SERVER)
        cmd = mock_cmd.call_args[0][0]
        assert AdminCommandNames.REPORT_RESOURCES in cmd

    def test_returns_resources_by_site(self):
        from nvflare.fuel.hci.client.api import APIStatus
        from nvflare.fuel.hci.proto import ProtoKey

        session = _make_session()
        reply = {
            ResultKey.STATUS: APIStatus.SUCCESS,
            ResultKey.META: {MetaKey.STATUS: MetaStatusValue.OK},
            ProtoKey.DATA: [
                {
                    ProtoKey.TYPE: ProtoKey.TABLE,
                    ProtoKey.ROWS: [["Sites", "Resources"], ["server", "unlimited"]],
                }
            ],
        }
        with patch.object(session, "_do_command", return_value=reply):
            result = session.report_resources(TargetType.SERVER)
        assert "server" in result
        assert result["server"] == "unlimited"

    def test_preserves_multiple_client_targets(self):
        # Consistency regression: report_resources today escapes the original
        # over-quoting bug only because it uses " ".join(parts) instead of
        # join_args(parts). Lock in the parts.extend(targets) pattern so that
        # each name round-trips as its own token regardless of which joiner is
        # used. See TestCheckStatus.test_check_status_preserves_multiple_client_targets.
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.report_resources(TargetType.CLIENT, ["site-1", "site-2"])
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [
            AdminCommandNames.REPORT_RESOURCES,
            TargetType.CLIENT,
            "site-1",
            "site-2",
        ]


class TestReportVersion:
    def test_sends_report_version_command(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.report_version(TargetType.SERVER)
        cmd = mock_cmd.call_args[0][0]
        assert AdminCommandNames.REPORT_VERSION in cmd

    def test_preserves_multiple_client_targets(self):
        # Consistency regression: see TestReportResources.test_preserves_multiple_client_targets.
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.report_version(TargetType.CLIENT, ["site-1", "site-2"])
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [
            AdminCommandNames.REPORT_VERSION,
            TargetType.CLIENT,
            "site-1",
            "site-2",
        ]


class TestShutdown:
    def test_rejects_invalid_target(self):
        session = _make_session()
        session.close = MagicMock()
        with pytest.raises(ValueError, match="shutdown target_type"):
            session.shutdown("invalid-target")

    def test_shuts_down_server_closes_session(self):
        session = _make_session()
        session.close = MagicMock()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()):
            session.shutdown(TargetType.SERVER)
        session.close.assert_called_once()

    def test_shutdown_preserves_result_if_close_fails(self):
        session = _make_session()
        session.close = MagicMock(side_effect=RuntimeError("logout failed"))
        expected_meta = _ok_meta_result()[ResultKey.META]

        with patch.object(session, "_do_command", return_value=_ok_meta_result()):
            result = session.shutdown(TargetType.SERVER)

        assert result == expected_meta
        session.close.assert_called_once()

    def test_shutdown_system_sends_all_target_command(self):
        session = _make_session()
        stopped_info = MagicMock()
        stopped_info.server_info.status = "stopped"

        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            with patch.object(session, "_do_get_system_info", return_value=stopped_info):
                session.shutdown_system()

        mock_cmd.assert_called_once_with(f"{AdminCommandNames.SHUTDOWN} {TargetType.ALL}")

    def test_shutdown_client_quotes_single_target(self):
        session = _make_session()
        session.close = MagicMock()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.shutdown(TargetType.CLIENT, ["site-1"])
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [AdminCommandNames.SHUTDOWN, TargetType.CLIENT, "site-1"]
        session.close.assert_not_called()

    def test_shutdown_preserves_multiple_client_targets(self):
        # Regression: see TestCheckStatus.test_check_status_preserves_multiple_client_targets.
        # PR #4462 restored shutdown's multi-target support but reintroduced the
        # over-quoting pattern — this test guards against that regression.
        session = _make_session()
        session.close = MagicMock()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.shutdown(TargetType.CLIENT, ["site-1", "site-2", "site-3"])
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [
            AdminCommandNames.SHUTDOWN,
            TargetType.CLIENT,
            "site-1",
            "site-2",
            "site-3",
        ]
        session.close.assert_not_called()


class TestRestart:
    def test_sends_restart_command(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.restart(TargetType.SERVER)
        cmd = mock_cmd.call_args[0][0]
        assert AdminCommandNames.RESTART in cmd

    def test_rejects_invalid_target(self):
        session = _make_session()
        with pytest.raises(ValueError, match="restart target_type"):
            session.restart("invalid-target")

    def test_restart_client_quotes_single_target(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.restart(TargetType.CLIENT, ["site-1"])
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [AdminCommandNames.RESTART, TargetType.CLIENT, "site-1"]

    def test_restart_preserves_multiple_client_targets(self):
        # Regression: see TestCheckStatus.test_check_status_preserves_multiple_client_targets.
        # PR #4462 restored restart's multi-target support but reintroduced the
        # over-quoting pattern — this test guards against that regression.
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.restart(TargetType.CLIENT, ["site-1", "site-2"])
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [
            AdminCommandNames.RESTART,
            TargetType.CLIENT,
            "site-1",
            "site-2",
        ]


class TestRemoveClient:
    def test_sends_remove_client_command(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.remove_client("site-1")
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [AdminCommandNames.REMOVE_CLIENT, "site-1"]

    def test_quotes_spaced_client_name(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.remove_client("site 1")
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [AdminCommandNames.REMOVE_CLIENT, "site 1"]

    def test_raises_on_empty_client_name(self):
        session = _make_session()
        with pytest.raises(ValueError):
            session.remove_client("")


class TestShowStats:
    def test_sends_show_stats_command(self):
        session = _make_session()
        from nvflare.fuel.hci.client.api import APIStatus

        reply = {ResultKey.STATUS: APIStatus.SUCCESS, ResultKey.META: None, "data": []}
        with patch.object(session, "_do_command", return_value=reply) as mock_cmd:
            session.show_stats("job1", TargetType.SERVER)
        cmd = mock_cmd.call_args[0][0]
        assert AdminCommandNames.SHOW_STATS in cmd
        assert "job1" in cmd

    def test_quotes_client_targets(self):
        session = _make_session()
        from nvflare.fuel.hci.client.api import APIStatus

        reply = {ResultKey.STATUS: APIStatus.SUCCESS, ResultKey.META: None, "data": []}
        with patch.object(session, "_do_command", return_value=reply) as mock_cmd:
            session.show_stats("job-1", TargetType.CLIENT, ["site-1"])
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [AdminCommandNames.SHOW_STATS, "job-1", TargetType.CLIENT, "site-1"]

    def test_preserves_multiple_client_targets(self):
        # Regression: see TestCheckStatus.test_check_status_preserves_multiple_client_targets.
        # show_stats / show_errors share _collect_info helper in flare_api.py, which
        # was the third caller affected by the over-quoting bug.
        session = _make_session()
        from nvflare.fuel.hci.client.api import APIStatus

        reply = {ResultKey.STATUS: APIStatus.SUCCESS, ResultKey.META: None, "data": []}
        with patch.object(session, "_do_command", return_value=reply) as mock_cmd:
            session.show_stats("job-1", TargetType.CLIENT, ["site-1", "site-2"])
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [
            AdminCommandNames.SHOW_STATS,
            "job-1",
            TargetType.CLIENT,
            "site-1",
            "site-2",
        ]


class TestShowErrors:
    def test_sends_show_errors_command(self):
        session = _make_session()
        from nvflare.fuel.hci.client.api import APIStatus

        reply = {ResultKey.STATUS: APIStatus.SUCCESS, ResultKey.META: None, "data": []}
        with patch.object(session, "_do_command", return_value=reply) as mock_cmd:
            session.show_errors("job1", TargetType.SERVER)
        cmd = mock_cmd.call_args[0][0]
        assert AdminCommandNames.SHOW_ERRORS in cmd
        assert "job1" in cmd

    def test_quotes_client_targets(self):
        session = _make_session()
        from nvflare.fuel.hci.client.api import APIStatus

        reply = {ResultKey.STATUS: APIStatus.SUCCESS, ResultKey.META: None, "data": []}
        with patch.object(session, "_do_command", return_value=reply) as mock_cmd:
            session.show_errors("job-1", TargetType.CLIENT, ["site-1"])
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [AdminCommandNames.SHOW_ERRORS, "job-1", TargetType.CLIENT, "site-1"]

    def test_preserves_multiple_client_targets(self):
        # Regression: see TestCheckStatus.test_check_status_preserves_multiple_client_targets.
        session = _make_session()
        from nvflare.fuel.hci.client.api import APIStatus

        reply = {ResultKey.STATUS: APIStatus.SUCCESS, ResultKey.META: None, "data": []}
        with patch.object(session, "_do_command", return_value=reply) as mock_cmd:
            session.show_errors("job-1", TargetType.CLIENT, ["site-1", "site-2"])
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [
            AdminCommandNames.SHOW_ERRORS,
            "job-1",
            TargetType.CLIENT,
            "site-1",
            "site-2",
        ]


class TestGetJobLogs:
    def test_sends_get_job_log_command(self):
        session = _make_session()
        from nvflare.fuel.hci.client.api import APIStatus

        reply = {ResultKey.STATUS: APIStatus.SUCCESS, ResultKey.META: None, "data": []}
        with patch.object(session, "_do_command", return_value=reply) as mock_cmd:
            session.get_job_logs("job1")
        cmd = mock_cmd.call_args[0][0]
        assert "get_job_log" in cmd
        assert "job1" in cmd

    def test_includes_tail_lines_flag(self):
        session = _make_session()
        from nvflare.fuel.hci.client.api import APIStatus

        reply = {ResultKey.STATUS: APIStatus.SUCCESS, ResultKey.META: None, "data": []}
        with patch.object(session, "_do_command", return_value=reply) as mock_cmd:
            session.get_job_logs("job1", tail_lines=50)
        cmd = mock_cmd.call_args[0][0]
        assert "-n" in cmd
        assert "50" in cmd

    @pytest.mark.parametrize("tail_lines", [0, -1])
    def test_rejects_non_positive_tail_lines(self, tail_lines):
        session = _make_session()

        with pytest.raises(ValueError, match="greater than 0"):
            session.get_job_logs("job1", tail_lines=tail_lines)

    def test_rejects_non_integer_tail_lines(self):
        session = _make_session()

        with pytest.raises(ValueError, match="tail_lines must be int"):
            session.get_job_logs("job1", tail_lines=2.5)

    def test_includes_grep_pattern_flag(self):
        session = _make_session()
        from nvflare.fuel.hci.client.api import APIStatus

        reply = {ResultKey.STATUS: APIStatus.SUCCESS, ResultKey.META: None, "data": []}
        with patch.object(session, "_do_command", return_value=reply) as mock_cmd:
            session.get_job_logs("job1", grep_pattern="ERROR")
        cmd = mock_cmd.call_args[0][0]
        assert "-g" in cmd
        assert "ERROR" in cmd

    def test_quotes_multi_word_grep_pattern(self):
        session = _make_session()
        from nvflare.fuel.hci.client.api import APIStatus

        reply = {ResultKey.STATUS: APIStatus.SUCCESS, ResultKey.META: None, "data": []}
        with patch.object(session, "_do_command", return_value=reply) as mock_cmd:
            session.get_job_logs("job1", grep_pattern="CUDA out of memory")
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [
            AdminCommandNames.GET_JOB_LOG,
            "job1",
            "-g",
            "CUDA out of memory",
        ]

    def test_quotes_grep_pattern_with_embedded_double_quote(self):
        session = _make_session()
        from nvflare.fuel.hci.client.api import APIStatus

        reply = {ResultKey.STATUS: APIStatus.SUCCESS, ResultKey.META: None, "data": []}
        with patch.object(session, "_do_command", return_value=reply) as mock_cmd:
            session.get_job_logs("job1", grep_pattern='foo"bar')
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [
            AdminCommandNames.GET_JOB_LOG,
            "job1",
            "-g",
            'foo"bar',
        ]

    def test_ignores_empty_grep_pattern(self):
        session = _make_session()
        from nvflare.fuel.hci.client.api import APIStatus

        reply = {ResultKey.STATUS: APIStatus.SUCCESS, ResultKey.META: None, "data": []}
        with patch.object(session, "_do_command", return_value=reply) as mock_cmd:
            session.get_job_logs("job1", grep_pattern="")
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [AdminCommandNames.GET_JOB_LOG, "job1"]

    def test_grep_target_quotes_pattern_with_embedded_double_quote(self):
        session = _make_session()
        from nvflare.fuel.hci.client.api import APIStatus

        reply = {
            ResultKey.STATUS: APIStatus.SUCCESS,
            ResultKey.META: None,
            "data": [{"type": "string", "data": "matched"}],
        }
        with patch.object(session, "_do_command", return_value=reply) as mock_cmd:
            result = session.grep_target("server", pattern='foo"bar', file="log.txt")

        assert result == "matched"
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == ["grep", "server", 'foo"bar', "log.txt"]

    def test_returns_logs_dict(self):
        session = _make_session()
        from nvflare.fuel.hci.client.api import APIStatus

        reply = {ResultKey.STATUS: APIStatus.SUCCESS, ResultKey.META: None, "data": []}
        with patch.object(session, "_do_command", return_value=reply):
            result = session.get_job_logs("job1")
        assert "logs" in result

    def test_rejects_non_server_target(self):
        session = _make_session()

        with pytest.raises(ValueError, match="only supports target='server'"):
            session.get_job_logs("job1", target="all")


class TestConfigureJobLog:
    def test_sends_configure_job_log_command(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.configure_job_log("job1", "INFO")
        cmd = mock_cmd.call_args[0][0]
        assert AdminCommandNames.CONFIGURE_JOB_LOG in cmd
        assert "job1" in cmd
        assert "INFO" in cmd

    def test_sends_dict_config_as_json(self):
        session = _make_session()
        import json

        config = {"version": 1, "disable_existing_loggers": False}
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.configure_job_log("job1", config)
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [
            AdminCommandNames.CONFIGURE_JOB_LOG,
            "job1",
            "all",
            json.dumps(config),
        ]

    def test_uses_target_parameter(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.configure_job_log("job1", "DEBUG", target="site-1")
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [
            AdminCommandNames.CONFIGURE_JOB_LOG,
            "job1",
            "client",
            "site-1",
            "DEBUG",
        ]

    def test_site_named_client_is_treated_as_explicit_client(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.configure_job_log("job1", "DEBUG", target="client")
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [
            AdminCommandNames.CONFIGURE_JOB_LOG,
            "job1",
            "client",
            "client",
            "DEBUG",
        ]

    def test_disables_meta_enforcement(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.configure_job_log("job1", "INFO")
        assert mock_cmd.call_args.kwargs["enforce_meta"] is False

    def test_quotes_string_config_with_spaces(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.configure_job_log("job1", "/my workspace/log.conf")
        cmd = mock_cmd.call_args[0][0]
        assert split_to_args(cmd) == [
            AdminCommandNames.CONFIGURE_JOB_LOG,
            "job1",
            "all",
            "/my workspace/log.conf",
        ]


class TestConfigureSiteLog:
    def test_sends_configure_site_log_command(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.configure_site_log("WARNING")
        cmd = mock_cmd.call_args[0][0]
        assert AdminCommandNames.CONFIGURE_SITE_LOG in cmd
        assert "WARNING" in cmd

    def test_rejects_dict_config(self):
        session = _make_session()
        config = {"version": 1, "disable_existing_loggers": False}
        with pytest.raises(ValueError, match="configure_site_log only supports log levels and built-in log modes"):
            session.configure_site_log(config)

    def test_uses_target_parameter(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.configure_site_log("INFO", target="site-2")
        cmd = mock_cmd.call_args[0][0]
        assert "site-2" in cmd

    def test_disables_meta_enforcement(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.configure_site_log("WARNING")
        assert mock_cmd.call_args.kwargs["enforce_meta"] is False

    def test_rejects_file_path_style_config(self):
        session = _make_session()
        with pytest.raises(ValueError, match="configure_site_log only supports log levels and built-in log modes"):
            session.configure_site_log("/my workspace/log.conf", target="site-2")


class TestWaitForJob:
    def test_job_timeout_is_not_internal_error(self):
        from nvflare.fuel.flare_api.api_spec import InternalError, JobTimeout

        assert not issubclass(JobTimeout, InternalError)

    def test_returns_job_meta_on_finish(self):
        session = _make_session()
        job_meta = {"status": "FINISHED_OK", "id": "job1"}
        with patch.object(
            session,
            "monitor_job_and_return_job_meta",
            return_value=(MonitorReturnCode.JOB_FINISHED, job_meta),
        ):
            result = session.wait_for_job("job1")
        assert result == job_meta

    def test_raises_timeout_error_on_timeout(self):
        session = _make_session()
        from nvflare.fuel.flare_api.api_spec import JobTimeout

        with patch.object(
            session,
            "monitor_job_and_return_job_meta",
            return_value=(MonitorReturnCode.TIMEOUT, None),
        ):
            with pytest.raises(JobTimeout):
                session.wait_for_job("job1", timeout=5.0)
