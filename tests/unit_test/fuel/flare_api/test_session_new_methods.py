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


class TestShutdown:
    def test_sends_shutdown_command(self):
        session = _make_session()
        session.close = MagicMock()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.shutdown(TargetType.CLIENT)
        cmd = mock_cmd.call_args[0][0]
        assert AdminCommandNames.SHUTDOWN in cmd

    def test_shuts_down_server_closes_session(self):
        session = _make_session()
        session.close = MagicMock()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()):
            session.shutdown(TargetType.SERVER)
        session.close.assert_called_once()


class TestRestart:
    def test_sends_restart_command(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.restart(TargetType.SERVER)
        cmd = mock_cmd.call_args[0][0]
        assert AdminCommandNames.RESTART in cmd


class TestRemoveClient:
    def test_sends_remove_client_command(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.remove_client("site-1")
        cmd = mock_cmd.call_args[0][0]
        assert AdminCommandNames.REMOVE_CLIENT in cmd
        assert "site-1" in cmd

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
        assert "-g" in cmd
        assert "'CUDA out of memory'" in cmd

    def test_returns_logs_dict(self):
        session = _make_session()
        from nvflare.fuel.hci.client.api import APIStatus

        reply = {ResultKey.STATUS: APIStatus.SUCCESS, ResultKey.META: None, "data": []}
        with patch.object(session, "_do_command", return_value=reply):
            result = session.get_job_logs("job1")
        assert "logs" in result


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
        import shlex

        config = {"version": 1, "disable_existing_loggers": False}
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.configure_job_log("job1", config)
        cmd = mock_cmd.call_args[0][0]
        assert AdminCommandNames.CONFIGURE_JOB_LOG in cmd
        assert "job1" in cmd
        assert shlex.quote(json.dumps(config)) in cmd

    def test_uses_target_parameter(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.configure_job_log("job1", "DEBUG", target="site-1")
        cmd = mock_cmd.call_args[0][0]
        assert f"{AdminCommandNames.CONFIGURE_JOB_LOG} job1 client site-1 DEBUG" == cmd

    def test_disables_meta_enforcement(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.configure_job_log("job1", "INFO")
        assert mock_cmd.call_args.kwargs["enforce_meta"] is False


class TestConfigureSiteLog:
    def test_sends_configure_site_log_command(self):
        session = _make_session()
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.configure_site_log("WARNING")
        cmd = mock_cmd.call_args[0][0]
        assert AdminCommandNames.CONFIGURE_SITE_LOG in cmd
        assert "WARNING" in cmd

    def test_sends_dict_config_as_json(self):
        session = _make_session()
        import json
        import shlex

        config = {"version": 1, "disable_existing_loggers": False}
        with patch.object(session, "_do_command", return_value=_ok_meta_result()) as mock_cmd:
            session.configure_site_log(config)
        cmd = mock_cmd.call_args[0][0]
        assert AdminCommandNames.CONFIGURE_SITE_LOG in cmd
        assert shlex.quote(json.dumps(config)) in cmd

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


class TestWaitForJob:
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
        with patch.object(
            session,
            "monitor_job_and_return_job_meta",
            return_value=(MonitorReturnCode.TIMEOUT, None),
        ):
            with pytest.raises(TimeoutError):
                session.wait_for_job("job1", timeout=5.0)
