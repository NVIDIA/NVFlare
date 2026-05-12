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

from nvflare.apis.fl_constant import AdminCommandNames, SUBMIT_TOKEN_JOB_DELETED_STATUS
from nvflare.apis.fl_exception import FLCommunicationError
from nvflare.apis.job_def import JobMetaKey
from nvflare.fuel.flare_api.api_spec import (
    AuthenticationError,
    ClientInfo,
    InternalError,
    InvalidArgumentError,
    InvalidJobDefinition,
    JobNotFound,
    MonitorReturnCode,
    NoConnection,
    ServerInfo,
    SubmitTokenJobDeleted,
)
from nvflare.fuel.flare_api.flare_api import Session, new_session
from nvflare.fuel.hci.client.api import APIStatus, ResultKey
from nvflare.fuel.hci.proto import MetaKey, MetaStatusValue


@pytest.mark.parametrize("job_name", ["fox training poc", ".hidden-job", "-flaggy-job"])
def test_submit_job_rejects_invalid_job_folder_names(tmp_path, job_name):
    job_dir = tmp_path / job_name
    job_dir.mkdir()

    session = Session.__new__(Session)
    session.upload_dir = str(tmp_path)
    session._do_command = MagicMock()

    with pytest.raises(InvalidJobDefinition, match=rf"job folder name '{job_name}'.*no spaces"):
        session.submit_job(str(job_dir))

    session._do_command.assert_not_called()


def test_submit_job_accepts_valid_job_folder_name(tmp_path):
    job_dir = tmp_path / "fox-training_poc.1"
    job_dir.mkdir()

    session = Session.__new__(Session)
    session.upload_dir = str(tmp_path)
    session._do_command = MagicMock(
        return_value={
            ResultKey.STATUS: "SUCCESS",
            ResultKey.META: {MetaKey.STATUS: MetaStatusValue.OK, MetaKey.JOB_ID: "job-1"},
        }
    )

    assert session.submit_job(str(job_dir)) == "job-1"
    session._do_command.assert_called_once()


def test_submit_job_forwards_submit_token(tmp_path):
    job_dir = tmp_path / "fox-training"
    job_dir.mkdir()

    session = Session.__new__(Session)
    session.upload_dir = str(tmp_path)
    session._do_command = MagicMock(
        return_value={
            ResultKey.STATUS: "SUCCESS",
            ResultKey.META: {MetaKey.STATUS: MetaStatusValue.OK, MetaKey.JOB_ID: "job-1"},
        }
    )

    assert session.submit_job(str(job_dir), submit_token="retry.01:A_b-c") == "job-1"

    command = session._do_command.call_args.args[0]
    assert command.startswith("submit_job ")
    assert str(job_dir) in command
    assert "--submit-token retry.01:A_b-c" in command


@pytest.mark.parametrize("token", ["", "bad token", "bad/token", "x" * 129])
def test_submit_job_rejects_invalid_submit_token(tmp_path, token):
    job_dir = tmp_path / "fox-training"
    job_dir.mkdir()

    session = Session.__new__(Session)
    session.upload_dir = str(tmp_path)
    session._do_command = MagicMock()

    with pytest.raises(InvalidArgumentError, match="submit_token"):
        session.submit_job(str(job_dir), submit_token=token)

    session._do_command.assert_not_called()


def test_list_jobs_forwards_submit_token():
    session = Session.__new__(Session)
    session._do_command = MagicMock(
        return_value={
            ResultKey.STATUS: "SUCCESS",
            ResultKey.META: {MetaKey.STATUS: MetaStatusValue.OK, MetaKey.JOBS: []},
        }
    )

    assert session.list_jobs(submit_token="retry.01:A_b-c") == []

    assert session._do_command.call_args.args[0] == "list_jobs --submit-token retry.01:A_b-c"


@pytest.mark.parametrize("token", ["", "bad token", "bad/token", "x" * 129])
def test_list_jobs_rejects_invalid_submit_token(token):
    session = Session.__new__(Session)
    session._do_command = MagicMock()

    with pytest.raises(InvalidArgumentError, match="submit_token"):
        session.list_jobs(submit_token=token)

    session._do_command.assert_not_called()


def test_delete_job_returns_submit_record_tombstone_count():
    session = Session.__new__(Session)
    session._do_command = MagicMock(
        return_value={
            ResultKey.STATUS: "SUCCESS",
            ResultKey.META: {
                MetaKey.STATUS: MetaStatusValue.OK,
                MetaKey.JOB_ID: "job-1",
                "submit_records_marked_deleted": 1,
            },
        }
    )

    assert session.delete_job("job-1") == {"job_id": "job-1", "submit_records_marked_deleted": 1}
    session._do_command.assert_called_once_with("delete_job job-1")


@pytest.mark.parametrize("status", ["FINISHED:COMPLETED", "FAILED", "ABORTED", "ABANDONED", "FINISHED_EXCEPTION"])
def test_monitor_job_stops_on_terminal_statuses(status):
    session = Session.__new__(Session)
    meta = {JobMetaKey.STATUS.value: status}
    session.get_job_meta = MagicMock(return_value=meta)

    with patch("nvflare.fuel.flare_api.flare_api.time.sleep") as sleep:
        rc, returned_meta = session.monitor_job_and_return_job_meta("job-1", timeout=60, poll_interval=10)

    assert rc == MonitorReturnCode.JOB_FINISHED
    assert returned_meta is meta
    session.get_job_meta.assert_called_once_with("job-1")
    sleep.assert_not_called()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"timeout": -1}, "timeout must be >= 0"),
        ({"poll_interval": 0}, "poll_interval must be > 0"),
        ({"poll_interval": -1}, "poll_interval must be > 0"),
    ],
)
def test_monitor_job_rejects_invalid_polling_options(kwargs, match):
    session = Session.__new__(Session)
    session.get_job_meta = MagicMock()

    with pytest.raises(InvalidArgumentError, match=match):
        session.monitor_job_and_return_job_meta("job-1", **kwargs)

    session.get_job_meta.assert_not_called()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"timeout": -1}, "timeout must be >= 0"),
        ({"poll_interval": 0}, "poll_interval must be > 0"),
    ],
)
def test_wait_for_job_rejects_invalid_polling_options(kwargs, match):
    session = Session.__new__(Session)
    session.get_job_meta = MagicMock()

    with pytest.raises(InvalidArgumentError, match=match):
        session.wait_for_job("job-1", **kwargs)

    session.get_job_meta.assert_not_called()


def test_do_command_includes_syntax_error_details():
    session = Session.__new__(Session)
    session.api = MagicMock()
    session.api.closed = False
    session.api.do_command.return_value = {
        ResultKey.STATUS: APIStatus.ERROR_SYNTAX,
        ResultKey.DETAILS: "usage: submit_job job_folder",
    }

    with pytest.raises(InternalError, match=r"protocol error: ERROR_SYNTAX: usage: submit_job job_folder"):
        session._do_command("submit_job /tmp/job", enforce_meta=False)


def test_do_command_raises_no_connection_for_server_connection_error():
    session = Session.__new__(Session)
    session.api = MagicMock()
    session.api.closed = False
    session.api.do_command.return_value = {
        ResultKey.STATUS: APIStatus.ERROR_SERVER_CONNECTION,
        ResultKey.DETAILS: "connection refused",
    }

    with pytest.raises(NoConnection, match=r"cannot connect to server: ERROR_SERVER_CONNECTION"):
        session._do_command("list_jobs", enforce_meta=False)


@pytest.mark.parametrize(
    "command",
    [
        f"{AdminCommandNames.ABORT_JOB} job-1",
        f"{AdminCommandNames.SHUTDOWN} all",
    ],
)
def test_do_command_retries_idempotent_command_connection_failure(command):
    session = Session.__new__(Session)
    session.api = MagicMock()
    session.api.closed = False
    session.api.do_command.side_effect = [
        {
            ResultKey.STATUS: APIStatus.ERROR_SERVER_CONNECTION,
            ResultKey.DETAILS: "Server did not respond",
        },
        {
            ResultKey.STATUS: APIStatus.ERROR_SERVER_CONNECTION,
            ResultKey.DETAILS: "Server did not respond",
        },
        {
            ResultKey.STATUS: APIStatus.SUCCESS,
            ResultKey.META: {MetaKey.STATUS: MetaStatusValue.OK, MetaKey.INFO: "Command completed."},
        },
    ]

    with patch("nvflare.fuel.flare_api.flare_api.time.sleep") as sleep:
        result = session._do_command(command)

    assert result[ResultKey.STATUS] == APIStatus.SUCCESS
    assert session.api.do_command.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [0.5, 1.0]


def test_do_command_does_not_retry_submit_job_connection_failure():
    session = Session.__new__(Session)
    session.api = MagicMock()
    session.api.closed = False
    session.api.do_command.return_value = {
        ResultKey.STATUS: APIStatus.ERROR_SERVER_CONNECTION,
        ResultKey.DETAILS: "Server did not respond",
    }

    with patch("nvflare.fuel.flare_api.flare_api.time.sleep") as sleep:
        with pytest.raises(NoConnection, match=r"cannot connect to server: ERROR_SERVER_CONNECTION"):
            session._do_command(f"{AdminCommandNames.SUBMIT_JOB} /tmp/job", enforce_meta=False)

    session.api.do_command.assert_called_once()
    sleep.assert_not_called()


def test_try_connect_maps_transient_communication_failure_to_no_connection():
    session = Session.__new__(Session)
    session.api = MagicMock()
    session.api.closed = False
    session.api.connect.side_effect = FLCommunicationError("cannot connect to server for 4.0 seconds")

    with pytest.raises(NoConnection, match="cannot connect to server for 4.0 seconds"):
        session.try_connect(4.0)


def test_no_connection_is_connection_error():
    assert issubclass(NoConnection, ConnectionError)


def test_do_command_raises_authentication_error_for_error_cert():
    session = Session.__new__(Session)
    session.api = MagicMock()
    session.api.closed = False
    session.api.do_command.return_value = {
        ResultKey.STATUS: APIStatus.ERROR_CERT,
        ResultKey.DETAILS: "certificate validation failed",
    }

    with pytest.raises(AuthenticationError, match="certificate validation failed"):
        session._do_command("list_jobs", enforce_meta=False)


def test_do_command_maps_deleted_submit_token_status():
    session = Session.__new__(Session)
    session.api = MagicMock()
    session.api.closed = False
    session.api.do_command.return_value = {
        ResultKey.STATUS: APIStatus.SUCCESS,
        ResultKey.META: {
            MetaKey.STATUS: SUBMIT_TOKEN_JOB_DELETED_STATUS,
            MetaKey.INFO: "submit token refers to a deleted job",
            MetaKey.JOB_ID: "job-1",
            "submit_record_state": "job_deleted",
            "deleted_time": "2026-04-30T10:00:00-07:00",
        },
    }

    with pytest.raises(SubmitTokenJobDeleted) as exc_info:
        session._do_command("list_jobs --submit-token retry-1")

    assert exc_info.value.job_id == "job-1"
    assert exc_info.value.state == "job_deleted"
    assert exc_info.value.deleted_time == "2026-04-30T10:00:00-07:00"


def test_new_session_closes_session_on_connect_failure():
    fake_session = MagicMock()
    fake_session.try_connect.side_effect = NoConnection("cannot connect")

    with patch("nvflare.fuel.flare_api.flare_api.Session", return_value=fake_session):
        with pytest.raises(NoConnection):
            new_session("admin@nvidia.com", "/tmp/startup", timeout=5.0)

    fake_session.close.assert_called_once()


def test_new_session_preserves_connect_failure_when_close_also_fails():
    fake_session = MagicMock()
    fake_session.try_connect.side_effect = NoConnection("cannot connect")
    fake_session.close.side_effect = RuntimeError("logout failed")
    fake_logger = MagicMock()

    with patch("nvflare.fuel.flare_api.flare_api.Session", return_value=fake_session):
        with patch("nvflare.fuel.flare_api.flare_api.get_obj_logger", return_value=fake_logger):
            with pytest.raises(NoConnection, match="cannot connect"):
                new_session("admin@nvidia.com", "/tmp/startup", timeout=5.0)

    fake_session.close.assert_called_once()
    fake_logger.debug.assert_called_once_with(
        "failed to close partially initialized session during cleanup: %s",
        fake_session.close.side_effect,
    )


def test_new_session_applies_cli_relevant_session_options():
    fake_session = MagicMock()
    fake_session.api = MagicMock()

    with patch("nvflare.fuel.flare_api.flare_api.Session", return_value=fake_session):
        returned = new_session(
            "admin@nvidia.com",
            "/tmp/startup",
            timeout=5.0,
            command_timeout=2.5,
            auto_login_max_tries=1,
        )

    fake_session.set_timeout.assert_called_once_with(2.5)
    assert fake_session.api.auto_login_max_tries == 1
    fake_session.try_connect.assert_called_once_with(5.0)
    assert returned is fake_session


def test_validate_job_id_rejects_non_string_job_id():
    with pytest.raises(JobNotFound, match="invalid job_id 0"):
        Session._validate_job_id(0)


def test_server_info_str_handles_none_start_time():
    assert str(ServerInfo("running", None)) == "status: running, start_time: unknown"


def test_client_info_str_handles_none_last_connect_time():
    assert str(ClientInfo("site-1", None)) == "site-1 (last_connect_time: unknown)"


def _make_session_with_meta(meta: dict):
    session = Session.__new__(Session)
    session._do_command = MagicMock(
        return_value={
            ResultKey.STATUS: "SUCCESS",
            ResultKey.META: {MetaKey.STATUS: MetaStatusValue.OK, **meta},
        }
    )
    session.close = MagicMock()
    return session


def test_new_poll_session_preserves_debug_setting():
    sess = Session.__new__(Session)
    sess.username = "admin@nvidia.com"
    sess.startup_path = "/tmp/startup"
    sess.secure_mode = True
    sess._debug = True
    sess._study = "cancer"

    with patch("nvflare.fuel.flare_api.flare_api.Session") as session_cls:
        sess._new_poll_session()

    session_cls.assert_called_once_with(
        username="admin@nvidia.com",
        startup_path="/tmp/startup",
        secure_mode=True,
        debug=True,
        study="cancer",
    )


class TestSessionShutdown:
    def test_shutdown_server_sends_correct_command(self):
        sess = _make_session_with_meta({})
        sess.shutdown("server", wait=False)
        sess._do_command.assert_called_once_with("shutdown server")

    def test_shutdown_client_sends_correct_command(self):
        sess = _make_session_with_meta({})
        sess.shutdown("client", wait=False)
        sess._do_command.assert_called_once_with("shutdown client")

    def test_shutdown_all_sends_correct_command(self):
        sess = _make_session_with_meta({})
        sess.shutdown("all", wait=False)
        sess._do_command.assert_called_once_with("shutdown all")

    def test_shutdown_client_with_names_sends_correct_command(self):
        sess = _make_session_with_meta({})
        sess.shutdown("client", client_names=["site-1", "site-2"], wait=False)
        cmd = sess._do_command.call_args[0][0]
        assert cmd.startswith("shutdown client")
        assert "site-1" in cmd
        assert "site-2" in cmd

    def test_shutdown_server_closes_session(self):
        sess = _make_session_with_meta({})
        sess._wait_for_server_down = MagicMock()
        sess.shutdown("server")
        sess.close.assert_called_once()
        sess._wait_for_server_down.assert_called_once()

    def test_shutdown_rejects_zero_wait_timeout_before_command(self):
        sess = _make_session_with_meta({})
        with pytest.raises(ValueError, match="timeout must be a positive number"):
            sess.shutdown("server", timeout=0)
        sess._do_command.assert_not_called()

    def test_shutdown_all_closes_session(self):
        sess = _make_session_with_meta({})
        sess._wait_for_server_down = MagicMock()
        sess.shutdown("all")
        sess.close.assert_called_once()
        sess._wait_for_server_down.assert_called_once()

    def test_shutdown_client_does_not_close_session(self):
        sess = _make_session_with_meta({})
        sess.shutdown("client", wait=False)
        sess.close.assert_not_called()

    def test_shutdown_client_waits_by_default(self):
        sess = _make_session_with_meta({})
        sess._wait_for_clients_shutdown = MagicMock()
        sess.shutdown("client")
        sess._wait_for_clients_shutdown.assert_called_once_with(None, 30.0)

    def test_shutdown_invalid_target_raises_value_error(self):
        sess = _make_session_with_meta({})
        with pytest.raises(ValueError, match="shutdown target_type must be one of"):
            sess.shutdown("relay")


class TestSessionRestart:
    def test_restart_server_sends_correct_command(self):
        sess = _make_session_with_meta({})
        sess.restart("server", wait=False)
        sess._do_command.assert_called_once_with("restart server")

    def test_restart_client_sends_correct_command(self):
        sess = _make_session_with_meta({})
        sess.restart("client", wait=False)
        sess._do_command.assert_called_once_with("restart client")

    def test_restart_all_sends_correct_command(self):
        sess = _make_session_with_meta({})
        sess.restart("all", wait=False)
        sess._do_command.assert_called_once_with("restart all")

    def test_restart_client_with_names_sends_correct_command(self):
        sess = _make_session_with_meta({})
        sess.restart("client", client_names=["site-1"], wait=False)
        cmd = sess._do_command.call_args[0][0]
        assert cmd.startswith("restart client")
        assert "site-1" in cmd

    def test_restart_server_waits_by_default(self):
        sess = _make_session_with_meta({})
        sess.get_system_info = MagicMock(return_value=MagicMock(server_info=MagicMock(start_time=123)))
        sess._wait_for_server_restart = MagicMock()
        sess.restart("server")
        sess._wait_for_server_restart.assert_called_once_with(123, 30.0)

    def test_restart_rejects_zero_wait_timeout_before_command(self):
        sess = _make_session_with_meta({})
        with pytest.raises(ValueError, match="timeout must be a positive number"):
            sess.restart("server", timeout=0)
        sess._do_command.assert_not_called()

    def test_restart_all_waits_for_server_and_clients(self):
        sess = _make_session_with_meta({})
        sys_info = MagicMock(
            server_info=MagicMock(start_time=123),
            client_info=[ClientInfo("site-1", 10), ClientInfo("site-2", 20)],
        )
        sess.get_system_info = MagicMock(return_value=sys_info)
        sess._wait_for_server_restart = MagicMock()
        sess._wait_for_clients_restart = MagicMock()

        sess.restart("all")

        sess._wait_for_server_restart.assert_called_once_with(123, 30.0)
        sess._wait_for_clients_restart.assert_called_once_with(
            {"site-1": 10, "site-2": 20}, 30.0, use_poll_session=True
        )

    def test_wait_for_server_restart_accepts_immediate_success_when_previous_start_unknown(self):
        sess = _make_session_with_meta({})
        sys_info = MagicMock(server_info=MagicMock(start_time=123))
        sess._poll_system_info = MagicMock(return_value=sys_info)

        assert sess._wait_for_server_restart(None, timeout=1.0) is sys_info
        sess._poll_system_info.assert_called_once()

    def test_wait_for_clients_restart_uses_poll_session_through_transient_no_connection(self):
        sess = _make_session_with_meta({})
        sys_info = MagicMock(client_info=[ClientInfo("site-1", 456)])
        sess._poll_system_info = MagicMock(side_effect=[NoConnection("server unavailable"), sys_info])

        with patch("nvflare.fuel.flare_api.flare_api.time.sleep") as sleep:
            sess._wait_for_clients_restart({"site-1": 123}, timeout=1.0, use_poll_session=True)

        assert sess._poll_system_info.call_count == 2
        sleep.assert_called_once()

    def test_restart_client_waits_by_default(self):
        sess = _make_session_with_meta({})
        sess._client_last_connect_times = MagicMock(return_value={"site-1": 123})
        sess._wait_for_clients_restart = MagicMock()
        sess.restart("client", client_names=["site-1"])
        sess._wait_for_clients_restart.assert_called_once_with({"site-1": 123}, 30.0)

    def test_restart_invalid_target_raises_value_error(self):
        sess = _make_session_with_meta({})
        with pytest.raises(ValueError, match="restart target_type must be one of"):
            sess.restart("relay")
