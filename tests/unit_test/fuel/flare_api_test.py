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

from nvflare.fuel.flare_api.api_spec import (
    AuthenticationError,
    ClientInfo,
    InternalError,
    InvalidJobDefinition,
    JobNotFound,
    NoConnection,
    ServerInfo,
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
    assert str(ClientInfo("site-1", None)) == "site-1(last_connect_time: unknown)"
