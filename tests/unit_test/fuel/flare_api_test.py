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

from nvflare.fuel.flare_api.api_spec import AuthenticationError, InternalError, InvalidJobDefinition, NoConnection
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
