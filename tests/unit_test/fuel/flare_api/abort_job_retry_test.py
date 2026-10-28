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

import re
import time
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from nvflare.fuel.flare_api.api_spec import AuthorizationError, InternalError
from nvflare.fuel.flare_api.flare_api import Session
from nvflare.fuel.hci.client.api_status import APIStatus
from nvflare.fuel.hci.proto import MetaStatusValue, ReplyKeyword

JOB_ID = "12345678-1234-5678-1234-567812345678"
STARTING = f"Job {JOB_ID} {ReplyKeyword.JOB_STARTING_RETRY_ABORT}"


def _reply(status, info):
    return {"status": APIStatus.SUCCESS, "meta": {"status": status, "info": info}}


@pytest.fixture
def abort_session(monkeypatch):
    clock = SimpleNamespace(now=0.0)

    def advance(delay):
        clock.now += delay

    sleep = Mock(side_effect=advance)
    monkeypatch.setattr(time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(time, "sleep", sleep)
    # Keep the real server-response-to-exception mapping; mock only the transport.
    session = Session.__new__(Session)
    session.username = "admin@nvidia.com"
    session.api = Mock(closed=False)
    return session, clock, sleep


def test_abort_job_without_retry(abort_session):
    session, clock, sleep = abort_session
    session.api.do_command.return_value = _reply(MetaStatusValue.OK, "Abort signal has been sent.")

    assert session.abort_job(JOB_ID) == "Abort signal has been sent."

    session.api.do_command.assert_called_once_with(f"abort_job {JOB_ID}", props=None)
    sleep.assert_not_called()


@pytest.mark.parametrize("message", ["Abort signal has been sent.", f"Job for {JOB_ID} is already completed."])
def test_abort_job_retries_while_the_job_is_starting(abort_session, message):
    session, clock, sleep = abort_session
    session.api.do_command.side_effect = [_reply(MetaStatusValue.ERROR, STARTING), _reply(MetaStatusValue.OK, message)]

    assert session.abort_job(JOB_ID) == message

    assert session.api.do_command.call_args_list == [call(f"abort_job {JOB_ID}", props=None)] * 2
    sleep.assert_called_once_with(0.5)


def test_abort_job_retries_then_reports_authorization_denial(abort_session):
    session, clock, sleep = abort_session
    session.api.do_command.side_effect = [
        _reply(MetaStatusValue.ERROR, STARTING),
        _reply(MetaStatusValue.NOT_AUTHORIZED, "not allowed"),
    ]

    with pytest.raises(AuthorizationError):
        session.abort_job(JOB_ID)

    assert session.api.do_command.call_count == 2
    sleep.assert_called_once_with(0.5)


def test_abort_job_gives_up_after_the_retry_window(abort_session):
    session, clock, sleep = abort_session
    session.api.do_command.return_value = _reply(MetaStatusValue.ERROR, STARTING)

    with pytest.raises(InternalError, match=re.escape(STARTING)):
        session.abort_job(JOB_ID)

    assert clock.now == 30.0
    assert sleep.call_args_list == [call(0.5)] * 60
    assert session.api.do_command.call_count == 61


@pytest.mark.parametrize(
    "status,info,exception",
    [
        (MetaStatusValue.ERROR, f"Job another-job {ReplyKeyword.JOB_STARTING_RETRY_ABORT}", InternalError),
        (MetaStatusValue.ERROR, "unrelated abort failure", InternalError),
        (MetaStatusValue.INTERNAL_ERROR, STARTING, InternalError),
        (MetaStatusValue.NOT_AUTHORIZED, "not allowed", AuthorizationError),
    ],
)
def test_abort_job_does_not_retry_other_errors(abort_session, status, info, exception):
    session, clock, sleep = abort_session
    session.api.do_command.return_value = _reply(status, info)

    with pytest.raises(exception):
        session.abort_job(JOB_ID)

    session.api.do_command.assert_called_once_with(f"abort_job {JOB_ID}", props=None)
    sleep.assert_not_called()
