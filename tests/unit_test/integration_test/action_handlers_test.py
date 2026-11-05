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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import yaml

from nvflare.apis.job_def import RunStatus
from nvflare.fuel.flare_api.api_spec import InternalError
from nvflare.fuel.flare_api.flare_api import Session
from nvflare.fuel.hci.client.api_status import APIStatus
from nvflare.fuel.hci.proto import MetaStatusValue
from tests.integration_test.src import action_handlers, nvf_test_driver

JOB_ID = "12345678-1234-5678-1234-567812345678"
STARTING_INFO = f"Job {JOB_ID} is starting; retry abort."
ABORT_CONFIG = yaml.safe_load(
    (Path(__file__).resolve().parents[2] / "integration_test/data/test_configs/authorization/abort_job.yml").read_text()
)


def _reply(status=MetaStatusValue.OK, info=""):
    return {"status": APIStatus.SUCCESS, "meta": {"status": status, "info": info}}


def _session(*responses, username="admin@a.org"):
    # Exercise Session.abort_job/_do_command's real response-to-exception mapping.
    session = Session.__new__(Session)
    session.username = username
    session.api = Mock(closed=False)
    session.api.do_command.side_effect = responses
    return session


def _gate_stats(phase="waiting_for_release"):
    return {"server": {"AbortTestController": {"tasks": {}, "phase": phase}}}


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(now=0.0, sleeps=[])

    def sleep(delay):
        if len(state.sleeps) >= 25:
            pytest.fail("retry or event wait failed to stop")
        assert delay >= 0
        state.sleeps.append(delay)
        state.now += delay

    fake_time = SimpleNamespace(monotonic=lambda: state.now, time=lambda: state.now, sleep=Mock(side_effect=sleep))
    monkeypatch.setattr(action_handlers, "time", fake_time)
    monkeypatch.setattr(nvf_test_driver, "time", fake_time)
    return state


@pytest.fixture
def controller():
    return SimpleNamespace(job_id=JOB_ID, admin_api_response={"message": "previous response"})


def test_abort_succeeds_without_retry(controller, clock):
    session = _session(_reply())

    action_handlers._AbortJobHandler().handle([], controller, session)

    session.api.do_command.assert_called_once_with(f"abort_job {JOB_ID}", props=None)
    assert controller.admin_api_response is None
    assert clock.sleeps == []


def test_abort_retries_starting_response_then_succeeds(controller, clock):
    session = _session(_reply(MetaStatusValue.ERROR, STARTING_INFO), _reply())

    action_handlers._AbortJobHandler().handle([], controller, session)

    assert session.api.do_command.call_args_list == [call(f"abort_job {JOB_ID}", props=None)] * 2
    assert controller.admin_api_response is None
    assert clock.sleeps == [0.5]


def test_abort_startup_retry_stops_at_deadline(controller, clock, monkeypatch):
    monkeypatch.setattr(action_handlers, "_ABORT_STARTUP_RETRY_TIMEOUT", 0.75)
    session = _session(
        _reply(MetaStatusValue.ERROR, STARTING_INFO),
        _reply(MetaStatusValue.ERROR, STARTING_INFO),
        AssertionError("must not issue another abort after the retry deadline"),
    )

    with pytest.raises(InternalError) as caught:
        action_handlers._AbortJobHandler().handle([], controller, session)

    assert str(caught.value) == f"error: {STARTING_INFO}"
    assert session.api.do_command.call_count == 2
    assert clock.sleeps == [0.5, 0.25]
    assert clock.now == 0.75


@pytest.mark.parametrize(
    "status,info",
    [
        (MetaStatusValue.ERROR, "Job another-job is starting; retry abort."),
        (MetaStatusValue.ERROR, "unrelated abort failure"),
        (MetaStatusValue.INTERNAL_ERROR, STARTING_INFO),
    ],
)
def test_abort_does_not_retry_other_internal_errors(controller, clock, status, info):
    session = _session(_reply(status, info))

    with pytest.raises(InternalError):
        action_handlers._AbortJobHandler().handle([], controller, session)

    session.api.do_command.assert_called_once_with(f"abort_job {JOB_ID}", props=None)
    assert clock.sleeps == []


@pytest.mark.parametrize("starting_first", [False, True])
def test_abort_preserves_authorization_denial(controller, clock, starting_first):
    responses = [_reply(MetaStatusValue.NOT_AUTHORIZED, "not allowed")]
    if starting_first:
        responses.insert(0, _reply(MetaStatusValue.ERROR, STARTING_INFO))
    session = _session(*responses, username="trainer@b.org")

    action_handlers._AbortJobHandler().handle([], controller, session)

    assert controller.admin_api_response == {
        "message": "Error: PermissionError: Authorization Error: user 'trainer@b.org' is not authorized for 'abort_job'"
    }
    assert session.api.do_command.call_count == len(responses)
    assert clock.sleeps == ([0.5] if starting_first else [])


def test_app_command_sends_release_to_current_job_and_records_response(controller):
    response = _reply()
    response["data"] = [{"type": "dict", "data": {"released": True}}]
    session = _session(response)

    action_handlers._AppCommandHandler().handle(["release_abort_test"], controller, session)

    session.api.do_command.assert_called_once_with(f"app_command {JOB_ID} release_abort_test", props=None)
    assert controller.admin_api_response == {"released": True}


CONFIG_EVENTS = [
    pytest.param(case["event_sequence"][1], RunStatus.DISPATCHED.value, RunStatus.RUNNING.value, id=case["test_name"])
    for case in ABORT_CONFIG["tests"]
] + [
    pytest.param(
        case["event_sequence"][2],
        RunStatus.RUNNING.value,
        RunStatus.FINISHED_ABORTED.value,
        id=case["test_name"] + " completion",
    )
    for case in ABORT_CONFIG["tests"]
    if case["event_sequence"][2]["actions"] == ["mark_test_done"]
]


@pytest.mark.parametrize("event,initial_status,expected_status", CONFIG_EVENTS)
def test_abort_config_waits_for_current_job_state(event, initial_status, expected_status, clock):
    driver = nvf_test_driver.NVFTestDriver("unused", Mock(), poll_period=0.5, event_sequence_timeout=5.0)
    driver.job_id = JOB_ID
    driver.server_status = Mock(return_value="started")
    driver._get_site_log = Mock(return_value=["Started run: previous-job", "Abort previous-job"])
    statuses = iter([initial_status, expected_status, expected_status])
    observed_states = []

    def poll_state(state):
        _, state = nvf_test_driver._update_run_state(_gate_stats(), state, next(statuses))
        return state

    driver._get_run_state = Mock(side_effect=poll_state)

    def execute_actions(actions, admin_user_name=None):
        observed_states.append(driver._get_run_state.call_args.args[0]["job_status"])
        driver.admin_api_response = event["result"].get("data")
        driver.test_done = True

    driver.execute_actions = Mock(side_effect=execute_actions)

    driver.run_event_sequence([event])

    assert observed_states == [expected_status]
    driver._get_site_log.assert_not_called()
    driver.execute_actions.assert_called_once_with(
        actions=event["actions"], admin_user_name=event.get("admin_user_name")
    )


@pytest.mark.parametrize("case", ABORT_CONFIG["tests"], ids=lambda case: case["test_name"])
def test_abort_config_waits_for_gate_even_when_metadata_is_running(case, clock):
    driver = nvf_test_driver.NVFTestDriver("unused", Mock(), poll_period=0.5, event_sequence_timeout=5.0)
    driver.job_id = JOB_ID
    driver.server_status = Mock(return_value="started")
    phases = iter(["initializing", "waiting_for_release", "waiting_for_release"])
    observed_phases = []
    event = case["event_sequence"][1]

    def poll_state(state):
        _, state = nvf_test_driver._update_run_state(_gate_stats(next(phases)), state, RunStatus.RUNNING.value)
        return state

    def execute_actions(actions, admin_user_name=None):
        observed_phases.append(driver._get_run_state.call_args.args[0]["workflows"]["AbortTestController"]["phase"])
        driver.admin_api_response = event["result"].get("data")
        driver.test_done = True

    driver._get_run_state = Mock(side_effect=poll_state)
    driver.execute_actions = Mock(side_effect=execute_actions)

    driver.run_event_sequence([event])

    assert observed_phases == ["waiting_for_release"]


DENIAL_CASES = [
    case for case in ABORT_CONFIG["tests"] if case["event_sequence"][1]["result"]["type"] == "admin_api_response"
]


@pytest.mark.parametrize("case", DENIAL_CASES, ids=lambda case: case["test_name"])
@pytest.mark.parametrize("abort_incorrectly_allowed", [False, True])
def test_abort_config_releases_only_after_verifying_permission_denial(
    case, abort_incorrectly_allowed, clock, monkeypatch
):
    driver = nvf_test_driver.NVFTestDriver("unused", Mock(), poll_period=0.5, event_sequence_timeout=5.0)
    driver.job_id = JOB_ID
    driver.server_status = Mock(return_value="started")
    denial_event = case["event_sequence"][1]
    user = denial_event["admin_user_name"]
    abort_status = MetaStatusValue.OK if abort_incorrectly_allowed else MetaStatusValue.NOT_AUTHORIZED
    abort_session = _session(_reply(abort_status), username=user)
    release_session = _session(username="super@test.org")
    driver.admin_apis = {user: abort_session}
    driver.super_admin_api = release_session
    current_status = RunStatus.RUNNING.value

    def poll_state(state):
        _, state = nvf_test_driver._update_run_state(_gate_stats(), state, current_status)
        return state

    def release(command, props=None):
        nonlocal current_status
        abort_session.api.do_command.assert_called_once_with(f"abort_job {JOB_ID}", props=None)
        current_status = RunStatus.FINISHED_COMPLETED.value
        response = _reply()
        response["data"] = [{"type": "dict", "data": {"released": True}}]
        return response

    release_session.api.do_command.side_effect = release
    driver._get_run_state = Mock(side_effect=poll_state)
    monkeypatch.setattr(action_handlers, "check_job_done", Mock(return_value=True))

    if abort_incorrectly_allowed:
        with pytest.raises(nvf_test_driver.NVFTestError, match="Missing admin_api_response"):
            driver.run_event_sequence(case["event_sequence"][1:])
        release_session.api.do_command.assert_not_called()
        assert current_status == RunStatus.RUNNING.value
    else:
        driver.run_event_sequence(case["event_sequence"][1:])
        release_session.api.do_command.assert_called_once_with(f"app_command {JOB_ID} release_abort_test", props=None)
        assert driver.test_done is True


@pytest.mark.parametrize("event,initial_status,expected_status", CONFIG_EVENTS)
@pytest.mark.parametrize("terminal_status", [RunStatus.FINISHED_COMPLETED.value, RunStatus.FINISHED_ABNORMAL.value])
def test_abort_config_rejects_terminal_job_without_waiting_for_timeout(
    event, initial_status, expected_status, terminal_status, clock
):
    driver = nvf_test_driver.NVFTestDriver("unused", Mock(), poll_period=0.5, event_sequence_timeout=5.0)
    driver.job_id = JOB_ID
    driver.last_job_name = "abort_test_job"
    driver.server_status = Mock(return_value="started")
    driver.client_status = Mock(return_value="started")
    driver.execute_actions = Mock()
    statuses = iter([initial_status, terminal_status])

    def poll_state(state):
        _, state = nvf_test_driver._update_run_state(_gate_stats(), state, next(statuses))
        return state

    driver._get_run_state = Mock(side_effect=poll_state)

    with pytest.raises(nvf_test_driver.NVFTestError) as caught:
        driver.run_event_sequence([event])

    assert f"terminal status {terminal_status!r}" in str(caught.value)
    assert f"waiting for job_status={expected_status!r}" in str(caught.value)
    assert JOB_ID in str(caught.value)
    driver.execute_actions.assert_not_called()
    assert clock.sleeps == [0.5]


def test_abort_config_rejects_job_that_finishes_between_readiness_and_abort(clock):
    driver = nvf_test_driver.NVFTestDriver("unused", Mock(), poll_period=0.5, event_sequence_timeout=5.0)
    driver.job_id = JOB_ID
    driver.last_job_name = "abort_test_job"
    driver.server_status = Mock(return_value="started")
    driver.client_status = Mock(return_value="started")
    session = _session(_reply(info=f"Job for {JOB_ID} is already completed."))
    statuses = iter([RunStatus.RUNNING.value, RunStatus.FINISHED_COMPLETED.value, RunStatus.FINISHED_COMPLETED.value])

    def poll_state(state):
        _, state = nvf_test_driver._update_run_state(_gate_stats(), state, next(statuses))
        return state

    def execute_actions(actions, admin_user_name=None):
        assert actions == ["abort_job"]
        action_handlers._AbortJobHandler().handle([], driver, session)

    driver._get_run_state = Mock(side_effect=poll_state)
    driver.execute_actions = Mock(side_effect=execute_actions)

    with pytest.raises(nvf_test_driver.NVFTestError) as caught:
        driver.run_event_sequence(ABORT_CONFIG["tests"][0]["event_sequence"][1:])

    assert "terminal status 'FINISHED:COMPLETED'" in str(caught.value)
    assert "waiting for job_status='FINISHED:ABORTED'" in str(caught.value)
    session.api.do_command.assert_called_once_with(f"abort_job {JOB_ID}", props=None)
    assert driver.test_done is False
    assert clock.sleeps == [0.5]
