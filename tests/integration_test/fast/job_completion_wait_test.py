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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.job_def import RunStatus
from tests.integration_test.src import action_handlers, nvf_test_driver


@pytest.fixture
def completion_driver(monkeypatch):
    monkeypatch.delenv("NVFLARE_EVENT_SEQUENCE_TIMEOUT", raising=False)
    clock = SimpleNamespace(now=0.0)

    def sleep(seconds):
        clock.now += seconds
        assert clock.now <= 10, "Completion polling did not stop"

    clock.time = lambda: clock.now
    clock.sleep = sleep
    monkeypatch.setattr(action_handlers, "time", clock)
    monkeypatch.setattr(nvf_test_driver, "time", clock)
    driver = nvf_test_driver.NVFTestDriver("", None, poll_period=0.5, event_sequence_timeout=5)
    driver.job_id = "completed-job"
    driver.super_admin_api = Mock()
    driver.super_admin_api.get_system_info.return_value.server_info.status = "stopped"
    driver.super_admin_api.get_job_status.return_value = RunStatus.FINISHED_COMPLETED.value
    driver._get_run_state = Mock(return_value={"run_finished": True})
    driver.server_status = Mock(return_value=object())
    driver.client_status = Mock(return_value=[])
    return driver, clock


def completion_event(action="ensure_current_job_done"):
    return {
        "trigger": {"type": "run_state", "data": {"run_finished": True}},
        "actions": [action],
        "result": {"type": "run_state", "data": {"run_finished": True}},
    }


@pytest.mark.parametrize("timeout", [5, None])
def test_completion_waits_for_delayed_client_cleanup(completion_driver, timeout):
    driver, clock = completion_driver
    driver.event_sequence_timeout = timeout
    driver.super_admin_api.get_client_job_status.side_effect = lambda: [
        {"client_name": "site-1", "status": "no_reply" if clock.now < 2 else "no_jobs"}
    ]

    driver.run_event_sequence([completion_event()])

    assert driver.test_done
    assert 2 <= clock.now < 5
    assert driver.super_admin_api.get_client_job_status.call_count > 2


@pytest.mark.parametrize(
    "action_timeout,elapsed,deadline", [("", 4, 5), (" 1", 0, 1.5), (" 0.1", 0, 0.6), (" 10", 4, 5)]
)
def test_completion_wait_respects_remaining_deadline(completion_driver, action_timeout, elapsed, deadline):
    driver, clock = completion_driver
    driver.super_admin_api.get_client_job_status.return_value = [{"client_name": "site-1", "status": "no_reply"}]

    with pytest.raises(TimeoutError, match="completed-job"):
        driver.run_event_sequence(
            [completion_event(f"sleep {elapsed}"), completion_event(f"ensure_current_job_done{action_timeout}")]
        )

    assert not driver.test_done
    assert clock.now == deadline
