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

import importlib
from pathlib import Path
from subprocess import TimeoutExpired
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.fuel.flare_api.api_spec import SessionClosed


@pytest.mark.parametrize("setup_duration", [0, 1900])
def test_setup_keeps_admin_session_active(monkeypatch, setup_duration):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    system_test = importlib.import_module("tests.integration_test.system_test")
    clock = SimpleNamespace(now=0, last_active=0)

    def check_session():
        if clock.now - clock.last_active > 1800:
            raise SessionClosed("session expired during setup")
        clock.last_active = clock.now

    def wait(timeout=None):
        remaining = setup_duration - clock.now
        clock.now += remaining if timeout is None else min(remaining, timeout)
        if clock.now < setup_duration:
            raise TimeoutExpired("dataset setup", timeout)
        return 0

    process = Mock()
    process.wait.side_effect = wait
    monkeypatch.setattr(system_test, "run_command_in_subprocess", Mock(return_value=process))
    driver = Mock()
    driver.super_admin_api.get_system_info.side_effect = check_session
    driver.run_event_sequence.side_effect = lambda events: check_session()
    test_cases = [("slow setup", None, ["dataset setup"], [], [], True)]

    system_test.TestSystem().test_run_job_complete((test_cases, Mock(), driver, "test.yml"))

    assert clock.now == setup_duration
    driver.run_event_sequence.assert_called_once_with([])
