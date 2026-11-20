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

import errno
from unittest.mock import Mock

import psutil
import pytest

from tests.integration_test.src import utils
from tests.timing_utils import ManualClock

pytestmark = pytest.mark.timeout(10)


@pytest.fixture
def owned_process(monkeypatch):
    clock = ManualClock()
    child = Mock(pid=1234)
    child.is_running.return_value = True
    child.status.return_value = psutil.STATUS_RUNNING
    monkeypatch.setattr(utils, "time", clock)
    monkeypatch.setattr(clock, "sleep", clock.advance)
    monkeypatch.setattr(utils, "find_site_processes", lambda *_: {child})

    def pidfd_error(*args, **kwargs):
        assert wait.call_count <= 5, "cleanup exceeded its poll budget"
        raise OSError(errno.EINVAL, "Invalid argument")

    wait = Mock(side_effect=pidfd_error)
    monkeypatch.setattr(psutil, "wait_procs", wait)
    return child, clock, wait


@pytest.mark.parametrize("exited_state", [psutil.STATUS_ZOMBIE, "gone"])
def test_pidfd_error_falls_back_to_process_state_and_reaps_launcher(tmp_path, owned_process, exited_state):
    child, clock, wait = owned_process
    launcher = Mock()
    launcher.poll.return_value = 0

    def status():
        if clock.monotonic() >= 1000.2:
            if exited_state == "gone":
                raise psutil.NoSuchProcess(child.pid)
            return exited_state
        return psutil.STATUS_RUNNING

    child.status.side_effect = status
    utils.stop_site_processes(launcher, str(tmp_path), kill_timeout=0.3)

    assert child.kill.call_count == wait.call_count == 2
    assert clock.monotonic() == pytest.approx(1000.2)
    launcher.wait.assert_called_once_with(timeout=0.3)


def test_pidfd_error_does_not_hide_surviving_process_or_extend_deadline(tmp_path, owned_process):
    child, clock, wait = owned_process
    with pytest.raises(RuntimeError, match="processes survived cleanup.*1234"):
        utils.stop_site_processes(None, str(tmp_path), kill_timeout=0.3)

    assert clock.monotonic() == pytest.approx(1000.3)
    assert child.kill.call_count == wait.call_count == 3
    assert all(0 <= call.kwargs["timeout"] <= 0.1 for call in wait.call_args_list)


def test_other_wait_errors_propagate_and_launcher_reaping_is_still_attempted(tmp_path, owned_process):
    _, clock, wait = owned_process
    wait.side_effect = OSError(errno.EIO, "I/O error")
    launcher = Mock()
    launcher.poll.return_value = 0

    with pytest.raises(RuntimeError, match="Stop workspace job processes.*I/O error"):
        utils.stop_site_processes(launcher, str(tmp_path), kill_timeout=0.3)

    assert clock.monotonic() == 1000.0
    launcher.wait.assert_called_once_with(timeout=0.3)
