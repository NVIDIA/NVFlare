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
import subprocess
from unittest.mock import Mock

import psutil
import pytest

from tests import process_test_utils as processes
from tests.timing_utils import ManualClock

pytestmark = pytest.mark.timeout(10)


@pytest.fixture
def owned_child(monkeypatch):
    clock = ManualClock()
    clock.sleep = clock.advance
    monkeypatch.setattr(processes, "time", clock)
    owner = processes.OwnedProcesses(grace=0.1)
    monkeypatch.setattr(owner, "capture", lambda **_: None)
    child = Mock(pid=1234)
    child.is_running.return_value = True
    child.status.return_value = psutil.STATUS_RUNNING
    child.kill.side_effect = lambda: setattr(child.is_running, "return_value", False)
    owner.identities.add(child)
    return owner, child, clock


def test_pidfd_error_still_escalates_live_child_and_keeps_deadline(owned_child, monkeypatch):
    owner, child, clock = owned_child
    monkeypatch.setattr(processes.psutil, "wait_procs", Mock(side_effect=OSError(errno.EINVAL, "pidfd")))
    owner.close()
    child.terminate.assert_called_once()
    child.kill.assert_called_once()
    assert clock.monotonic() == pytest.approx(1000.1)


def test_survivor_is_reported_and_other_children_are_stopped(owned_child, monkeypatch):
    owner, child, _ = owned_child
    other = Mock(pid=5678)
    other.is_running.return_value = False
    owner.identities.add(other)
    child.kill.side_effect = psutil.AccessDenied(child.pid)
    monkeypatch.setattr(processes.psutil, "wait_procs", Mock(return_value=([], [child])))
    with pytest.raises(RuntimeError, match="survivors=\\[1234\\]"):
        owner.close()
    child.kill.assert_called_once()


def test_cleanup_failure_preserves_primary_timeout_and_runs_all_callbacks(owned_child, monkeypatch):
    owner, child, _ = owned_child
    first = Mock(side_effect=ValueError("admin close failed"))
    second = Mock()
    owner.callbacks.extend([first, second])
    monkeypatch.setattr(processes.psutil, "wait_procs", Mock(side_effect=OSError(errno.EIO, "wait failed")))
    original = subprocess.TimeoutExpired("original job", 300)
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        with owner:
            raise original
    assert caught.value is original
    assert "admin close failed" in caught.value.cleanup_error
    assert "wait failed" in caught.value.cleanup_error
    second.assert_called_once()
    child.kill.assert_called_once()


def test_unexpected_wait_failure_is_not_reclassified_as_exit(owned_child, monkeypatch):
    _, child, _ = owned_child
    monkeypatch.setattr(processes.psutil, "wait_procs", Mock(side_effect=OSError(errno.EIO, "wait failed")))
    with pytest.raises(OSError, match="wait failed"):
        processes.wait_for_processes([child], 0.1)
