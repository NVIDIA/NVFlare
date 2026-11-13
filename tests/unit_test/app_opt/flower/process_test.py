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

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from nvflare.app_common.tie.process_mgr import CommandDescriptor
from nvflare.app_opt.flower.process import FlowerProcessManager
from tests.timing_utils import isolate_time


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_failed_start_preserves_launch_error(tmp_path, cleanup_fails):
    manager = FlowerProcessManager(CommandDescriptor("flower-superlink"))
    context = MagicMock()
    context.get_engine.return_value.get_workspace.return_value.get_run_dir.return_value = str(tmp_path)
    launch_error = OSError("launch failed")
    cleanup_error = RuntimeError("cleanup failed")
    with (
        patch.object(manager, "_start_guard", side_effect=launch_error),
        patch.object(manager, "stop", side_effect=cleanup_error if cleanup_fails else None) as stop,
    ):
        with pytest.raises(OSError, match="launch failed") as caught:
            manager.start(context)
    assert caught.value is launch_error
    if cleanup_fails:
        assert caught.value.__cause__ is cleanup_error
    stop.assert_called_once_with(timeout=0.1)


@pytest.mark.parametrize("hung", [False, True])
def test_failed_guard_is_reaped_and_error_consumed(hung):
    manager = FlowerProcessManager(CommandDescriptor("flower-superlink"))
    guard = manager._guard = MagicMock()
    guard.poll.return_value = -9 if hung else 1
    guard.wait.side_effect = (
        [subprocess.TimeoutExpired("guard", 10), subprocess.TimeoutExpired("guard", 1), -9] if hung else [1]
    )
    with pytest.raises(RuntimeError, match="timed out" if hung else "guard failed"):
        manager.stop()
    assert manager._guard is None
    assert manager.stop() == 0
    guard.stdin.close.assert_called_once()
    guard.stdout.close.assert_called_once()
    if hung:
        guard.terminate.assert_called_once()
        guard.kill.assert_called_once()
        assert guard.wait.call_count == 3


def test_unreaped_guard_remains_owned_for_later_stop():
    manager = FlowerProcessManager(CommandDescriptor("flower-superlink"))
    guard = manager._guard = MagicMock()
    guard.poll.return_value = None
    guard.wait.side_effect = subprocess.TimeoutExpired("guard", 1)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        manager.stop()
    assert manager._guard is guard
    guard.poll.return_value = -9
    guard.wait.side_effect = None
    guard.wait.return_value = -9
    with pytest.raises(RuntimeError, match="guard failed"):
        manager.stop()
    assert manager._guard is None
    assert manager.stop() == 0


@pytest.mark.parametrize("persistent", [False, True])
def test_stop_retries_incomplete_scan_and_releases_log_thread(monkeypatch, persistent):
    from nvflare.app_opt.flower import process as subject

    manager = FlowerProcessManager(CommandDescriptor("flower-superlink"))
    manager.process = MagicMock()
    manager.process.poll.return_value = 0
    manager.process.returncode = 0
    child = MagicMock()
    manager._descendants = {child}
    manager._log_writer = MagicMock()
    manager._log_writer.is_alive.return_value = False
    clock = isolate_time(monkeypatch, subject)
    clock.monotonic = MagicMock(side_effect=[0, 0, 2, 3])
    clock.sleep = MagicMock()
    scans = []

    def capture(roots, owner, errors):
        scans.append(roots)
        if persistent or len(scans) == 1:
            errors.append(OSError("inspection failed for PID 2"))
        return set(roots)

    with (
        patch.object(subject, "capture_owned", side_effect=capture),
        patch.object(subject, "_is_alive", side_effect=lambda p: p.kill.call_count == 0),
    ):
        if persistent:
            with pytest.raises(RuntimeError, match="inspection failed for PID 2"):
                manager.stop(timeout=0.1)
        else:
            assert manager.stop(timeout=0.1) == 0
    child.kill.assert_called_once()
    assert len(scans) > 1
    assert manager._log_stop.is_set()
    manager._log_writer.join.assert_called()
    manager.process.stdout.close.assert_called_once()
