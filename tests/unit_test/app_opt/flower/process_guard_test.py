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
import io
from unittest.mock import MagicMock, patch

import psutil
import pytest

from nvflare.app_opt.flower import process_guard as guard
from tests.timing_utils import isolate_time


def process(pid, parent=0):
    result = MagicMock()
    result.pid = pid
    result.info = {"ppid": parent}
    result.is_running.return_value = True
    result.status.return_value = psutil.STATUS_RUNNING
    result.environ.return_value = {}
    return result


@pytest.mark.parametrize("error", [OSError(errno.EIO, "inspection failed"), psutil.Error("inspection failed")])
def test_scan_keeps_owned_workers_and_continues_after_inspection_error(error, caplog):
    root, unrelated, orphan = process(1), process(2), process(3)
    unrelated.environ.side_effect = error
    orphan.environ.return_value = {guard.OWNER_ENV: "job"}
    errors = []
    with patch.object(guard.psutil, "process_iter", return_value=[root, unrelated, orphan]):
        assert guard.capture_owned([root], "job", errors) == {root, orphan}
    assert len(errors) == 1
    assert "PID 2" in str(errors[0]) and "inspection failed" in caplog.text


def test_partial_snapshot_keeps_retained_and_discovered_workers():
    root, child = process(1), process(2, 1)

    def snapshot(*args):
        yield child
        raise OSError(errno.EIO, "snapshot failed")

    errors = []
    with patch.object(guard.psutil, "process_iter", snapshot):
        assert guard.capture_owned([root], errors=errors) == {root, child}
    assert len(errors) == 1


@pytest.mark.parametrize("persistent", [False, True])
def test_cleanup_kills_known_workers_despite_inspection_errors(monkeypatch, persistent):
    root, unrelated = process(1), process(2)
    unrelated.environ.side_effect = (
        OSError(errno.EIO, "scan failed") if persistent else [OSError(errno.EIO, "scan failed"), {}, {}]
    )
    root.kill.side_effect = lambda: setattr(root.is_running, "return_value", False)
    clock = isolate_time(monkeypatch, guard)
    clock.monotonic = MagicMock(side_effect=[0, 0, 2, 3])
    clock.sleep = MagicMock()
    with patch.object(guard.psutil, "process_iter", return_value=[root, unrelated]):
        if persistent:
            with pytest.raises(RuntimeError, match="PID 2"):
                guard.cleanup({root}, "job", timeout=1)
        else:
            guard.cleanup({root}, "job", timeout=1)
    root.kill.assert_called_once()
    unrelated.kill.assert_not_called()


def test_guard_retries_scan_error_without_killing_healthy_tree():
    root = process(1)
    scans = []
    cleanup = MagicMock()

    def capture(owned):
        scans.append(set(owned))
        cleanup.assert_not_called()
        if len(scans) == 1:
            return {root}
        if len(scans) == 2:
            raise OSError(errno.EIO, "scan failed")
        assert owned == {root}
        return owned

    with (
        patch.object(guard.sys, "stdin", io.BytesIO()),
        patch.object(guard.os, "set_blocking"),
        patch.object(guard.os, "close"),
        patch.object(guard.os, "read", return_value=b""),
        patch.object(guard.selectors, "DefaultSelector") as selector,
        patch.object(guard, "capture_owned", side_effect=capture),
        patch.object(guard, "cleanup", cleanup),
    ):
        guard.sys.stdin.fileno = lambda: 0
        selector.return_value.__enter__.return_value.select.side_effect = [[], [], [], [True]]
        guard.main("job", 42)
    assert len(scans) == 3
    cleanup.assert_called_once_with({root}, "job")
