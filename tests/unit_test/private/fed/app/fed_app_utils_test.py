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

from unittest.mock import Mock

import pytest

from nvflare.private.fed.app import utils


@pytest.mark.parametrize("version_info", [(3, 10, 0), (3, 13, 0), (3, 14, 0)])
def test_version_check_accepts_supported_python_versions(monkeypatch, version_info):
    monkeypatch.setattr(utils.sys, "version_info", version_info)

    utils.version_check()


@pytest.mark.parametrize(
    "version_info, expected_message",
    [
        (
            (3, 9, 18),
            f"Python versions {utils._format_python_version(utils._HIGHEST_UNSUPPORTED_PYTHON_VERSION)} "
            "and below are not supported",
        ),
        (
            (3, 15, 0),
            f"Python versions {utils._format_python_version(utils._LOWEST_UNSUPPORTED_PYTHON_VERSION)} "
            "and above are not yet supported",
        ),
    ],
)
def test_version_check_rejects_unsupported_python_versions(monkeypatch, version_info, expected_message):
    monkeypatch.setattr(utils.sys, "version_info", version_info)

    with pytest.raises(RuntimeError, match=expected_message):
        utils.version_check()


@pytest.mark.parametrize("stop_requested", [False, True])
def test_linux_simulator_watchdog_reaches_shutdown_without_procfs(monkeypatch, stop_requested, caplog):
    stop = Mock()
    stop.is_set.side_effect = [False, stop_requested]
    # The launcher is present at the first poll, then the worker is reparented.
    ppid = Mock(side_effect=[123, 1])
    group_kill = Mock()
    tree_kill = Mock(side_effect=PermissionError("procfs restricted"))
    monkeypatch.setattr(utils, "sys", Mock(platform="linux"))
    monkeypatch.setattr(utils, "os", Mock(getppid=ppid, getpid=lambda: 456, getpgid=lambda pid: pid, killpg=group_kill))
    monkeypatch.setattr(utils, "time", Mock())
    monkeypatch.setattr(utils.psutil, "pid_exists", Mock(side_effect=AssertionError("unexpected procfs probe")))
    monkeypatch.setattr(utils, "kill_child_processes", tree_kill)

    utils.check_parent_alive(123, stop)

    assert ppid.call_count == (1 if stop_requested else 2)
    utils.time.sleep.assert_called_once_with(1)
    tree_kill.assert_called_once_with(456)
    group_kill.assert_called_once_with(456, 9)
    assert "Cannot inspect all simulator worker descendants" in caplog.text


def test_cleanup_continues_after_descendant_exits(monkeypatch):
    exited = Mock()
    exited.kill.side_effect = utils.psutil.NoSuchProcess(789)
    living = Mock()
    parent = Mock()
    parent.children.return_value = [exited, living]
    monkeypatch.setattr(utils.psutil, "Process", lambda pid: parent)

    utils.kill_child_processes(456)

    living.kill.assert_called_once_with()


def test_watchdog_does_not_fall_back_to_signalling_a_shared_group(monkeypatch):
    monkeypatch.setattr(utils, "sys", Mock(platform="linux"))
    monkeypatch.setattr(utils, "os", Mock(getpid=lambda: 456, getpgid=lambda pid: 123))
    monkeypatch.setattr(utils, "kill_child_processes", Mock(side_effect=PermissionError()))

    with pytest.raises(PermissionError):
        utils.check_parent_alive(123, Mock(is_set=lambda: True))

    utils.os.killpg.assert_not_called()
