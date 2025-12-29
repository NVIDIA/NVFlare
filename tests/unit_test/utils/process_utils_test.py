# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
import signal
from unittest import mock

import pytest

from nvflare.utils.process_utils import ProcessAdapter, spawn_process


class TestProcessAdapterWithPopen:
    """Test ProcessAdapter when initialized with a subprocess.Popen object."""

    def test_init_with_process(self):
        mock_process = mock.Mock()
        mock_process.pid = 1234

        adapter = ProcessAdapter(process=mock_process)

        assert adapter.process is mock_process
        assert adapter.pid == 1234

    def test_poll_delegates_to_process(self):
        mock_process = mock.Mock()
        mock_process.pid = 1234
        mock_process.poll.return_value = 0

        adapter = ProcessAdapter(process=mock_process)

        assert adapter.poll() == 0
        mock_process.poll.assert_called_once()

    def test_poll_returns_none_when_running(self):
        mock_process = mock.Mock()
        mock_process.pid = 1234
        mock_process.poll.return_value = None

        adapter = ProcessAdapter(process=mock_process)

        assert adapter.poll() is None

    def test_wait_delegates_to_process(self):
        mock_process = mock.Mock()
        mock_process.pid = 1234

        adapter = ProcessAdapter(process=mock_process)
        adapter.wait()

        mock_process.wait.assert_called_once()


class TestProcessAdapterWithPidOnly:
    """Test ProcessAdapter when initialized with only a PID (posix_spawn case)."""

    def test_init_with_pid_only(self):
        adapter = ProcessAdapter(pid=5678)

        assert adapter.process is None
        assert adapter.pid == 5678

    def test_init_without_process_or_pid_raises(self):
        with pytest.raises(ValueError, match="requires either a process object or a pid"):
            ProcessAdapter()

    def test_poll_with_process_still_running(self, monkeypatch):
        adapter = ProcessAdapter(pid=5678)

        # waitpid returns (0, 0) when process is still running with WNOHANG
        monkeypatch.setattr(
            "nvflare.utils.process_utils.os.waitpid",
            lambda pid, opts: (0, 0),
        )

        assert adapter.poll() is None

    def test_poll_with_process_exited(self, monkeypatch):
        adapter = ProcessAdapter(pid=5678)

        # Simulate process exited with status 0
        def fake_waitpid(pid, opts):
            if opts == os.WNOHANG:
                # Return the pid and a status that decodes to 0
                return pid, 0
            return pid, 0

        monkeypatch.setattr("nvflare.utils.process_utils.os.waitpid", fake_waitpid)
        monkeypatch.setattr("nvflare.utils.process_utils.os.WIFEXITED", lambda s: True)
        monkeypatch.setattr("nvflare.utils.process_utils.os.WEXITSTATUS", lambda s: 0)

        # Remove waitstatus_to_exitcode to test fallback path
        monkeypatch.delattr("nvflare.utils.process_utils.os.waitstatus_to_exitcode", raising=False)

        result = adapter.poll()
        assert result == 0

    def test_poll_caches_return_code(self, monkeypatch):
        adapter = ProcessAdapter(pid=5678)
        call_count = [0]

        def fake_waitpid(pid, opts):
            call_count[0] += 1
            return pid, 0

        monkeypatch.setattr("nvflare.utils.process_utils.os.waitpid", fake_waitpid)
        monkeypatch.setattr("nvflare.utils.process_utils.os.WIFEXITED", lambda s: True)
        monkeypatch.setattr("nvflare.utils.process_utils.os.WEXITSTATUS", lambda s: 42)
        monkeypatch.delattr("nvflare.utils.process_utils.os.waitstatus_to_exitcode", raising=False)

        # First call
        assert adapter.poll() == 42
        # Second call should use cached value
        assert adapter.poll() == 42
        # waitpid should only be called once
        assert call_count[0] == 1

    def test_poll_handles_child_process_error(self, monkeypatch):
        adapter = ProcessAdapter(pid=5678)

        def raise_child_error(pid, opts):
            raise ChildProcessError("No child processes")

        monkeypatch.setattr("nvflare.utils.process_utils.os.waitpid", raise_child_error)

        # Should return -1 when process already reaped
        result = adapter.poll()
        assert result == -1

    def test_wait_with_pid_only(self, monkeypatch):
        adapter = ProcessAdapter(pid=5678)

        wait_calls = []

        def fake_waitpid(pid, opts):
            wait_calls.append((pid, opts))
            return pid, 0

        monkeypatch.setattr("nvflare.utils.process_utils.os.waitpid", fake_waitpid)
        monkeypatch.setattr("nvflare.utils.process_utils.os.WIFEXITED", lambda s: True)
        monkeypatch.setattr("nvflare.utils.process_utils.os.WEXITSTATUS", lambda s: 0)
        monkeypatch.delattr("nvflare.utils.process_utils.os.waitstatus_to_exitcode", raising=False)

        adapter.wait()

        # Should call waitpid with blocking (0 options)
        assert (5678, 0) in wait_calls


class TestProcessAdapterTerminate:
    """Test ProcessAdapter.terminate() behavior."""

    def test_terminate_kills_process_group(self, monkeypatch):
        adapter = ProcessAdapter(pid=1234)

        recorded = {}

        monkeypatch.setattr("nvflare.utils.process_utils.os.getpgid", lambda pid: pid + 100)

        def fake_killpg(pgid, sig):
            recorded["pgid"] = pgid
            recorded["sig"] = sig

        monkeypatch.setattr("nvflare.utils.process_utils.os.killpg", fake_killpg)

        adapter.terminate()

        assert recorded["pgid"] == 1334
        assert recorded["sig"] == signal.SIGKILL

    def test_terminate_handles_getpgid_failure(self, monkeypatch):
        adapter = ProcessAdapter(pid=1234)

        def raise_error(pid):
            raise ProcessLookupError("No such process")

        monkeypatch.setattr("nvflare.utils.process_utils.os.getpgid", raise_error)

        adapter.terminate()

        # getpgid failure now short-circuits with no kill attempt
        # so reaching here without exception is success.

    def test_terminate_handles_killpg_failure_gracefully(self, monkeypatch):
        adapter = ProcessAdapter(pid=1234)

        monkeypatch.setattr("nvflare.utils.process_utils.os.getpgid", lambda pid: pid)

        def raise_error(pgid, sig):
            raise ProcessLookupError("No such process")

        monkeypatch.setattr("nvflare.utils.process_utils.os.killpg", raise_error)

        # Should not raise exception
        adapter.terminate()


class TestSpawnProcess:
    """Test the spawn_process utility function."""

    def test_spawn_uses_posix_spawn_when_available(self, monkeypatch):
        spawned = {}

        def fake_posix_spawn(
            path,
            argv,
            env,
            *,
            file_actions=None,
            setpgroup=0,
            resetids=False,
            setsid=False,
            sched_param=None,
            sched_scheduler=None,
            sigmask=(),
            sigdef=(),
        ):
            spawned["path"] = path
            spawned["argv"] = argv
            spawned["setsid"] = setsid
            return 9999

        monkeypatch.setattr("nvflare.utils.process_utils._POSIX_SPAWN_SUPPORTED", True)
        monkeypatch.setattr("nvflare.utils.process_utils.os.posix_spawn", fake_posix_spawn)

        adapter = spawn_process(["/bin/echo", "hello"], {"PATH": "/usr/bin"})

        assert adapter.pid == 9999
        assert adapter.process is None
        assert spawned["setsid"] is True
        assert spawned["path"] == "/bin/echo"

    def test_spawn_falls_back_on_posix_spawn_failure(self, monkeypatch):
        def failing_spawn(*args, **kwargs):
            raise OSError("posix_spawn not supported")

        mock_popen = mock.Mock()
        mock_popen.pid = 8888

        monkeypatch.setattr("nvflare.utils.process_utils._POSIX_SPAWN_SUPPORTED", True)
        monkeypatch.setattr("nvflare.utils.process_utils.os.posix_spawn", failing_spawn)
        monkeypatch.setattr(
            "nvflare.utils.process_utils.subprocess.Popen",
            mock.Mock(return_value=mock_popen),
        )

        adapter = spawn_process(["/bin/echo", "hello"], {"PATH": "/usr/bin"})

        assert adapter.process is mock_popen
        assert adapter.pid == 8888

    def test_spawn_uses_popen_when_posix_spawn_not_supported(self, monkeypatch):
        mock_popen = mock.Mock()
        mock_popen.pid = 7777

        monkeypatch.setattr("nvflare.utils.process_utils._POSIX_SPAWN_SUPPORTED", False)
        popen_mock = mock.Mock(return_value=mock_popen)
        monkeypatch.setattr("nvflare.utils.process_utils.subprocess.Popen", popen_mock)

        adapter = spawn_process(["/bin/echo", "hello"], {"PATH": "/usr/bin"})

        assert adapter.process is mock_popen
        assert adapter.pid == 7777
        popen_mock.assert_called_once()

    def test_spawn_with_empty_cmd_uses_popen(self, monkeypatch):
        mock_popen = mock.Mock()
        mock_popen.pid = 6666

        monkeypatch.setattr("nvflare.utils.process_utils._POSIX_SPAWN_SUPPORTED", True)
        popen_mock = mock.Mock(return_value=mock_popen)
        monkeypatch.setattr("nvflare.utils.process_utils.subprocess.Popen", popen_mock)

        adapter = spawn_process([], {"PATH": "/usr/bin"})

        assert adapter.process is mock_popen
        popen_mock.assert_called_once()

    def test_spawn_sets_setsid_in_popen_fallback(self, monkeypatch):
        mock_popen = mock.Mock()
        mock_popen.pid = 5555

        monkeypatch.setattr("nvflare.utils.process_utils._POSIX_SPAWN_SUPPORTED", False)
        popen_mock = mock.Mock(return_value=mock_popen)
        monkeypatch.setattr("nvflare.utils.process_utils.subprocess.Popen", popen_mock)

        spawn_process(["/bin/echo"], {"PATH": "/usr/bin"})

        # Verify preexec_fn is set to os.setsid
        call_kwargs = popen_mock.call_args[1]
        assert call_kwargs["preexec_fn"] is os.setsid
