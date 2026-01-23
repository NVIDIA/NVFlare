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

from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JobReturnCode
from nvflare.app_common.job_launcher.process_launcher import ProcessHandle, ProcessJobLauncher


class DummyWorkspace:
    def __init__(self, custom_path: str = ""):
        self.custom_path = custom_path
        self.calls = []

    def get_app_custom_dir(self, job_id):
        self.calls.append(job_id)
        return self.custom_path


class DummyLauncher(ProcessJobLauncher):
    def __init__(self, command: str = "/usr/bin/python worker.py"):
        super().__init__()
        self._command = command

    def get_command(self, job_meta, fl_ctx) -> str:
        return self._command


def _build_fl_ctx(workspace):
    fl_ctx = FLContext()
    fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace, private=True, sticky=False)
    return fl_ctx


def test_launch_job_uses_posix_spawn_when_supported(monkeypatch):
    launcher = DummyLauncher()

    workspace = DummyWorkspace()
    fl_ctx = _build_fl_ctx(workspace)
    job_meta = {JobConstants.JOB_ID: "job-posix"}

    spawned = {}

    def fake_spawn(
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
        spawned["argv"] = tuple(argv)
        spawned["setsid"] = setsid
        spawned["env"] = env
        return 4321

    # Mock at the correct module path where spawn_process is defined
    monkeypatch.setattr(
        "nvflare.utils.process_utils.os.posix_spawn",
        fake_spawn,
    )
    popen_mock = mock.Mock()
    monkeypatch.setattr(
        "nvflare.utils.process_utils.subprocess.Popen",
        popen_mock,
    )

    handle = launcher.launch_job(job_meta, fl_ctx)

    # Access via adapter
    assert handle.adapter.process is None
    assert handle.adapter.pid == 4321
    assert spawned["setsid"] is True
    assert spawned["argv"][0].endswith("python")
    popen_mock.assert_not_called()


def test_launch_job_falls_back_when_spawn_fails(monkeypatch):
    launcher = DummyLauncher()

    workspace = DummyWorkspace()
    fl_ctx = _build_fl_ctx(workspace)
    job_meta = {JobConstants.JOB_ID: "job-fallback"}

    # Mock at the correct module path
    monkeypatch.setattr(
        "nvflare.utils.process_utils.os.posix_spawn",
        mock.Mock(side_effect=OSError("boom")),
    )
    popen_instance = mock.Mock()
    popen_instance.pid = 9999
    monkeypatch.setattr(
        "nvflare.utils.process_utils.subprocess.Popen",
        mock.Mock(return_value=popen_instance),
    )

    handle = launcher.launch_job(job_meta, fl_ctx)

    # Access via adapter
    assert handle.adapter.process is popen_instance
    assert handle.adapter.pid == popen_instance.pid


def test_process_handle_with_pid_wait_and_poll(monkeypatch):
    handle = ProcessHandle(pid=1234)

    calls = []

    def fake_waitpid(pid, options):
        calls.append(options)
        if options == os.WNOHANG:
            return 0, 0
        return pid, 0

    # Mock at the correct module path
    monkeypatch.setattr(
        "nvflare.utils.process_utils.os.waitpid",
        fake_waitpid,
    )

    assert handle.poll() == JobReturnCode.UNKNOWN

    handle.wait()
    assert handle.poll() == JobReturnCode.SUCCESS
    assert os.WNOHANG in calls


def test_process_handle_terminate_uses_process_group(monkeypatch):
    handle = ProcessHandle(pid=4321)

    # Mock at the correct module path
    monkeypatch.setattr(
        "nvflare.utils.process_utils.os.getpgid",
        lambda pid: pid + 1,
    )
    recorded = {}

    def fake_killpg(pgid, sig):
        recorded["pgid"] = pgid
        recorded["sig"] = sig

    monkeypatch.setattr(
        "nvflare.utils.process_utils.os.killpg",
        fake_killpg,
    )

    handle.terminate()

    assert recorded["pgid"] == 4322
    assert recorded["sig"] == signal.SIGKILL
