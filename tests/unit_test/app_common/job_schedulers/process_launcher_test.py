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
import shlex
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pytest
import yaml

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JobProcessArgs, JobProcessEnv, JobReturnCode
from nvflare.app_common.job_launcher.client_process_launcher import ClientProcessJobLauncher
from nvflare.app_common.job_launcher.process_launcher import ProcessHandle, ProcessJobLauncher
from nvflare.app_common.job_launcher.server_process_launcher import ServerProcessJobLauncher
from nvflare.client.cell.bootstrap import bootstrap_file_name
from tests.timing_utils import isolate_time


@pytest.mark.parametrize("group_state", ["gone", "alive", "unknown"])
def test_restart_requires_confirmed_empty_owned_group(group_state):
    adapter = mock.Mock()
    adapter.poll.return_value = -signal.SIGKILL
    handle = ProcessHandle(process_adapter=adapter, owns_process_group=True)
    error = {"gone": ProcessLookupError(), "alive": None, "unknown": PermissionError()}[group_state]
    with mock.patch("nvflare.app_common.job_launcher.process_launcher.os.killpg", side_effect=error):
        assert handle.can_restart() is (group_state == "gone")


@pytest.mark.parametrize("bootstrap", [False, True])
def test_restart_requires_no_uncleared_external_trainer(tmp_path, bootstrap):
    adapter = mock.Mock()
    adapter.poll.return_value = -signal.SIGKILL
    handle = ProcessHandle(process_adapter=adapter, owns_process_group=True, app_dir=str(tmp_path))
    (tmp_path / "checkpoint.json").write_text("checkpoint")
    if bootstrap:
        (tmp_path / bootstrap_file_name(1)).write_text("launch in progress")
    with mock.patch("nvflare.app_common.job_launcher.process_launcher.os.killpg", side_effect=ProcessLookupError):
        assert handle.can_restart() is not bootstrap


@pytest.mark.skipif(not hasattr(os, "waitid"), reason="Process tasks require waitid/WNOWAIT")
def test_cj_exit_does_not_prove_its_separate_task_worker_group_settled(tmp_path):
    from nvflare.utils.process_utils import spawn_process

    app_dir = tmp_path / "job" / "app_site-1"
    app_dir.mkdir(parents=True)
    script = """
from pathlib import Path
import os, sys
from nvflare.apis.task_launcher_spec import TaskLaunchRequest
from nvflare.app_common.task_launcher.process_launcher import ProcessTaskLauncher
root = Path(sys.argv[1])
(root / 'job' / '.nvflare' / 'task-execution' / 'attempts').mkdir(parents=True)
worker = ProcessTaskLauncher().launch_task(TaskLaunchRequest(
    'job', 'site-1', 'task', 'attempt',
    (sys.executable, '-c', 'import time; time.sleep(60)')))
(root / 'worker.pid').write_text(str(worker.process_group_id))
os._exit(17)
"""
    adapter = spawn_process([sys.executable, "-c", script, str(tmp_path)], os.environ.copy())
    handle = ProcessHandle(process_adapter=adapter, owns_process_group=True, app_dir=str(app_dir))
    try:
        handle.wait()
        worker_pid = int((tmp_path / "worker.pid").read_text())
        assert os.getpgid(worker_pid) == worker_pid
        assert handle.poll() == JobReturnCode.EXECUTION_ERROR
        assert not handle.can_restart()
    finally:
        record = tmp_path / "worker.pid"
        if record.exists():
            worker_pid = int(record.read_text())
            try:
                if os.getpgid(worker_pid) == worker_pid:
                    os.killpg(worker_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_restart_requires_readable_trainer_launch_directory(tmp_path):
    adapter = mock.Mock()
    adapter.poll.return_value = -signal.SIGKILL
    handle = ProcessHandle(process_adapter=adapter, owns_process_group=True, app_dir=str(tmp_path))
    with (
        mock.patch("nvflare.app_common.job_launcher.process_launcher.os.killpg", side_effect=ProcessLookupError),
        mock.patch("nvflare.app_common.job_launcher.process_launcher.os.listdir", side_effect=PermissionError),
    ):
        assert not handle.can_restart()


@pytest.mark.parametrize("cleanup", ["delayed", "forced", "stuck", "unknown", "cancelled"])
def test_failed_process_waits_for_owned_group_cleanup(monkeypatch, cleanup):
    adapter = mock.Mock(pid=4321)
    adapter.poll.return_value = -signal.SIGKILL
    handle = ProcessHandle(process_adapter=adapter, owns_process_group=True)
    clock = isolate_time(monkeypatch, "nvflare.app_common.job_launcher.process_launcher")
    elapsed = [0.0]
    kills = []

    def killpg(pgid, sig):
        assert pgid == adapter.pid
        if cleanup == "unknown":
            raise PermissionError("group cannot be inspected")
        if sig == signal.SIGKILL:
            kills.append(elapsed[0])
        elif (cleanup == "delayed" and elapsed[0] >= 0.3) or (kills and cleanup != "stuck"):
            raise ProcessLookupError()

    def sleep(seconds):
        elapsed[0] += seconds
        if cleanup == "cancelled" and not kills:
            handle.terminate()

    clock.monotonic = lambda: elapsed[0]
    clock.sleep = sleep
    monkeypatch.setattr("nvflare.app_common.job_launcher.process_launcher.os.killpg", killpg)
    handle.wait()
    adapter.wait.assert_called_once()
    assert handle.can_restart() is (cleanup not in ("stuck", "unknown"))
    if cleanup == "delayed":
        assert elapsed[0] == pytest.approx(0.3)
        assert not kills
    elif cleanup == "cancelled":
        assert kills == [0.1]
        assert elapsed[0] == pytest.approx(0.1)
    else:
        assert elapsed[0] == pytest.approx(10.0 if cleanup == "forced" else 20.0)
        assert kills == ([] if cleanup == "unknown" else [10.0])


class DummyWorkspace:
    def __init__(self, root_dir, custom_path: str = ""):
        self.root_dir = str(root_dir)
        self.custom_path = custom_path
        self.calls = []

    def get_root_dir(self):
        return self.root_dir

    def get_resources_file_path(self):
        return None

    def get_app_dir(self, job_id):
        return os.path.join(self.root_dir, job_id, "app")

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


def _write_runtime(tmp_path, studies):
    local = tmp_path / "local"
    local.mkdir(exist_ok=True)
    path = local / "study_runtime.yaml"
    path.write_text(yaml.safe_dump({"format_version": 2, "studies": studies}))
    return path


@pytest.mark.parametrize("launcher_class", [ClientProcessJobLauncher, ServerProcessJobLauncher])
@pytest.mark.parametrize("study", [None, "study-a", "study-b", "unconfigured"])
def test_study_environment(launcher_class, study, tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("DB_HOST", "inherited-host")
    monkeypatch.setenv("STUDY_PASSWORD", "synthetic-secret\nwith spaces")
    monkeypatch.setenv("EMPTY_SOURCE", "")
    parent_env = os.environ.copy()
    _write_runtime(
        tmp_path,
        {
            name: {
                "env": {"DB_HOST": name, "STUDY_PASSWORD": "overridden-source"},
                "secret_env": {"DB_PASSWORD": {"source": "STUDY_PASSWORD"}, "EMPTY": {"source": "EMPTY_SOURCE"}},
            }
            for name in ("default", "study-a", "study-b")
        },
    )
    custom_path = str(tmp_path / "custom")
    ctx = _build_fl_ctx(DummyWorkspace(tmp_path, custom_path))
    ctx.set_prop(FLContextKey.JOB_PROCESS_ARGS, {JobProcessArgs.AUTH_TOKEN: ("--token", "bootstrap-token")})
    meta = {JobConstants.JOB_ID: "job-test"}
    if study is not None:
        meta["study"] = study
    launcher = launcher_class()
    with (
        mock.patch.object(launcher, "get_command", return_value="python worker.py"),
        mock.patch("nvflare.app_common.job_launcher.process_launcher.spawn_process") as spawn,
    ):
        launcher.launch_job(meta, ctx)

    argv, child_env = spawn.call_args.args
    assert argv == ["python", "worker.py"]
    assert child_env[JobProcessEnv.AUTH_TOKEN] == "bootstrap-token"
    assert custom_path in child_env["PYTHONPATH"].split(os.pathsep)
    if study == "unconfigured":
        assert child_env["DB_HOST"] == "inherited-host"
        assert child_env["STUDY_PASSWORD"] == parent_env["STUDY_PASSWORD"]
        assert "no entry for study 'unconfigured'" in caplog.text
    else:
        assert child_env["DB_HOST"] == (study or "default")
        assert child_env["STUDY_PASSWORD"] == "overridden-source"
        assert child_env["DB_PASSWORD"] == parent_env["STUDY_PASSWORD"]
        assert child_env["EMPTY"] == ""
    assert os.environ == parent_env
    assert parent_env["STUDY_PASSWORD"] not in caplog.text


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("runtime", [False, True])
def test_runtime_is_optional_and_legacy_is_ignored(tmp_path, monkeypatch, legacy, runtime):
    monkeypatch.setenv("STUDY_PASSWORD", "inherited-secret")
    if legacy:
        (tmp_path / "local").mkdir()
        (tmp_path / "local" / "study_data.yaml").write_text("unused legacy config")
    expected_env = os.environ.copy()
    if runtime:
        _write_runtime(tmp_path, {"default": {"env": {"DB_HOST": "study-host"}}})
        expected_env["DB_HOST"] = "study-host"
    with mock.patch("nvflare.app_common.job_launcher.process_launcher.spawn_process") as spawn:
        DummyLauncher().launch_job({JobConstants.JOB_ID: "job"}, _build_fl_ctx(DummyWorkspace(tmp_path)))
    assert spawn.call_args.args[1] == expected_env


@pytest.mark.parametrize("failure", ["missing-secret", "malformed", "unsupported", "read-error"])
def test_runtime_errors_prevent_spawn(tmp_path, monkeypatch, failure, caplog):
    monkeypatch.delenv("MISSING_STUDY_SOURCE", raising=False)
    monkeypatch.setenv("DB_PASSWORD", "inherited-destination-must-not-mask-error")
    path = _write_runtime(tmp_path, {"default": {"secret_env": {"DB_PASSWORD": {"source": "MISSING_STUDY_SOURCE"}}}})
    if failure == "malformed":
        path.write_text("studies: [")
    elif failure == "unsupported":
        _write_runtime(tmp_path, {"other-study": {"datasets": {}}})
    elif failure == "read-error":
        path.unlink()
        path.mkdir()
    with mock.patch("nvflare.app_common.job_launcher.process_launcher.spawn_process") as spawn:
        with pytest.raises(ValueError) as error:
            DummyLauncher().launch_job({JobConstants.JOB_ID: "job"}, _build_fl_ctx(DummyWorkspace(tmp_path)))
    spawn.assert_not_called()
    if failure == "missing-secret":
        assert "MISSING_STUDY_SOURCE" in str(error.value)
    assert os.environ["DB_PASSWORD"] not in str(error.value) + caplog.text


def test_independent_launches_and_reload(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDY_PASSWORD", "shared-parent-secret")
    parent_env = os.environ.copy()
    studies = {
        "study-a": {"env": {"STUDY_SETTING": "a"}},
        "study-b": {"env": {"STUDY_SETTING": "b"}},
        "unused": {"secret_env": {"PASSWORD": {"source": "MISSING_STUDY_SOURCE"}}},
    }
    _write_runtime(tmp_path, studies)
    launcher = DummyLauncher()

    def launch(study):
        launcher.launch_job({JobConstants.JOB_ID: study, "study": study}, _build_fl_ctx(DummyWorkspace(tmp_path)))

    with mock.patch("nvflare.app_common.job_launcher.process_launcher.spawn_process") as spawn:
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(launch, ["study-a", "study-b"]))
        envs = [call.args[1] for call in spawn.call_args_list]
        assert {env["STUDY_SETTING"] for env in envs} == {"a", "b"}
        assert all(env["STUDY_PASSWORD"] == "shared-parent-secret" for env in envs)
        studies["study-a"]["env"]["STUDY_SETTING"] = "updated"
        _write_runtime(tmp_path, studies)
        launch("study-a")
        assert spawn.call_args.args[1]["STUDY_SETTING"] == "updated"
        assert {env["STUDY_SETTING"] for env in envs} == {"a", "b"}
    assert os.environ == parent_env


@pytest.mark.parametrize("use_posix_spawn", [False, True])
def test_study_environment_reaches_real_child(tmp_path, monkeypatch, use_posix_spawn):
    if use_posix_spawn and not hasattr(os, "posix_spawn"):
        pytest.skip("posix_spawn is unavailable")
    monkeypatch.setattr("nvflare.utils.process_utils._POSIX_SPAWN_SUPPORTED", use_posix_spawn)
    monkeypatch.setenv("STUDY_PASSWORD", "synthetic secret\nwith spaces")
    _write_runtime(
        tmp_path,
        {
            "default": {
                "env": {"STUDY_SETTING": "configured", "EMPTY": ""},
                "secret_env": {"PASSWORD": {"source": "STUDY_PASSWORD"}},
            }
        },
    )
    script = tmp_path / "check_env.py"
    script.write_text(
        "import os, sys\n"
        "valid = (os.environ['STUDY_SETTING'] == 'configured' and os.environ['EMPTY'] == ''\n"
        "         and os.environ['PASSWORD'] == os.environ['STUDY_PASSWORD'])\n"
        "sys.exit(0 if valid else 1)\n"
    )
    launcher = DummyLauncher(shlex.join([sys.executable, str(script)]))
    handle = launcher.launch_job({JobConstants.JOB_ID: "job"}, _build_fl_ctx(DummyWorkspace(tmp_path)))
    try:
        handle.wait()
        assert handle.poll() == JobReturnCode.SUCCESS
    finally:
        if handle.poll() == JobReturnCode.UNKNOWN:
            handle.terminate()


def test_launch_job_uses_posix_spawn_when_supported(monkeypatch, tmp_path):
    launcher = DummyLauncher()

    workspace = DummyWorkspace(tmp_path)
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
    assert handle.app_dir == workspace.get_app_dir("job-posix")
    assert spawned["setsid"] is True
    assert spawned["argv"][0].endswith("python")
    popen_mock.assert_not_called()


def test_launch_job_falls_back_when_spawn_fails(monkeypatch, tmp_path):
    launcher = DummyLauncher()

    workspace = DummyWorkspace(tmp_path)
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
    assert handle.app_dir == workspace.get_app_dir("job-fallback")


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


def test_process_launcher_handle_event_always_adds_launcher(monkeypatch):
    launcher = DummyLauncher()
    fl_ctx = FLContext()
    recorded = {}

    def fake_add_launcher(selected_launcher, ctx):
        recorded["launcher"] = selected_launcher
        recorded["ctx"] = ctx

    monkeypatch.setattr("nvflare.app_common.job_launcher.process_launcher.add_launcher", fake_add_launcher)

    launcher.handle_event(EventType.BEFORE_JOB_LAUNCH, fl_ctx)

    assert recorded["launcher"] is launcher
    assert recorded["ctx"] is fl_ctx
