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

import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import psutil
import pytest

from nvflare.apis.task_launcher_spec import (
    TaskExecutionPhase,
    TaskLauncherError,
    TaskLaunchRequest,
    TaskResourceRequest,
    TaskSettlementError,
    UnsupportedTaskResourceError,
)
from nvflare.app_common.task_launcher.process_launcher import (
    ProcessTaskHandle,
    ProcessTaskLauncher,
    _OwnedProcessAdapter,
)
from nvflare.utils.process_utils import ProcessAdapter, spawn_process

pytestmark = pytest.mark.skipif(os.name != "posix", reason="ProcessTaskLauncher requires POSIX process groups")
requires_waitid = pytest.mark.skipif(
    not all(hasattr(os, name) for name in ("waitid", "WNOWAIT", "P_PID", "WEXITED", "WNOHANG", "CLD_EXITED")),
    reason="requires waitid/WNOWAIT (Linux, or macOS with Python >= 3.13)",
)


@pytest.fixture(autouse=True)
def cleanup_launched_groups(monkeypatch):
    handles = []
    launch = ProcessTaskLauncher.launch_task

    def tracked_launch(launcher, request):
        handle = launch(launcher, request)
        handles.append(handle)
        return handle

    monkeypatch.setattr(ProcessTaskLauncher, "launch_task", tracked_launch)
    yield
    for handle in handles:
        handle.cancel()


def _request(tmp_path, attempt_id, code, *args, environment=None):
    return TaskLaunchRequest(
        job_id="job-1",
        site_name="site-1",
        task_id="task-1",
        attempt_id=attempt_id,
        argv=(sys.executable, "-c", code, *map(str, args)),
        environment=dict(os.environ) if environment is None else environment,
        cwd=str(tmp_path),
    )


def _wait_for_file(path, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            pytest.fail(f"timed out waiting for {path}")
        time.sleep(0.01)


def _assert_pid_not_running(pid):
    try:
        if psutil.Process(pid).status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
            return
    except psutil.NoSuchProcess:
        return
    pytest.fail(f"process {pid} is still running after confirmed process-group settlement")


@pytest.mark.parametrize("field", ["stop_grace_period", "descendant_settle_timeout", "poll_interval"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 0, -1, True, False, None, "0.5"])
def test_launcher_rejects_invalid_timers(field, value):
    with pytest.raises(ValueError, match=field):
        ProcessTaskLauncher(**{field: value})


@pytest.mark.parametrize("field", ["stop_grace_period", "descendant_settle_timeout", "poll_interval"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_handle_rejects_nonfinite_timers(field, value, tmp_path):
    timers = {"stop_grace_period": 0.2, "descendant_settle_timeout": 0.1, "poll_interval": 0.01}
    timers[field] = value
    with pytest.raises(ValueError, match=field):
        ProcessTaskHandle(_request(tmp_path, "invalid-timer", "pass"), Mock(pid=1234), **timers)


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf"), -1, True, False, "0.5"])
def test_settlement_wait_rejects_invalid_timeout_before_polling(timeout, tmp_path):
    adapter = Mock(pid=1234)
    handle = ProcessTaskHandle(
        _request(tmp_path, "invalid-wait", "pass"),
        adapter,
        stop_grace_period=0.2,
        descendant_settle_timeout=0.1,
        poll_interval=0.01,
    )
    with pytest.raises(ValueError, match="timeout"):
        handle.wait_for_settlement(timeout=timeout)
    adapter.poll.assert_not_called()


@pytest.mark.parametrize("timeout", [None, 0, 0.1, 1])
def test_settlement_wait_preserves_valid_timeout_boundaries(timeout, tmp_path, monkeypatch):
    adapter = Mock(pid=1234)
    adapter.poll.return_value = 0
    handle = ProcessTaskHandle(
        _request(tmp_path, "valid-wait", "pass"),
        adapter,
        stop_grace_period=0.2,
        descendant_settle_timeout=0.1,
        poll_interval=0.01,
    )
    monkeypatch.setattr(handle, "_group_exists", lambda: False)
    assert handle.wait_for_settlement(timeout=timeout).succeeded


@requires_waitid
def test_clean_exit_honors_exact_environment_and_working_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOULD_NOT_LEAK", "resident-secret")
    output_path = tmp_path / "observation.json"
    code = (
        "import json, os, pathlib, sys, time; "
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({'cwd': os.getcwd(), 'value': os.environ['TASK_VALUE'], "
        "'unrequested': os.environ.get('SHOULD_NOT_LEAK')})); time.sleep(0.1)"
    )
    request = _request(
        tmp_path,
        "attempt-clean",
        code,
        output_path,
        environment={"TASK_VALUE": "present"},
    )

    handle = ProcessTaskLauncher().launch_task(request)
    assert os.getpgid(handle.process_group_id) == handle.process_group_id
    status = handle.wait_for_settlement(timeout=5.0)

    assert status.phase == TaskExecutionPhase.TERMINAL
    assert status.exit_code == 0
    assert status.termination_signal is None
    assert status.cancel_requested is False
    assert status.settled is True
    assert status.failure_reason is None
    assert status.succeeded is True
    assert json.loads(output_path.read_text()) == {
        "cwd": str(tmp_path),
        "value": "present",
        "unrequested": None,
    }
    assert handle.request.identity == ("job-1", "site-1", "task-1", "attempt-clean")
    assert handle.execution_id == f"process:{handle.process_group_id}"


@requires_waitid
def test_nonzero_and_signal_exits_are_distinct(tmp_path):
    launcher = ProcessTaskLauncher()
    nonzero = launcher.launch_task(_request(tmp_path, "attempt-nonzero", "import sys; sys.exit(23)"))
    signalled = launcher.launch_task(
        _request(tmp_path, "attempt-signal", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)")
    )

    nonzero_status = nonzero.wait_for_settlement(timeout=5.0)
    signal_status = signalled.wait_for_settlement(timeout=5.0)

    assert nonzero_status.exit_code == 23
    assert nonzero_status.termination_signal is None
    assert nonzero_status.settled is True
    assert nonzero_status.succeeded is False
    assert signal_status.exit_code is None
    assert signal_status.termination_signal == signal.SIGTERM
    assert signal_status.settled is True
    assert signal_status.succeeded is False


@requires_waitid
def test_cancel_terminates_leader_and_child_and_confirms_settlement(tmp_path):
    child_pid_path = tmp_path / "child.pid"
    ready_path = tmp_path / "leader.ready"
    child_code = (
        "import os, pathlib, signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
    )
    leader_code = (
        "import pathlib, signal, subprocess, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]); "
        "p = pathlib.Path(sys.argv[2]); "
        'exec("while not p.exists():\\n time.sleep(0.01)"); '
        "pathlib.Path(sys.argv[3]).write_text('ready'); time.sleep(30)"
    )
    launcher = ProcessTaskLauncher(stop_grace_period=0.5, descendant_settle_timeout=0.1, poll_interval=0.01)
    handle = launcher.launch_task(
        _request(tmp_path, "attempt-cancel", leader_code, child_code, child_pid_path, ready_path)
    )
    _wait_for_file(ready_path)
    _wait_for_file(child_pid_path)
    child_pid = int(child_pid_path.read_text())

    status = handle.cancel()

    assert status.phase == TaskExecutionPhase.TERMINAL
    assert status.cancel_requested is True
    assert status.settled is True
    assert status.succeeded is False
    assert status.termination_signal == signal.SIGKILL
    _assert_pid_not_running(child_pid)


@requires_waitid
def test_leader_exit_is_not_settlement_while_descendant_survives(tmp_path):
    child_pid_path = tmp_path / "lingering-child.pid"
    child_code = (
        "import os, pathlib, sys, time; " "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
    )
    leader_code = (
        "import pathlib, subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]); "
        "p = pathlib.Path(sys.argv[2]); "
        'exec("while not p.exists():\\n time.sleep(0.01)")'
    )
    launcher = ProcessTaskLauncher(stop_grace_period=0.5, descendant_settle_timeout=0.05, poll_interval=0.01)
    handle = launcher.launch_task(_request(tmp_path, "attempt-lingering", leader_code, child_code, child_pid_path))
    _wait_for_file(child_pid_path)
    child_pid = int(child_pid_path.read_text())

    status = handle.wait_for_settlement(timeout=5.0)

    assert status.exit_code == 0
    assert status.cancel_requested is False
    assert status.settled is True
    assert status.succeeded is False
    assert status.failure_reason == "task leader exited while descendants remained alive"
    _assert_pid_not_running(child_pid)


@requires_waitid
def test_wait_timeout_does_not_claim_settlement_or_cancel(tmp_path):
    launcher = ProcessTaskLauncher(stop_grace_period=0.2, poll_interval=0.01)
    handle = launcher.launch_task(_request(tmp_path, "attempt-timeout", "import time; time.sleep(30)"))

    with pytest.raises(TimeoutError, match="did not settle"):
        handle.wait_for_settlement(timeout=0.05)

    status = handle.poll()
    assert status.phase == TaskExecutionPhase.RUNNING
    assert status.cancel_requested is False
    assert status.settled is False
    assert handle.cancel().settled is True


@requires_waitid
def test_indefinite_settlement_wait_does_not_block_concurrent_cancel(tmp_path):
    ready_path = tmp_path / "waiter.ready"
    code = "import pathlib, sys, time; pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(30)"
    launcher = ProcessTaskLauncher(stop_grace_period=0.2, poll_interval=0.01)
    handle = launcher.launch_task(_request(tmp_path, "attempt-concurrent-cancel", code, ready_path))
    _wait_for_file(ready_path)
    wait_result = {}
    wait_started = threading.Event()

    def wait_for_worker():
        wait_started.set()
        try:
            wait_result["status"] = handle.wait_for_settlement()
        except Exception as e:
            wait_result["error"] = e

    waiter = threading.Thread(target=wait_for_worker, daemon=True)
    waiter.start()
    assert wait_started.wait(timeout=1.0)
    time.sleep(0.05)

    cancel_status = handle.cancel()
    waiter.join(timeout=2.0)

    assert not waiter.is_alive(), "indefinite wait held the handle lock and blocked cancellation"
    assert "error" not in wait_result
    assert cancel_status.cancel_requested is True
    assert cancel_status.settled is True
    assert wait_result["status"].cancel_requested is True
    assert wait_result["status"].settled is True


@pytest.mark.parametrize(
    "resources",
    [
        TaskResourceRequest(cpu_cores=1),
        TaskResourceRequest(memory_mb=512),
        TaskResourceRequest(gpu_count=1),
    ],
)
@requires_waitid
def test_resource_requests_are_rejected_without_admission(resources, tmp_path):
    request = _request(tmp_path, f"resource-{resources}", "pass")
    request = TaskLaunchRequest(
        job_id=request.job_id,
        site_name=request.site_name,
        task_id=request.task_id,
        attempt_id=request.attempt_id,
        argv=request.argv,
        environment=request.environment,
        cwd=request.cwd,
        resources=resources,
    )

    with pytest.raises(UnsupportedTaskResourceError, match="admission/reservation"):
        ProcessTaskLauncher().launch_task(request)


@requires_waitid
def test_attempt_identity_cannot_be_reused(tmp_path):
    launcher = ProcessTaskLauncher()
    request = _request(tmp_path, "attempt-once", "pass")
    handle = launcher.launch_task(request)
    assert handle.wait_for_settlement(timeout=5.0).succeeded

    with pytest.raises(TaskLauncherError, match="already launched"):
        launcher.launch_task(request)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("job_id", "", "job_id"),
        ("site_name", "\x00", "site_name"),
        ("task_id", " ", "task_id"),
        ("attempt_id", "", "attempt_id"),
    ],
)
def test_request_rejects_invalid_identity(field, value, message):
    values = {"job_id": "j", "site_name": "s", "task_id": "t", "attempt_id": "a"}
    values[field] = value
    with pytest.raises(ValueError, match=message):
        TaskLaunchRequest(**values, argv=(sys.executable,), environment={})


@pytest.mark.parametrize("cpu_cores", [float("nan"), float("inf"), float("-inf"), 0, -1, True, "1"])
def test_resource_request_rejects_invalid_cpu_count(cpu_cores):
    with pytest.raises(ValueError, match="cpu_cores.*finite positive"):
        TaskResourceRequest(cpu_cores=cpu_cores)


@pytest.mark.parametrize("cpu_cores", [None, 0.5, 1, 2.0])
def test_resource_request_accepts_finite_cpu_count(cpu_cores):
    assert TaskResourceRequest(cpu_cores=cpu_cores).cpu_cores == cpu_cores


@pytest.mark.parametrize("argv", ["python", b"python", bytearray(b"python")])
def test_request_rejects_scalar_argv(argv):
    with pytest.raises(ValueError, match="argv.*sequence"):
        TaskLaunchRequest(job_id="j", site_name="s", task_id="t", attempt_id="a", argv=argv)


@pytest.mark.parametrize(
    "statuses, settled", [([psutil.STATUS_ZOMBIE], True), ([psutil.STATUS_ZOMBIE, psutil.STATUS_RUNNING], False)]
)
def test_settlement_distinguishes_zombies_from_live_group_members(tmp_path, monkeypatch, statuses, settled):
    handle = ProcessTaskHandle(
        _request(tmp_path, "zombie", "pass"),
        Mock(pid=1234, poll=lambda: 0),
        stop_grace_period=0.2,
        descendant_settle_timeout=0.1,
        poll_interval=0.01,
    )
    monkeypatch.setattr(os, "killpg", lambda *_args: None)
    monkeypatch.setattr(os, "getpgid", lambda _pid: 1234)
    processes = {i: Mock(pid=i, status=lambda s=s: s, create_time=lambda: 1.0) for i, s in enumerate(statuses)}
    monkeypatch.setattr(psutil, "pids", lambda: list(processes))
    monkeypatch.setattr(psutil, "Process", processes.__getitem__)
    assert handle.poll().settled is settled


def test_group_inspection_permission_failure_does_not_claim_settlement(tmp_path, monkeypatch):
    process = Mock(pid=1234)
    process.status.side_effect = psutil.AccessDenied(pid=1234)
    handle = ProcessTaskHandle(
        _request(tmp_path, "unknown-status", "pass"),
        Mock(pid=1234, poll=lambda: 0),
        stop_grace_period=0.2,
        descendant_settle_timeout=0.1,
        poll_interval=0.01,
    )
    monkeypatch.setattr(os, "killpg", lambda *_args: None)
    monkeypatch.setattr(os, "getpgid", lambda _pid: 1234)
    monkeypatch.setattr(psutil, "pids", lambda: [process.pid])
    monkeypatch.setattr(psutil, "Process", lambda _pid: process)
    assert handle.poll().settled is False


def test_process_enumeration_permission_failure_does_not_claim_settlement(tmp_path, monkeypatch):
    handle = ProcessTaskHandle(
        _request(tmp_path, "uninspectable-group", "pass"),
        Mock(pid=1234, poll=lambda: 0),
        stop_grace_period=0.2,
        descendant_settle_timeout=0.1,
        poll_interval=0.01,
    )
    monkeypatch.setattr(os, "killpg", lambda *_args: None)
    monkeypatch.setattr(psutil, "pids", Mock(side_effect=PermissionError("cannot enumerate processes")))
    assert handle.poll().settled is False


@pytest.mark.skipif(sys.platform != "linux", reason="Linux zombie process-group regression")
@requires_waitid
def test_real_zombie_group_settles_without_signalling_or_waiting_for_init(tmp_path, monkeypatch):
    process = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    try:
        observation = psutil.Process(process.pid)
        deadline = time.monotonic() + 5
        while observation.status() != psutil.STATUS_ZOMBIE:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        handle = ProcessTaskHandle(
            _request(tmp_path, "real-zombie", "pass"),
            _OwnedProcessAdapter(ProcessAdapter(process=process)),
            stop_grace_period=0.2,
            descendant_settle_timeout=0.1,
            poll_interval=0.01,
        )
        killpg = Mock(wraps=os.killpg)
        monkeypatch.setattr(os, "killpg", killpg)
        assert handle.wait_for_settlement(timeout=1).succeeded
        assert all(call.args[1] == 0 for call in killpg.call_args_list)
        with pytest.raises(ChildProcessError):
            os.waitpid(process.pid, os.WNOHANG)
    finally:
        process.wait(timeout=5)


@requires_waitid
@pytest.mark.parametrize("settle_with", ["wait", "cancel"])
def test_zombie_member_with_nonreaping_parent_does_not_block_settlement(tmp_path, monkeypatch, settle_with):
    # The test remains the live parent of both children. Put them in one group
    # in our session so a zombie sibling survives the owned leader's reap,
    # without requiring PID 1 or the test parent to reap it for settlement.
    release = tmp_path / "leader.release"
    code = "import pathlib, sys, time; p = pathlib.Path(sys.argv[1]); "
    code += 'exec("while not p.exists():\\n time.sleep(0.01)")'
    leader_pid = os.posix_spawn(
        sys.executable, [sys.executable, "-c", code, str(release)], dict(os.environ), setpgroup=0
    )
    sibling_pid = None
    owned = _OwnedProcessAdapter(ProcessAdapter(pid=leader_pid))
    try:
        sibling_pid = os.posix_spawn(
            sys.executable, [sys.executable, "-c", "pass"], dict(os.environ), setpgroup=leader_pid
        )
        assert os.getpgid(leader_pid) == leader_pid
        sibling = psutil.Process(sibling_pid)
        deadline = time.monotonic() + 5
        while sibling.status() != psutil.STATUS_ZOMBIE:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        release.write_text("exit")
        while owned.poll() is None:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        handle = ProcessTaskHandle(
            _request(tmp_path, "unreaped-zombie-member", "pass"),
            owned,
            stop_grace_period=0.1,
            descendant_settle_timeout=0.1,
            poll_interval=0.01,
        )
        probe = Mock(wraps=os.killpg)
        monkeypatch.setattr(os, "killpg", probe)
        status = handle.wait_for_settlement(timeout=1) if settle_with == "wait" else handle.cancel()
        assert status.succeeded
        assert owned.reaped
        assert sibling.status() == psutil.STATUS_ZOMBIE
        if sys.platform == "linux":
            # Darwin removes a zombie's PGID before its parent reaps it.
            assert os.getpgid(sibling_pid) == leader_pid
        assert all(call.args[1] == 0 for call in probe.call_args_list)
        assert handle.poll() == status
        assert handle.wait_for_settlement(timeout=0) == status
        assert handle.cancel() == status
        with pytest.raises(ChildProcessError):
            os.waitpid(leader_pid, os.WNOHANG)
    finally:
        if not owned.reaped:
            try:
                os.killpg(leader_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.waitpid(leader_pid, 0)
        if sibling_pid is not None:
            os.waitpid(sibling_pid, 0)


@pytest.mark.parametrize("churn", ["new-zombie", "reused-pid", "vanished-before-inspection"])
def test_unstable_dead_member_snapshots_do_not_prove_settlement(tmp_path, monkeypatch, churn):
    handle = _mock_handle(tmp_path)
    handle._adapter.owns_exited_leader = True
    monkeypatch.setattr(os, "killpg", lambda *_args: None)
    monkeypatch.setattr(os, "getpgid", lambda _pid: 1234)
    member = Mock(status=lambda: psutil.STATUS_ZOMBIE, create_time=Mock(return_value=1.0))
    monkeypatch.setattr(psutil, "Process", lambda _pid: member)
    snapshots = Mock(side_effect=[[5678], [5678, 5679]]) if churn == "new-zombie" else Mock(return_value=[5678])
    monkeypatch.setattr(psutil, "pids", snapshots)
    if churn == "reused-pid":
        member.create_time.side_effect = [1.0, 2.0]
    elif churn == "vanished-before-inspection":
        monkeypatch.setattr(psutil, "Process", Mock(side_effect=psutil.NoSuchProcess(5678)))
    assert not handle.poll().settled
    handle._adapter.reap.assert_not_called()


def test_terminal_member_with_missing_group_id_is_confirmed_twice(tmp_path, monkeypatch):
    handle = _mock_handle(tmp_path)
    handle._adapter.owns_exited_leader = True
    monkeypatch.setattr(os, "killpg", Mock(side_effect=PermissionError()))
    monkeypatch.setattr(os, "getpgid", Mock(side_effect=ProcessLookupError()))
    monkeypatch.setattr(psutil, "pids", lambda: [1234, 5678])
    member = Mock(status=lambda: psutil.STATUS_ZOMBIE, create_time=lambda: 1.0)
    inspect = Mock(return_value=member)
    monkeypatch.setattr(psutil, "Process", inspect)
    assert handle.poll().settled
    assert inspect.call_count == 2


def _mock_handle(tmp_path):
    return ProcessTaskHandle(
        _request(tmp_path, "fault", "pass"),
        Mock(pid=1234, poll=lambda: 0),
        stop_grace_period=0.1,
        descendant_settle_timeout=0.1,
        poll_interval=0.01,
    )


@pytest.mark.parametrize("error", [ProcessLookupError(), PermissionError("denied")])
def test_group_signalling_handles_missing_or_inaccessible_group(tmp_path, monkeypatch, error):
    handle = _mock_handle(tmp_path)
    monkeypatch.setattr(os, "killpg", Mock(side_effect=error))
    handle._signal_group(signal.SIGTERM)
    if isinstance(error, PermissionError):
        assert handle._group_exists() is True


def test_group_inspection_tolerates_process_exit_between_enumeration_and_probe(tmp_path, monkeypatch):
    handle = _mock_handle(tmp_path)
    monkeypatch.setattr(os, "killpg", lambda *_args: None)
    monkeypatch.setattr(os, "getpgid", Mock(side_effect=ProcessLookupError()))
    monkeypatch.setattr(psutil, "pids", lambda: [10])
    monkeypatch.setattr(psutil, "Process", Mock(side_effect=psutil.NoSuchProcess(10)))
    assert handle._group_exists() is True  # No observed dead members: fail closed.


@pytest.mark.parametrize("members", [[], [10]])
def test_group_disappearing_during_inspection_is_settled(tmp_path, monkeypatch, members):
    handle = _mock_handle(tmp_path)
    # The initial signal probe succeeds, but init reaps the last member before
    # inspection can observe it. A fresh signal probe confirms the empty scope.
    probe = Mock(side_effect=[None, ProcessLookupError()])
    monkeypatch.setattr(os, "killpg", probe)
    monkeypatch.setattr(os, "getpgid", Mock(side_effect=ProcessLookupError()))
    monkeypatch.setattr(psutil, "pids", lambda: members)
    monkeypatch.setattr(psutil, "Process", Mock(side_effect=psutil.NoSuchProcess(10)))
    if members:
        # A vanished unidentified parent is uncertain until the next probe
        # confirms that the whole group disappeared.
        assert not handle.poll().settled
    assert handle.poll().settled
    assert probe.call_count == 2


@pytest.mark.parametrize("result", [None, PermissionError("cannot confirm absence")])
def test_empty_process_snapshot_does_not_prove_settlement(tmp_path, monkeypatch, result):
    handle = _mock_handle(tmp_path)
    monkeypatch.setattr(os, "killpg", Mock(side_effect=[None, result]))
    monkeypatch.setattr(psutil, "pids", lambda: [])
    assert not handle.poll().settled


def test_confirmed_settlement_is_not_reopened_by_a_later_probe(tmp_path, monkeypatch):
    handle = _mock_handle(tmp_path)
    adapter_poll = Mock(return_value=0)
    monkeypatch.setattr(handle._adapter, "poll", adapter_poll)
    # A later scan could be inconclusive, or the numeric PGID could be reused.
    # Neither is part of the execution scope whose settlement was already proved.
    probe = Mock(side_effect=[False, True])
    monkeypatch.setattr(handle, "_group_exists", probe)
    signals = Mock()
    monkeypatch.setattr(handle, "_signal_group", signals)
    settled = handle.poll()
    assert settled.succeeded
    assert handle.poll() == settled
    assert handle.cancel() == settled
    assert handle.wait_for_settlement(timeout=0) == settled
    signals.assert_not_called()
    probe.assert_called_once()
    adapter_poll.assert_called_once()


def test_termination_keeps_settlement_proved_by_its_wait(tmp_path, monkeypatch):
    handle = _mock_handle(tmp_path)
    probe = Mock(side_effect=[True, True, False, True, True])
    monkeypatch.setattr(handle, "_group_exists", probe)
    signals = Mock()
    monkeypatch.setattr(handle, "_signal_group", signals)
    status = handle.cancel()
    assert status.settled
    assert status.cancel_requested
    assert status.failure_reason is None
    signals.assert_called_once_with(signal.SIGTERM)
    assert probe.call_count == 3


def test_final_cleanup_observation_can_confirm_settlement(tmp_path, monkeypatch):
    handle = _mock_handle(tmp_path)
    monkeypatch.setattr(handle, "_group_exists", Mock(side_effect=[True, True, False]))
    monkeypatch.setattr(handle, "_wait_for_group_exit", lambda _timeout: False)
    signals = Mock()
    monkeypatch.setattr(handle, "_signal_group", signals)
    status = handle.cancel()
    assert status.settled
    assert status.failure_reason is None
    assert [call.args[0] for call in signals.call_args_list] == [signal.SIGTERM, signal.SIGKILL]


def test_settlement_at_descendant_deadline_is_not_marked_as_failure(tmp_path, monkeypatch):
    handle = _mock_handle(tmp_path)
    monkeypatch.setattr(handle, "_group_exists", Mock(side_effect=[True, False]))
    monkeypatch.setattr(handle, "_wait_for_group_exit", lambda _timeout: False)
    signals = Mock()
    monkeypatch.setattr(handle, "_signal_group", signals)
    assert handle.wait_for_settlement().succeeded
    signals.assert_not_called()


def test_failed_termination_reports_unsettled_status_and_keeps_failure_reason(tmp_path, monkeypatch):
    handle = _mock_handle(tmp_path)
    monkeypatch.setattr(handle, "_group_exists", lambda: True)
    monkeypatch.setattr(handle, "_wait_for_group_exit", lambda _timeout: False)
    signals = Mock()
    monkeypatch.setattr(handle, "_signal_group", signals)
    with pytest.raises(TaskSettlementError, match="did not settle") as error:
        handle.cancel()
    assert not error.value.status.settled
    assert error.value.status.cancel_requested
    assert [call.args[0] for call in signals.call_args_list] == [signal.SIGTERM, signal.SIGKILL]
    handle._failure_reason = "earlier failure"
    with pytest.raises(TaskSettlementError, match="earlier failure"):
        handle.cancel()


def test_cancel_and_termination_are_idempotent_after_settlement(tmp_path, monkeypatch):
    handle = _mock_handle(tmp_path)
    monkeypatch.setattr(handle, "_group_exists", lambda: False)
    assert handle.cancel().succeeded
    assert handle._terminate_group().succeeded


@pytest.mark.parametrize("settles", [False, True])
def test_leader_exit_wait_respects_descendant_deadline(tmp_path, monkeypatch, settles):
    handle = _mock_handle(tmp_path)
    probe = Mock(side_effect=[True, False]) if settles else Mock(return_value=True)
    monkeypatch.setattr(handle, "_group_exists", probe)
    monkeypatch.setattr(handle, "_wait_for_group_exit", lambda _timeout: settles)
    if settles:
        assert handle.wait_for_settlement(timeout=0).succeeded
    else:
        with pytest.raises(TimeoutError, match="did not settle"):
            handle.wait_for_settlement(timeout=0)


def test_launcher_rejects_invalid_request_and_unsupported_platform(tmp_path, monkeypatch):
    launcher = ProcessTaskLauncher()
    with pytest.raises(TypeError, match="TaskLaunchRequest"):
        launcher.launch_task({})
    monkeypatch.delattr(os, "killpg")
    with pytest.raises(TaskLauncherError, match="POSIX"):
        launcher.launch_task(_request(tmp_path, "unsupported", "pass"))


@requires_waitid
def test_spawn_failure_allows_retry_of_same_attempt(tmp_path, monkeypatch):
    launcher = ProcessTaskLauncher(max_attempt_identities=1)
    request = _request(tmp_path, "retry-spawn", "pass")
    monkeypatch.setattr(
        "nvflare.app_common.task_launcher.process_launcher.spawn_process", Mock(side_effect=OSError("spawn failed"))
    )
    for _ in range(2):
        with pytest.raises(OSError, match="spawn failed"):
            launcher.launch_task(request)
        assert request.identity not in launcher._attempt_identities


@pytest.mark.parametrize("bounded_wait", [False, True])
@requires_waitid
def test_exited_leader_keeps_pid_reserved_until_stubborn_descendant_settles(tmp_path, bounded_wait):
    child_pid_path = tmp_path / "retained-child.pid"
    child_code = (
        "import os, pathlib, signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
    )
    leader_code = (
        "import pathlib, subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]); p = pathlib.Path(sys.argv[2]); "
        'exec("while not p.exists():\\n time.sleep(0.01)")'
    )
    handle = ProcessTaskLauncher(
        stop_grace_period=1.0 if bounded_wait else 0.1, descendant_settle_timeout=0.01, poll_interval=0.01
    ).launch_task(_request(tmp_path, "retained-leader", leader_code, child_code, child_pid_path))
    _wait_for_file(child_pid_path)
    deadline = time.monotonic() + 5
    while handle.poll().phase != TaskExecutionPhase.TERMINAL:
        assert time.monotonic() < deadline
        time.sleep(0.01)

    assert not handle.poll().settled
    observed = os.waitid(os.P_PID, handle.process_group_id, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    assert observed.si_pid == handle.process_group_id
    assert observed.si_status == 0
    if bounded_wait:
        started = time.monotonic()
        try:
            status = handle.wait_for_settlement(timeout=0.12)
        except TimeoutError:
            # The caller can finish observing cleanup after its wait expires.
            # The launcher must retain the leader until that proof is complete.
            status = handle.poll()
        assert time.monotonic() - started < 0.5
        assert not status.succeeded
        if status.settled:
            _assert_pid_not_running(int(child_pid_path.read_text()))
    assert handle.cancel().settled
    _assert_pid_not_running(int(child_pid_path.read_text()))
    with pytest.raises(ChildProcessError):
        os.waitpid(handle.process_group_id, os.WNOHANG)


@pytest.mark.parametrize("return_code", [None, 0])
@requires_waitid
def test_external_reaper_prevents_group_signals_and_settlement(tmp_path, monkeypatch, return_code):
    adapter = _OwnedProcessAdapter(Mock(pid=1234))
    adapter._return_code = return_code
    handle = ProcessTaskHandle(
        _request(tmp_path, "lost-identity", "pass"),
        adapter,
        stop_grace_period=0.1,
        descendant_settle_timeout=0.1,
        poll_interval=0.01,
    )
    monkeypatch.setattr(os, "waitid", Mock(side_effect=ChildProcessError()))
    killpg = Mock()
    monkeypatch.setattr(os, "killpg", killpg)

    assert not handle.poll().settled
    with pytest.raises(TaskSettlementError, match="ownership is lost") as error:
        handle.cancel()
    assert not error.value.status.settled
    assert error.value.status.cancel_requested
    killpg.assert_not_called()


@requires_waitid
@pytest.mark.parametrize("backend", ["popen", "posix_spawn"])
def test_external_reaper_between_verification_and_reap_fails_closed(tmp_path, monkeypatch, backend):
    request = _request(tmp_path, "external-reap-race", "pass")
    if backend == "posix_spawn":
        request = replace(request, cwd=None)
    process = spawn_process(list(request.argv), dict(request.environment), cwd=request.cwd)
    if backend == "posix_spawn" and process.process is not None:
        process.wait()
        pytest.skip("host posix_spawn with setsid support selected the Popen fallback")
    owned = _OwnedProcessAdapter(process)
    handle = ProcessTaskHandle(request, owned, stop_grace_period=0.1, descendant_settle_timeout=0.1, poll_interval=0.01)
    waitpid = os.waitpid
    stolen = False
    try:
        deadline = time.monotonic() + 5
        while owned.poll() is None:
            assert time.monotonic() < deadline
            time.sleep(0.01)

        def reap_elsewhere_then_wait(pid, flags):
            nonlocal stolen
            assert pid == process.pid
            waitpid(pid, 0)
            stolen = True
            return waitpid(pid, flags)

        monkeypatch.setattr(os, "waitpid", reap_elsewhere_then_wait)
        wrapped_poll = Mock(wraps=process.poll)
        monkeypatch.setattr(process, "poll", wrapped_poll)
        probe = Mock(wraps=os.killpg)
        monkeypatch.setattr(os, "killpg", probe)
        status = handle.poll()
        # Unrelated process-table churn can postpone the settlement proof.
        # Wait until reaping actually reaches the injected ownership race.
        while not stolen:
            assert not status.settled
            assert time.monotonic() < deadline
            time.sleep(0.01)
            status = handle.poll()
        assert not status.settled
        assert status.exit_code == 0
        assert status.termination_signal is None
        assert "externally reaped" in status.failure_reason
        assert not owned.reaped
        wrapped_poll.assert_not_called()
        # Lost ownership remains final even if a reused numeric PID becomes
        # observable as another child. Never signal or inspect it again.
        observation = Mock(return_value=Mock(si_pid=process.pid, si_status=0, si_code=os.CLD_EXITED))
        monkeypatch.setattr(os, "waitid", observation)
        with pytest.raises(TaskSettlementError, match="ownership is lost"):
            handle.cancel()
        assert not handle.poll().settled
        observation.assert_not_called()
        assert all(call.args[1] == 0 for call in probe.call_args_list)
    finally:
        monkeypatch.setattr(os, "waitpid", waitpid)
        if not stolen and owned.poll() is None:
            try:
                os.kill(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()


@requires_waitid
def test_polling_retains_leader_and_reaps_only_after_group_settlement(tmp_path, monkeypatch):
    process = ProcessAdapter(pid=1234)
    poll = Mock(wraps=process.poll)
    monkeypatch.setattr(process, "poll", poll)
    adapter = _OwnedProcessAdapter(process)
    monkeypatch.setattr(os, "waitid", Mock(return_value=Mock(si_pid=1234, si_status=23, si_code=os.CLD_EXITED)))
    reap = Mock(return_value=(1234, 23 << 8))
    monkeypatch.setattr(os, "waitpid", reap)
    handle = ProcessTaskHandle(
        _request(tmp_path, "reap-order", "pass"),
        adapter,
        stop_grace_period=0.1,
        descendant_settle_timeout=0.1,
        poll_interval=0.01,
    )
    monkeypatch.setattr(handle, "_group_exists", Mock(side_effect=[True, True, False]))

    assert handle.poll().exit_code == 23
    assert not handle.poll().settled
    reap.assert_not_called()
    poll.assert_not_called()
    assert handle.poll().settled
    reap.assert_called_once_with(1234, os.WNOHANG)
    assert process.poll() == 23
    poll.assert_called_once()  # Only this explicit adapter poll uses the wrapper.
    assert handle.cancel().settled
    reap.assert_called_once()


@pytest.mark.parametrize("missing", ["waitid", "WNOWAIT"])
def test_missing_nonreaping_wait_support_is_rejected_before_spawn(tmp_path, monkeypatch, missing):
    monkeypatch.delattr(os, missing, raising=False)
    spawn = Mock()
    monkeypatch.setattr("nvflare.app_common.task_launcher.process_launcher.spawn_process", spawn)
    with pytest.raises(TaskLauncherError, match="waitid/WNOWAIT"):
        ProcessTaskLauncher().launch_task(_request(tmp_path, "unsupported-wait", "pass"))
    spawn.assert_not_called()


def test_remembered_live_member_blocks_settlement_after_group_probe_disappears(tmp_path, monkeypatch):
    handle = _mock_handle(tmp_path)
    member = Mock(pid=5678, is_running=Mock(return_value=True), status=Mock(return_value=psutil.STATUS_RUNNING))
    handle._group_members[member.pid] = member
    monkeypatch.setattr(os, "killpg", Mock(side_effect=ProcessLookupError()))
    assert not handle.poll().settled
    member.status.return_value = psutil.STATUS_ZOMBIE
    assert handle.poll().settled


@requires_waitid
def test_cancel_does_not_affect_an_unrelated_process_group(tmp_path):
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        handle = ProcessTaskLauncher(stop_grace_period=0.1).launch_task(
            _request(tmp_path, "separate-group", "import time; time.sleep(30)")
        )
        assert handle.process_group_id != unrelated.pid
        assert handle.cancel().settled
        assert handle.cancel().settled
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait(timeout=5)


@requires_waitid
def test_subprocess_imports_the_isolated_checkout(tmp_path):
    import nvflare

    checkout = Path(__file__).resolve().parents[4]
    assert Path(nvflare.__file__).resolve().is_relative_to(checkout)
    output = tmp_path / "import.json"
    code = (
        "import json, pathlib, sys; import nvflare; from nvflare.app_common.task_launcher import ProcessTaskLauncher; "
        "import nvflare.app_common.task_launcher.process_launcher as launcher; "
        "pathlib.Path(sys.argv[1]).write_text(json.dumps([nvflare.__file__, launcher.__file__]))"
    )
    handle = ProcessTaskLauncher().launch_task(
        _request(
            tmp_path,
            "import-check",
            code,
            output,
            environment={"PYTHONPATH": str(checkout), "PYTHONDONTWRITEBYTECODE": "1"},
        )
    )
    assert handle.wait_for_settlement(timeout=5).succeeded
    assert all(Path(path).resolve().is_relative_to(checkout) for path in json.loads(output.read_text()))


@requires_waitid
@pytest.mark.parametrize("backend", ["popen", "posix_spawn"])
@pytest.mark.parametrize("exit_code", [-1, 256, 511, -257])
def test_full_exit_values_settle_once_on_both_spawn_paths(tmp_path, monkeypatch, backend, exit_code):
    request = _request(tmp_path, "full-exit", f"import sys; sys.exit({exit_code})")
    if backend == "posix_spawn":
        request = replace(request, cwd=None)
    handle = ProcessTaskLauncher().launch_task(request)
    if backend == "posix_spawn" and handle._adapter._adapter.process is not None:
        pytest.skip("host posix_spawn with setsid support selected the Popen fallback")
    assert (handle._adapter._adapter.process is None) == (backend == "posix_spawn")
    reap = Mock(wraps=handle._adapter.reap)
    monkeypatch.setattr(handle._adapter, "reap", reap)

    status = handle.wait_for_settlement(timeout=5)

    assert status.exit_code == exit_code & 0xFF
    assert status.termination_signal is None
    assert status.settled
    assert status.succeeded == (exit_code & 0xFF == 0)
    assert handle.poll() == status
    assert handle.wait_for_settlement(timeout=0) == status
    assert handle.cancel() == status
    reap.assert_called_once()
    with pytest.raises(ChildProcessError):
        os.waitpid(handle.process_group_id, os.WNOHANG)


@requires_waitid
@pytest.mark.parametrize("observed_exit", [256, 23])
def test_completed_reap_is_final_even_when_observed_exit_differs(tmp_path, monkeypatch, observed_exit):
    process = ProcessAdapter(pid=1234)
    adapter = _OwnedProcessAdapter(process)
    observation = Mock(return_value=Mock(si_pid=1234, si_status=observed_exit, si_code=os.CLD_EXITED))
    monkeypatch.setattr(os, "waitid", observation)
    reap = Mock(return_value=(1234, 0))
    monkeypatch.setattr(os, "waitpid", reap)
    handle = ProcessTaskHandle(
        _request(tmp_path, "completed-reap", "pass"),
        adapter,
        stop_grace_period=0.1,
        descendant_settle_timeout=0.1,
        poll_interval=0.01,
    )
    monkeypatch.setattr(handle, "_group_exists", lambda: False)
    assert handle.poll().succeeded
    assert adapter.reaped
    assert adapter.poll() == 0
    assert adapter.reap() == 0
    observation.assert_called()
    assert observation.call_count == 2
    reap.assert_called_once_with(1234, os.WNOHANG)
    assert process.poll() == 0
    assert handle.cancel().succeeded


@requires_waitid
def test_membership_churn_retains_leader_for_group_cleanup(tmp_path, monkeypatch):
    process = ProcessAdapter(pid=1234)
    adapter = _OwnedProcessAdapter(process)
    monkeypatch.setattr(os, "waitid", Mock(return_value=Mock(si_pid=1234, si_status=0, si_code=os.CLD_EXITED)))
    reap = Mock(return_value=(1234, 0))
    monkeypatch.setattr(os, "waitpid", reap)
    handle = ProcessTaskHandle(
        _request(tmp_path, "fork-during-snapshot", "pass"),
        adapter,
        stop_grace_period=0.1,
        descendant_settle_timeout=0.1,
        poll_interval=0.01,
    )
    # The snapshot omits a child forked by a disappearing parent. The retained
    # leader's terminal state cannot make that snapshot a settlement proof.
    monkeypatch.setattr(psutil, "pids", lambda: [5678])
    monkeypatch.setattr(psutil, "Process", Mock(side_effect=psutil.NoSuchProcess(5678)))
    monkeypatch.setattr(os, "getpgid", Mock(side_effect=ProcessLookupError()))
    probe = Mock()
    monkeypatch.setattr(os, "killpg", probe)
    assert not handle.poll().settled
    assert not adapter.reaped
    reap.assert_not_called()
    handle._signal_group(signal.SIGTERM)
    probe.assert_called_with(1234, signal.SIGTERM)

    # The leader is reaped only after the owned group is confirmed settled.
    probe.side_effect = ProcessLookupError()
    status = handle.poll()
    assert status.settled
    assert status.succeeded
    assert handle.wait_for_settlement(timeout=0) == status
    assert handle.cancel() == status
    reap.assert_called_once()


@pytest.mark.parametrize("capacity", [None, True, 0, -1, 1.5, "10"])
def test_identity_capacity_must_be_a_positive_integer(capacity):
    with pytest.raises(ValueError, match="max_attempt_identities"):
        ProcessTaskLauncher(max_attempt_identities=capacity)


@requires_waitid
def test_identity_capacity_is_bounded_without_evicting_settled_attempts(tmp_path, monkeypatch):
    launcher = ProcessTaskLauncher(max_attempt_identities=1)
    request = _request(tmp_path, "retained-at-capacity", "pass")
    assert launcher.launch_task(request).wait_for_settlement(timeout=5).succeeded
    spawn = Mock()
    monkeypatch.setattr("nvflare.app_common.task_launcher.process_launcher.spawn_process", spawn)
    with pytest.raises(TaskLauncherError, match="already launched"):
        launcher.launch_task(request)
    with pytest.raises(TaskLauncherError, match="capacity exhausted"):
        launcher.launch_task(replace(request, attempt_id="new-attempt"))
    assert launcher._attempt_identities == {request.identity}
    spawn.assert_not_called()


@requires_waitid
def test_real_worker_forked_after_snapshot_cannot_be_reported_settled(tmp_path, monkeypatch):
    fork_ready = tmp_path / "forker.ready"
    fork_now = tmp_path / "fork.now"
    child_pid_path = tmp_path / "late-worker.pid"
    forker_code = (
        "import os, pathlib, subprocess, sys, time; "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); p = pathlib.Path(sys.argv[2]); "
        'exec("while not p.exists():\\n time.sleep(0.01)"); '
        "subprocess.Popen([sys.executable, '-c', "
        "'import os, pathlib, sys, time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)', "
        "sys.argv[3]]); p = pathlib.Path(sys.argv[3]); "
        'exec("while not p.exists():\\n time.sleep(0.01)")'
    )
    leader_code = "import subprocess, sys; subprocess.Popen([sys.executable, '-c', sys.argv[1], *sys.argv[2:]])"
    request = _request(tmp_path, "real-churn", leader_code, forker_code, fork_ready, fork_now, child_pid_path)
    process = spawn_process(list(request.argv), dict(request.environment), cwd=request.cwd)
    owned = _OwnedProcessAdapter(process)
    handle = ProcessTaskHandle(request, owned, stop_grace_period=0.1, descendant_settle_timeout=0.1, poll_interval=0.01)
    child_pid = None
    try:
        _wait_for_file(fork_ready)
        deadline = time.monotonic() + 5
        while owned.poll() is None:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        forker = psutil.Process(int(fork_ready.read_text()))
        pids = psutil.pids

        def fork_during_snapshot():
            snapshot = pids()
            fork_now.write_text("fork")
            _wait_for_file(child_pid_path)
            deadline = time.monotonic() + 5
            while True:
                try:
                    if not forker.is_running() or forker.status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                        break
                except psutil.NoSuchProcess:
                    break
                assert time.monotonic() < deadline
                time.sleep(0.01)
            return snapshot

        monkeypatch.setattr(psutil, "pids", fork_during_snapshot)
        probe = Mock(wraps=os.killpg)
        monkeypatch.setattr(os, "killpg", probe)
        status = handle.poll()
        child_pid = int(child_pid_path.read_text())
        assert psutil.Process(child_pid).is_running()
        assert psutil.Process(child_pid).status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
        assert not status.settled
        assert not status.succeeded
        assert not owned.reaped
        assert handle.cancel().settled
        _assert_pid_not_running(child_pid)
        assert owned.reaped
        assert any(call.args[1] == signal.SIGTERM for call in probe.call_args_list)
    finally:
        if child_pid is None and child_pid_path.exists():
            child_pid = int(child_pid_path.read_text())
        if not owned.reaped:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        elif child_pid is not None:
            # Fallback teardown for an assertion failure: successful cleanup
            # above must be performed by the handle, not by this fixture.
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if child_pid is not None:
            deadline = time.monotonic() + 5
            while True:
                try:
                    if psutil.Process(child_pid).status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                        break
                except psutil.NoSuchProcess:
                    break
                assert time.monotonic() < deadline
                time.sleep(0.01)
