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
from nvflare.utils.process_utils import ProcessAdapter

pytestmark = pytest.mark.skipif(os.name != "posix", reason="ProcessTaskLauncher requires POSIX process groups")


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
    monkeypatch.setattr(
        psutil, "process_iter", lambda: [Mock(pid=i, status=lambda s=s: s) for i, s in enumerate(statuses)]
    )
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
    monkeypatch.setattr(psutil, "process_iter", lambda: [process])
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
    monkeypatch.setattr(psutil, "process_iter", Mock(side_effect=PermissionError("cannot enumerate processes")))
    assert handle.poll().settled is False


@pytest.mark.skipif(sys.platform != "linux", reason="Linux zombie process-group regression")
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
    monkeypatch.setattr(psutil, "process_iter", lambda: [Mock(pid=10)])
    assert handle._group_exists() is True  # No observed dead members: fail closed.


@pytest.mark.parametrize("members", [[], [Mock(pid=10)]])
def test_group_disappearing_during_inspection_is_settled(tmp_path, monkeypatch, members):
    handle = _mock_handle(tmp_path)
    # The initial signal probe succeeds, but init reaps the last member before
    # inspection can observe it. A fresh signal probe confirms the empty scope.
    probe = Mock(side_effect=[None, ProcessLookupError()])
    monkeypatch.setattr(os, "killpg", probe)
    monkeypatch.setattr(os, "getpgid", Mock(side_effect=ProcessLookupError()))
    monkeypatch.setattr(psutil, "process_iter", lambda: members)
    assert handle.poll().settled
    assert probe.call_count == 2


@pytest.mark.parametrize("result", [None, PermissionError("cannot confirm absence")])
def test_empty_process_snapshot_does_not_prove_settlement(tmp_path, monkeypatch, result):
    handle = _mock_handle(tmp_path)
    monkeypatch.setattr(os, "killpg", Mock(side_effect=[None, result]))
    monkeypatch.setattr(psutil, "process_iter", lambda: [])
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


def test_spawn_failure_allows_retry_of_same_attempt(tmp_path, monkeypatch):
    launcher = ProcessTaskLauncher()
    request = _request(tmp_path, "retry-spawn", "pass")
    monkeypatch.setattr(
        "nvflare.app_common.task_launcher.process_launcher.spawn_process", Mock(side_effect=OSError("spawn failed"))
    )
    for _ in range(2):
        with pytest.raises(OSError, match="spawn failed"):
            launcher.launch_task(request)
        assert request.identity not in launcher._attempt_identities


@pytest.mark.parametrize("bounded_wait", [False, True])
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


def test_polling_retains_leader_and_reaps_only_after_group_settlement(tmp_path, monkeypatch):
    process = Mock(pid=1234, poll=Mock(return_value=23))
    adapter = _OwnedProcessAdapter(process)
    monkeypatch.setattr(os, "waitid", Mock(return_value=Mock(si_pid=1234, si_status=23, si_code=os.CLD_EXITED)))
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
    process.poll.assert_not_called()
    assert handle.poll().settled
    process.poll.assert_called_once()
    assert handle.cancel().settled
    process.poll.assert_called_once()


def test_missing_nonreaping_wait_support_is_rejected_before_spawn(tmp_path, monkeypatch):
    monkeypatch.delattr(os, "WNOWAIT")
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
