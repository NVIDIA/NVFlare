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
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

from nvflare.apis.task_launcher_spec import TaskExecutionPhase, TaskLauncherError, TaskLaunchError, TaskSettlementError
from nvflare.utils.process_utils import ProcessAdapter, spawn_process
from tests.local_task_launcher import ProcessTaskHandle, ProcessTaskLauncher, _OwnedProcessAdapter
from tests.process_launcher_test_utils import cleanup_launched_groups  # noqa: F401 (pytest fixture registration)
from tests.process_launcher_test_utils import (
    _assert_pid_not_running,
    _process_running,
    _request,
    _wait_for_file,
    requires_waitid,
)
from tests.timing_utils import WAIT_TIMEOUT, ManualClock, ThreadGroup


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
def test_indefinite_settlement_wait_does_not_block_concurrent_cancel(tmp_path, monkeypatch):
    from tests import local_task_launcher as launcher_module

    ready_path = tmp_path / "waiter.ready"
    code = "import pathlib, sys, time; pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(30)"
    launcher = ProcessTaskLauncher(stop_grace_period=0.2, poll_interval=0.01)
    handle = launcher.launch_task(_request(tmp_path, "attempt-concurrent-cancel", code, ready_path))
    wait_result = {}
    poll_sleep_entered = threading.Event()
    release_poll = threading.Event()
    cancel_done = threading.Event()
    waiter_context = threading.local()

    def controlled_sleep(seconds):
        if getattr(waiter_context, "active", False):
            poll_sleep_entered.set()
            assert release_poll.wait(WAIT_TIMEOUT)
        else:
            time.sleep(seconds)

    def wait_for_worker():
        waiter_context.active = True
        wait_result["status"] = handle.wait_for_settlement()

    def cancel_worker():
        wait_result["cancel"] = handle.cancel()
        cancel_done.set()

    local_time = SimpleNamespace(**vars(time))
    local_time.sleep = controlled_sleep
    monkeypatch.setattr(launcher_module, "time", local_time)
    with ThreadGroup(release_poll) as workers:
        try:
            _wait_for_file(ready_path)
            workers.start(wait_for_worker)
            assert poll_sleep_entered.wait(WAIT_TIMEOUT), "waiter never reached its settlement poll"
            workers.start(cancel_worker)
            assert cancel_done.wait(WAIT_TIMEOUT), "indefinite wait blocked concurrent cancellation"
        finally:
            release_poll.set()
            # Even a regression holding the handle lock must not strand the child.
            if not cancel_done.is_set():
                try:
                    os.killpg(handle.process_group_id, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    assert wait_result["cancel"].cancel_requested
    assert wait_result["cancel"].settled
    assert wait_result["status"].cancel_requested
    assert wait_result["status"].settled


@requires_waitid
def test_attempt_identity_cannot_be_reused(tmp_path):
    launcher = ProcessTaskLauncher()
    request = _request(tmp_path, "attempt-once", "pass")
    handle = launcher.launch_task(request)
    assert handle.wait_for_settlement(timeout=5.0).succeeded

    with pytest.raises(TaskLauncherError, match="already launched"):
        launcher.launch_task(request)


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
        assert status.settled
        assert status.exit_code == 0
        assert status.failure_reason is None
        # Process churn can delay settlement until after cancel records its request.
        if settle_with == "wait":
            assert status.succeeded
        assert owned.reaped
        assert sibling.status() == psutil.STATUS_ZOMBIE
        if sys.platform == "linux":
            # Darwin removes a zombie's PGID before its parent reaps it.
            assert os.getpgid(sibling_pid) == leader_pid
        # A scan can be inconclusive while unrelated processes churn. Cancel
        # may signal terminal members before settlement, but once settled the
        # handle must not probe or signal the released process-group ID again.
        calls_at_settlement = probe.call_count
        assert handle.poll() == status
        assert handle.wait_for_settlement(timeout=0) == status
        assert handle.cancel() == status
        assert probe.call_count == calls_at_settlement
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


@requires_waitid
@pytest.mark.parametrize("backend", ["popen", "posix_spawn"])
@pytest.mark.parametrize("log_target", ["spawn_helper", "launcher"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_post_spawn_logging_fault_preserves_fence_and_cleanup_handle(
    tmp_path, monkeypatch, backend, log_target, cleanup_fails
):
    class BrokenHandler(logging.Handler):
        def emit(self, record):
            raise OSError("injected post-spawn logging failure")

    request = _request(tmp_path, "post-spawn-fault", "import time; time.sleep(30)")
    if backend == "posix_spawn":
        request = replace(request, cwd=None)
    launcher = ProcessTaskLauncher(stop_grace_period=0.1, poll_interval=0.01)
    created = []
    popen = subprocess.Popen
    posix_spawn = os.posix_spawn

    def track_popen(*args, **kwargs):
        process = popen(*args, **kwargs)
        created.append((process.pid, process))
        return process

    def track_posix_spawn(*args, **kwargs):
        pid = posix_spawn(*args, **kwargs)
        created.append((pid, None))
        return pid

    monkeypatch.setattr(subprocess, "Popen", track_popen)
    monkeypatch.setattr(os, "posix_spawn", track_posix_spawn)
    from nvflare.utils import process_utils

    logger = process_utils.log if log_target == "spawn_helper" else launcher.logger
    level = logger.level
    handler = BrokenHandler()
    cancel = ProcessTaskHandle.cancel
    if cleanup_fails:
        monkeypatch.setattr(ProcessTaskHandle, "cancel", Mock(side_effect=OSError("injected cleanup failure")))
    handle = None
    try:
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        with pytest.raises(TaskLaunchError, match="after acquiring") as caught:
            launcher.launch_task(request)
        handle = caught.value.handle
        logger.removeHandler(handler)
        logger.setLevel(level)
        assert len(created) == 1
        assert handle.process_group_id == created[0][0]
        if backend == "posix_spawn" and created[0][1] is not None:
            pytest.skip("host posix_spawn with setsid support selected the Popen fallback")
        assert (created[0][1] is None) == (backend == "posix_spawn")
        assert request.identity in launcher._attempt_identities
        with pytest.raises(TaskLauncherError, match="already launched"):
            launcher.launch_task(request)
        assert len(created) == 1
        if cleanup_fails:
            assert "cleanup unconfirmed" in str(caught.value)
            assert handle.poll().phase == TaskExecutionPhase.RUNNING
            assert not handle.poll().settled
        else:
            assert "settled during cleanup" in str(caught.value)
            assert handle.poll().settled
            _assert_pid_not_running(handle.process_group_id)
            with pytest.raises(ChildProcessError):
                os.waitpid(handle.process_group_id, os.WNOHANG)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
        monkeypatch.setattr(ProcessTaskHandle, "cancel", cancel)
        if handle is not None:
            assert handle.cancel().settled
        else:
            # Teardown for a regression that loses the acquired handle.
            for pid, process in created:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                if process is not None:
                    process.wait(timeout=5)
                else:
                    os.waitpid(pid, 0)


@requires_waitid
@pytest.mark.parametrize("backend", ["popen", "posix_spawn"])
def test_real_process_creation_failure_releases_attempt_fence(tmp_path, backend):
    request = replace(_request(tmp_path, "missing-command", "pass"), argv=(str(tmp_path / "missing-command"),))
    if backend == "posix_spawn":
        request = replace(request, cwd=None)
    launcher = ProcessTaskLauncher(max_attempt_identities=1)
    for _ in range(2):
        with pytest.raises(OSError):
            launcher.launch_task(request)
        assert request.identity not in launcher._attempt_identities


@pytest.mark.parametrize("bounded_wait", [False, True])
@requires_waitid
def test_exited_leader_keeps_pid_reserved_until_stubborn_descendant_settles(tmp_path, bounded_wait, monkeypatch):
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
        import tests.local_task_launcher as process_launcher

        clock = ManualClock()
        started = clock.monotonic()
        with monkeypatch.context() as clock_patch:
            clock.sleep = clock.advance
            clock_patch.setattr(process_launcher, "time", clock)
            try:
                status = handle.wait_for_settlement(timeout=0.12)
            except TimeoutError:
                # Retain the leader until the remaining descendant is cleaned up.
                status = handle.poll()
        assert clock.monotonic() - started <= 0.12 + 1e-9
        assert not status.succeeded
        if status.settled:
            _assert_pid_not_running(int(child_pid_path.read_text()))
    assert handle.cancel().settled
    _assert_pid_not_running(int(child_pid_path.read_text()))
    with pytest.raises(ChildProcessError):
        os.waitpid(handle.process_group_id, os.WNOHANG)


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
@pytest.mark.parametrize("backend", ["popen", "posix_spawn"])
def test_external_reaping_between_verification_and_signal_is_detected_later(tmp_path, monkeypatch, backend):
    # This deliberately violates the exclusive-reaper contract. It documents
    # fault detection without promising an atomic verify-and-signal syscall.
    ready = tmp_path / "signal-boundary-child.pid"
    child_code = "import os, pathlib, sys, time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
    leader_code = (
        "import pathlib, subprocess, sys, time; subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]); "
        "p = pathlib.Path(sys.argv[2]); "
        'exec("while not p.exists():\\n time.sleep(0.01)")'
    )
    request = _request(tmp_path, "external-reaper-signal-boundary", leader_code, child_code, ready)
    if backend == "posix_spawn":
        request = replace(request, cwd=None)
    process = spawn_process(list(request.argv), dict(request.environment), cwd=request.cwd)
    owned = _OwnedProcessAdapter(process)
    handle = ProcessTaskHandle(request, owned, stop_grace_period=0.1, descendant_settle_timeout=0.1, poll_interval=0.01)
    killpg = os.killpg
    stolen = False
    child = None
    try:
        if backend == "posix_spawn" and process.process is not None:
            pytest.skip("host posix_spawn with setsid support selected the Popen fallback")
        _wait_for_file(ready)
        child = psutil.Process(int(ready.read_text()))
        deadline = time.monotonic() + 5
        while owned.poll() is None:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        verify = owned.verify_identity

        def verify_then_external_reap():
            nonlocal stolen
            verify()
            _, exit_status = os.waitpid(process.pid, 0)
            stolen = True
            # The external reaper knows the exit status. Keep fixture Popen
            # finalization from waiting on the released numeric PID again.
            process._return_code = os.waitstatus_to_exitcode(exit_status)
            if process.process is not None:
                process.process.returncode = process._return_code

        emitted = []

        def signal_after_external_reap(pgid, sig):
            assert stolen
            assert pgid == os.getpgid(child.pid)
            with pytest.raises(ChildProcessError):
                os.waitid(os.P_PID, pgid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            emitted.append(sig)
            killpg(pgid, sig)

        monkeypatch.setattr(owned, "verify_identity", verify_then_external_reap)
        monkeypatch.setattr(os, "killpg", signal_after_external_reap)
        handle._signal_group(signal.SIGTERM)
        assert emitted == [signal.SIGTERM]
        status = handle.poll()
        assert not status.settled
        assert "externally reaped" in status.failure_reason
        with pytest.raises(TaskSettlementError, match="ownership is lost"):
            handle._signal_group(signal.SIGKILL)
        assert emitted == [signal.SIGTERM]
        while _process_running(child):
            assert time.monotonic() < deadline
            time.sleep(0.01)
    finally:
        monkeypatch.setattr(os, "killpg", killpg)
        if not stolen:
            try:
                killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        process.wait()
        if child is not None:
            try:
                if _process_running(child):
                    child.kill()
            except psutil.NoSuchProcess:
                pass


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

    checkout = Path(__file__).resolve().parents[3]
    assert Path(nvflare.__file__).resolve().is_relative_to(checkout)
    output = tmp_path / "import.json"
    code = (
        "import json, pathlib, sys; import nvflare; from tests.local_task_launcher import ProcessTaskLauncher; "
        "import tests.local_task_launcher as launcher; "
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
def test_identity_capacity_is_bounded_without_evicting_settled_attempts(tmp_path, monkeypatch):
    launcher = ProcessTaskLauncher(max_attempt_identities=1)
    request = _request(tmp_path, "retained-at-capacity", "pass")
    assert launcher.launch_task(request).wait_for_settlement(timeout=5).succeeded
    spawn = Mock()
    monkeypatch.setattr("tests.local_task_launcher.spawn_process", spawn)
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
            while _process_running(forker):
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


pytestmark = [pytest.mark.timeout(120), pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")]
