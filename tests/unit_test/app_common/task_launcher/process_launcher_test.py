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


import os
import signal
import sys
from unittest.mock import Mock

import psutil
import pytest

from nvflare.apis.task_launcher_spec import (
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
from tests.process_launcher_test_utils import cleanup_launched_groups  # noqa: F401 (pytest fixture registration)
from tests.process_launcher_test_utils import _mock_handle, _process_running, _request, requires_waitid

pytestmark = pytest.mark.skipif(os.name != "posix", reason="ProcessTaskLauncher requires POSIX process groups")


def test_process_exit_probe_accepts_disappearance_between_identity_and_status():
    process = Mock(pid=1234)
    process.is_running.return_value = True
    process.status.side_effect = psutil.NoSuchProcess(process.pid)

    assert not _process_running(process)
    process.status.assert_called_once()


@pytest.mark.parametrize("status", [psutil.STATUS_RUNNING, psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD])
def test_process_exit_probe_distinguishes_live_and_terminal_processes(status):
    process = Mock()
    process.is_running.return_value = True
    process.status.return_value = status

    assert _process_running(process) == (status == psutil.STATUS_RUNNING)


def test_process_exit_probe_does_not_treat_access_denied_as_exit():
    process = Mock(pid=1234)
    process.is_running.return_value = True
    process.status.side_effect = psutil.AccessDenied(process.pid)

    with pytest.raises(psutil.AccessDenied):
        _process_running(process)


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
