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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.executors import multi_process_executor as subject
from nvflare.fuel.common.multi_process_executor_constants import CommunicationMetaData as Meta
from nvflare.fuel.f3.cellnet.core_cell import make_reply as cell_reply
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellRC
from nvflare.private.defs import new_cell_message


class LocalExecutor(subject.MultiProcessExecutor):
    def get_multi_process_command(self):
        return "python"


@pytest.fixture
def env(monkeypatch):
    executor = LocalExecutor(num_of_processes=2)
    executor.executor = object()
    executor.targets = ["site-1.job.0", "site-1.job.1"]
    executor.exe_process = Mock()
    executor.exe_process.poll.return_value = None
    engine = Mock()
    engine.client.cell.fire_and_forget.return_value = {}
    context = FLContext()
    context.get_engine = lambda: engine
    executor.engine = engine
    abort = Signal()
    ticks = [0]
    actions = []

    def step(*args):
        ticks[0] += 1
        assert ticks[0] < 15, "executor failed to stop waiting"
        if actions:
            actions.pop(0)()
        return False

    monkeypatch.setattr(subject, "time", SimpleNamespace(monotonic=lambda: ticks[0], sleep=step))
    monkeypatch.setattr(executor._result_ready, "wait", step)
    return SimpleNamespace(executor=executor, context=context, engine=engine, abort=abort, actions=actions)


def deliver(executor, rank, result=None, execution_id=None):
    return executor.receive_execute_result(
        new_cell_message(
            {},
            {
                Meta.EXECUTION_ID: execution_id or executor._execution_id,
                Meta.RANK_NUMBER: rank,
                Meta.SHAREABLE: result,
                Meta.FL_CTX: FLContext(),
            },
        )
    )


def run(env):
    return env.executor.execute("train", Shareable(), env.context, env.abort)


def test_waits_for_every_rank_and_returns_rank_zero_result(env):
    expected = Shareable({"weights": [1, 2]})

    def rank_zero():
        deliver(env.executor, 0, expected)
        assert not env.executor.execute_complete

    env.actions.extend([rank_zero, lambda: deliver(env.executor, 1)])
    assert run(env) == expected
    assert not env.executor.finalized


@pytest.mark.parametrize("rank", [0, 1])
def test_rank_failure_stops_task_and_cleans_up(env, rank):
    env.actions.append(lambda: deliver(env.executor, rank, make_reply(ReturnCode.EXECUTION_EXCEPTION)))
    assert run(env).get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert env.executor.finalized
    env.executor.exe_process.wait.assert_called_once_with(timeout=5.0)


def test_failure_is_not_overwritten_by_late_rank_zero_success(env):
    def replies():
        deliver(env.executor, 1, make_reply(ReturnCode.EXECUTION_EXCEPTION))
        deliver(env.executor, 0, Shareable())

    env.actions.append(replies)
    assert run(env).get_return_code() == ReturnCode.EXECUTION_EXCEPTION


@pytest.mark.parametrize("before_dispatch", [True, False])
def test_abort_stops_wait_and_cleans_up(env, before_dispatch):
    if before_dispatch:
        env.abort.trigger(True)
    else:
        env.actions.append(lambda: env.abort.trigger(True))
    assert run(env).get_return_code() == ReturnCode.TASK_ABORTED
    assert env.executor.finalized
    if before_dispatch:
        assert env.engine.client.cell.fire_and_forget.call_count == 1  # CLOSE only


@pytest.mark.parametrize("raises", [True, False])
def test_dispatch_failure_returns_without_waiting(env, raises):
    if raises:
        env.engine.client.cell.fire_and_forget.side_effect = RuntimeError("send failed")
    else:
        env.engine.client.cell.fire_and_forget.return_value = {"site-1.job.1": CellRC.TARGET_UNREACHABLE}
    assert run(env).get_return_code() == ReturnCode.EXECUTION_RESULT_ERROR
    assert env.executor.finalized


def test_launcher_exit_stops_result_wait(env):
    env.executor.exe_process.poll.return_value = 1
    assert run(env).get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert env.executor.finalized


def test_configured_task_timeout_stops_live_but_stalled_workers(env):
    env.executor.task_timeout = 2
    assert run(env).get_return_code() == ReturnCode.TIMEOUT
    assert env.executor.finalized


def test_stale_and_duplicate_rank_results_are_rejected(env):
    def replies():
        assert deliver(env.executor, 0, Shareable(), "old-task").get_header(MessageHeaderKey.RETURN_CODE) != CellRC.OK
        deliver(env.executor, 0, Shareable({"round": 1}))
        deliver(env.executor, 0, Shareable({"round": 0}))
        deliver(env.executor, 1)

    env.actions.append(replies)
    assert run(env)["round"] == 1


@pytest.mark.parametrize("timeout", [0, -1, True, "5", float("nan"), float("inf")])
def test_invalid_task_timeout_is_rejected(timeout):
    with pytest.raises(ValueError, match="task_timeout"):
        LocalExecutor(task_timeout=timeout)


def test_initialization_failure_is_propagated_and_launcher_reaped(env, monkeypatch):
    env.context.get_identity_name = lambda: "site-1"
    env.context.get_job_id = lambda: "job"
    env.context.set_prop(FLContextKey.ARGS, SimpleNamespace(workspace="/tmp/workspace"))
    cell = env.engine.client.cell
    cell.get_fqcn.return_value = "site-1.job"
    cell.is_cell_reachable.return_value = True
    cell.broadcast_request.return_value = {"site-1.job.0": cell_reply(CellRC.PROCESS_EXCEPTION)}
    env.executor.exe_process.pid = 987654
    monkeypatch.setattr(subject, "popen_in_new_session", lambda *a, **k: env.executor.exe_process)
    with pytest.raises(RuntimeError, match="initializing multi-process rank"):
        env.executor._initialize_multi_process(env.context)
    env.executor.exe_process.terminate.assert_called_once()
    env.executor.exe_process.wait.assert_called_once_with(timeout=5.0)


def test_finalize_cleans_surviving_ranks_even_after_launcher_exit(env, monkeypatch):
    rank = Mock()
    env.executor._child_processes = {12345: rank}
    monkeypatch.setattr(subject.psutil, "wait_procs", lambda children, timeout: (children, []))
    env.executor.exe_process.poll.return_value = 0
    env.engine.client.cell.fire_and_forget.side_effect = RuntimeError("CLOSE failed")
    env.executor.finalize(env.context)
    env.executor.finalize(env.context)
    rank.terminate.assert_called_once()
    env.executor.exe_process.terminate.assert_called_once()
    env.executor.exe_process.wait.assert_called_once_with(timeout=5.0)


def test_end_run_is_acknowledged_before_close_and_reaping(env):
    cell = env.engine.client.cell
    cell.broadcast_request.return_value = {target: cell_reply(CellRC.OK) for target in env.executor.targets}
    env.executor.handle_event("_end_run", env.context)
    assert env.executor.finalized
    assert cell.mock_calls[0][0] == "broadcast_request"
    assert cell.mock_calls[1][0] == "fire_and_forget"
    assert env.executor.exe_process.wait.call_count == 2


def test_cleanup_escalates_when_rank_and_launcher_ignore_termination(env, monkeypatch):
    rank = Mock()
    env.executor._child_processes = {12345: rank}
    env.executor.exe_process.wait.side_effect = [subject.subprocess.TimeoutExpired("launcher", 5), 0]
    waits = iter([([], [rank]), ([rank], [])])
    monkeypatch.setattr(subject.psutil, "wait_procs", lambda *a, **k: next(waits))
    env.executor.finalize(env.context)
    rank.terminate.assert_called_once()
    rank.kill.assert_called_once()
    env.executor.exe_process.kill.assert_called_once()
    assert env.executor.exe_process.wait.call_count == 2


def test_descendant_discovery_keeps_original_launcher_identity(env, monkeypatch):
    launcher = Mock()
    launcher.is_running.return_value = False
    env.executor._launcher_process = launcher
    monkeypatch.setattr(subject.psutil, "Process", Mock(side_effect=AssertionError("must not rediscover by PID")))
    env.executor._remember_child_processes()
    launcher.children.assert_not_called()


def test_cleanup_kills_owned_rank_session_before_term_handler_can_fork(env, monkeypatch):
    rank = Mock(pid=12345)
    rank.is_running.return_value = True
    env.executor._child_processes = {rank.pid: rank}
    order = Mock()
    monkeypatch.setattr(subject.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(subject.os, "killpg", order.killpg)
    monkeypatch.setattr(subject.psutil, "wait_procs", lambda children, timeout: (children, []))
    rank.terminate.side_effect = lambda: order.terminate()
    env.executor.finalize(env.context)
    assert order.mock_calls[0].args == (rank.pid, subject.signal.SIGKILL)
    assert order.mock_calls[1][0] == "terminate"
