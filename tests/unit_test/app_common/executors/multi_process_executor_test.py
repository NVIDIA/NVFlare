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

"""Regressions for bounded, orderly multi-process worker shutdown."""

import signal
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.executors import multi_process_executor as module
from nvflare.app_common.executors.multi_process_executor import MultiProcessExecutor
from nvflare.fuel.common.multi_process_executor_constants import MultiProcessCommandNames


class _Executor(MultiProcessExecutor):
    def get_multi_process_command(self):
        return "unused"


@pytest.fixture
def executor(monkeypatch):
    executor = _Executor()
    executor.engine = SimpleNamespace(client=SimpleNamespace(cell=Mock()))
    executor.targets = ["site-1.job-1.0", "site-1.job-1.1"]
    executor.engine.client.cell.fire_and_forget.return_value = {}
    executor.logger = Mock()
    executor.log_info = Mock()
    executor.log_warning = Mock()
    executor.log_error = Mock()
    executor.exe_process = Mock(spec=subprocess.Popen)
    executor.exe_process.pid = 123456
    executor.exe_process.returncode = None
    executor.exe_process.poll.return_value = None
    executor.exe_process.wait.return_value = 0
    monkeypatch.setattr(module.os, "killpg", Mock())
    return executor


def test_normal_shutdown_delivers_close_before_waiting_and_preserves_launcher(executor):
    events = []
    executor.engine.client.cell.fire_and_forget.side_effect = lambda **_kwargs: events.append("close")
    executor.exe_process.wait.side_effect = lambda **_kwargs: events.append("reaped") or 0

    executor.finalize(FLContext())

    assert events == ["close", "reaped"]
    close = executor.engine.client.cell.fire_and_forget.call_args.kwargs
    assert close["targets"] == executor.targets
    assert close["topic"] == MultiProcessCommandNames.CLOSE
    executor.exe_process.wait.assert_called_once_with(timeout=module._WORKER_SHUTDOWN_TIMEOUT)
    module.os.killpg.assert_not_called()
    executor.exe_process.kill.assert_not_called()
    executor.exe_process.terminate.assert_not_called()
    assert executor.finalized and executor.stop_execute


def test_slow_shutdown_kills_worker_group_and_reaps_launcher(executor):
    executor.exe_process.wait.side_effect = [
        subprocess.TimeoutExpired("worker", module._WORKER_SHUTDOWN_TIMEOUT),
        -signal.SIGKILL,
    ]

    executor.finalize(FLContext())

    module.os.killpg.assert_called_once_with(executor.exe_process.pid, signal.SIGKILL)
    assert executor.exe_process.wait.call_args_list == [
        call(timeout=module._WORKER_SHUTDOWN_TIMEOUT),
        call(timeout=module._WORKER_KILL_TIMEOUT),
    ]
    executor.exe_process.kill.assert_not_called()
    executor.exe_process.terminate.assert_not_called()


@pytest.mark.parametrize("failure", ["error reply", "exception"])
def test_close_delivery_failure_kills_and_reaps_without_a_grace_period(executor, failure):
    if failure == "error reply":
        executor.engine.client.cell.fire_and_forget.return_value = {executor.targets[-1]: "target unreachable"}
    else:
        executor.engine.client.cell.fire_and_forget.side_effect = RuntimeError("cell disconnected")

    executor.finalize(FLContext())

    module.os.killpg.assert_called_once_with(executor.exe_process.pid, signal.SIGKILL)
    executor.exe_process.wait.assert_called_once_with(timeout=module._WORKER_KILL_TIMEOUT)
    assert executor.finalized


def test_failed_group_signal_falls_back_to_killing_and_reaping_direct_child(executor):
    executor.exe_process.wait.side_effect = [
        subprocess.TimeoutExpired("worker", module._WORKER_SHUTDOWN_TIMEOUT),
        -signal.SIGKILL,
    ]
    module.os.killpg.side_effect = OSError("group inaccessible")

    executor.finalize(FLContext())

    module.os.killpg.assert_called_once_with(executor.exe_process.pid, signal.SIGKILL)
    executor.exe_process.kill.assert_called_once_with()
    assert executor.exe_process.wait.call_args_list == [
        call(timeout=module._WORKER_SHUTDOWN_TIMEOUT),
        call(timeout=module._WORKER_KILL_TIMEOUT),
    ]


def test_already_reaped_launcher_still_closes_ranks_without_signaling(executor):
    executor.exe_process.returncode = 0
    executor.exe_process.poll.return_value = 0

    executor.finalize(FLContext())

    executor.engine.client.cell.fire_and_forget.assert_called_once()
    close = executor.engine.client.cell.fire_and_forget.call_args.kwargs
    assert close["targets"] == executor.targets
    assert close["topic"] == MultiProcessCommandNames.CLOSE
    executor.exe_process.wait.assert_not_called()
    module.os.killpg.assert_not_called()
    executor.exe_process.kill.assert_not_called()
    executor.exe_process.terminate.assert_not_called()


def test_abort_skips_the_normal_grace_period_and_still_reaps(executor):
    abort_signal = Signal()
    abort_signal.trigger("test abort")

    executor._execute_multi_process("train", Shareable(), FLContext(), abort_signal)

    module.os.killpg.assert_called_once_with(executor.exe_process.pid, signal.SIGKILL)
    executor.exe_process.wait.assert_called_once_with(timeout=module._WORKER_KILL_TIMEOUT)
    assert executor.finalized


def test_end_run_waits_for_rank_handlers_before_close_and_finalizes_once(executor):
    events = []

    def acknowledge_handlers(**_kwargs):
        events.append("rank handlers complete")
        return {target: module.F3make_reply(module.F3ReturnCode.OK) for target in executor.targets}

    executor.engine.client.cell.broadcast_request.side_effect = acknowledge_handlers
    executor.engine.client.cell.fire_and_forget.side_effect = lambda **_kwargs: events.append("close")
    executor.exe_process.wait.side_effect = lambda **_kwargs: events.append("reaped") or 0

    executor.handle_event(EventType.END_RUN, FLContext())
    executor.handle_event(EventType.END_RUN, FLContext())
    executor.finalize(FLContext())

    assert events == ["rank handlers complete", "close", "reaped"]
    executor.engine.client.cell.broadcast_request.assert_called_once()
    broadcast = executor.engine.client.cell.broadcast_request.call_args.kwargs
    assert broadcast["targets"] == executor.targets
    assert broadcast["topic"] == MultiProcessCommandNames.FIRE_EVENT
    assert broadcast["timeout"] == module._WORKER_SHUTDOWN_TIMEOUT
    executor.engine.client.cell.fire_and_forget.assert_called_once()
    executor.exe_process.wait.assert_called_once_with(timeout=module._WORKER_SHUTDOWN_TIMEOUT)
    module.os.killpg.assert_not_called()


def test_end_run_timeout_reply_forces_cleanup_without_a_grace_period(executor):
    replies = {target: module.F3make_reply(module.F3ReturnCode.OK) for target in executor.targets}
    replies[executor.targets[-1]] = module.F3make_reply(module.F3ReturnCode.TIMEOUT)
    executor.engine.client.cell.broadcast_request.return_value = replies

    executor.handle_event(EventType.END_RUN, FLContext())

    executor.engine.client.cell.broadcast_request.assert_called_once()
    module.os.killpg.assert_called_once_with(executor.exe_process.pid, signal.SIGKILL)
    executor.exe_process.wait.assert_called_once_with(timeout=module._WORKER_KILL_TIMEOUT)


def test_run_aborted_between_tasks_skips_end_run_handshake_and_grace_period(executor):
    fl_ctx = FLContext()
    fl_ctx.set_prop(FLContextKey.RUN_ABORT_REQUESTED, True, private=True, sticky=False)

    executor.handle_event(EventType.END_RUN, fl_ctx)

    executor.engine.client.cell.broadcast_request.assert_not_called()
    executor.engine.client.cell.fire_and_forget.assert_called_once()
    assert executor.engine.client.cell.fire_and_forget.call_args.kwargs["topic"] == MultiProcessCommandNames.CLOSE
    module.os.killpg.assert_called_once_with(executor.exe_process.pid, signal.SIGKILL)
    executor.exe_process.wait.assert_called_once_with(timeout=module._WORKER_KILL_TIMEOUT)


def test_failed_launcher_exit_during_grace_still_cleans_up_rank_group(executor):
    executor.exe_process.wait.side_effect = [1, 1]

    executor.finalize(FLContext())

    module.os.killpg.assert_called_once_with(executor.exe_process.pid, signal.SIGKILL)
    assert executor.exe_process.wait.call_args_list == [
        call(timeout=module._WORKER_SHUTDOWN_TIMEOUT),
        call(timeout=module._WORKER_KILL_TIMEOUT),
    ]
