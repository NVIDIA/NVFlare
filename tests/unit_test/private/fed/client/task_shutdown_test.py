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

"""Shared native shutdown ownership, independent of optional backend packages."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.task_launcher_spec import (
    TaskExecutionPhase,
    TaskExecutionStatus,
    TaskHandleSpec,
    TaskLauncherSpec,
    TaskLaunchRequest,
    TaskSettlementError,
)
from nvflare.apis.utils.event import fire_event_to_components
from nvflare.private.fed.client import task_worker_executor as executor_module
from nvflare.private.fed.client.client_runner import ClientRunner
from nvflare.private.fed.client.task_execution_policy import TaskExecutionPolicy
from nvflare.private.fed.client.task_worker_executor import TaskWorkerExecutor
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore
from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity
from nvflare.private.fed.task_worker.supervisor import _Attempt
from nvflare.private.fed.tbi import TBI


@pytest.mark.parametrize("mode", ["docker", "k8s", "slurm"])
@pytest.mark.parametrize("pending_launch", [False, True], ids=["api-outage", "late-handle"])
def test_native_end_run_retains_owner_past_readiness_until_scope_and_executor_settle(
    tmp_path, monkeypatch, mode, pending_launch
):
    class SiteLauncher(TaskLauncherSpec):
        launch_mode = mode

        def launch_task(self, request):
            raise AssertionError("the test starts with an already-owned attempt")

    clock = SimpleNamespace(now=0.0)
    recovery_time = 10 if pending_launch else 240
    executor_done_time = recovery_time + 3
    request = TaskLaunchRequest("job", "site", "task", "attempt", ("unused",))

    class NativeHandle(TaskHandleSpec):
        settled = False

        @property
        def request(self):
            return request

        @property
        def execution_id(self):
            return "native:test-owned-execution"

        def poll(self):
            return TaskExecutionStatus(
                TaskExecutionPhase.TERMINAL if self.settled else TaskExecutionPhase.RUNNING,
                cancel_requested=True,
                settled=self.settled,
            )

        def cancel(self):
            if clock.now < recovery_time:
                raise TaskSettlementError("backend observation unavailable", self.poll())
            self.settled = True
            return self.poll()

        def wait_for_settlement(self, timeout=None):
            return self.cancel()

    handle = NativeHandle()
    adapter = TaskWorkerExecutor(
        {}, (), TaskExecutionPolicy(SiteLauncher(), poll_interval=0.5, settlement_timeout=0.01)
    )
    adapter._done.clear()
    adapter.system_panic = Mock()
    identity = TaskAttemptIdentity("job", "site", "task", "train", "attempt")
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    store.create_attempt(identity)
    attempt = _Attempt(identity, store, str(tmp_path), handle=None if pending_launch else handle)
    adapter.supervisor._active = attempt
    adapter.supervisor._attempts[identity] = attempt
    context = FLContext()
    engine = SimpleNamespace(
        new_context=lambda: context,
        fire_event=lambda event, ctx: fire_event_to_components(event, [adapter], ctx),
    )
    context.set_prop(ReservedKey.ENGINE, engine)
    runner = ClientRunner.__new__(ClientRunner)
    TBI.__init__(runner)
    runner.engine = engine
    runner.task_lock = threading.Lock()
    runner.running_tasks = {}
    runner._run_abort_requested = True

    def sleep(delay):
        clock.now += delay
        assert clock.now < 400, "shutdown did not finish after backend recovery"
        if pending_launch and clock.now >= recovery_time:
            attempt.handle = handle
        if clock.now >= executor_done_time:
            adapter._done.set()

    simulated_time = SimpleNamespace(time=lambda: clock.now, monotonic=lambda: clock.now, sleep=sleep)
    monkeypatch.setattr("nvflare.private.fed.tbi.time", simulated_time)
    monkeypatch.setattr(executor_module, "time", simulated_time)
    runner.end_run_events_sequence()
    assert clock.now >= executor_done_time
    assert handle.settled
    assert not adapter.supervisor.owns_execution
    assert adapter._done.is_set() and adapter._ending
    assert adapter.supervisor._stopping
    if not pending_launch:
        assert adapter.supervisor.settlement_unconfirmed
        adapter.system_panic.assert_called_once()
