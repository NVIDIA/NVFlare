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


"""Shared setup for policy unit tests and real process integration scenarios."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.task_launcher_spec import TaskExecutionPhase, TaskExecutionStatus, TaskLauncherSpec, TaskLaunchRequest
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore
from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity, WorkerBootstrap, read_bootstrap

SETTLED = TaskExecutionStatus(TaskExecutionPhase.TERMINAL, exit_code=0, settled=True)


UNSETTLED = TaskExecutionStatus(TaskExecutionPhase.TERMINAL, exit_code=0)


RUNNING = TaskExecutionStatus(TaskExecutionPhase.RUNNING)


class RecordingLauncher(TaskLauncherSpec):
    launch_mode = "test"

    def __init__(self, on_launch):
        super().__init__()
        self.launch_task = Mock(side_effect=on_launch)

    def launch_task(self, request):
        raise NotImplementedError()


def _attempt(tmp_path, *, attempt_id="attempt-1"):
    identity = TaskAttemptIdentity("job", "owner", "task", "compute", attempt_id)
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    bootstrap = WorkerBootstrap(
        identity=identity,
        artifact_root=store.root_dir,
        workspace_root=str(tmp_path),
        executor={"path": "unused.Compute", "args": {}},
    )
    handle = SimpleNamespace(
        execution_id=f"test:{attempt_id}",
        poll=Mock(return_value=UNSETTLED),
        cancel=Mock(return_value=replace(SETTLED, cancel_requested=True)),
        wait_for_settlement=Mock(return_value=SETTLED),
    )

    def on_launch(request):
        handle.request = request
        store.commit_result(identity, Shareable({"value": 2}))
        return handle

    def request_factory(actual_identity, bootstrap_path):
        assert actual_identity == identity
        assert read_bootstrap(bootstrap_path).identity == identity
        return TaskLaunchRequest(
            identity.job_id,
            identity.site_name,
            identity.task_id,
            identity.attempt_id,
            ("trusted-worker", bootstrap_path),
            environment={},
        )

    return {
        "bootstrap": bootstrap,
        "store": store,
        "input_payload": Shareable({"value": 1}),
        "runtime_root": str(tmp_path / "runtime"),
        "launcher": RecordingLauncher(on_launch),
        "abort_signal": Signal(),
        "request_factory": request_factory,
    }, handle


def _payload(attempt):
    return Path(attempt["store"].attempt_dir(attempt["bootstrap"].identity)) / "result.fobs"


def _records(attempt):
    return [json.loads(line) for line in (Path(attempt["runtime_root"]) / "diagnostics.jsonl").read_text().splitlines()]
