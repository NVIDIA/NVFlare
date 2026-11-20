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
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from nvflare.apis.fl_constant import TaskResultReceipt
from nvflare.apis.task_launcher_spec import TaskLaunchRequest
from nvflare.private.fed.task_worker.supervisor import TaskSupervisor, TaskSupervisorOptions
from tests.local_task_launcher import ProcessTaskLauncher
from tests.task_supervisor_test_utils import _attempt, _payload


@pytest.mark.skipif(
    os.name != "posix" or not all(hasattr(os, name) for name in ("waitid", "WNOWAIT", "P_PID")),
    reason="Process launcher requires supported POSIX waitid/WNOWAIT (macOS Python >= 3.13)",
)
@pytest.mark.parametrize("fault", ["none", "exit", "signal", "fsync", "cancel"])
def test_real_p3_worker_p2_launcher_completion_and_settlement(tmp_path, fault):
    supervisor = TaskSupervisor()
    attempt, _ = _attempt(tmp_path)
    workspace = tmp_path / "workspace"
    (workspace / "startup").mkdir(parents=True)
    (workspace / "local").mkdir()
    custom = workspace / "job" / "app_owner" / "custom"
    custom.mkdir(parents=True)
    (custom.parent / "config").mkdir()
    (workspace / "job" / "meta.json").write_text(json.dumps({"byoc": True}))
    (custom / "p4_probe.py").write_text(
        "import os, nvflare\n"
        "from nvflare.apis.executor import Executor\n"
        "from nvflare.apis.shareable import Shareable\n"
        "class Probe(Executor):\n"
        "    def execute(self, task_name, shareable, fl_ctx, abort_signal):\n"
        "        return Shareable({'value': shareable['value'] + 1, 'pid': os.getpid(), 'source': nvflare.__file__})\n"
    )
    attempt["bootstrap"] = replace(
        attempt["bootstrap"],
        workspace_root=str(workspace),
        executor={"path": "p4_probe.Probe", "args": {}},
    )
    attempt["launcher"] = ProcessTaskLauncher(stop_grace_period=1.0, poll_interval=0.01)
    attempt["options"] = TaskSupervisorOptions(worker_timeout=30, poll_interval=0.01)
    repo_root = Path(__file__).resolve().parents[3]
    marker = tmp_path / "completed"
    # Inject failures at the actual completion boundary without replacing the
    # real worker entry, authorization, or Process launcher.
    injected_entry = f"""
import os, signal, stat, time
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore
original = FileTaskArtifactStore.commit_staged_result
def commit(self, identity, *args, **kwargs):
    if {fault!r} == 'fsync':
        sync = os.fsync
        def fail_once(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode) and os.path.exists(
                os.path.join(self.attempt_dir(identity), 'completion.json')
            ):
                os.fsync = sync
                raise OSError('injected directory fsync failure after completion')
            return sync(fd)
        os.fsync = fail_once
    result = original(self, identity, *args, **kwargs)
    if {fault!r} == 'exit':
        os._exit(7)
    if {fault!r} == 'signal':
        os.kill(os.getpid(), signal.SIGKILL)
    if {fault!r} == 'cancel':
        with open({str(marker)!r}, 'w') as stream:
            stream.write('completed')
        while True:
            time.sleep(0.01)
    return result
FileTaskArtifactStore.commit_staged_result = commit
from nvflare.private.fed.app.client.task_worker_process import main
main()
"""

    def request_factory(identity, bootstrap_path):
        argv = (
            (sys.executable, "-m", "nvflare.private.fed.app.client.task_worker_process")
            if fault == "none"
            else (sys.executable, "-c", injected_entry)
        )
        return TaskLaunchRequest(
            identity.job_id,
            identity.site_name,
            identity.task_id,
            identity.attempt_id,
            (*argv, "--bootstrap", bootstrap_path, "--execution_scope", "native"),
            environment={"PYTHONPATH": str(repo_root)},
            cwd=str(repo_root),
        )

    attempt["request_factory"] = request_factory
    if fault == "cancel":
        with ThreadPoolExecutor() as pool:
            running = pool.submit(supervisor.run, **attempt)
            try:
                deadline = time.monotonic() + 20
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert marker.exists(), "worker did not publish completion"
                supervisor.cancel_active()
                outcome = running.result(timeout=10)
            finally:
                supervisor.end_run()
        assert outcome.aborted and outcome.result is None
        assert outcome.execution_status.settled
        assert supervisor._active is None
        assert _payload(attempt).exists()
        assert (_payload(attempt).parent / "completion.json").exists()
        return
    outcome = supervisor.run(**attempt)
    assert outcome.result["value"] == 2
    assert outcome.result["pid"] != os.getpid()
    assert outcome.result["pid"] == outcome.completion.worker_pid
    assert Path(outcome.result["source"]).resolve() == repo_root / "nvflare" / "__init__.py"
    assert outcome.execution_status.settled
    if fault == "exit":
        assert outcome.execution_status.exit_code == 7
    elif fault == "signal":
        assert outcome.execution_status.termination_signal == signal.SIGKILL
    elif fault == "fsync":
        assert outcome.execution_status.exit_code == 1
        assert (_payload(attempt).parent / "failure.json").exists()
    else:
        assert outcome.execution_status.succeeded
    assert supervisor._active is None
    supervisor.record_receipt(outcome.identity, TaskResultReceipt.RECEIVED)
    supervisor.end_run()
    assert _payload(attempt).exists() is (fault != "none")


pytestmark = [pytest.mark.timeout(120), pytest.mark.usefixtures("isolated_fobs")]
