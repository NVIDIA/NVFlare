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

"""Real worker, test-only local launcher, sidecar and participant rollup."""

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from nvflare.apis.fl_constant import TaskResultReceipt
from nvflare.apis.task_launcher_spec import TaskLaunchRequest
from nvflare.private.fed.resource_stats import collector, cpu_consumed, task_rollup
from nvflare.private.fed.resource_stats.collector import JobResourceCollector, assemble_participant_summary
from nvflare.private.fed.resource_stats.contract import (
    capacity_to_resource_time,
    derive_job_totals,
    participant_measurements,
)
from nvflare.private.fed.task_worker.supervisor import TaskSupervisor, TaskSupervisorOptions
from tests.local_task_launcher import ProcessTaskLauncher as LocalProcessTaskLauncher
from tests.task_supervisor_test_utils import _attempt

pytestmark = [
    pytest.mark.timeout(120),
    pytest.mark.usefixtures("isolated_fobs"),
    pytest.mark.skipif(
        os.name != "posix" or not all(hasattr(os, name) for name in ("waitid", "WNOWAIT", "P_PID")),
        reason="Local task harness requires POSIX waitid/WNOWAIT (macOS Python >= 3.13)",
    ),
]


@pytest.mark.parametrize(
    "fail,completion_budget",
    [(False, False), (True, False), (False, True)],
    ids=["completed", "python-failure", "finite-completion-deadline"],
)
def test_task_worker_resource_report_reconciles_without_double_counting(tmp_path, monkeypatch, fail, completion_budget):
    # Isolate parent accounting; the child controls only its GPU inventory boundary.
    monkeypatch.setattr(collector, "_active_collector", None)
    monkeypatch.setattr(cpu_consumed, "_active", None)
    monkeypatch.setattr(task_rollup, "_ACTIVE", None)
    job = JobResourceCollector(
        tmp_path,
        task_based=True,
        capacity_probe=lambda: {"cpu": {"units": 1}, "memory": {"bytes": 1024}, "gpu": {"count": 0}},
    )
    supervisor = TaskSupervisor(resource_reducer=job._task_reducer)
    attempt, _ = _attempt(tmp_path)
    workspace = tmp_path / "workspace"
    (workspace / "startup").mkdir(parents=True)
    (workspace / "local").mkdir()
    custom = workspace / "job" / "app_owner" / "custom"
    custom.mkdir(parents=True)
    (custom.parent / "config").mkdir()
    (workspace / "job" / "meta.json").write_text(json.dumps({"byoc": True}))
    release = tmp_path / "inventory-release"
    inventory_reads = tmp_path / "inventory-reads"
    late_inventory = tmp_path / "late-inventory"
    application_done = tmp_path / "application-done"
    deadline_armed = tmp_path / "deadline-armed"
    entry = tmp_path / "worker_with_inventory.py"
    entry.write_text(
        "import time\n"
        "from pathlib import Path\n"
        "from nvflare.private.fed.resource_stats.probes import gpu_nvidia\n"
        "from nvflare.private.fed.app.client.task_worker_process import main\n"
        "queries = 0\n"
        "def inventory(*, timeout):\n"
        "    global queries\n"
        "    assert timeout == 2.0\n"
        "    queries += 1\n"
        f"    Path({str(inventory_reads)!r}).write_text(str(queries))\n"
        f"    if {completion_budget!r} and Path({str(application_done)!r}).exists():\n"
        f"        Path({str(late_inventory)!r}).touch()\n"
        f"        while not Path({str(release)!r}).exists(): time.sleep(0.01)\n"
        "    return [0]\n"
        "gpu_nvidia.get_host_gpu_ids = inventory\n"
        "main()\n"
    )
    worker_command = (sys.executable, str(entry))
    (custom / "resource_probe.py").write_text(
        "import time\n"
        "from pathlib import Path\n"
        "from nvflare.apis.executor import Executor\n"
        "from nvflare.apis.shareable import Shareable\n"
        "class Probe(Executor):\n"
        "    def execute(self, *args):\n"
        "        total = sum(range(2000000))\n"
        f"        if {fail!r}: raise ValueError('deliberate measured failure')\n"
        f"        Path({str(application_done)!r}).touch()\n"
        f"        if {completion_budget!r}:\n"
        "            deadline = time.monotonic() + 10\n"
        f"            while not Path({str(deadline_armed)!r}).exists():\n"
        "                if time.monotonic() >= deadline: raise TimeoutError('deadline was not armed')\n"
        "                time.sleep(0.01)\n"
        "        return Shareable({'total': total})\n"
    )
    attempt["bootstrap"] = replace(
        attempt["bootstrap"],
        workspace_root=str(workspace),
        executor={"path": "resource_probe.Probe", "args": {}},
    )
    attempt["launcher"] = LocalProcessTaskLauncher(stop_grace_period=1.0, poll_interval=0.01)
    attempt["options"] = TaskSupervisorOptions(worker_timeout=30, poll_interval=0.01)
    repo_root = Path(__file__).resolve().parents[3]
    records = []
    read = attempt["store"].read_resource_stats

    def capture(identity):
        record = read(identity)
        if record is not None:
            records.append(record)
        return record

    monkeypatch.setattr(attempt["store"], "read_resource_stats", capture)

    def request_factory(identity, bootstrap_path):
        return TaskLaunchRequest(
            identity.job_id,
            identity.site_name,
            identity.task_id,
            identity.attempt_id,
            (
                *worker_command,
                "--bootstrap",
                bootstrap_path,
                "--parent_pid",
                str(os.getpid()),
            ),
            environment={"PYTHONPATH": str(repo_root), "CUDA_VISIBLE_DEVICES": "0"},
            cwd=str(repo_root),
        )

    attempt["request_factory"] = request_factory
    if completion_budget:
        completion_deadline = None

        def deadline_provider():
            nonlocal completion_deadline
            if completion_deadline is None and application_done.exists():
                completion_deadline = time.monotonic() + 1.0
                deadline_armed.touch()
            return completion_deadline

        attempt["deadline_provider"] = deadline_provider
    try:
        if fail:
            with pytest.raises(RuntimeError, match="deliberate measured failure"):
                supervisor.run(**attempt)
        else:
            outcome = supervisor.run(**attempt)
            assert outcome.execution_status.settled
            assert outcome.result["total"] == sum(range(2000000))
        worker = job._task_reducer.snapshot()
        assert records, "the real worker must publish its sidecar before settlement returns"
        record = records[0]
        assert inventory_reads.read_text() == "1"
        assert record["resource_capacity"]["gpu"] == {"count": 1}
        assert not late_inventory.exists(), "GPU inventory must not be queried after application completion"
        if completion_budget:
            assert application_done.exists() and not release.exists()
            assert completion_deadline is not None
        worker_cpu = float(record["cpu_consumed"]["cpu_seconds"])
        assert worker_cpu > 0
        assert worker["cpu_consumed"]["cpu_seconds"] == record["cpu_consumed"]["cpu_seconds"]
        assert worker["resource_time"] == capacity_to_resource_time(
            record["resource_capacity"], record["duration_seconds"], scope_issues=record["issues"]
        )
        assert worker["duration_contained"]
        assert not Path(attempt["store"].attempt_dir(attempt["bootstrap"].identity), "resource_stats.json").exists()
        if not fail:
            assert float(str(outcome.completion.diagnostics["cpu_seconds"])) == worker_cpu
            assert float(str(outcome.completion.diagnostics["elapsed_seconds"])) == float(record["duration_seconds"])
            before_retry = job._task_reducer.snapshot()
            assert supervisor.read_saved_result(outcome.identity)["total"] == outcome.result["total"]
            assert job._task_reducer.snapshot() == before_retry
            supervisor.record_receipt(outcome.identity, TaskResultReceipt.RECEIVED)
        coordinator = job.stop()
        handoff = job.finish(message_traffic={"sent_bytes": 0, "received_bytes": 0})
        expected_cpu = float(coordinator["cpu_consumed"]["cpu_seconds"]) + worker_cpu
        assert float(handoff["resource_utilization"]["cpu_consumed"]["cpu_seconds"]) == expected_cpu
        assert handoff["duration_seconds"] == coordinator["duration_seconds"]
        participant = assemble_participant_summary(job_id="job", participant_name="owner", child_handoff=handoff)
        entry = {
            "participant_name": "owner",
            "role": "client",
            "received_at": participant["reported_at"],
            **participant_measurements(participant),
        }
        totals = derive_job_totals([entry])
        assert totals["resource_time"] == participant["resource_time"]
        assert float(totals["resource_utilization"]["cpu_consumed"]["cpu_seconds"]) == expected_cpu
    finally:
        release.touch()
        supervisor.end_run()
