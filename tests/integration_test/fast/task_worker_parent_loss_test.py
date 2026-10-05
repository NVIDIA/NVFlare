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

"""Exercise the Process guardian with an exited, deliberately unreaped owner."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

_WORKER_SOURCE = """
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from nvflare.private.fed.app.client.task_worker_process import _start_parent_guard

_start_parent_guard(os.getppid())
guardian = psutil.Process().children()[0]
descendant = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
record = Path(sys.argv[1])
staged = record.with_suffix(".tmp")
staged.write_text(json.dumps({
    "worker": os.getpid(), "guardian": guardian.pid, "descendant": descendant.pid,
}))
staged.replace(record)
time.sleep(60)
"""

_JOB_SOURCE = """
import subprocess
import sys
import time

subprocess.Popen([sys.executable, "-c", sys.argv[1], sys.argv[2]], start_new_session=True)
time.sleep(60)
"""


def _alive(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.mark.skipif(not hasattr(os, "fork"), reason="Process guardian requires POSIX fork")
@pytest.mark.parametrize("repeat", range(2))
def test_guardian_settles_worker_group_without_waiting_for_job_reaping(tmp_path, repeat):
    # Do not call poll/wait until settlement: Darwin can retain the original
    # parent relationship while the killed job is an unreaped zombie.
    record = tmp_path / f"owned-processes-{repeat}.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT), env.get("PYTHONPATH", "")))
    job = subprocess.Popen(
        [sys.executable, "-c", _JOB_SOURCE, _WORKER_SOURCE, str(record)],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    job_identity = psutil.Process(job.pid)
    owned = {}
    try:
        deadline = time.monotonic() + 10
        while not record.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert record.exists(), "worker did not arm its guardian"
        pids = json.loads(record.read_text())
        owned = {name: psutil.Process(pid) for name, pid in pids.items()}
        assert all(_alive(process) for process in owned.values())
        assert owned["worker"].ppid() == job.pid
        assert os.getpgid(owned["worker"].pid) == owned["worker"].pid
        assert all(os.getpgid(process.pid) == owned["worker"].pid for process in owned.values())
        job.kill()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and any(_alive(process) for process in owned.values()):
            time.sleep(0.05)
        assert job_identity.status() == psutil.STATUS_ZOMBIE, "owner was reaped before checking settlement"
        assert not any(_alive(process) for process in owned.values()), {
            name: {"pid": process.pid, "ppid": process.ppid(), "status": process.status()}
            for name, process in owned.items()
            if _alive(process)
        }
    finally:
        # Kill only still-live processes identified at startup, not an arbitrary
        # reused PID or process group. Preserve the owner until this cleanup.
        # Include children of a still-live owner if setup failed before its
        # record was published, so a failed import cannot leak the test worker.
        try:
            if _alive(job_identity):
                for child in job_identity.children(recursive=True):
                    owned.setdefault(str(child.pid), child)
        except psutil.NoSuchProcess:
            pass
        for process in reversed(list(owned.values())):
            if _alive(process):
                try:
                    process.send_signal(signal.SIGKILL)
                except psutil.NoSuchProcess:
                    pass
        if _alive(job_identity):
            job.kill()
        job.wait(timeout=5)
