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

"""Real observer processes, with blocked filesystem calls gated explicitly."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nvflare.private.fed.resource_stats import filesystem_probe

pytestmark = [pytest.mark.timeout(30), pytest.mark.skipif(os.name != "posix", reason="POSIX helper pipe ownership")]

UNAVAILABLE = {"status": "unavailable", "issues": ["observation_incomplete"]}


def await_file(path):
    deadline = time.monotonic() + 10
    while not path.exists():
        if time.monotonic() >= deadline:
            pytest.fail(f"observer did not reach checkpoint: {path.name}")
        time.sleep(0.01)


@pytest.fixture
def blocked_observer(tmp_path, monkeypatch):
    ready = tmp_path / "entered"
    release = tmp_path / "release"
    wrapper = tmp_path / "gated_filesystem.py"
    wrapper.write_text(
        "import os\n"
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n"
        "from nvflare.private.fed.resource_stats import filesystem_probe_worker as worker\n"
        "def blocked(path):\n"
        f"    Path({str(ready)!r}).write_text(str(os.getpid()))\n"
        f"    while not Path({str(release)!r}).exists(): time.sleep(0.01)\n"
        "    return {'bytes': 999}\n"
        "if os.environ.get('BLOCK_CAPACITY') == '1':\n"
        "    worker.observe_workspace_filesystem = blocked\n"
        "else:\n"
        "    worker.observe_retained_content = blocked\n"
        "worker.main(sys.argv[1], int(sys.argv[2]))\n"
    )
    monkeypatch.setattr(filesystem_probe, "_HELPER_MODULE", "gated_filesystem")
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join((str(tmp_path), str(Path.cwd()), os.environ.get("PYTHONPATH", "")))
    )
    return ready, release


def test_completed_observations_are_read_and_helper_is_reaped(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "input").write_bytes(b"123")
    probe = filesystem_probe.FilesystemProbe(run, timeout=5)
    try:
        result = probe.result()
        assert result["retained_content"] == {"bytes": 3}
        assert int(result["workspace_filesystem"]["capacity_bytes"]) > 0
        # Let normal interpreter exit flush subprocess coverage before cleanup.
        assert probe._process.wait(timeout=10) == 0
    finally:
        probe.close()
    assert probe._process.returncode is not None


@pytest.mark.parametrize("capacity_blocked", [False, True])
def test_blocked_syscall_does_not_hold_finalization(blocked_observer, tmp_path, monkeypatch, capacity_blocked):
    ready, release = blocked_observer
    if capacity_blocked:
        monkeypatch.setenv("BLOCK_CAPACITY", "1")
    run = tmp_path / "run"
    run.mkdir()
    probe = filesystem_probe.FilesystemProbe(run, timeout=10)
    try:
        await_file(ready)
        start = time.monotonic()
        result = probe.result(timeout=0.15)
        probe.close()
        assert time.monotonic() - start < 1.5
        assert not release.exists()  # The syscall stayed blocked through abandonment.
        assert result["retained_content"] == UNAVAILABLE
        if capacity_blocked:
            assert result["workspace_filesystem"] == UNAVAILABLE
        else:
            assert int(result["workspace_filesystem"]["capacity_bytes"]) > 0
        assert probe._process.returncode == -signal.SIGKILL
        assert probe.result() == result  # A late observation cannot replace the snapshot.
    finally:
        release.touch()
        probe.close()


@pytest.mark.parametrize("outcome", [0, 7])
def test_job_interpreter_exits_after_observer_timeout(blocked_observer, tmp_path, outcome):
    ready, release = blocked_observer
    owner = tmp_path / "owner.py"
    report = tmp_path / "report.json"
    owner.write_text(
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        "from nvflare.private.fed.resource_stats import collector, filesystem_probe\n"
        "filesystem_probe._HELPER_MODULE = 'gated_filesystem'\n"
        "ticks = iter((0, 1000000000))\n"
        "capacity = {'cpu': {'units': 2}, 'memory': {'bytes': 1024}, 'gpu': {'count': 0}}\n"
        "job = collector.JobResourceCollector(sys.argv[1], clock_ns=lambda: next(ticks), capacity_probe=lambda: capacity)\n"
        f"Path({str(report)!r}).write_text(json.dumps(job.finish()))\n"
        f"sys.exit({outcome})\n"
    )
    process = subprocess.Popen(
        [sys.executable, str(owner), str(tmp_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        await_file(ready)
        assert not release.exists()
        _, stderr = process.communicate(timeout=5)
        assert process.returncode == outcome, stderr.decode()
        result = json.loads(report.read_text())
        assert result["duration_seconds"] == 1  # The final probe wait is outside the window.
        assert result["resource_capacity"]["cpu"] == {"units": 2}
        assert "cpu_seconds" in result["resource_utilization"]["cpu_consumed"]
        assert result["resource_utilization"]["retained_content"] == {"status": "unavailable"}
        assert {"metric": "retained_content", "code": "observation_incomplete"} in result["issues"]
    finally:
        release.touch()
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


def test_observer_exits_if_owning_pipe_closes(blocked_observer, tmp_path):
    ready, release = blocked_observer
    probe = filesystem_probe.FilesystemProbe(tmp_path, timeout=10)
    try:
        await_file(ready)
        # Closing the only parent writer also models an abruptly killed owner.
        probe._process.stdin.close()
        assert probe._process.wait(timeout=2) == 0
        assert not release.exists()
    finally:
        release.touch()
        probe.close()


@pytest.mark.parametrize("blocked_phase", ["startup", "reap"])
def test_job_interpreter_does_not_join_a_blocked_startup_or_reaper(blocked_observer, tmp_path, blocked_phase):
    ready, release = blocked_observer
    owner = tmp_path / "blocked_owner.py"
    checkpoint = tmp_path / "owner_finished"
    owner.write_text(
        "import subprocess\n"
        "import sys\n"
        "import threading\n"
        "from pathlib import Path\n"
        "from nvflare.private.fed.resource_stats import filesystem_probe\n"
        "filesystem_probe._HELPER_MODULE = 'gated_filesystem'\n"
        "def blocked_start(*args, **kwargs):\n"
        "    threading.Event().wait()\n"
        "def blocked_reap(*, timeout=None):\n"
        "    if timeout is not None: raise subprocess.TimeoutExpired('observer', timeout)\n"
        "    threading.Event().wait()\n"
        "if sys.argv[1] == 'startup':\n"
        "    filesystem_probe.subprocess.Popen = blocked_start\n"
        "probe = filesystem_probe.FilesystemProbe(sys.argv[2], timeout=0.2)\n"
        "probe.result()\n"
        "if sys.argv[1] == 'reap':\n"
        "    assert probe._process is not None\n"
        "    probe._process.wait = blocked_reap\n"
        "probe.close()\n"
        f"Path({str(checkpoint)!r}).touch()\n"
    )
    process = subprocess.Popen([sys.executable, str(owner), blocked_phase, str(tmp_path)], stderr=subprocess.PIPE)
    try:
        _, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, stderr.decode()
        assert checkpoint.exists()
        assert not release.exists()
    finally:
        release.touch()
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


def test_blocked_observer_dies_with_owner_when_a_descendant_inherits_stdin(blocked_observer, tmp_path):
    ready, release = blocked_observer
    holder_file = tmp_path / "holder"
    driver = tmp_path / "owner_death.py"
    driver.write_text(
        "import ctypes\n"
        "import os\n"
        "import signal\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n"
        "def wait_for(predicate):\n"
        "    deadline = time.monotonic() + 10\n"
        "    while not predicate():\n"
        "        assert time.monotonic() < deadline\n"
        "        time.sleep(0.01)\n"
        "linux = sys.platform == 'linux'\n"
        "if linux:\n"
        "    assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0\n"
        "owner = os.fork()\n"
        "if owner == 0:\n"
        "    os.setsid()\n"
        f"    helper = subprocess.Popen([sys.executable, '-m', 'gated_filesystem', {str(tmp_path)!r}, str(os.getpid())], stdin=subprocess.PIPE, stdout=subprocess.PIPE)\n"
        "    holder = os.fork()\n"
        "    if holder == 0:\n"
        "        time.sleep(20)\n"
        "        os._exit(0)\n"
        f"    Path({str(holder_file)!r}).write_text(str(holder))\n"
        "    time.sleep(20)\n"
        "    os._exit(1)\n"
        "helper_pid = None\n"
        "owner_reaped = False\n"
        "try:\n"
        f"    wait_for(lambda: Path({str(ready)!r}).exists() and Path({str(holder_file)!r}).exists())\n"
        f"    helper_pid = int(Path({str(ready)!r}).read_text())\n"
        f"    holder = int(Path({str(holder_file)!r}).read_text())\n"
        "    os.kill(holder, 0)\n"
        "    os.kill(owner, signal.SIGKILL)\n"
        "    os.waitpid(owner, 0)\n"
        "    owner_reaped = True\n"
        "    deadline = time.monotonic() + 3\n"
        "    while True:\n"
        "        if linux:\n"
        "            pid, status = os.waitpid(helper_pid, os.WNOHANG)\n"
        "            if pid:\n"
        "                assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0\n"
        "                break\n"
        "        else:\n"
        "            try: os.kill(helper_pid, 0)\n"
        "            except ProcessLookupError: break\n"
        "        assert time.monotonic() < deadline\n"
        "        time.sleep(0.01)\n"
        "    os.kill(holder, 0)\n"  # This live writer proves EOF did not cause helper exit.
        f"    assert not Path({str(release)!r}).exists()\n"
        "finally:\n"
        "    try: os.killpg(owner, signal.SIGKILL)\n"
        "    except ProcessLookupError: pass\n"
        "    try: os.kill(owner, signal.SIGKILL)\n"
        "    except ProcessLookupError: pass\n"
        "    if not owner_reaped: os.waitpid(owner, 0)\n"
        "    deadline = time.monotonic() + 3\n"
        "    while True:\n"
        "        try: pid, _ = os.waitpid(-1, os.WNOHANG)\n"
        "        except ChildProcessError: break\n"
        "        assert time.monotonic() < deadline\n"
        "        if not pid: time.sleep(0.01)\n"
    )
    subprocess.run([sys.executable, str(driver)], check=True, timeout=20)
