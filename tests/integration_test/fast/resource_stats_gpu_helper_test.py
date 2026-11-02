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


"""Real helper startup and ownership, replacing only the native GPU query."""

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from nvflare.private.fed.resource_stats.probes import gpu_nvidia, gpu_probe_process
from tests.timing_utils import ThreadGroup, wait_for

pytestmark = [pytest.mark.timeout(30), pytest.mark.skipif(os.name != "posix", reason="POSIX GPU helper ownership")]


def test_helper_preserves_python_startup_without_importing_nvflare(tmp_path):
    marker = tmp_path / "startup.txt"
    (tmp_path / "sitecustomize.py").write_text(
        "import ctypes\n"
        "import importlib.abc\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "class RejectNVFlare(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'nvflare' or fullname.startswith('nvflare.'):\n"
        "            raise RuntimeError('GPU helper must not bootstrap NVFlare')\n"
        "\n"
        "def unavailable_library(*args, **kwargs):\n"
        "    raise OSError('No GPU provider in this startup contract test')\n"
        "\n"
        "sys.meta_path.insert(0, RejectNVFlare())\n"
        "ctypes.CDLL = unavailable_library\n"
        f"Path({str(marker)!r}).write_text(os.environ['PROBE_STARTUP_CANARY'])\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path)
    env["PROBE_STARTUP_CANARY"] = "ordinary startup"
    completed = subprocess.run(
        [sys.executable, gpu_nvidia.__file__],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert marker.read_text() == "ordinary startup"
    assert json.loads(completed.stdout) is None


@pytest.fixture
def gated_probe(tmp_path, monkeypatch):
    ready, release = tmp_path / "entered", tmp_path / "release"
    wrapper = tmp_path / "gated_gpu.py"
    wrapper.write_text(
        "import os\nimport runpy\nimport sys\nimport time\nfrom pathlib import Path\n"
        f"native = runpy.run_path({gpu_nvidia.__file__!r})\n"
        "def query():\n"
        f"    checkpoint = Path({str(ready)!r})\n"
        "    checkpoint.with_suffix('.mask').write_text(os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>'))\n"
        "    checkpoint.with_suffix('.tmp').write_text(str(os.getpid()))\n"
        "    checkpoint.with_suffix('.tmp').replace(checkpoint)\n"
        f"    while not Path({str(release)!r}).exists(): time.sleep(0.01)\n"
        "    if os.environ.get('GPU_TEST_CRASH') == '1': os._exit(7)\n"
        "    start = time.process_time()\n"
        "    while time.process_time() - start < 0.2: pass\n"
        "    return {'count': int(os.environ.get('GPU_TEST_COUNT', '2'))}\n"
        "native['main'].__globals__['probe_gpu'] = query\n"
        "native['main'](parent_pid=int(sys.argv[2]), deadline=float(sys.argv[3]))\n"
    )
    monkeypatch.delenv("GPU_TEST_CRASH", raising=False)
    monkeypatch.delenv("GPU_TEST_COUNT", raising=False)
    monkeypatch.setattr(gpu_nvidia, "__file__", str(wrapper))
    return ready, release, wrapper


def test_discovery_overlaps_execution_and_holds_positive_initialization(gated_probe):
    ready, release, _ = gated_probe
    session = gpu_probe_process.GpuProbe()
    try:
        wait_for(ready.exists)
        assert not release.exists()
        release.touch()
        assert session.result() == {"count": 2}
        assert session._process.poll() is None
        assert session.result() == {"count": 2}
    finally:
        session.close()
    assert session._process.returncode == 0


@pytest.mark.parametrize("prefix", ["GPU", "MIG"])
def test_helper_keeps_its_launch_mask_when_parent_changes_to_a_same_count_mask(gated_probe, monkeypatch, prefix):
    ready, release, _ = gated_probe
    first = ",".join(f"{prefix}-00000000-0000-0000-0000-{index:012d}" for index in (1, 2))
    later = ",".join(f"{prefix}-00000000-0000-0000-0000-{index:012d}" for index in (3, 4))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", first)
    session = gpu_probe_process.GpuProbe()
    try:
        wait_for(ready.exists)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", later)
        assert ready.with_suffix(".mask").read_text() == first
        release.touch()
        assert session.result() == {"count": 2}
        assert session.result() == {"count": 2}
        assert os.environ["CUDA_VISIBLE_DEVICES"] == later
    finally:
        session.close()


def test_zero_count_exits_without_holding_initialization(gated_probe, monkeypatch):
    ready, release, _ = gated_probe
    monkeypatch.setenv("GPU_TEST_COUNT", "0")
    session = gpu_probe_process.GpuProbe()
    try:
        wait_for(ready.exists)
        release.touch()
        assert session.result() == {"count": 0}
        assert session._process.wait(timeout=3) == 0
    finally:
        session.close()


def test_owner_can_cancel_in_progress_discovery(gated_probe):
    ready, _, _ = gated_probe
    session = gpu_probe_process.GpuProbe()
    try:
        wait_for(ready.exists)
        started = time.monotonic()
        session.close()
        assert time.monotonic() - started < 3
        assert session._process.returncode in (0, -signal.SIGKILL)
        assert session.result() is None
    finally:
        session.close()


def test_native_discovery_has_a_failure_deadline(gated_probe, monkeypatch):
    ready, _, _ = gated_probe
    monkeypatch.setattr(gpu_probe_process, "_TIMEOUT_SECONDS", 3.0)
    session = gpu_probe_process.GpuProbe()
    try:
        wait_for(ready.exists)
        started = time.monotonic()
        assert session.result(timeout=3.0) is None
        assert session._process.wait(timeout=1) == -signal.SIGALRM
        session.close()
        assert time.monotonic() - started < 5
    finally:
        session.close()


def test_creator_thread_exit_does_not_kill_a_live_owners_helper(gated_probe):
    ready, release, _ = gated_probe
    sessions = []

    def launch():
        sessions.append(gpu_probe_process.GpuProbe())
        wait_for(ready.exists)

    try:
        with ThreadGroup() as threads:
            threads.start(launch)
        assert not release.exists()  # Query remained blocked until its creator thread exited.
        release.touch()
        assert sessions[0].result() == {"count": 2}
        assert sessions[0]._process.poll() is None
    finally:
        for session in sessions:
            session.close()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper proves orphan reaping")
@pytest.mark.parametrize("phase", ["discovery", "hold"])
def test_helper_dies_with_owner_even_when_a_descendant_holds_stdin(gated_probe, tmp_path, phase):
    ready, release, wrapper = gated_probe
    holder_file, phase_file = tmp_path / "holder", tmp_path / "phase"
    owner_script = tmp_path / "owner.py"
    owner_script.write_text(
        "import ctypes\nimport json\nimport os\nimport signal\nimport subprocess\nimport sys\nimport time\nfrom pathlib import Path\n"
        "def wait_for(predicate, timeout=10):\n"
        "    deadline = time.monotonic()+timeout\n"
        "    while not predicate():\n"
        "        assert time.monotonic() < deadline\n"
        "        time.sleep(0.01)\n"
        "libc = ctypes.CDLL(None)\n"
        "assert libc.prctl(36, 1, 0, 0, 0) == 0\n"
        "owner = os.fork()\n"
        "if owner == 0:\n"
        "    os.setsid()\n"
        f"    helper = subprocess.Popen([sys.executable, {str(wrapper)!r}, '--hold', str(os.getpid()), str(time.monotonic()+20)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)\n"
        "    holder = os.fork()\n"
        "    if holder == 0:\n"
        "        time.sleep(20)\n"
        "        os._exit(0)\n"
        f"    Path({str(holder_file)!r}).write_text(str(holder))\n"
        + (
            f"    Path({str(release)!r}).touch()\n" "    assert json.loads(helper.stdout.readline()) == {'count': 2}\n"
            if phase == "hold"
            else ""
        )
        + f"    Path({str(phase_file)!r}).touch()\n"
        "    time.sleep(20)\n"
        "    os._exit(1)\n"
        "helper_pid = None\n"
        "try:\n"
        f"    wait_for(lambda: Path({str(ready)!r}).exists() and Path({str(phase_file)!r}).exists())\n"
        f"    helper_pid = int(Path({str(ready)!r}).read_text())\n"
        f"    holder = int(Path({str(holder_file)!r}).read_text())\n"
        "    os.kill(holder, 0)\n"
        "    os.kill(owner, signal.SIGKILL)\n"
        "    os.waitid(os.P_PID, owner, os.WEXITED | os.WNOWAIT)\n"
        "    deadline = time.monotonic()+3\n"
        "    while True:\n"
        "        pid, status = os.waitpid(helper_pid, os.WNOHANG)\n"
        "        if pid:\n"
        "            helper_pid = None\n"
        "            assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0\n"
        "            break\n"
        "        assert time.monotonic() < deadline\n"
        "        time.sleep(0.01)\n"
        "    os.kill(holder, 0)  # EOF cannot explain helper exit: this writer is still alive.\n"
        "finally:\n"
        "    try: os.killpg(owner, signal.SIGKILL)\n"
        "    except ProcessLookupError: pass\n"
        "    try: os.kill(owner, signal.SIGKILL)\n"
        "    except ProcessLookupError: pass\n"
        "    os.waitpid(owner, 0)\n"
        "    deadline = time.monotonic()+3\n"
        "    while True:\n"
        "        try: pid, _ = os.waitpid(-1, os.WNOHANG)\n"
        "        except ChildProcessError: break\n"
        "        assert time.monotonic() < deadline\n"
        "        if not pid: time.sleep(0.01)\n"
    )
    subprocess.run([sys.executable, str(owner_script)], check=True, timeout=20)
