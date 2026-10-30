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

"""Real helper startup, with native GPU access replaced at the library boundary."""

import json
import os
import signal
import subprocess
import sys
import time
from decimal import Decimal

import pytest

try:
    import resource
except ImportError:
    resource = None

from nvflare.private.fed.resource_stats import collector
from nvflare.private.fed.resource_stats.cpu_consumed import CpuConsumedAccountant
from nvflare.private.fed.resource_stats.probes import gpu_nvidia, gpu_probe_process


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


pytestmark = [pytest.mark.timeout(30), pytest.mark.skipif(os.name != "posix", reason="POSIX GPU helper ownership")]


def _await_file(path):
    deadline = time.monotonic() + 10
    while not path.exists():
        if time.monotonic() >= deadline:
            pytest.fail(f"helper did not reach checkpoint: {path.name}")
        time.sleep(0.01)


@pytest.fixture
def gated_probe(tmp_path, monkeypatch):
    # Execute the real helper lifecycle, replacing only its native query.
    # A filesystem gate establishes readiness before release/cancel/death.
    ready = tmp_path / "entered"
    release = tmp_path / "release"
    wrapper = tmp_path / "gated_gpu.py"
    wrapper.write_text(
        "import os\n"
        "import runpy\n"
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n"
        f"native = runpy.run_path({gpu_nvidia.__file__!r})\n"
        "def query():\n"
        f"    Path({str(ready)!r}).write_text(str(os.getpid()))\n"
        f"    while not Path({str(release)!r}).exists(): time.sleep(0.01)\n"
        "    if os.environ.get('GPU_TEST_CRASH') == '1': os._exit(7)\n"
        "    start = time.process_time()\n"
        "    while time.process_time() - start < 0.2: pass\n"
        "    return {'groups': [{'kind': 'full_gpu', 'count': 2}]}\n"
        "native['main'].__globals__['probe_gpu'] = query\n"
        "native['main'](parent_pid=int(sys.argv[2]), deadline=float(sys.argv[3]))\n"
    )
    monkeypatch.setattr(gpu_nvidia, "__file__", str(wrapper))
    return ready, release, wrapper


def test_discovery_overlaps_execution_and_waits_for_valid_result(gated_probe):
    ready, release, _ = gated_probe
    session = gpu_probe_process.GpuProbe()
    try:
        _await_file(ready)
        assert not release.exists()  # start() returned while discovery was blocked.
        release.touch()
        assert session.result() == {"groups": [{"kind": "full_gpu", "count": 2}]}
        assert session._process.poll() is None  # Holds initialization through execution.
        assert session.result() == {"groups": [{"kind": "full_gpu", "count": 2}]}
    finally:
        session.close()
    assert session._process.returncode == 0


def test_collector_excludes_probe_cpu_and_final_wait(gated_probe, monkeypatch):
    ready, release, _ = gated_probe

    def usage():
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        return Decimal(0), Decimal(str(child.ru_utime)) + Decimal(str(child.ru_stime))

    before = usage()[1]
    monkeypatch.setattr(
        collector, "CpuConsumedAccountant", lambda **kw: CpuConsumedAccountant(usage_reader=usage, **kw)
    )
    clock = iter((0, 1_000_000_000))
    instance = collector.ExecutionResourceCollector(clock_ns=lambda: next(clock))
    try:
        _await_file(ready)
        release.touch()
        result = instance.stop()
        assert result["cpu_consumed"] == {"cpu_seconds": "0"}
        assert result["duration_seconds"] == "1"
        assert result["resource_capacity"]["gpu"]["groups"][0]["count"] == "2"
        assert usage()[1] - before >= Decimal("0.15")  # CPU appears only after stop's cleanup.
        assert instance.stop() == result
    finally:
        instance._gpu_probe.close()


def test_owner_can_cancel_in_progress_discovery(gated_probe):
    ready, _, _ = gated_probe
    session = gpu_probe_process.GpuProbe()
    try:
        _await_file(ready)
        start = time.monotonic()
        session.close()
        assert time.monotonic() - start < 3
        assert session._process.returncode == -signal.SIGKILL
        assert session.result() is None
    finally:
        session.close()


def test_native_failure_preserves_other_capacity(gated_probe, monkeypatch):
    ready, release, _ = gated_probe
    monkeypatch.setenv("GPU_TEST_CRASH", "1")
    monkeypatch.setattr(collector.cgroup_linux, "probe_cpu", lambda *a, **kw: {"groups": [{"units": "2"}]})
    monkeypatch.setattr(collector.cgroup_linux, "probe_memory", lambda *a, **kw: {"bytes": "1024"})
    instance = collector.ExecutionResourceCollector()
    try:
        _await_file(ready)
        release.touch()
        result = instance.stop()
        assert "gpu" not in result["resource_capacity"]
        assert result["resource_capacity"]["status"] == "partial"
        assert "cpu_seconds" in result["cpu_consumed"]
        assert instance._gpu_probe._process.returncode == 7
    finally:
        instance._gpu_probe.close()


def test_native_discovery_has_a_failure_deadline(gated_probe, monkeypatch):
    ready, _, _ = gated_probe
    monkeypatch.setattr(gpu_probe_process, "_TIMEOUT_SECONDS", 3.0)
    session = gpu_probe_process.GpuProbe()
    try:
        _await_file(ready)
        start = time.monotonic()
        assert session.result(timeout=3.0) is None
        session.close()
        assert time.monotonic() - start < 5
        assert session._process.returncode in (-signal.SIGALRM, -signal.SIGKILL)
    finally:
        session.close()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux parent-death signal")
def test_helper_dies_with_owner_during_native_discovery(gated_probe, tmp_path):
    ready, _, wrapper = gated_probe
    owner_script = tmp_path / "owner.py"
    # The test is a temporary subreaper so it can prove reaping of the orphan.
    # Do not modify the pytest process's subreaper state.
    owner_script.write_text(
        "import ctypes\n"
        "import os\n"
        "import signal\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n"
        "libc = ctypes.CDLL(None)\n"
        "assert libc.prctl(36, 1, 0, 0, 0) == 0\n"
        "owner = os.fork()\n"
        "if owner == 0:\n"
        f"    subprocess.Popen([sys.executable, {str(wrapper)!r}, '--hold', str(os.getpid()), str(time.monotonic()+20)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)\n"
        "    time.sleep(20)\n"
        "    os._exit(1)\n"
        "helper = None\n"
        "try:\n"
        "    deadline = time.monotonic()+10\n"
        f"    while not Path({str(ready)!r}).exists():\n"
        "        assert time.monotonic() < deadline\n"
        "        time.sleep(0.01)\n"
        f"    helper = int(Path({str(ready)!r}).read_text())\n"
        "    os.kill(owner, signal.SIGKILL)\n"
        "    os.waitpid(owner, 0)\n"
        "    owner = None\n"
        "    deadline = time.monotonic()+3\n"
        "    while True:\n"
        "        pid, status = os.waitpid(helper, os.WNOHANG)\n"
        "        if pid:\n"
        "            helper = None\n"
        "            assert os.WTERMSIG(status) == signal.SIGKILL\n"
        "            break\n"
        "        assert time.monotonic() < deadline\n"
        "        time.sleep(0.01)\n"
        "finally:\n"
        "    if owner is not None:\n"
        "        os.kill(owner, signal.SIGKILL)\n"
        "        os.waitpid(owner, 0)\n"
        "    if helper is not None:\n"
        "        os.kill(helper, signal.SIGKILL)\n"
        "        os.waitpid(helper, 0)\n"
    )
    subprocess.run([sys.executable, str(owner_script)], check=True, timeout=20)
