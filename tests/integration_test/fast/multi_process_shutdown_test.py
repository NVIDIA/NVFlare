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

"""Exercise production shutdown with real launchers in isolated accounting processes.

CLOSE delivery uses a file sentinel; these tests cover launcher reaping and OS
CPU-counter propagation, not the complete FLARE sub-worker protocol.
"""

import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "killpg") or importlib.util.find_spec("resource") is None,
    reason="requires Unix process groups and resource accounting",
)

_RANK_SCRIPT = textwrap.dedent(
    """
    import json
    import os
    import resource
    import sys
    import time
    from pathlib import Path

    directory = Path(sys.argv[1])
    rank = os.environ.get("LOCAL_RANK", sys.argv[2] if len(sys.argv) > 2 else "0")
    before = time.process_time()
    while time.process_time() - before < 0.2:
        sum(i * i for i in range(1000))
    used = time.process_time() - before
    (directory / f"ready-{rank}").write_text(str(used))
    deadline = time.monotonic() + 40.0
    while not (directory / "close").exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("CLOSE was not delivered")
        time.sleep(0.01)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    (directory / f"finished-{rank}").write_text(json.dumps({"cpu": usage.ru_utime + usage.ru_stime}))
    """
)

_LAUNCHER_SCRIPT = textwrap.dedent(
    """
    import json
    import resource
    import subprocess
    import sys
    from pathlib import Path

    directory = Path(sys.argv[1])
    children = [
        subprocess.Popen([sys.executable, str(directory / "rank.py"), str(directory), str(rank)])
        for rank in range(3)
    ]
    try:
        for child in children:
            if child.wait(timeout=45.0):
                raise RuntimeError("rank failed")
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5.0)
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    (directory / "launcher-usage.json").write_text(json.dumps({
        "self_cpu": own.ru_utime + own.ru_stime,
        "children_cpu": children.ru_utime + children.ru_stime,
    }))
    (directory / "launcher-finished").touch()
    """
)


_TORCHRUN_WRAPPER_SCRIPT = textwrap.dedent(
    """
    import atexit
    import json
    import resource
    import runpy
    import sys
    from pathlib import Path

    directory = Path(sys.argv.pop(1))

    def record_usage():
        own = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        (directory / "launcher-usage.json").write_text(json.dumps({
            "self_cpu": own.ru_utime + own.ru_stime,
            "children_cpu": children.ru_utime + children.ru_stime,
        }))

    atexit.register(record_usage)
    runpy.run_module("torch.distributed.run", run_name="__main__")
    """
)


def _run_isolated(mode, directory):
    """Keep cumulative RUSAGE_CHILDREN independent of earlier pytest children."""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    repo = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo), env.get("PYTHONPATH")]))
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), mode, str(directory)],
        cwd=repo,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        output, errors = process.communicate(timeout=80.0)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            output, errors = process.communicate(timeout=8.0)
        except subprocess.TimeoutExpired:
            process.kill()
            output, errors = process.communicate(timeout=5.0)
        pytest.fail(f"isolated shutdown harness timed out: {output}\n{errors}")
    assert process.returncode == 0, f"{output}\n{errors}"
    return json.loads(next(line.removeprefix("RESULT ") for line in output.splitlines() if line.startswith("RESULT ")))


def _assert_clean_cpu_shutdown(result, ranks):
    assert result["returncode"] == 0
    assert result["reaped"]
    assert result["finished_ranks"] == ranks
    assert result["rank_cpu"] >= ranks * 0.19
    # The launcher must wait its ranks, and finalize must wait the launcher,
    # before those descendants' CPU becomes available to the job process.
    assert result["launcher_children_cpu"] >= result["rank_cpu"] - 0.03
    assert result["children_cpu"] >= result["launcher_self_cpu"] + result["rank_cpu"] - 0.03


def test_finalize_reaps_cpu_burning_descendants(tmp_path):
    result = _run_isolated("descendants", tmp_path)
    _assert_clean_cpu_shutdown(result, ranks=3)
    assert result["launcher_finished"]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires optional PyTorch launcher")
def test_finalize_reaps_real_torchrun_cpu_ranks(tmp_path):
    result = _run_isolated("torchrun", tmp_path)
    _assert_clean_cpu_shutdown(result, ranks=2)


def test_finalize_kills_and_reaps_unresponsive_launcher(tmp_path):
    result = _run_isolated("hung", tmp_path)
    assert result["returncode"] == -signal.SIGKILL
    assert result["reaped"]
    assert result["elapsed"] < 3.0


def _harness(mode, directory):
    import resource

    from nvflare.apis.fl_context import FLContext
    from nvflare.app_common.executors import multi_process_executor as mpe
    from nvflare.fuel.common.multi_process_executor_constants import MultiProcessCommandNames

    class SentinelCell:
        def fire_and_forget(self, **kwargs):
            assert kwargs["topic"] == MultiProcessCommandNames.CLOSE
            (directory / "close").touch()

    class TestExecutor(mpe.MultiProcessExecutor):
        def get_multi_process_command(self):
            return ""

        def log_info(self, fl_ctx, message, fire_event=True):
            self.logger.info(message)

    def interrupted(_signum, _frame):
        raise SystemExit("shutdown harness interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    if mode == "hung":
        command = [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys,time; Path(sys.argv[1]).touch(); time.sleep(60)",
            str(directory / "ready-0"),
        ]
        ranks = 1
        mpe._WORKER_SHUTDOWN_TIMEOUT = 0.1
        mpe._WORKER_KILL_TIMEOUT = 2.0
    else:
        (directory / "rank.py").write_text(_RANK_SCRIPT)
        if mode == "torchrun":
            ranks = 2
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                master_port = listener.getsockname()[1]
            (directory / "torchrun_wrapper.py").write_text(_TORCHRUN_WRAPPER_SCRIPT)
            command = [
                sys.executable,
                str(directory / "torchrun_wrapper.py"),
                str(directory),
                "--rdzv-backend=static",
                "--master-addr=127.0.0.1",
                f"--master-port={master_port}",
                "--nnodes=1",
                "--nproc-per-node=2",
                "--max-restarts=0",
                str(directory / "rank.py"),
                str(directory),
            ]
        else:
            ranks = 3
            (directory / "launcher.py").write_text(_LAUNCHER_SCRIPT)
            command = [sys.executable, str(directory / "launcher.py"), str(directory)]
    with (directory / "worker.log").open("w") as log:
        worker = subprocess.Popen(command, start_new_session=True, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 45.0
            while not all((directory / f"ready-{rank}").exists() for rank in range(ranks)):
                if worker.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError(f"worker readiness failed: {(directory / 'worker.log').read_text()}")
                time.sleep(0.01)
            executor = TestExecutor()
            executor.targets = [str(rank) for rank in range(ranks)]
            executor.engine = SimpleNamespace(client=SimpleNamespace(cell=SentinelCell()))
            executor.exe_process = worker
            start = time.monotonic()
            executor.finalize(FLContext())
            elapsed = time.monotonic() - start
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            try:
                os.waitpid(worker.pid, os.WNOHANG)
            except ChildProcessError:
                reaped = True
            else:
                reaped = False
            receipts = [json.loads(path.read_text()) for path in directory.glob("finished-*")]
            launcher_usage = directory / "launcher-usage.json"
            launcher_receipt = json.loads(launcher_usage.read_text()) if launcher_usage.exists() else {}
            result = {
                "returncode": worker.returncode,
                "reaped": reaped,
                "elapsed": elapsed,
                "children_cpu": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
                "rank_cpu": sum(receipt["cpu"] for receipt in receipts),
                "launcher_self_cpu": launcher_receipt.get("self_cpu"),
                "launcher_children_cpu": launcher_receipt.get("children_cpu"),
                "finished_ranks": len(receipts),
                "launcher_finished": (directory / "launcher-finished").exists(),
            }
            print("RESULT " + json.dumps(result), flush=True)
        finally:
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            worker.wait(timeout=5.0)


if __name__ == "__main__":
    _harness(sys.argv[1], Path(sys.argv[2]))
