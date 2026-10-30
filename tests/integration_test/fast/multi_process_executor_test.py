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

"""Real two-rank initialization, task, failure and shutdown contracts, without GPUs."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from nvflare.apis.fl_constant import ReturnCode


@pytest.mark.parametrize(
    "mode,simulator,backend",
    [
        ("success", "false", "local"),
        ("success", "true", "local"),
        ("failure", "false", "local"),
        ("exit", "false", "local"),
        ("block", "false", "local"),
        ("startup_failure", "false", "local"),
        *[
            pytest.param(
                mode,
                "false",
                "torchrun",
                marks=pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires CPU PyTorch"),
            )
            for mode in ("success", "failure", "block")
        ],
    ],
)
def test_two_rank_lifecycle(tmp_path, mode, simulator, backend):
    output = tmp_path / "driver.log"
    with output.open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tests.multiprocess_test_utils",
                "contract",
                str(tmp_path),
                mode,
                simulator,
                backend,
            ],
            cwd=Path(__file__).resolve().parents[3],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            if mode == "block":
                deadline = time.monotonic() + 20
                while not all((tmp_path / f"executing-{rank}").exists() for rank in range(2)):
                    assert process.poll() is None, output.read_text()
                    assert time.monotonic() < deadline, output.read_text()
                    time.sleep(0.02)
                (tmp_path / "abort").touch()
                # Measured from the observed in-flight task's actual cancellation.
                process.wait(timeout=8)
            else:
                process.wait(timeout=35)
            assert process.returncode == 0, output.read_text()
            deadline = time.monotonic() + 5
            while True:
                live = []
                for pid_file in tmp_path.glob("rank-pid-*"):
                    try:
                        identity = json.loads(pid_file.read_text())
                        rank_process = psutil.Process(identity["pid"])
                        if rank_process.create_time() != identity["create_time"]:
                            continue
                        if rank_process.status() != psutil.STATUS_ZOMBIE:
                            live.append(rank_process.pid)
                    except psutil.NoSuchProcess:
                        pass
                if not live:
                    break
                assert time.monotonic() < deadline, f"ranks survived shutdown: {live}"
                time.sleep(0.02)
            if mode == "startup_failure":
                assert (tmp_path / "startup-rejected").exists()
            elif mode == "success":
                assert all((tmp_path / f"ended-{rank}").exists() for rank in range(2))
                assert json.loads((tmp_path / "result.json").read_text())["round"] == 1
            else:
                result = json.loads((tmp_path / "result.json").read_text())
                expected = ReturnCode.TASK_ABORTED if mode == "block" else ReturnCode.EXECUTION_EXCEPTION
                assert result["__headers__"]["__rc__"] == expected
        finally:
            # The rank launcher owns a separate session from the test driver.
            # Clean both groups even if startup or an assertion fails.
            owned_ranks = []
            for marker in tmp_path.glob("rank-pid-*"):
                try:
                    identity = json.loads(marker.read_text())
                    rank_process = psutil.Process(identity["pid"])
                    if rank_process.create_time() == identity["create_time"]:
                        owned_ranks.extend(rank_process.children(recursive=True))
                        owned_ranks.append(rank_process)
                except psutil.NoSuchProcess:
                    pass
            for rank_process in owned_ranks:
                try:
                    rank_process.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(owned_ranks, timeout=2)
            groups = [process.pid] if process.poll() is None else []
            try:
                for child in psutil.Process(process.pid).children(recursive=True):
                    try:
                        groups.append(os.getpgid(child.pid))
                    except ProcessLookupError:
                        pass
            except psutil.NoSuchProcess:
                pass
            pid_file = tmp_path / "launcher.pid"
            if pid_file.exists():
                identity = json.loads(pid_file.read_text())
                try:
                    launcher = psutil.Process(identity["pid"])
                    if launcher.create_time() == identity["create_time"]:
                        groups.append(os.getpgid(launcher.pid))
                except (psutil.NoSuchProcess, ProcessLookupError):
                    pass
            for group in groups:
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=5)
