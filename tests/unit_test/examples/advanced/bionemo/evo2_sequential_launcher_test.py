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

import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

_PROCESS_TIMEOUT = 30.0


def _example_dir() -> Path:
    return Path(__file__).parents[5] / "examples" / "advanced" / "bionemo" / "evo2"


def _wait_for_file(path: Path, process: subprocess.Popen, timeout: float = _PROCESS_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"Sequential launcher exited before the inner trainer was ready:\n{process.stdout.read()}")
        time.sleep(0.01)
    assert path.exists(), f"Inner trainer did not start within {timeout:.1f} seconds."


def test_workspace_lock_descriptor_survives_exec_until_the_inner_trainer_exits(tmp_path):
    lock_file = tmp_path / "shared workspace" / ".evo2_training.lock"
    ready_file = tmp_path / "inner-ready"
    release_file = tmp_path / "release-inner"
    inner_script = tmp_path / "hold_lock.py"
    inner_script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "import time\n"
        "Path(sys.argv[1]).touch()\n"
        "while not Path(sys.argv[2]).exists():\n"
        "    time.sleep(0.01)\n",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(_example_dir() / "sequential_launcher.py"),
            "--lock-file",
            str(lock_file),
            "--",
            sys.executable,
            str(inner_script),
            str(ready_file),
            str(release_file),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_file(ready_file, process)

        competing_fd = os.open(lock_file, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competing_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            release_file.touch()
            assert process.wait(timeout=_PROCESS_TIMEOUT) == 0
            fcntl.flock(competing_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(competing_fd)
    finally:
        release_file.touch(exist_ok=True)
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=_PROCESS_TIMEOUT)


def test_workspace_lock_releases_when_the_execed_inner_trainer_is_terminated(tmp_path):
    lock_file = tmp_path / "workspace" / ".evo2_training.lock"
    pid_file = tmp_path / "inner-pid"
    inner_script = tmp_path / "wait_forever.py"
    inner_script.write_text(
        "from pathlib import Path\n"
        "import os\n"
        "import sys\n"
        "import time\n"
        "Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(_example_dir() / "sequential_launcher.py"),
            "--lock-file",
            str(lock_file),
            "--",
            sys.executable,
            str(inner_script),
            str(pid_file),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_file(pid_file, process)
        assert int(pid_file.read_text(encoding="utf-8")) == process.pid

        competing_fd = os.open(lock_file, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competing_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            process.terminate()
            assert process.wait(timeout=_PROCESS_TIMEOUT) != 0
            fcntl.flock(competing_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(competing_fd)
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=_PROCESS_TIMEOUT)
