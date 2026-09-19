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
#
# Authors: Anbang Liu, Junhan Zhao, and Ziyue Xu

"""Short local IPC paths and a checked launcher for NVIDIA FLARE jobs."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from fabric_common import SITES, inside, write_json


def create_runtime_tmp(root: Path, run_root: Path) -> Path:
    """Keep IPC inside our copy without nesting it under long experiment names."""
    parent = inside(root, ".tmp")
    # CPython's Manager uses <TMPDIR>/pymp-XXXXXXXX/listener-XXXXXXXX.
    # Linux sockaddr_un allows at most 107 pathname bytes plus its terminator.
    longest_socket = parent / "r_XXXXXXXX/pymp-XXXXXXXX/listener-XXXXXXXX"
    if len(os.fsencode(longest_socket)) > 107:
        raise RuntimeError(f"Project path too long for multiprocessing IPC: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    directory = inside(parent, tempfile.mkdtemp(prefix="r_", dir=parent))
    write_json(
        run_root / "runtime_paths.json",
        {
            "temporary_directory": str(directory),
            "reason": "Short TMPDIR avoids the Linux AF_UNIX 107-byte pathname limit",
            "all_paths_inside": str(root.resolve()),
        },
    )
    return directory


def launch_simulator(job_dir: Path, workspace: Path, log_path: Path) -> Path:
    """Launch the exported job, preserving the SDK's three clients/one thread.

    NVFLARE 2.7.2's SimEnv waits for the same module but discards its exit code
    and does not tee its subprocess stderr into our Python-level log stream.
    Invoke that module directly so startup failures cannot look like success.
    """
    command = [
        sys.executable,
        "-B",
        "-u",
        "-m",
        "nvflare.private.fed.app.simulator.simulator",
        str(job_dir),
        "-w",
        str(workspace),
        "-c",
        ",".join(SITES),
        "-n",
        str(len(SITES)),
        "-t",
        "1",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    recent_lines = deque(maxlen=25)
    record = {
        "command": command,
        "workspace": str(workspace),
        "log": str(log_path),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "returncode": None,
    }
    process = None
    try:
        with log_path.open("x", encoding="utf-8", buffering=1) as handle:
            header = f"[FABRIC FLARE] {shlex.join(command)}\n"
            handle.write(header)
            print(header, end="", flush=True)
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
            try:
                for line in process.stdout:
                    handle.write(line)
                    print(line, end="", flush=True)
                    recent_lines.append(line.rstrip())
                record["returncode"] = process.wait()
            finally:
                # Stop only the process group we launched if interrupted or if
                # the launcher fails while the FLARE job is still running.
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                record["returncode"] = process.returncode
                process.stdout.close()
        if record["returncode"] != 0:
            raise RuntimeError(
                f"FLARE job exited with code {record['returncode']}; "
                f"no results will be evaluated. Full startup/training log: {log_path}\n" + "\n".join(recent_lines)
            )
        if not workspace.is_dir():
            raise RuntimeError(f"FLARE job produced no workspace. Inspect {log_path}")
        return workspace
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(log_path.with_suffix(".json"), record)
