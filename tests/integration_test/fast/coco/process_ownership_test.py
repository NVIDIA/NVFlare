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

"""A watchdog failure must report progress and leave no live child behind."""

import signal
import subprocess
import sys

import psutil
import pytest

from tests.coco_test_utils.process import run_coco_command
from tests.timing_utils import wait_for


def test_watchdog_reaps_the_shell_tree_and_reports_last_tool(tmp_path, monkeypatch):
    ready = tmp_path / "child.pid"
    calls = tmp_path / "tool-calls.jsonl"
    calls.write_text('{"tool": "docker", "args": ["pull", "fixture"]}\n')
    child = (
        "import os, signal; from pathlib import Path; "
        f"Path({str(ready)!r}).write_text(str(os.getpid())); signal.pause()"
    )
    parent = (
        f"import subprocess, sys; subprocess.Popen([sys.executable, '-c', {child!r}]); import signal; signal.pause()"
    )
    communicate = subprocess.Popen.communicate
    observed = []

    def expire_after_readiness(process, *args, **kwargs):
        if not observed:
            observed.append(process.pid)
            wait_for(
                lambda: ready.exists() and ready.read_text().isdigit(),
                message="watchdog probe child did not become ready",
            )
            raise subprocess.TimeoutExpired(process.args, kwargs["timeout"])
        return communicate(process, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "communicate", expire_after_readiness)

    def stopped(pid):
        try:
            return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return True

    try:
        with pytest.raises(AssertionError, match='"tool": "docker"'):
            run_coco_command([sys.executable, "-c", parent], capture_output=True, text=True, diagnostic_paths=(calls,))
        assert stopped(observed[0]), "watchdog did not reap its direct child"
        wait_for(lambda: stopped(int(ready.read_text())), message="watchdog left a live descendant")
    finally:
        # Keep the regression test itself safe if group cleanup is broken.
        owned = observed + ([int(ready.read_text())] if ready.exists() else [])
        for pid in owned:
            if not stopped(pid):
                try:
                    psutil.Process(pid).send_signal(signal.SIGKILL)
                except psutil.NoSuchProcess:
                    pass
