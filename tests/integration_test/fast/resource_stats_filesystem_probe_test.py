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

"""Verify interpreter shutdown while an optional filesystem observation is blocked."""

import subprocess
import sys


def test_daemon_observer_does_not_require_a_join_at_process_exit():
    script = """
import os
import threading
from nvflare.private.fed.resource_stats import collector

entered = threading.Event()
reader, writer = os.pipe()

def blocked_scan(_):
    entered.set()
    os.read(reader, 1)  # The writer stays open, with no data and no EOF.
    raise AssertionError("blocked read unexpectedly returned")

collector.observe_workspace_filesystem = lambda _: {"capacity_bytes": 1024}
collector.observe_retained_content = blocked_scan
result = collector.observe_filesystem("unused", timeout=2.0)
assert entered.is_set(), "observer did not reach the blocked call"
assert result == {
    "workspace_filesystem": {"capacity_bytes": 1024},
    "retained_content": {"status": "unavailable", "issues": ["observation_incomplete"]},
}
print("observation returned", flush=True)
# Normal interpreter shutdown must not join the blocked daemon thread.
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "observation returned" in result.stdout
