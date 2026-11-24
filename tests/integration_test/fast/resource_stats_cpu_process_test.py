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

"""CPU accounting across real interpreter, child-process, and fork boundaries."""

import json
import multiprocessing
import subprocess
import sys
import time
from contextlib import ExitStack

import pytest

from nvflare.private.fed.resource_stats import cpu_consumed as cpu_module
from nvflare.private.fed.resource_stats.cpu_consumed import CpuConsumedAccountant, resource
from tests.resource_stats_cpu_test_utils import cleanup_subprocess, exercise_forked_cpu_hooks
from tests.timing_utils import PROCESS_STARTUP_TIMEOUT, stop_processes


@pytest.mark.skipif(resource is None, reason="waited-child CPU accounting requires Unix")
def test_cpu_work_vs_sleep_and_short_lived_waited_children():
    sleeping = CpuConsumedAccountant()
    time.sleep(0.06)
    sleep_cpu = float(sleeping.finish()["cpu_seconds"])

    working = CpuConsumedAccountant()
    sum(range(18_000_000))
    work_cpu = float(working.finish()["cpu_seconds"])
    assert work_cpu > sleep_cpu + float("0.02")

    # Both workers have exited before the one and only final measurement.
    children = CpuConsumedAccountant()
    for _ in range(2):
        subprocess.run([sys.executable, "-c", "sum(range(12000000))"], check=True, timeout=PROCESS_STARTUP_TIMEOUT)
    child_cpu = float(children.finish()["cpu_seconds"])
    assert child_cpu > float("0.05")


@pytest.mark.skipif(resource is None, reason="waited-child CPU accounting requires Unix")
def test_two_concurrent_job_processes_do_not_count_shared_parent_cpu():
    program = (
        "import json,time; "
        "from nvflare.private.fed.resource_stats.cpu_consumed import CpuConsumedAccountant; "
        "a=CpuConsumedAccountant(); "
        "sum(range(12000000)); "
        "print(json.dumps(a.finish()))"
    )
    with ExitStack() as cleanup:
        processes = []
        for _ in range(2):
            process = subprocess.Popen([sys.executable, "-c", program], stdout=subprocess.PIPE, text=True)
            cleanup.callback(cleanup_subprocess, process)
            processes.append(process)
        outputs = [process.communicate(timeout=PROCESS_STARTUP_TIMEOUT)[0] for process in processes]
        assert all(process.returncode == 0 for process in processes)
        records = [json.loads(output) for output in outputs]
        assert all(("status" not in record and float(record["cpu_seconds"]) > 0 for record in records))


def test_import_without_unix_resource_keeps_cpu_measurement_unavailable():
    # A missing optional counter must not break ordinary executor imports.
    # Fail only this helper's import; other dependencies can still use resource.
    script = """
import builtins
original_import = builtins.__import__
def without_resource(name, globals=None, locals=None, fromlist=(), level=0):
    caller = (globals or {}).get("__name__")
    if name == "resource" and caller == "nvflare.private.fed.resource_stats.cpu_consumed":
        raise ModuleNotFoundError("simulated platform without resource")
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = without_resource
from nvflare.private.fed.resource_stats.cpu_consumed import CpuConsumedAccountant, resource
assert resource is None
assert CpuConsumedAccountant().finish() == {
    "status": "unavailable", "issues": ["observation_incomplete"]
}
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=PROCESS_STARTUP_TIMEOUT)


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="requires POSIX fork")
def test_fork_discards_inherited_accountant_without_changing_parent(monkeypatch):
    accountant = CpuConsumedAccountant(usage_reader=lambda: (float(0), float(0)))
    monkeypatch.setattr(cpu_module, "_active", accountant)
    context = multiprocessing.get_context("fork")
    reader, writer = context.Pipe(duplex=False)
    child = context.Process(target=exercise_forked_cpu_hooks, args=(writer,))
    try:
        accountant._lock.acquire()
        try:
            child.start()
        finally:
            accountant._lock.release()
        writer.close()
        assert reader.poll(5), "a CPU hook blocked on an inherited parent lock"
        assert reader.recv() == (
            True,
            True,
            {"status": "partial", "issues": ["attribution_incomplete"], "cpu_seconds": 0},
        )
        child.join(5)
        assert child.exitcode == 0
        assert cpu_module._active is accountant
        assert accountant.finish() == {"cpu_seconds": 0}
    finally:
        try:
            stop_processes(child, timeout=5)
        finally:
            reader.close()
            writer.close()
