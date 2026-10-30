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

import json
import os
import select
import shlex
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

from tests.integration_test import system_test
from tests.integration_test.src import poc_site_launcher, site_launcher, utils
from tests.integration_test.src.poc_site_launcher import POCSiteLauncher
from tests.integration_test.src.site_launcher import SiteLauncher, SiteProperties
from tests.process_test_utils import OwnedProcesses, wait_for_processes

pytestmark = pytest.mark.timeout(30)

_PROCESS_SCRIPT = """
import json
import os
import signal
import subprocess
import sys
import time

role = sys.argv[1]
if role == "trainer":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print(json.dumps({"trainer": os.getpid()}), flush=True)
elif role == "job":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = subprocess.Popen(
        [sys.executable, __file__, "trainer"],
        start_new_session=True, stdout=subprocess.PIPE, text=True,
    )
    state = json.loads(child.stdout.readline())
    state["job"] = os.getpid()
    print(json.dumps(state), flush=True)
else:
    child = subprocess.Popen(
        [sys.executable, __file__, "job", "-m", sys.argv[2]],
        start_new_session=True, stdout=subprocess.PIPE, text=True,
    )
    print(child.stdout.readline().strip(), flush=True)
time.sleep(120)
"""


@pytest.mark.skipif(os.name != "posix", reason="site sessions use POSIX process groups")
@pytest.mark.parametrize("missing_launcher", [False, True])
@pytest.mark.parametrize("stop_script_exits_job", [False, True])
def test_cleanup_reaps_detached_workspace_job_and_trainer_without_touching_sibling(
    tmp_path, missing_launcher, stop_script_exits_job
):
    root = tmp_path / "workspace with spaces" / "site-1"
    root.mkdir(parents=True)
    script = tmp_path / "process_tree.py"
    script.write_text(_PROCESS_SCRIPT)
    owner = OwnedProcesses(grace=1.0)

    def launch(*arguments):
        process = owner.start([sys.executable, str(script), *arguments], stdout=subprocess.PIPE, text=True)
        assert select.select([process.stdout], [], [], 10)[0], "test process did not reach readiness"
        state = json.loads(process.stdout.readline())
        return process, state

    try:
        sibling, _ = launch("trainer", f"--workspace={root}0")
        if missing_launcher:
            process, state = launch("job", "-m", str(root / "job-123"))
            launcher = None
        else:
            process, state = launch("site", str(root))
            launcher = process
            assert os.getpgid(state["job"]) != process.pid
        assert os.getpgid(state["trainer"]) != os.getpgid(state["job"])
        jobs = [psutil.Process(state[name]) for name in ("job", "trainer")]

        properties = SiteProperties("site-1", str(root), launcher)

        def stop_script():
            jobs[0].kill()
            wait_for_processes([jobs[0]], timeout=5)

        site_launcher.kill_process(properties, graceful_stop=stop_script if stop_script_exits_job else None)

        assert properties.process is None
        assert properties.processes_stopped
        assert all(not utils._process_running(child) for child in jobs)
        assert sibling.poll() is None, "cleanup killed a different site's process"
    finally:
        try:
            owner.close()
        finally:
            for process in owner.processes:
                process.stdout.close()


@pytest.mark.skipif(os.name != "posix", reason="site sessions use POSIX process groups")
def test_cleanup_retry_retains_orphaned_trainer_identity(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    root = workspace / "site-1"
    root.mkdir(parents=True)
    script = tmp_path / "process_tree.py"
    script.write_text(_PROCESS_SCRIPT)
    owner = OwnedProcesses(grace=1.0)
    job = None
    trainer = None
    original_kill = psutil.Process.kill
    try:
        job = owner.start(
            [sys.executable, str(script), "job", "-m", str(root)],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert select.select([job.stdout], [], [], 10)[0], "job did not reach readiness"
        trainer = psutil.Process(json.loads(job.stdout.readline())["trainer"])
        assert all(str(root) not in argument for argument in trainer.cmdline())

        launcher = POCSiteLauncher.__new__(POCSiteLauncher)
        SiteLauncher.__init__(launcher)
        launcher.poc_dir = str(workspace)
        launcher.poc_temp_dir = str(workspace)
        properties = SiteProperties("site-1", str(root), None)
        launcher.client_properties["site-1"] = properties
        monkeypatch.setattr(poc_site_launcher, "cleanup_job_and_snapshot", lambda *_: None)
        deny_trainer_once = True

        def fail_first_trainer_kill(process):
            nonlocal deny_trainer_once
            if process.pid == trainer.pid and deny_trainer_once:
                deny_trainer_once = False
                raise psutil.AccessDenied(process.pid)
            return original_kill(process)

        monkeypatch.setattr(psutil.Process, "kill", fail_first_trainer_kill)
        with pytest.raises(RuntimeError, match="Preserving workspace") as caught:
            launcher.cleanup()
        assert job.poll() is not None, str(caught.value)
        assert utils._process_running(trainer)
        assert trainer in properties.owned_processes
        assert not properties.processes_stopped
        assert workspace.exists()

        launcher.cleanup()
        assert not workspace.exists()
        assert not utils._process_running(trainer)
        assert properties.processes_stopped
        assert not properties.owned_processes
    finally:
        try:
            owner.close()
        finally:
            if job is not None:
                job.stdout.close()


@pytest.mark.parametrize("reset_job_info", [False, True])
def test_teardown_runs_all_commands_and_reset_after_multiple_failures(tmp_path, monkeypatch, reset_job_info):
    background = Mock(side_effect=RuntimeError("background cleanup failed"))
    monkeypatch.setattr(system_test, "_stop_background_processes", background)
    marker = tmp_path / "last-command-ran"
    teardown = [
        shlex.join([sys.executable, "-c", "raise SystemExit(7)"]),
        shlex.join([sys.executable, "-c", "raise SystemExit(9)"]),
        shlex.join(
            [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).touch()", str(marker)]
        ),
    ]
    driver = SimpleNamespace(event_sequence_timeout=10, reset_test_info=Mock())
    with pytest.raises(RuntimeError) as caught:
        system_test._teardown_test_case([], teardown, driver, reset_job_info)
    assert marker.exists()
    assert "background cleanup failed" in str(caught.value)
    assert "exited with code 7" in str(caught.value)
    assert "exited with code 9" in str(caught.value)
    driver.reset_test_info.assert_called_once_with(reset_job_info=reset_job_info)
