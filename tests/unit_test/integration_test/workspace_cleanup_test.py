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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

from tests.integration_test import system_test
from tests.integration_test.src import provision_site_launcher, site_launcher, utils
from tests.integration_test.src.provision_site_launcher import ProvisionSiteLauncher
from tests.integration_test.src.site_launcher import SiteLauncher, SiteProperties
from tests.timing_utils import ManualClock

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
    started = []
    identities = []

    def launch(*arguments):
        process = subprocess.Popen(
            [sys.executable, str(script), *arguments], start_new_session=True, stdout=subprocess.PIPE, text=True
        )
        started.append(process)
        identities.append(psutil.Process(process.pid))
        assert select.select([process.stdout], [], [], 10)[0], "test process did not reach readiness"
        state = json.loads(process.stdout.readline())
        identities.extend(psutil.Process(pid) for pid in state.values())
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
            psutil.wait_procs([jobs[0]], timeout=5)

        site_launcher.kill_process(properties, graceful_stop=stop_script if stop_script_exits_job else None)

        assert properties.process is None
        assert properties.processes_stopped
        assert all(not child.is_running() or child.status() == psutil.STATUS_ZOMBIE for child in jobs)
        assert sibling.poll() is None, "cleanup killed a different site's process"
    finally:
        for child in set(identities):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(set(identities), timeout=5)
        for process in started:
            process.wait(timeout=5)
            process.stdout.close()


def test_departed_launcher_never_looks_up_or_signals_reused_pid(tmp_path, monkeypatch):
    process = Mock(pid=1234, returncode=0)
    process.poll.return_value = 0
    lookup = Mock(side_effect=AssertionError("looked up a departed launcher PID"))
    signal_group = Mock(side_effect=AssertionError("signalled a departed launcher group"))
    monkeypatch.setattr(psutil, "Process", lookup)
    monkeypatch.setattr(psutil, "process_iter", lambda *_: [])
    monkeypatch.setattr(utils, "stop_process_group", signal_group)
    utils.stop_site_processes(process, str(tmp_path))
    lookup.assert_not_called()
    signal_group.assert_not_called()
    process.wait.assert_called_once_with(timeout=5)


@pytest.mark.parametrize("force_kill_denied", [False, True])
def test_group_probe_permission_error_falls_back_to_verified_process_cleanup(tmp_path, monkeypatch, force_kill_denied):
    process = Mock(pid=1234)
    identity = Mock(pid=1234)
    alive = [True]
    process.poll.side_effect = lambda: None if alive[0] else 0
    identity.is_running.side_effect = lambda: alive[0]
    identity.status.return_value = psutil.STATUS_RUNNING
    identity.children.return_value = []

    def kill():
        if force_kill_denied:
            raise psutil.AccessDenied(identity.pid)
        alive[0] = False

    def reap(timeout):
        if alive[0]:
            raise subprocess.TimeoutExpired("site", timeout)
        return 0

    identity.kill.side_effect = kill
    process.wait.side_effect = reap
    monkeypatch.setattr(psutil, "Process", lambda *_: identity)
    monkeypatch.setattr(psutil, "process_iter", lambda *_: [])
    monkeypatch.setattr(psutil, "wait_procs", lambda *_, **__: ([], []))
    monkeypatch.setattr(utils, "stop_process_group", Mock(side_effect=PermissionError("group probe failed")))
    if force_kill_denied:
        with pytest.raises(RuntimeError, match="processes survived cleanup") as caught:
            utils.stop_site_processes(process, str(tmp_path))
        assert "Reap site launcher" in str(caught.value)
    else:
        utils.stop_site_processes(process, str(tmp_path))
        assert not alive[0]
    process.wait.assert_called_once_with(timeout=5)


def test_unstoppable_workspace_process_fails_at_bounded_deadline(tmp_path, monkeypatch):
    child = Mock(pid=1234, info={"cmdline": ["python", "-m", str(tmp_path)]})
    child.is_running.return_value = True
    child.status.return_value = psutil.STATUS_RUNNING
    child.children.return_value = []
    clock = ManualClock()
    monkeypatch.setattr(utils, "time", clock)
    monkeypatch.setattr(psutil, "process_iter", lambda *_: [child])
    waits = []

    def wait(procs, timeout):
        waits.append(timeout)
        clock.advance(timeout)
        return [], list(procs)

    monkeypatch.setattr(psutil, "wait_procs", wait)
    with pytest.raises(RuntimeError, match="processes survived cleanup.*1234"):
        utils.stop_site_processes(None, str(tmp_path), kill_timeout=0.3)
    assert clock.monotonic() == pytest.approx(1000.3)
    assert waits and all(0 <= timeout <= 0.1 for timeout in waits)


@pytest.fixture
def provisioned(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(provision_site_launcher, "WORKSPACE", str(workspace))
    launcher = ProvisionSiteLauncher.__new__(ProvisionSiteLauncher)
    SiteLauncher.__init__(launcher)
    launcher.project_yaml = {"name": "cleanup_test"}
    for name in ["site-1", "site-2"]:
        launcher.client_properties[name] = SiteProperties(name, str(workspace / name), Mock())
    launcher.server_properties["server"] = SiteProperties("server", str(workspace / "server"), Mock())
    storage = Mock()
    monkeypatch.setattr(provision_site_launcher, "cleanup_job_and_snapshot", storage)
    return launcher, workspace, storage


def test_failing_stop_scripts_still_stop_every_site_and_remove_workspace(provisioned, monkeypatch):
    launcher, workspace, storage = provisioned
    attempted = []

    def stop_script(site):
        if site.name in {"site-1", "server"}:
            raise RuntimeError(f"{site.name} stop script failed")

    monkeypatch.setattr(provision_site_launcher, "_stop_site", stop_script)
    monkeypatch.setattr(
        site_launcher, "stop_site_processes", lambda process, root, **_: attempted.append(Path(root).name)
    )
    with pytest.raises(RuntimeError) as caught:
        launcher.cleanup()
    assert "site-1 stop script failed" in str(caught.value)
    assert "server stop script failed" in str(caught.value)
    assert attempted == ["site-1", "site-2", "server"]
    storage.assert_called_once()
    assert not workspace.exists()


def test_failed_process_cleanup_attempts_other_sites_and_preserves_workspace(provisioned, monkeypatch):
    launcher, workspace, storage = provisioned
    attempted = []
    monkeypatch.setattr(provision_site_launcher, "_stop_site", lambda _: None)

    def stop_processes(process, root, **_):
        name = Path(root).name
        attempted.append(name)
        if name == "site-1":
            raise RuntimeError("site-1 surviving trainer")

    monkeypatch.setattr(site_launcher, "stop_site_processes", stop_processes)
    with pytest.raises(RuntimeError, match="Preserving workspace") as caught:
        launcher.cleanup()
    assert "site-1 surviving trainer" in str(caught.value)
    assert attempted == ["site-1", "site-2", "server"]
    storage.assert_not_called()
    assert workspace.exists()


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


def test_fixture_teardown_attempts_sites_and_workspace_after_driver_failure(tmp_path, monkeypatch):
    config = {
        "cleanup": True,
        "project_yaml": "data/projects/dummy.yml",
        "jobs_root_dir": ".",
        "tests": [],
    }
    monkeypatch.setattr(system_test, "get_test_config", lambda _: config)
    launcher = Mock(client_properties={"site-1": SiteProperties("site-1", str(tmp_path), None)})
    launcher.prepare_workspace.return_value = str(tmp_path)
    driver = Mock()
    driver.finalize.side_effect = RuntimeError("driver finalize failed")
    monkeypatch.setattr(system_test, "ProvisionSiteLauncher", lambda **_: launcher)
    monkeypatch.setattr(system_test, "NVFTestDriver", lambda **_: driver)
    test_directory = tmp_path / "driver"
    test_directory.mkdir()
    monkeypatch.setattr(system_test.tempfile, "mkdtemp", lambda: str(test_directory))
    fixture = system_test.setup_and_teardown_system.__wrapped__(SimpleNamespace(param="unused.yml"))
    next(fixture)
    with pytest.raises(RuntimeError, match="driver finalize failed"):
        next(fixture)
    launcher.stop_all_sites.assert_called_once()
    launcher.cleanup.assert_called_once()
    assert not test_directory.exists()
