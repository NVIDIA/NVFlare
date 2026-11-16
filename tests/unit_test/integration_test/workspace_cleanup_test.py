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
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

from tests.integration_test import system_test
from tests.integration_test.src import provision_site_launcher, site_launcher, utils
from tests.integration_test.src.poc_site_launcher import POCSiteLauncher
from tests.integration_test.src.provision_site_launcher import ProvisionSiteLauncher
from tests.integration_test.src.site_launcher import SiteLauncher, SiteProperties
from tests.timing_utils import ManualClock

pytestmark = pytest.mark.timeout(30)


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


@pytest.mark.parametrize("launcher_type", ["poc", "provision"])
def test_failed_external_storage_cleanup_preserves_path_record_for_retry(tmp_path, monkeypatch, launcher_type):
    if launcher_type == "poc":
        launcher = POCSiteLauncher.__new__(POCSiteLauncher)
        SiteLauncher.__init__(launcher)
        workspace = tmp_path / "poc"
        launcher.poc_temp_dir = str(workspace)
        launcher.poc_dir = str(workspace / "example_project" / "prod_00")
        resource_root = Path(launcher.poc_dir)
    else:
        launcher = ProvisionSiteLauncher.__new__(ProvisionSiteLauncher)
        SiteLauncher.__init__(launcher)
        workspace = tmp_path / "provision"
        monkeypatch.setattr(provision_site_launcher, "WORKSPACE", str(workspace))
        launcher.project_yaml = {"name": "cleanup_test"}
        resource_root = workspace / "cleanup_test" / "prod_00"
        server = SiteProperties("server", str(resource_root / "server"), None)
        server.processes_stopped = True
        launcher.server_properties["server"] = server

    launcher.stop_all_sites = Mock()
    resource_file = resource_root / "server" / "local" / utils.RESOURCE_CONFIG
    resource_file.parent.mkdir(parents=True)
    job_store = tmp_path / "external_jobs"
    snapshot = tmp_path / "external_snapshot"
    job_store.mkdir()
    snapshot.mkdir()
    resource_file.write_text(
        json.dumps(
            {
                "components": [{"id": "job_manager", "args": {"uri_root": str(job_store)}}],
                "snapshot_persistor": {
                    "args": {"storage": {"path": utils.FILE_STORAGE, "args": {"root_dir": str(snapshot)}}}
                },
            }
        )
    )
    cleanup_path = utils.cleanup_path
    first_failure = True

    def fail_once(path):
        nonlocal first_failure
        if path == str(snapshot) and first_failure:
            first_failure = False
            raise PermissionError("snapshot removal denied")
        cleanup_path(path)

    monkeypatch.setattr(utils, "cleanup_path", fail_once)
    with pytest.raises(RuntimeError, match="snapshot removal denied"):
        launcher.cleanup()
    assert resource_file.exists()
    assert snapshot.exists()

    launcher.cleanup()
    assert not workspace.exists()
    assert not snapshot.exists()


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
