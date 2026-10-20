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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.workspace import Workspace
from nvflare.fuel.flare_api.api_spec import MonitorReturnCode
from tests.integration_test.fast import study_session_test as study


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(now=0.0, wall=1000.0, after_sleep=lambda: None)

    def sleep(delay):
        state.now += delay
        state.wall += 1_000_000  # Simulate a wall-clock correction during the wait.
        state.after_sleep()

    monkeypatch.setattr(
        study,
        "time",
        SimpleNamespace(monotonic=lambda: state.now, time=lambda: state.wall, sleep=sleep),
    )
    return state


def _runtime_meta_path(tmp_path, site="site-1", job_id="job-id"):
    root = tmp_path / site
    (root / "startup").mkdir(parents=True, exist_ok=True)
    (root / "local").mkdir(exist_ok=True)
    workspace = Workspace(root_dir=str(root), site_name=site)
    path = Path(workspace.get_job_meta_path(job_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.parametrize("source", ["admin", "session"])
def test_metadata_wait_survives_wall_clock_correction(monkeypatch, clock, source):
    expected = {"job_id": "job-id"}
    get_meta = Mock(side_effect=[{}, expected])
    if source == "admin":
        monkeypatch.setattr(study, "get_job_meta", get_meta)
        result = study._wait_for_job_meta(object(), "job-id", timeout=2)
    else:
        result = study._wait_for_session_job_meta(SimpleNamespace(get_job_meta=get_meta), "job-id", timeout=2)

    assert result == expected
    assert get_meta.call_count == 2


@pytest.mark.parametrize("initial_content", [None, "{", '{"job_id": "previous-job"}'])
def test_runtime_metadata_waits_for_complete_current_job_file(tmp_path, clock, initial_content):
    path = _runtime_meta_path(tmp_path)
    if initial_content is not None:
        path.write_text(initial_content)
    expected = {"job_id": "job-id", "study": "study-a"}
    clock.after_sleep = lambda: path.write_text(json.dumps(expected))

    assert study._wait_for_runtime_job_meta(str(tmp_path), "site-1", "job-id", timeout=2) == expected


def test_runtime_metadata_wait_is_bounded_when_wrong_job_remains(tmp_path, clock):
    _runtime_meta_path(tmp_path).write_text('{"job_id": "previous-job"}')

    with pytest.raises(AssertionError, match="Timed out waiting for runtime job meta"):
        study._wait_for_runtime_job_meta(str(tmp_path), "site-1", "job-id", timeout=1)

    assert clock.now == 1


@pytest.mark.parametrize("deployed_to_excluded_site", [False, True])
def test_excluded_site_is_checked_after_job_finishes(tmp_path, deployed_to_excluded_site):
    path = _runtime_meta_path(tmp_path, site="site-3")

    def finish(*args, **kwargs):
        if deployed_to_excluded_site:
            path.write_text('{"job_id": "job-id"}')
        return MonitorReturnCode.JOB_FINISHED

    session = SimpleNamespace(monitor_job=Mock(side_effect=finish))
    if deployed_to_excluded_site:
        with pytest.raises(AssertionError, match="Unexpected runtime job meta"):
            study._assert_runtime_job_meta_absent(str(tmp_path), "site-3", "job-id", session=session)
    else:
        study._assert_runtime_job_meta_absent(str(tmp_path), "site-3", "job-id", session=session)

    session.monitor_job.assert_called_once_with("job-id", timeout=60.0)


@pytest.mark.parametrize("result", [MonitorReturnCode.TIMEOUT, MonitorReturnCode.ENDED_BY_CB])
def test_excluded_site_check_rejects_unfinished_job(tmp_path, result):
    _runtime_meta_path(tmp_path, site="site-3")
    session = SimpleNamespace(monitor_job=Mock(return_value=result))

    with pytest.raises(AssertionError, match="did not finish"):
        study._assert_runtime_job_meta_absent(str(tmp_path), "site-3", "job-id", session=session)


def test_admin_prompt_wait_survives_wall_clock_correction(monkeypatch, clock):
    timeouts = []
    chunks = iter([b"User ", b"Name:"])

    def ready(read, write, error, timeout):
        timeouts.append(timeout)
        clock.now += 0.5
        clock.wall += 1_000_000
        return read, [], []

    monkeypatch.setattr(study, "select", SimpleNamespace(select=ready))
    monkeypatch.setattr(study, "os", SimpleNamespace(read=lambda *args: next(chunks)))

    assert study._read_until(123, ["User Name:"], timeout=5) == "User Name:"
    assert timeouts == [5, 4.5]


@pytest.mark.parametrize(
    "failed_stage",
    [
        "prepare_workspace",
        "start_servers",
        "start_clients",
        "initialize_super_user",
        "body",
        "finalize",
        "stop_all_sites",
    ],
)
def test_provisioned_fixture_attempts_all_cleanup_after_failure(tmp_path, monkeypatch, failed_stage):
    launcher = Mock()
    launcher.prepare_workspace.return_value = str(tmp_path)
    launcher.admin_user_names = ["admin@nvidia.com"]
    launcher.client_properties = {"site-1": object()}
    launcher.server_properties = {"server": object()}
    driver = Mock()
    owner = driver if failed_stage in ("initialize_super_user", "finalize") else launcher
    if failed_stage != "body":
        getattr(owner, failed_stage).side_effect = RuntimeError(f"injected {failed_stage} failure")

    downloads = tmp_path / "downloads"
    downloads.mkdir()
    monkeypatch.setattr(study, "tempfile", SimpleNamespace(mkdtemp=lambda **kwargs: str(downloads)))
    monkeypatch.setattr(study, "ProvisionSiteLauncher", Mock(return_value=launcher))
    monkeypatch.setattr(study, "NVFTestDriver", Mock(return_value=driver))
    session = Mock()

    with pytest.raises(RuntimeError, match=f"injected {failed_stage} failure"):
        with study._running_provisioned_system("project.yml", "admin@nvidia.com") as system:
            system["sessions"].append(session)
            if failed_stage == "body":
                raise RuntimeError("injected body failure")

    launcher.stop_all_sites.assert_called_once()
    launcher.cleanup.assert_called_once()
    assert not downloads.exists()
    if failed_stage in ("initialize_super_user", "body", "finalize", "stop_all_sites"):
        driver.finalize.assert_called_once()
    if failed_stage in ("body", "finalize", "stop_all_sites"):
        session.close.assert_called_once()
