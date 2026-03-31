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
import pty
import re
import select
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager, suppress

import pytest

from nvflare.apis.job_def import DEFAULT_JOB_STUDY, JobMetaKey
from nvflare.apis.workspace import Workspace
from nvflare.fuel.flare_api.flare_api import new_secure_session
from nvflare.fuel.hci.client.api_spec import AdminConfigKey
from nvflare.fuel.hci.client.config import secure_load_admin_config
from tests.integration_test.src import NVFTestDriver, POCSiteLauncher
from tests.integration_test.src.utils import _get_job_store_path_from_workspace, get_job_meta

JOBS_ROOT_DIR = os.path.join(os.path.dirname(__file__), "data", "jobs")
JOB_NAME = "hello-numpy-sag"
JOB_DIR = os.path.join(JOBS_ROOT_DIR, JOB_NAME)
ADMIN_NAME = "admin@nvidia.com"
POC_CLIENTS = 2


def _wait_for_job_meta(admin_api, job_id: str, timeout: float = 30.0) -> dict:
    end_time = time.time() + timeout
    while time.time() < end_time:
        meta = get_job_meta(admin_api, job_id)
        if meta.get(JobMetaKey.JOB_ID.value) == job_id:
            return meta
        time.sleep(0.5)
    raise AssertionError(f"Timed out waiting for job meta for {job_id}")


def _find_job(jobs: list[dict], job_id: str) -> dict:
    for job in jobs:
        if job.get("job_id") == job_id:
            return job
    raise AssertionError(f"job_id {job_id} not found in job list: {jobs}")


def _remove_study_from_persisted_job_meta(workspace_root: str, job_id: str):
    job_store_path = _get_job_store_path_from_workspace(workspace_root, "server")
    meta_path = os.path.join(job_store_path, job_id, "meta")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta.pop(JobMetaKey.STUDY.value, None)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)


def _read_until(master_fd: int, patterns: list[str], timeout: float = 30.0) -> str:
    end_time = time.time() + timeout
    output = ""
    while time.time() < end_time:
        remaining = max(0.1, end_time - time.time())
        ready, _, _ = select.select([master_fd], [], [], remaining)
        if not ready:
            continue
        chunk = os.read(master_fd, 4096).decode("utf-8", errors="replace")
        output += chunk
        if any(pattern in output for pattern in patterns):
            return output
    raise AssertionError(f"Timed out waiting for patterns {patterns!r}. Output so far:\n{output}")


def _send_line(master_fd: int, text: str):
    os.write(master_fd, f"{text}\n".encode("utf-8"))


def _login_to_admin_shell(master_fd: int, admin_name: str) -> str:
    output = _read_until(master_fd, ["User Name:", "> "], timeout=30.0)
    if "User Name:" in output:
        _send_line(master_fd, admin_name)
        output += _read_until(master_fd, ["> "], timeout=30.0)
    return output


def _run_admin_shell_command(master_fd: int, command: str) -> str:
    _send_line(master_fd, command)
    return _read_until(master_fd, ["> "], timeout=60.0)


def _get_admin_upload_dir(admin_root: str) -> str:
    conf = secure_load_admin_config(Workspace(root_dir=admin_root))
    admin_config = conf.get_admin_config()
    return admin_config[AdminConfigKey.UPLOAD_DIR]


@contextmanager
def _admin_shell(admin_root: str, study: str):
    startup_dir = os.path.join(admin_root, "startup")
    script_path = os.path.join(startup_dir, "fl_admin.sh")
    master_fd, slave_fd = pty.openpty()
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([os.path.dirname(sys.executable), env.get("PATH", "")])
    process = subprocess.Popen(
        ["bash", script_path, "--study", study],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        cwd=startup_dir,
        env=env,
    )
    os.close(slave_fd)
    try:
        yield process, master_fd
    finally:
        with suppress(Exception):
            _send_line(master_fd, "bye")
        with suppress(Exception):
            process.terminate()
        with suppress(Exception):
            process.wait(timeout=10)
        with suppress(Exception):
            os.close(master_fd)


@pytest.fixture
def running_poc_system(tmp_path):
    site_launcher = POCSiteLauncher(n_servers=1, n_clients=POC_CLIENTS)
    workspace_root = site_launcher.prepare_workspace()
    site_launcher.start_servers()
    site_launcher.start_clients()

    download_root_dir = tmp_path / "download_result"
    download_root_dir.mkdir()
    test_driver = NVFTestDriver(download_root_dir=str(download_root_dir), site_launcher=site_launcher, poll_period=1)

    sessions = []
    try:
        test_driver.initialize_super_user(
            workspace_root_dir=workspace_root,
            upload_root_dir=JOBS_ROOT_DIR,
            super_user_name=ADMIN_NAME,
        )
        test_driver.ensure_clients_started(num_clients=POC_CLIENTS, timeout=300)
        yield {
            "workspace_root": workspace_root,
            "admin_root": os.path.join(workspace_root, ADMIN_NAME),
            "old_admin_api": test_driver.super_admin_api,
            "sessions": sessions,
        }
    finally:
        for session in sessions:
            with suppress(Exception):
                session.close()
        test_driver.finalize()
        site_launcher.stop_all_sites()
        site_launcher.cleanup()


@pytest.mark.xdist_group(name="system_tests_group")
class TestStudySessionIntegration:
    def test_fladminapi_submit_defaults_study_and_list_jobs_exposes_it(self, running_poc_system):
        admin_api = running_poc_system["old_admin_api"]

        response = admin_api.submit_job(JOB_NAME)
        assert response["status"] == "SUCCESS"
        job_id = response["details"]["job_id"]

        meta = _wait_for_job_meta(admin_api, job_id)
        assert meta[JobMetaKey.STUDY.value] == DEFAULT_JOB_STUDY

        jobs = admin_api.list_jobs()["details"]
        listed_job = _find_job(jobs, job_id)
        assert listed_job[JobMetaKey.STUDY.value] == DEFAULT_JOB_STUDY

    def test_legacy_job_without_study_is_normalized_to_default(self, running_poc_system):
        admin_api = running_poc_system["old_admin_api"]

        response = admin_api.submit_job(JOB_NAME)
        assert response["status"] == "SUCCESS"
        job_id = response["details"]["job_id"]

        _wait_for_job_meta(admin_api, job_id)
        _remove_study_from_persisted_job_meta(running_poc_system["workspace_root"], job_id)

        normalized_meta = _wait_for_job_meta(admin_api, job_id)
        assert normalized_meta[JobMetaKey.STUDY.value] == DEFAULT_JOB_STUDY

        jobs = admin_api.list_jobs()["details"]
        listed_job = _find_job(jobs, job_id)
        assert listed_job[JobMetaKey.STUDY.value] == DEFAULT_JOB_STUDY

    def test_session_scopes_list_jobs_and_clone_preserves_source_study(self, running_poc_system):
        admin_root = running_poc_system["admin_root"]

        session_a = new_secure_session(ADMIN_NAME, admin_root, study="study-a")
        session_b = new_secure_session(ADMIN_NAME, admin_root, study="study-b")
        running_poc_system["sessions"].extend([session_a, session_b])

        job_a = session_a.submit_job(JOB_DIR)
        job_b = session_b.submit_job(JOB_DIR)

        meta_a = session_a.get_job_meta(job_a)
        meta_b = session_b.get_job_meta(job_b)
        assert meta_a[JobMetaKey.STUDY.value] == "study-a"
        assert meta_b[JobMetaKey.STUDY.value] == "study-b"

        jobs_a = session_a.list_jobs()
        jobs_b = session_b.list_jobs()

        assert {job["job_id"] for job in jobs_a} == {job_a}
        assert {job["job_id"] for job in jobs_b} == {job_b}

        cloned_job_id = session_b.clone_job(job_a)
        cloned_meta = session_a.get_job_meta(cloned_job_id)
        assert cloned_meta[JobMetaKey.STUDY.value] == "study-a"

        jobs_a_after_clone = session_a.list_jobs()
        jobs_b_after_clone = session_b.list_jobs()
        assert {job["job_id"] for job in jobs_a_after_clone} == {job_a, cloned_job_id}
        assert {job["job_id"] for job in jobs_b_after_clone} == {job_b}

    def test_fl_admin_shell_study_flag_scopes_terminal_session(self, running_poc_system):
        admin_root = running_poc_system["admin_root"]
        upload_dir = _get_admin_upload_dir(admin_root)
        shell_job_dir = os.path.join(upload_dir, JOB_NAME)
        shutil.copytree(JOB_DIR, shell_job_dir, dirs_exist_ok=True)

        try:
            with _admin_shell(admin_root, "study-a") as (_process_a, master_fd_a):
                _login_to_admin_shell(master_fd_a, ADMIN_NAME)

                submit_output = _run_admin_shell_command(master_fd_a, f"submit_job {JOB_NAME}")
                match = re.search(r"Submitted job:\s*([0-9a-fA-F-]+)", submit_output)
                assert match, f"failed to parse job id from output:\n{submit_output}"
                job_id = match.group(1)

                detailed_output = _run_admin_shell_command(master_fd_a, "list_jobs -d")
                assert job_id in detailed_output
                assert '"study": "study-a"' in detailed_output

                meta = _wait_for_job_meta(running_poc_system["old_admin_api"], job_id)
                assert meta[JobMetaKey.STUDY.value] == "study-a"

            with _admin_shell(admin_root, "study-b") as (_process_b, master_fd_b):
                _login_to_admin_shell(master_fd_b, ADMIN_NAME)
                other_study_output = _run_admin_shell_command(master_fd_b, "list_jobs")
                assert job_id not in other_study_output
                assert "No jobs matching the specified criteria." in other_study_output
        finally:
            shutil.rmtree(shell_job_dir, ignore_errors=True)
