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

"""End-to-end multi-study security coverage.

Single-tenant deployment (`api_version: 4`, no `studies:`):
- default login succeeds and default jobs can be submitted through the FLARE API session
- non-default study login is rejected through Flare API
- `fl_admin.sh` without `--study` creates default-scoped jobs
- legacy jobs with no stored study normalize to `default`
- persisted non-default jobs are hidden from default sessions (`list_jobs`, `get_job_meta`, `clone_job`)

Multi-study deployment (`api_version: 4` with `studies:`):
- default login and mapped study login succeed
- unknown-study and unmapped-user logins are rejected
- default and non-default sessions see only their own jobs
- cross-study direct job access is hidden as not found
- study membership does not override the certificate role used for submit authorization
- `check_status client` is filtered to the enrolled sites of the active study
- `@ALL` scheduling is narrowed to study-enrolled sites only
- submit-time `deploy_map` validation rejects out-of-study sites
- ProdEnv and `fl_admin.sh --study ...` propagate study context end to end
"""

import json
import os
import pty
import re
import select
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager, suppress

import numpy as np
import pytest

from nvflare.apis.job_def import DEFAULT_STUDY, JobMetaKey
from nvflare.apis.workspace import Workspace
from nvflare.app_common.np.recipes import NumpyFedAvgRecipe
from nvflare.fuel.flare_api.api_spec import AuthorizationError, InvalidJobDefinition, JobNotFound
from nvflare.fuel.flare_api.flare_api import Session as FlareSession
from nvflare.fuel.flare_api.flare_api import new_secure_session
from nvflare.fuel.hci.client.api import APIStatus, ResultKey
from nvflare.fuel.hci.client.api_spec import AdminConfigKey
from nvflare.fuel.hci.client.config import secure_load_admin_config
from nvflare.fuel.hci.proto import MetaKey, MetaStatusValue
from nvflare.recipe import ProdEnv
from tests.integration_test.src import NVFTestDriver, ProvisionSiteLauncher
from tests.integration_test.src.utils import _get_job_store_path_from_workspace, get_job_meta

INTEGRATION_TEST_ROOT = os.path.dirname(os.path.dirname(__file__))

JOBS_ROOT_DIR = os.path.join(INTEGRATION_TEST_ROOT, "data", "jobs")
JOB_NAME = "hello-numpy-sag"
JOB_DIR = os.path.join(JOBS_ROOT_DIR, JOB_NAME)
NO_STUDIES_PROJECT_YAML = os.path.join(INTEGRATION_TEST_ROOT, "data", "projects", "study_session_no_studies.yml")
WITH_STUDIES_PROJECT_YAML = os.path.join(INTEGRATION_TEST_ROOT, "data", "projects", "study_session_with_studies.yml")

MAIN_ADMIN = "admin@nvidia.com"
LEAD_ADMIN = "lead@nvidia.com"
OUTSIDER_ADMIN = "outsider@nvidia.com"
MIN_CLIENTS = 2


def _wait_for_job_meta(admin_api, job_id: str, timeout: float = 30.0) -> dict:
    end_time = time.time() + timeout
    while time.time() < end_time:
        meta = get_job_meta(admin_api, job_id)
        if meta.get(JobMetaKey.JOB_ID.value) == job_id:
            return meta
        time.sleep(0.5)
    raise AssertionError(f"Timed out waiting for job meta for {job_id}")


def _wait_for_session_job_meta(session, job_id: str, timeout: float = 30.0) -> dict:
    end_time = time.time() + timeout
    while time.time() < end_time:
        try:
            meta = session.get_job_meta(job_id)
        except JobNotFound:
            meta = {}
        if meta.get(JobMetaKey.JOB_ID.value) == job_id:
            return meta
        time.sleep(0.5)
    raise AssertionError(f"Timed out waiting for session job meta for {job_id}")


def _find_job(jobs: list[dict], job_id: str) -> dict:
    for job in jobs:
        if job.get("job_id") == job_id:
            return job
    raise AssertionError(f"job_id {job_id} not found in job list: {jobs}")


def _job_ids(jobs: list[dict]) -> set[str]:
    return {job["job_id"] for job in jobs}


def _get_persisted_job_meta_path(workspace_root: str, server_name: str, job_id: str) -> str:
    job_store_path = _get_job_store_path_from_workspace(workspace_root, server_name)
    return os.path.join(job_store_path, job_id, "meta")


def _update_persisted_job_study(workspace_root: str, server_name: str, job_id: str, study):
    meta_path = _get_persisted_job_meta_path(workspace_root, server_name, job_id)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if study is None:
        meta.pop(JobMetaKey.STUDY.value, None)
    else:
        meta[JobMetaKey.STUDY.value] = study

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)


def _wait_for_runtime_job_meta(workspace_root: str, site_name: str, job_id: str, timeout: float = 30.0) -> dict:
    site_root = os.path.join(workspace_root, site_name)
    workspace = Workspace(root_dir=site_root, site_name=site_name)
    meta_path = workspace.get_job_meta_path(job_id)

    end_time = time.time() + timeout
    while time.time() < end_time:
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get(JobMetaKey.JOB_ID.value) == job_id:
                return meta
        time.sleep(0.5)
    raise AssertionError(f"Timed out waiting for runtime job meta for {job_id} at site {site_name}")


def _assert_runtime_job_meta_absent(workspace_root: str, site_name: str, job_id: str, timeout: float = 10.0):
    site_root = os.path.join(workspace_root, site_name)
    workspace = Workspace(root_dir=site_root, site_name=site_name)
    meta_path = workspace.get_job_meta_path(job_id)

    end_time = time.time() + timeout
    while time.time() < end_time:
        if os.path.exists(meta_path):
            raise AssertionError(f"Unexpected runtime job meta for {job_id} at site {site_name}: {meta_path}")
        time.sleep(0.5)


def _make_numpy_recipe(name: str, min_clients: int = MIN_CLIENTS) -> NumpyFedAvgRecipe:
    return NumpyFedAvgRecipe(
        name=name,
        min_clients=min_clients,
        num_rounds=1,
        model=np.array([0.0] * 10),
        train_script=os.path.join(INTEGRATION_TEST_ROOT, "client.py"),
    )


def _export_numpy_job(job_dir: str, name: str, min_clients: int = MIN_CLIENTS):
    recipe = _make_numpy_recipe(name=name, min_clients=min_clients)
    recipe.export(job_dir)


def _extract_table_rows(response: dict) -> list[list[str]]:
    for item in response.get("data", []):
        if isinstance(item, dict) and item.get("type") == "table":
            return item.get("rows", [])
    raise AssertionError(f"No table rows found in response: {response}")


def _extract_error_messages(response: dict) -> list[str]:
    return [
        item.get("data") for item in response.get("data", []) if isinstance(item, dict) and item.get("type") == "error"
    ]


def _extract_client_names_from_check_status(response: dict) -> set[str]:
    rows = _extract_table_rows(response)
    assert rows, f"Expected non-empty check_status table rows: {response}"
    assert rows[0] and rows[0][0] == "CLIENT", f"Unexpected check_status table header: {rows[0]}"
    return {row[0] for row in rows[1:]}


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
def _admin_shell(admin_root: str, study: str | None = None):
    startup_dir = os.path.join(admin_root, "startup")
    script_path = os.path.join(startup_dir, "fl_admin.sh")
    master_fd, slave_fd = pty.openpty()
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([os.path.dirname(sys.executable), env.get("PATH", "")])
    command = ["bash", script_path]
    if study is not None:
        command.extend(["--study", study])
    process = subprocess.Popen(
        command,
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


def _stage_job_for_admin_shell(admin_root: str, source_job_dir: str, folder_name: str) -> str:
    upload_dir = _get_admin_upload_dir(admin_root)
    job_dir = os.path.join(upload_dir, folder_name)
    shutil.copytree(source_job_dir, job_dir)
    return job_dir


def _login_flare_session(admin_root: str, username: str, study: str):
    session = FlareSession(username=username, startup_path=admin_root, study=study)
    session.api.connect(10.0)
    result = session.api.login()
    return session, result


def _assert_login_success(admin_root: str, username: str, study: str):
    session, result = _login_flare_session(admin_root, username, study)
    try:
        assert result[ResultKey.STATUS] == APIStatus.SUCCESS
        assert session.api.is_ready()
    finally:
        session.api.close()


def _assert_login_rejected(admin_root: str, username: str, study: str):
    session, result = _login_flare_session(admin_root, username, study)
    try:
        assert result[ResultKey.STATUS] == APIStatus.ERROR_AUTHENTICATION
        assert not session.api.is_ready()
    finally:
        session.api.close()


@contextmanager
def _running_provisioned_system(project_yaml: str, super_user_name: str):
    site_launcher = ProvisionSiteLauncher(project_yaml=project_yaml)
    workspace_root = os.path.abspath(site_launcher.prepare_workspace())
    site_launcher.start_servers()
    site_launcher.start_clients()

    download_root_dir = tempfile.mkdtemp(prefix="study-session-download-")
    test_driver = NVFTestDriver(download_root_dir=download_root_dir, site_launcher=site_launcher, poll_period=1)
    sessions = []
    try:
        test_driver.initialize_super_user(
            workspace_root_dir=workspace_root,
            upload_root_dir=JOBS_ROOT_DIR,
            super_user_name=super_user_name,
        )
        test_driver.ensure_clients_started(num_clients=len(site_launcher.client_properties), timeout=300)
        yield {
            "workspace_root": workspace_root,
            "admin_roots": {name: os.path.join(workspace_root, name) for name in site_launcher.admin_user_names},
            "old_admin_api": test_driver.super_admin_api,
            "sessions": sessions,
            "server_name": next(iter(site_launcher.server_properties)),
        }
    finally:
        for session in sessions:
            with suppress(Exception):
                session.close()
        test_driver.finalize()
        site_launcher.stop_all_sites()
        site_launcher.cleanup()
        shutil.rmtree(download_root_dir, ignore_errors=True)


@pytest.fixture(scope="class")
def single_tenant_system():
    with _running_provisioned_system(NO_STUDIES_PROJECT_YAML, MAIN_ADMIN) as system:
        yield system


@pytest.fixture(scope="class")
def multi_study_system():
    with _running_provisioned_system(WITH_STUDIES_PROJECT_YAML, MAIN_ADMIN) as system:
        yield system


@pytest.mark.xdist_group(name="system_tests_group")
class TestSingleTenantStudySessionIntegration:
    def test_fladminapi_submit_defaults_study_and_list_jobs_exposes_it(self, single_tenant_system):
        admin_api = single_tenant_system["old_admin_api"]

        job_id = admin_api.submit_job(JOB_DIR)

        meta = _wait_for_job_meta(admin_api, job_id)
        assert meta[JobMetaKey.STUDY.value] == DEFAULT_STUDY

        jobs = admin_api.list_jobs()
        listed_job = _find_job(jobs, job_id)
        assert listed_job[JobMetaKey.STUDY.value] == DEFAULT_STUDY

    def test_single_tenant_rejects_non_default_login_for_flare_api(self, single_tenant_system):
        admin_root = single_tenant_system["admin_roots"][MAIN_ADMIN]

        _assert_login_rejected(admin_root, MAIN_ADMIN, "study-a")

    def test_fl_admin_shell_defaults_to_default_study_without_flag(self, single_tenant_system):
        admin_root = single_tenant_system["admin_roots"][MAIN_ADMIN]
        shell_job_dir = _stage_job_for_admin_shell(admin_root, JOB_DIR, "shell-default-single-tenant")

        try:
            with _admin_shell(admin_root) as (_process_default, master_fd_default):
                _login_to_admin_shell(master_fd_default, MAIN_ADMIN)

                submit_output = _run_admin_shell_command(master_fd_default, "submit_job shell-default-single-tenant")
                match = re.search(r"Submitted job:\s*([0-9a-fA-F-]+)", submit_output)
                assert match, f"failed to parse job id from output:\n{submit_output}"
                job_id = match.group(1)

                detailed_output = _run_admin_shell_command(master_fd_default, "list_jobs -d")
                assert job_id in detailed_output
                assert f'"study": "{DEFAULT_STUDY}"' in detailed_output
        finally:
            shutil.rmtree(shell_job_dir, ignore_errors=True)

    def test_single_tenant_normalizes_legacy_jobs_and_hides_non_default_jobs(self, single_tenant_system):
        admin_root = single_tenant_system["admin_roots"][MAIN_ADMIN]
        server_name = single_tenant_system["server_name"]
        admin_api = single_tenant_system["old_admin_api"]

        session = new_secure_session(MAIN_ADMIN, admin_root)
        single_tenant_system["sessions"].append(session)

        legacy_job_id = session.submit_job(JOB_DIR)
        _wait_for_job_meta(admin_api, legacy_job_id)
        _update_persisted_job_study(single_tenant_system["workspace_root"], server_name, legacy_job_id, None)

        normalized_meta = session.get_job_meta(legacy_job_id)
        assert normalized_meta[JobMetaKey.STUDY.value] == DEFAULT_STUDY
        assert legacy_job_id in _job_ids(session.list_jobs())

        hidden_job_id = session.submit_job(JOB_DIR)
        _wait_for_job_meta(admin_api, hidden_job_id)
        _update_persisted_job_study(single_tenant_system["workspace_root"], server_name, hidden_job_id, "study-a")

        assert hidden_job_id not in _job_ids(session.list_jobs())
        with pytest.raises(JobNotFound):
            session.get_job_meta(hidden_job_id)
        with pytest.raises(JobNotFound):
            session.clone_job(hidden_job_id)


@pytest.mark.xdist_group(name="system_tests_group")
class TestMultiStudySessionIntegration:
    def test_multistudy_login_accepts_valid_contexts_and_rejects_invalid_ones(self, multi_study_system):
        main_admin_root = multi_study_system["admin_roots"][MAIN_ADMIN]
        lead_admin_root = multi_study_system["admin_roots"][LEAD_ADMIN]
        outsider_admin_root = multi_study_system["admin_roots"][OUTSIDER_ADMIN]

        _assert_login_success(main_admin_root, MAIN_ADMIN, DEFAULT_STUDY)
        _assert_login_success(main_admin_root, MAIN_ADMIN, "study-a")
        _assert_login_success(lead_admin_root, LEAD_ADMIN, "study-b")
        _assert_login_success(outsider_admin_root, OUTSIDER_ADMIN, DEFAULT_STUDY)
        _assert_login_rejected(main_admin_root, MAIN_ADMIN, "unknown-study")
        _assert_login_rejected(outsider_admin_root, OUTSIDER_ADMIN, "study-a")

    def test_multistudy_isolates_default_and_study_jobs_in_listings(self, multi_study_system):
        admin_root = multi_study_system["admin_roots"][MAIN_ADMIN]

        default_session = new_secure_session(MAIN_ADMIN, admin_root)
        study_a_session = new_secure_session(MAIN_ADMIN, admin_root, study="study-a")
        study_b_session = new_secure_session(MAIN_ADMIN, admin_root, study="study-b")
        multi_study_system["sessions"].extend([default_session, study_a_session, study_b_session])

        default_job_id = default_session.submit_job(JOB_DIR)
        study_a_job_id = study_a_session.submit_job(JOB_DIR)

        default_meta = _wait_for_session_job_meta(default_session, default_job_id)
        study_a_meta = _wait_for_session_job_meta(study_a_session, study_a_job_id)
        assert default_meta[JobMetaKey.STUDY.value] == DEFAULT_STUDY
        assert study_a_meta[JobMetaKey.STUDY.value] == "study-a"

        assert default_job_id in _job_ids(default_session.list_jobs())
        assert study_a_job_id not in _job_ids(default_session.list_jobs())

        assert study_a_job_id in _job_ids(study_a_session.list_jobs())
        assert default_job_id not in _job_ids(study_a_session.list_jobs())

        assert default_job_id not in _job_ids(study_b_session.list_jobs())
        assert study_a_job_id not in _job_ids(study_b_session.list_jobs())

    def test_multistudy_hides_cross_study_direct_job_access(self, multi_study_system):
        admin_root = multi_study_system["admin_roots"][MAIN_ADMIN]

        study_a_session = new_secure_session(MAIN_ADMIN, admin_root, study="study-a")
        study_b_session = new_secure_session(MAIN_ADMIN, admin_root, study="study-b")
        default_session = new_secure_session(MAIN_ADMIN, admin_root)
        multi_study_system["sessions"].extend([study_a_session, study_b_session, default_session])

        job_id = study_a_session.submit_job(JOB_DIR)

        with pytest.raises(JobNotFound):
            study_b_session.get_job_meta(job_id)
        with pytest.raises(JobNotFound):
            study_b_session.clone_job(job_id)
        with pytest.raises(JobNotFound):
            default_session.get_job_meta(job_id)

        abort_response = study_b_session.do_command(f"abort_job {job_id}")
        assert abort_response[ResultKey.STATUS] == APIStatus.SUCCESS
        assert abort_response[ResultKey.META][MetaKey.STATUS] == MetaStatusValue.INVALID_JOB_ID
        assert abort_response[ResultKey.META][MetaKey.INFO] == job_id
        assert _extract_error_messages(abort_response) == [f"no such job: {job_id}"]

    def test_multistudy_membership_does_not_override_certificate_role(self, multi_study_system):
        lead_admin_root = multi_study_system["admin_roots"][LEAD_ADMIN]

        default_session = new_secure_session(LEAD_ADMIN, lead_admin_root)
        study_a_session = new_secure_session(LEAD_ADMIN, lead_admin_root, study="study-a")
        study_b_session = new_secure_session(LEAD_ADMIN, lead_admin_root, study="study-b")
        multi_study_system["sessions"].extend([default_session, study_a_session, study_b_session])

        with pytest.raises(AuthorizationError):
            default_session.submit_job(JOB_DIR)

        with pytest.raises(AuthorizationError):
            study_a_session.submit_job(JOB_DIR)

        with pytest.raises(AuthorizationError):
            study_b_session.submit_job(JOB_DIR)

    def test_multistudy_filters_check_status_to_enrolled_sites(self, multi_study_system):
        admin_root = multi_study_system["admin_roots"][MAIN_ADMIN]

        study_a_session = new_secure_session(MAIN_ADMIN, admin_root, study="study-a")
        study_b_session = new_secure_session(MAIN_ADMIN, admin_root, study="study-b")
        multi_study_system["sessions"].extend([study_a_session, study_b_session])

        study_a_status = study_a_session.api.do_command("check_status client")
        study_b_status = study_b_session.api.do_command("check_status client")

        assert _extract_client_names_from_check_status(study_a_status) == {"site-1", "site-2"}
        assert _extract_client_names_from_check_status(study_b_status) == {"site-3"}

    def test_multistudy_scheduler_limits_all_sites_jobs_to_study_sites(self, multi_study_system):
        admin_root = multi_study_system["admin_roots"][MAIN_ADMIN]
        workspace_root = multi_study_system["workspace_root"]
        server_name = multi_study_system["server_name"]

        study_a_session = new_secure_session(MAIN_ADMIN, admin_root, study="study-a")
        multi_study_system["sessions"].append(study_a_session)

        job_id = study_a_session.submit_job(JOB_DIR)
        meta = _wait_for_session_job_meta(study_a_session, job_id)
        assert meta[JobMetaKey.STUDY.value] == "study-a"

        for site_name in [server_name, "site-1", "site-2"]:
            runtime_meta = _wait_for_runtime_job_meta(workspace_root, site_name, job_id)
            assert runtime_meta[JobMetaKey.STUDY.value] == "study-a"

        _assert_runtime_job_meta_absent(workspace_root, "site-3", job_id)

    def test_multistudy_rejects_deploy_map_sites_outside_study(self, multi_study_system, tmp_path):
        admin_root = multi_study_system["admin_roots"][MAIN_ADMIN]

        study_a_session = new_secure_session(MAIN_ADMIN, admin_root, study="study-a")
        multi_study_system["sessions"].append(study_a_session)

        export_root = tmp_path / "invalid_deploy_map_job_export"
        job_name = "invalid-deploy-map-job"
        _export_numpy_job(str(export_root), name=job_name)
        job_dir = export_root / job_name

        meta_path = job_dir / "meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["deploy_map"] = {"app": ["server", "site-1", "site-3"]}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        with pytest.raises(InvalidJobDefinition, match="site 'site-3' is not enrolled in study 'study-a'"):
            study_a_session.submit_job(str(job_dir))

    def test_multistudy_prod_env_propagates_study_and_runtime_meta(self, multi_study_system):
        admin_root = multi_study_system["admin_roots"][MAIN_ADMIN]
        workspace_root = multi_study_system["workspace_root"]
        server_name = multi_study_system["server_name"]

        recipe = _make_numpy_recipe("prod-env-study-a")
        env = ProdEnv(startup_kit_location=admin_root, username=MAIN_ADMIN, study="study-a", login_timeout=2.0)
        recipe.process_env(env)
        job_id = env.deploy(recipe.job)
        study_a_session = new_secure_session(MAIN_ADMIN, admin_root, study="study-a")
        multi_study_system["sessions"].append(study_a_session)

        meta = _wait_for_session_job_meta(study_a_session, job_id)
        assert meta[JobMetaKey.STUDY.value] == "study-a"

        for site_name in [server_name, "site-1", "site-2"]:
            runtime_meta = _wait_for_runtime_job_meta(workspace_root, site_name, job_id)
            assert runtime_meta[JobMetaKey.STUDY.value] == "study-a"

        _assert_runtime_job_meta_absent(workspace_root, "site-3", job_id)

    def test_multistudy_fl_admin_shell_scopes_terminal_session(self, multi_study_system):
        admin_root = multi_study_system["admin_roots"][MAIN_ADMIN]
        shell_job_dir = _stage_job_for_admin_shell(admin_root, JOB_DIR, "shell-study-a")

        try:
            with _admin_shell(admin_root, "study-a") as (_process_a, master_fd_a):
                _login_to_admin_shell(master_fd_a, MAIN_ADMIN)

                submit_output = _run_admin_shell_command(master_fd_a, "submit_job shell-study-a")
                match = re.search(r"Submitted job:\s*([0-9a-fA-F-]+)", submit_output)
                assert match, f"failed to parse job id from output:\n{submit_output}"
                job_id = match.group(1)

                detailed_output = _run_admin_shell_command(master_fd_a, "list_jobs -d")
                assert job_id in detailed_output
                assert '"study": "study-a"' in detailed_output

            with _admin_shell(admin_root, "study-b") as (_process_b, master_fd_b):
                _login_to_admin_shell(master_fd_b, MAIN_ADMIN)
                other_study_output = _run_admin_shell_command(master_fd_b, "list_jobs")
                assert job_id not in other_study_output
        finally:
            shutil.rmtree(shell_job_dir, ignore_errors=True)
