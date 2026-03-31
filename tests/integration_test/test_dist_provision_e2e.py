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

"""End-to-end integration test for distributed provisioning with hello-pt.

Provisions a 1-server / 2-client federation using the Manual Workflow
(nvflare cert init / cert csr / cert sign / package), starts the federation,
submits hello-pt, and asserts the job completes successfully.

Prerequisites:
  pip install torch torchvision
  python -c "from torchvision.datasets import CIFAR10; CIFAR10(root='~/data', download=True)"

Run:
  cd tests/integration_test
  python -m pytest test_dist_provision_e2e.py -v -s
"""

import os
import time

import pytest

JOBS_ROOT = os.path.join(os.path.dirname(__file__), "data", "jobs")
JOB_NAME = "hello-pt"
TIMEOUT_SERVER_START = 60  # seconds to wait for server to be ready
TIMEOUT_JOB_DONE = 600  # seconds to wait for hello-pt to finish
POLL_INTERVAL = 5


@pytest.fixture(scope="module")
def launcher():
    from src.dist_provision_site_launcher import DistProvisionSiteLauncher

    launcher = DistProvisionSiteLauncher(
        server_name="localhost",
        server_port=8002,
        client_names=["site-1", "site-2"],
        admin_name="admin",
        project_name="dist_prov_test",
    )
    workspace = launcher.prepare_workspace()
    print(f"\nWorkspace: {workspace}")
    yield launcher
    launcher.cleanup()


@pytest.fixture(scope="module")
def running_federation(launcher):
    """Start server + clients; yield; stop all."""
    launcher.start_servers()
    time.sleep(5)
    launcher.start_clients()
    time.sleep(5)
    yield launcher
    launcher.stop_clients()
    launcher.stop_servers()


@pytest.fixture(scope="module")
def admin_api(running_federation):
    """Connect and log in to the admin API."""
    import tempfile

    from nvflare.apis.workspace import Workspace
    from nvflare.fuel.hci.client.fl_admin_api import FLAdminAPI

    from src.utils import create_admin_api, ensure_admin_api_logged_in

    upload_dir = tempfile.mkdtemp()
    download_dir = tempfile.mkdtemp()

    api = create_admin_api(
        workspace_root_dir=running_federation.work_dir,
        upload_root_dir=upload_dir,
        download_root_dir=download_dir,
        admin_user_name=running_federation.admin_name,
    )
    assert ensure_admin_api_logged_in(api, timeout=TIMEOUT_SERVER_START), "Admin failed to log in"
    yield api
    try:
        api.logout()
    except Exception:
        pass


def _wait_for_clients_ready(admin_api, n_clients: int, timeout: int = 60) -> bool:
    from nvflare.fuel.hci.client.api_status import APIStatus
    from nvflare.fuel.hci.client.fl_admin_api_spec import TargetType

    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = admin_api.check_status(target_type=TargetType.CLIENT)
        if resp.get("status") == APIStatus.SUCCESS:
            statuses = resp.get("details", {}).get("client_statuses", [])
            # statuses[0] is the header row; rows 1+ are clients
            ready = [r for r in statuses[1:] if r[3] != "No Reply"]
            if len(ready) >= n_clients:
                return True
        time.sleep(POLL_INTERVAL)
    return False


def _submit_job(admin_api, job_dir: str) -> str:
    resp = admin_api.submit_job(job_dir)
    assert resp.get("status") == "SUCCESS", f"submit_job failed: {resp}"
    return resp["details"]["job_id"]


def _wait_for_job(admin_api, job_id: str, timeout: int = TIMEOUT_JOB_DONE) -> bool:
    from src.utils import check_job_done

    deadline = time.time() + timeout
    while time.time() < deadline:
        if check_job_done(job_id, admin_api):
            return True
        time.sleep(POLL_INTERVAL)
    return False


def _job_completed_successfully(admin_api, job_id: str) -> bool:
    from nvflare.apis.job_def import RunStatus
    from src.utils import get_job_meta

    meta = get_job_meta(admin_api, job_id)
    return meta.get("status") == RunStatus.FINISHED_COMPLETED.value


class TestDistProvisionHelloPT:
    def test_provisioning_created_startup_kits(self, launcher):
        """All startup kits must contain start.sh and a cert."""
        for name, prop in launcher.server_properties.items():
            assert os.path.exists(os.path.join(prop.root_dir, "startup", "start.sh")), \
                f"Missing start.sh for server {name}"
            assert os.path.exists(os.path.join(prop.root_dir, "startup", "server.crt")), \
                f"Missing server.crt for {name}"
            assert os.path.exists(os.path.join(prop.root_dir, "startup", "rootCA.pem")), \
                f"Missing rootCA.pem for {name}"

        for name, prop in launcher.client_properties.items():
            assert os.path.exists(os.path.join(prop.root_dir, "startup", "start.sh")), \
                f"Missing start.sh for client {name}"
            assert os.path.exists(os.path.join(prop.root_dir, "startup", "client.crt")), \
                f"Missing client.crt for {name}"

        admin_kit = os.path.join(launcher.work_dir, launcher.admin_name)
        assert os.path.exists(os.path.join(admin_kit, "startup", "fed_admin.json")), \
            "Missing fed_admin.json in admin kit"

    def test_clients_connect(self, admin_api, running_federation):
        """Both clients must connect to the server within the timeout."""
        n_clients = len(running_federation.client_names)
        connected = _wait_for_clients_ready(admin_api, n_clients, timeout=TIMEOUT_SERVER_START)
        assert connected, f"Only {n_clients} clients did not connect within {TIMEOUT_SERVER_START}s"

    def test_hello_pt_completes(self, admin_api):
        """Submit hello-pt and assert it finishes with FINISHED_COMPLETED status."""
        job_dir = os.path.join(JOBS_ROOT, JOB_NAME)
        assert os.path.isdir(job_dir), f"Job directory not found: {job_dir}"

        job_id = _submit_job(admin_api, job_dir)
        print(f"\nSubmitted job: {job_id}")

        done = _wait_for_job(admin_api, job_id, timeout=TIMEOUT_JOB_DONE)
        assert done, f"Job {job_id} did not finish within {TIMEOUT_JOB_DONE}s"

        success = _job_completed_successfully(admin_api, job_id)
        assert success, f"Job {job_id} did not complete successfully"
        print(f"Job {job_id} completed successfully.")
