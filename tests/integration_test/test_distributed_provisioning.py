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

"""Integration test: cert init → cert csr → cert sign → nvflare package.

Tests the full distributed provisioning workflow for a localhost 2-site
federation and verifies the resulting startup kits are self-consistent and
structurally equivalent to what `nvflare provision` would produce.

Run with:
    python3 -m pytest tests/integration_test/test_distributed_provisioning.py -v
"""

import json
import os
import signal
import stat
import subprocess
import sys
import time
import types
import unittest.mock

import pytest

from nvflare.tool.cert.cert_commands import handle_cert_csr, handle_cert_init, handle_cert_sign
from nvflare.tool.package.package_commands import handle_package

# ---------------------------------------------------------------------------
# Participants for the test federation
# ---------------------------------------------------------------------------

# (name, cert_type) — cert_type is embedded in CSR and signed cert; package derives it from cert
_PARTICIPANTS = [
    ("localhost", "server"),
    ("site-1", "client"),
    ("site-2", "client"),
    ("admin@myfl.com", "lead"),
]

_SERVER_NAME = "localhost"
_ENDPOINT = "grpc://localhost:8002"
_PROJECT = "myfl"

# Files expected in every startup/ directory
_STARTUP_CERTS = {"rootCA.pem"}
_STARTUP_SERVER = {"server.crt", "server.key", "fed_server.json", "start.sh", "sub_start.sh"}
_STARTUP_CLIENT = {"client.crt", "client.key", "fed_client.json", "start.sh", "sub_start.sh"}
_STARTUP_ADMIN = {"client.crt", "client.key", "fed_admin.json", "fl_admin.sh"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ns(**kwargs):
    """Build a SimpleNamespace with common defaults for CLI args."""
    defaults = dict(force=True, output_fmt=None, schema=False, org=None)
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


def _run_package(name, cert, key, rootca, workspace):
    """Call handle_package and return the output result dict.

    Kit type is derived automatically from the signed certificate's UNSTRUCTURED_NAME;
    no -t/--type argument is needed in the new workflow.
    """
    result = {}

    def _capture(r, fmt):
        result.update(r)

    args = _ns(
        kit_type=None,  # derived from signed cert
        name=name,
        endpoint=_ENDPOINT,
        dir=None,
        cert=cert,
        key=key,
        rootca=rootca,
        workspace=workspace,
        project_name=_PROJECT,
        admin_port=None,
    )
    with unittest.mock.patch("nvflare.tool.package.package_commands.output", side_effect=_capture):
        handle_package(args)
    return result


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.fixture()
def provisioned(tmp_path):
    """Run the full 4-step workflow and return a dict of kit_dir paths by participant name."""
    ca_dir = str(tmp_path / "ca")
    csr_dir = str(tmp_path / "csr")
    signed_dir = str(tmp_path / "signed")
    ws = str(tmp_path / "ws")

    # Step 1 — cert init
    handle_cert_init(_ns(project=_PROJECT, output_dir=ca_dir))

    # Step 2 — cert csr for each participant (propose type in CSR)
    for name, cert_type in _PARTICIPANTS:
        handle_cert_csr(_ns(name=name, output_dir=csr_dir, cert_type=cert_type))

    # Step 3 — cert sign for each participant (type read from CSR; no -t needed)
    for name, _ in _PARTICIPANTS:
        sign_out = os.path.join(signed_dir, name)
        handle_cert_sign(
            _ns(
                csr_path=os.path.join(csr_dir, f"{name}.csr"),
                ca_dir=ca_dir,
                output_dir=sign_out,
                cert_type=None,  # read from CSR
            )
        )

    # Step 4 — nvflare package for each participant (kit type derived from signed cert)
    kit_dirs = {}
    for name, _ in _PARTICIPANTS:
        sign_out = os.path.join(signed_dir, name)
        result = _run_package(
            name=name,
            cert=os.path.join(sign_out, f"{name}.crt"),
            key=os.path.join(csr_dir, f"{name}.key"),
            rootca=os.path.join(sign_out, "rootCA.pem"),
            workspace=ws,
        )
        kit_dirs[name] = result["output_dir"]

    return kit_dirs


class TestDistributedProvisioningWorkflow:
    """Full distributed provisioning workflow: cert init → csr → sign → package."""

    # ------------------------------------------------------------------
    # Step-level smoke tests (fail fast if a command itself breaks)
    # ------------------------------------------------------------------

    def test_cert_init_produces_ca_files(self, tmp_path):
        ca_dir = str(tmp_path / "ca")
        handle_cert_init(_ns(project=_PROJECT, output_dir=ca_dir))
        assert os.path.isfile(os.path.join(ca_dir, "rootCA.pem"))
        assert os.path.isfile(os.path.join(ca_dir, "rootCA.key"))
        assert os.path.isfile(os.path.join(ca_dir, "ca.json"))

    def test_cert_csr_produces_key_and_csr(self, tmp_path):
        csr_dir = str(tmp_path / "csr")
        handle_cert_csr(_ns(name="site-1", output_dir=csr_dir))
        assert os.path.isfile(os.path.join(csr_dir, "site-1.key"))
        assert os.path.isfile(os.path.join(csr_dir, "site-1.csr"))
        # Key must be private (permissions 0o600)
        mode = stat.S_IMODE(os.stat(os.path.join(csr_dir, "site-1.key")).st_mode)
        assert mode == 0o600

    def test_cert_sign_produces_cert_and_rootca(self, tmp_path):
        ca_dir = str(tmp_path / "ca")
        csr_dir = str(tmp_path / "csr")
        sign_out = str(tmp_path / "signed")
        handle_cert_init(_ns(project=_PROJECT, output_dir=ca_dir))
        handle_cert_csr(_ns(name="site-1", output_dir=csr_dir, cert_type="client"))
        # cert_type=None: type read from CSR UNSTRUCTURED_NAME
        handle_cert_sign(
            _ns(
                csr_path=os.path.join(csr_dir, "site-1.csr"),
                ca_dir=ca_dir,
                output_dir=sign_out,
                cert_type=None,
            )
        )
        assert os.path.isfile(os.path.join(sign_out, "site-1.crt"))
        assert os.path.isfile(os.path.join(sign_out, "rootCA.pem"))

    # ------------------------------------------------------------------
    # Kit structure tests
    # ------------------------------------------------------------------

    def test_all_kit_dirs_created(self, provisioned):
        for name, _ in _PARTICIPANTS:
            assert os.path.isdir(provisioned[name]), f"Kit dir missing for {name}"

    def test_server_kit_has_expected_files(self, provisioned):
        startup = os.path.join(provisioned[_SERVER_NAME], "startup")
        expected = _STARTUP_SERVER | _STARTUP_CERTS
        for f in expected:
            assert os.path.isfile(os.path.join(startup, f)), f"Missing {f} in server startup/"

    @pytest.mark.parametrize("name", ["site-1", "site-2"])
    def test_client_kit_has_expected_files(self, name, provisioned):
        startup = os.path.join(provisioned[name], "startup")
        expected = _STARTUP_CLIENT | _STARTUP_CERTS
        for f in expected:
            assert os.path.isfile(os.path.join(startup, f)), f"Missing {f} in {name}/startup/"

    def test_admin_kit_has_expected_files(self, provisioned):
        startup = os.path.join(provisioned["admin@myfl.com"], "startup")
        expected = _STARTUP_ADMIN | _STARTUP_CERTS
        for f in expected:
            assert os.path.isfile(os.path.join(startup, f)), f"Missing {f} in admin startup/"

    def test_all_kits_have_local_and_transfer_dirs(self, provisioned):
        for name, _ in _PARTICIPANTS:
            assert os.path.isdir(os.path.join(provisioned[name], "local")), f"Missing local/ for {name}"
            assert os.path.isdir(os.path.join(provisioned[name], "transfer")), f"Missing transfer/ for {name}"

    def test_server_key_permissions_600(self, provisioned):
        key = os.path.join(provisioned[_SERVER_NAME], "startup", "server.key")
        assert stat.S_IMODE(os.stat(key).st_mode) == 0o600

    # ------------------------------------------------------------------
    # Config consistency tests — server, client, admin configs must agree
    # ------------------------------------------------------------------

    def test_fed_server_json_target_and_sp_end_point(self, provisioned):
        with open(os.path.join(provisioned[_SERVER_NAME], "startup", "fed_server.json")) as f:
            cfg = json.load(f)
        svc = cfg["servers"][0]["service"]
        assert svc["target"] == "localhost:8002"
        assert svc["scheme"] == "grpc"
        sp = cfg["overseer_agent"]["args"]["sp_end_point"]
        assert sp.startswith("localhost:8002:")

    def test_fed_client_sp_end_point_matches_server(self, provisioned):
        with open(os.path.join(provisioned[_SERVER_NAME], "startup", "fed_server.json")) as f:
            srv_cfg = json.load(f)
        server_sp = srv_cfg["overseer_agent"]["args"]["sp_end_point"]

        for name in ("site-1", "site-2"):
            with open(os.path.join(provisioned[name], "startup", "fed_client.json")) as f:
                cli_cfg = json.load(f)
            client_sp = cli_cfg["overseer_agent"]["args"]["sp_end_point"]
            assert client_sp == server_sp, f"{name} sp_end_point {client_sp!r} != server sp_end_point {server_sp!r}"

    def test_fed_admin_json_port_matches_server(self, provisioned):
        with open(os.path.join(provisioned[_SERVER_NAME], "startup", "fed_server.json")) as f:
            srv_cfg = json.load(f)
        server_admin_port = srv_cfg["servers"][0]["admin_port"]

        with open(os.path.join(provisioned["admin@myfl.com"], "startup", "fed_admin.json")) as f:
            adm_cfg = json.load(f)
        assert adm_cfg["admin"]["port"] == server_admin_port

    def test_fed_admin_server_identity_matches_server_name(self, provisioned):
        """Admin must authenticate against the server's identity (cert CN)."""
        with open(os.path.join(provisioned[_SERVER_NAME], "startup", "fed_server.json")) as f:
            srv_cfg = json.load(f)
        server_identity = srv_cfg["servers"][0]["admin_server"]

        with open(os.path.join(provisioned["admin@myfl.com"], "startup", "fed_admin.json")) as f:
            adm_cfg = json.load(f)
        # fed_admin.json uses "cn" or "server_identity" to name the server for mTLS
        assert (
            adm_cfg["admin"].get("cn") == server_identity or adm_cfg["admin"].get("server_identity") == server_identity
        )

    def test_all_participants_use_same_ca(self, provisioned):
        """rootCA.pem must be identical in all kits (same signing authority)."""
        ref = None
        for name, _ in _PARTICIPANTS:
            rootca_path = os.path.join(provisioned[name], "startup", "rootCA.pem")
            content = open(rootca_path, "rb").read()
            if ref is None:
                ref = content
            else:
                assert content == ref, f"rootCA.pem in {name} differs from {_PARTICIPANTS[0][0]}"

    def test_no_server_placeholder_dir_in_non_server_kits(self, provisioned):
        """The server placeholder directory must be removed for non-server kits."""
        for name in ("site-1", "site-2", "admin@myfl.com"):
            prod_dir = os.path.dirname(provisioned[name])
            assert not os.path.exists(
                os.path.join(prod_dir, _SERVER_NAME)
            ), f"Server placeholder dir found in prod dir for {name}"


# ---------------------------------------------------------------------------
# End-to-end runtime test: packaged kits can run a real FL job
# ---------------------------------------------------------------------------

_JOB_TIMEOUT = 300  # seconds to wait for FL job completion
_STARTUP_WAIT = 5  # seconds to wait for server/clients to initialise
_ADMIN_CONNECT_TIMEOUT = 60  # seconds to retry admin API connection before giving up


def _kill_process_group(proc):
    """SIGTERM the process group; ignore errors if it's already gone."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        pass


@pytest.mark.slow
class TestDistributedProvisioningE2E:
    """End-to-end: packaged kits start server/clients and run hello-numpy-sag."""

    @pytest.fixture()
    def admin_api(self, provisioned, tmp_path):
        """Start server + 2 clients from packaged kits; yield a logged-in FLAdminAPI."""
        env = os.environ.copy()
        python_path = ":".join(p for p in sys.path if p)
        env["PYTHONPATH"] = python_path

        def _launch(cmd):
            return subprocess.Popen(
                cmd,
                shell=False,
                env=env,
                preexec_fn=os.setsid,
            )

        processes = []

        # Start server
        server_kit = provisioned[_SERVER_NAME]
        processes.append(
            _launch(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "nvflare.private.fed.app.server.server_train",
                    "-m",
                    server_kit,
                    "-s",
                    "fed_server.json",
                    "--set",
                    "secure_train=true",
                    "org=myorg",
                    "config_folder=config",
                ]
            )
        )
        time.sleep(_STARTUP_WAIT)

        # Start clients
        for client_name in ("site-1", "site-2"):
            processes.append(
                _launch(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "nvflare.private.fed.app.client.client_train",
                        "-m",
                        provisioned[client_name],
                        "-s",
                        "fed_client.json",
                        "--set",
                        "secure_train=true",
                        f"uid={client_name}",
                        "org=myorg",
                        "config_folder=config",
                    ]
                )
            )
        time.sleep(_STARTUP_WAIT)

        # Build admin API — retry until the server is accepting connections.
        # create_admin_api calls connect(10.0) which raises FLCommunicationError if the
        # server is not yet ready; poll until success or _ADMIN_CONNECT_TIMEOUT is exceeded.
        upload_dir = str(tmp_path / "upload")
        download_dir = str(tmp_path / "download")
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(download_dir, exist_ok=True)

        from tests.integration_test.src.utils import create_admin_api

        api = None
        deadline = time.time() + _ADMIN_CONNECT_TIMEOUT
        while time.time() < deadline:
            try:
                api = create_admin_api(
                    workspace_root_dir=os.path.dirname(provisioned["admin@myfl.com"]),
                    upload_root_dir=upload_dir,
                    download_root_dir=download_dir,
                    admin_user_name="admin@myfl.com",
                )
                break
            except Exception:
                time.sleep(2)
        if api is None:
            pytest.fail(f"Admin API did not connect within {_ADMIN_CONNECT_TIMEOUT} s")

        yield api

        # Teardown
        try:
            api.logout()
        except Exception:
            pass
        for p in processes:
            _kill_process_group(p)
        for p in processes:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass

    def test_hello_numpy_sag_completes(self, admin_api, tmp_path):
        """Submit hello-numpy-sag; verify it finishes with FINISHED_COMPLETED."""
        from nvflare.apis.job_def import RunStatus
        from nvflare.fuel.hci.client.api_status import APIStatus
        from nvflare.fuel.hci.client.fl_admin_api_spec import TargetType
        from tests.integration_test.src.utils import check_job_done, ensure_admin_api_logged_in, get_job_meta

        assert ensure_admin_api_logged_in(admin_api, timeout=60), "Admin API did not log in within 60 s"

        job_dir = os.path.join(os.path.dirname(__file__), "data", "jobs", "hello-numpy-sag")
        response = admin_api.submit_job(job_dir)
        assert response.get("status") == APIStatus.SUCCESS, f"Job submit failed: {response}"
        job_id = response["details"]["job_id"]

        # Poll until done; log status every 30 s so a timeout is diagnosable.
        deadline = time.time() + _JOB_TIMEOUT
        last_log = time.time()
        while time.time() < deadline:
            if check_job_done(job_id, admin_api):
                break
            if time.time() - last_log >= 30:
                srv = admin_api.check_status(target_type=TargetType.SERVER)
                cli = admin_api.check_status(target_type=TargetType.CLIENT)
                print(f"\n[E2E poll] job={job_id}  server={srv}  clients={cli}")
                last_log = time.time()
            time.sleep(5)
        else:
            srv = admin_api.check_status(target_type=TargetType.SERVER)
            cli = admin_api.check_status(target_type=TargetType.CLIENT)
            meta = get_job_meta(admin_api, job_id)
            pytest.fail(
                f"Job {job_id} did not complete within {_JOB_TIMEOUT} s\n"
                f"  server status : {srv}\n"
                f"  client status : {cli}\n"
                f"  job meta      : {meta}"
            )

        meta = get_job_meta(admin_api, job_id)
        assert (
            meta.get("status") == RunStatus.FINISHED_COMPLETED.value
        ), f"Expected FINISHED_COMPLETED, got {meta.get('status')!r}"
