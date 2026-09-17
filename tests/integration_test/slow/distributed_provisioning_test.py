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

"""Integration test: cert init → cert request → cert approve → nvflare package.

Tests the full distributed provisioning workflow for a localhost 2-site
federation and verifies the resulting startup kits are self-consistent and
structurally equivalent to what `nvflare provision` would produce.

Run with:
    python3 -m pytest tests/integration_test/slow/distributed_provisioning_test.py -v
"""

import json
import os
import signal
import socket
import stat
import subprocess
import sys
import time
import types
import unittest.mock
import zipfile

import pytest
import yaml

from nvflare.lighter.provision import provision
from nvflare.tool.cert.cert_commands import handle_cert_approve, handle_cert_init, handle_cert_request
from nvflare.tool.package.package_commands import handle_package

INTEGRATION_TEST_ROOT = os.path.dirname(os.path.dirname(__file__))


def _tcp_bind_available() -> bool:
    """Return True if the process can bind a TCP socket (required for internal Cell listener)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("", 0))
        return True
    except OSError:
        return False
    finally:
        s.close()


_TCP_BIND_SKIP = pytest.mark.skipif(
    not _tcp_bind_available(),
    reason="TCP socket binding not available in this environment; Cell internal listener cannot start",
)

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
_PROJECT = "myfl"
_FED_PORT = 8002
_ADMIN_PORT = 8003

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
    defaults = dict(force=False, output_fmt=None, schema=False, org=None)
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


def _request_participant(project: str, name: str, cert_type: str, org: str = "myorg") -> dict:
    if cert_type == "client":
        participant = {"name": name, "type": "client", "org": org}
    elif cert_type == "server":
        participant = {"name": name, "type": "server", "org": org}
    elif cert_type == "org_admin":
        participant = {"name": name, "type": "admin", "org": org, "role": "org_admin"}
    elif cert_type in {"lead", "member"}:
        participant = {"name": name, "type": "admin", "org": org, "role": cert_type}
    else:
        raise ValueError(f"unsupported cert_type: {cert_type}")
    return {
        "name": project,
        "participants": [participant],
    }


def _write_project_profile(path: str, project: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "name": project,
                "scheme": "grpc",
                "connection_security": "mtls",
                "server": {
                    "host": _SERVER_NAME,
                    "fed_learn_port": _FED_PORT,
                    "admin_port": _ADMIN_PORT,
                },
            },
            f,
            sort_keys=False,
        )


def _run_request(name: str, cert_type: str, project: str, request_root: str, org: str = "myorg"):
    request_dir = os.path.join(request_root, name)
    participant_path = os.path.join(request_root, f"{name}.yaml")
    os.makedirs(request_root, exist_ok=True)
    with open(participant_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_request_participant(project, name, cert_type, org), f, sort_keys=False)
    handle_cert_request(_ns(participant=participant_path, output_dir=request_dir))
    return request_dir, os.path.join(request_dir, f"{name}.request.zip")


def _run_approve(name: str, request_zip: str, ca_dir: str, signed_dir: str, profile: str):
    signed_zip = os.path.join(signed_dir, f"{name}.signed.zip")
    handle_cert_approve(
        _ns(request_zip=request_zip, ca_dir=ca_dir, profile=profile, signed_zip=signed_zip, valid_days=1095)
    )
    return signed_zip


def _run_package(signed_zip: str, request_dir: str, workspace: str, endpoint: str = None):
    """Call handle_package signed-zip mode and return the output result dict."""
    result = {}

    def _capture(data, exit_code=0, status="ok"):
        result.update(data)
        result["_exit_code"] = exit_code
        result["_status"] = status

    args = _ns(
        input=signed_zip,
        endpoint=endpoint,
        request_dir=request_dir,
        workspace=workspace,
        expected_fingerprint=None,
    )
    with unittest.mock.patch("nvflare.tool.package.package_commands.output_ok", side_effect=_capture):
        handle_package(args)
    return result


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.fixture()
def provisioned(tmp_path):
    """Run the full public workflow and return a dict of kit_dir paths by participant name."""
    ca_dir = str(tmp_path / "ca")
    request_root = str(tmp_path / "requests")
    signed_dir = str(tmp_path / "signed")
    ws = str(tmp_path / "ws")
    profile = str(tmp_path / "project_profile.yaml")

    # Step 1 — cert init (profile must exist before init so the project name can be read from it)
    _write_project_profile(profile, _PROJECT)
    handle_cert_init(_ns(profile=profile, output_dir=ca_dir))

    request_dirs = {}
    request_zips = {}
    signed_zips = {}
    # Step 2 — cert request for each participant. The private key remains in request_dir.
    for name, cert_type in _PARTICIPANTS:
        request_dirs[name], request_zips[name] = _run_request(name, cert_type, _PROJECT, request_root)

    # Step 3 — cert approve for each request zip.
    for name, _ in _PARTICIPANTS:
        signed_zips[name] = _run_approve(name, request_zips[name], ca_dir, signed_dir, profile)

    # Step 4 — nvflare package signed zip for each participant.
    kit_dirs = {}
    for name, _ in _PARTICIPANTS:
        result = _run_package(signed_zip=signed_zips[name], request_dir=request_dirs[name], workspace=ws)
        kit_dirs[name] = result["output_dir"]

    return kit_dirs


class TestDistributedProvisioningWorkflow:
    """Full distributed provisioning workflow: cert init → request → approve → package."""

    # ------------------------------------------------------------------
    # Step-level smoke tests (fail fast if a command itself breaks)
    # ------------------------------------------------------------------

    def test_cert_init_produces_ca_files(self, tmp_path):
        ca_dir = str(tmp_path / "ca")
        profile = str(tmp_path / "project_profile.yaml")
        _write_project_profile(profile, _PROJECT)
        handle_cert_init(_ns(profile=profile, output_dir=ca_dir))
        assert os.path.isfile(os.path.join(ca_dir, "rootCA.pem"))
        assert os.path.isfile(os.path.join(ca_dir, "rootCA.key"))
        assert os.path.isfile(os.path.join(ca_dir, "ca.json"))

    def test_cert_request_produces_key_csr_and_request_zip(self, tmp_path):
        request_root = str(tmp_path / "requests")
        request_dir, request_zip = _run_request("site-1", "client", _PROJECT, request_root)
        assert os.path.isfile(os.path.join(request_dir, "site-1.key"))
        assert os.path.isfile(os.path.join(request_dir, "site-1.csr"))
        assert os.path.isfile(os.path.join(request_dir, "request.json"))
        assert os.path.isfile(request_zip)
        # Key must be private (permissions 0o600)
        mode = stat.S_IMODE(os.stat(os.path.join(request_dir, "site-1.key")).st_mode)
        assert mode == 0o600

    def test_cert_approve_produces_signed_zip(self, tmp_path):
        ca_dir = str(tmp_path / "ca")
        request_root = str(tmp_path / "requests")
        signed_dir = str(tmp_path / "signed")
        profile = str(tmp_path / "project_profile.yaml")
        _write_project_profile(profile, _PROJECT)
        handle_cert_init(_ns(profile=profile, output_dir=ca_dir))
        _request_dir, request_zip = _run_request("site-1", "client", _PROJECT, request_root)
        signed_zip = _run_approve("site-1", request_zip, ca_dir, signed_dir, profile)
        assert os.path.isfile(signed_zip)
        with zipfile.ZipFile(signed_zip) as zf:
            signed_json = json.loads(zf.read("signed.json"))
        assert signed_json["server"] == {
            "host": _SERVER_NAME,
            "fed_learn_port": _FED_PORT,
            "admin_port": _ADMIN_PORT,
        }

    @pytest.mark.parametrize("name,cert_type", [("site-1", "client"), ("admin@myfl.com", "lead")])
    def test_client_and_user_request_site_yaml_do_not_need_server_blocks(self, tmp_path, name, cert_type):
        request_root = str(tmp_path / "requests")
        request_dir, request_zip = _run_request(name, cert_type, _PROJECT, request_root)

        with open(os.path.join(request_dir, "site.yaml"), encoding="utf-8") as f:
            local_site = yaml.safe_load(f)
        with zipfile.ZipFile(request_zip) as zf:
            request_site = yaml.safe_load(zf.read("site.yaml"))

        assert "server" not in local_site["participants"][0]
        assert "server" not in request_site["participants"][0]

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

    def test_fed_server_json_target(self, provisioned):
        with open(os.path.join(provisioned[_SERVER_NAME], "startup", "fed_server.json")) as f:
            cfg = json.load(f)
        svc = cfg["servers"][0]["service"]
        assert svc["target"] == "localhost:8002"
        assert svc["scheme"] == "grpc"

    def test_fed_client_target_matches_server(self, provisioned):
        with open(os.path.join(provisioned[_SERVER_NAME], "startup", "fed_server.json")) as f:
            srv_target = json.load(f)["servers"][0]["service"]["target"]

        for name in ("site-1", "site-2"):
            with open(os.path.join(provisioned[name], "startup", "fed_client.json")) as f:
                cli_target = json.load(f)["servers"][0]["service"]["target"]
            assert cli_target == srv_target, f"{name} target {cli_target!r} != server target {srv_target!r}"

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

    def test_all_participants_land_in_same_provision_version_dir(self, provisioned):
        """Distributed packages signed by the same CA/version land in one prod_00 directory."""
        prod_dirs = {os.path.dirname(provisioned[name]) for name, _ in _PARTICIPANTS}
        assert len(prod_dirs) == 1
        prod_dir = prod_dirs.pop()
        assert os.path.basename(prod_dir) == "prod_00"
        assert os.path.isdir(os.path.join(prod_dir, _SERVER_NAME))


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
@_TCP_BIND_SKIP
class TestDistributedProvisioningE2E:
    """End-to-end: packaged kits start server/clients and run hello-numpy-sag."""

    @pytest.fixture()
    def admin_api(self, provisioned, tmp_path):
        """Start server + 2 clients from packaged kits; yield a logged-in Session."""
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
            api.close()
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
        from tests.integration_test.src.utils import check_job_done, ensure_admin_api_logged_in, get_job_meta

        assert ensure_admin_api_logged_in(admin_api, timeout=60), "Admin API did not log in within 60 s"

        job_dir = os.path.join(INTEGRATION_TEST_ROOT, "data", "jobs", "hello-numpy-sag")
        job_id = admin_api.submit_job(job_dir)
        assert isinstance(job_id, str) and job_id, f"Job submit failed: {job_id!r}"

        # Poll until done; log status every 30 s so a timeout is diagnosable.
        deadline = time.time() + _JOB_TIMEOUT
        last_log = time.time()
        while time.time() < deadline:
            if check_job_done(job_id, admin_api):
                break
            if time.time() - last_log >= 30:
                try:
                    srv = admin_api.get_system_info()
                    cli = admin_api.get_client_job_status()
                except Exception as e:
                    srv = cli = f"<error: {e}>"
                print(f"\n[E2E poll] job={job_id}  server={srv}  clients={cli}")
                last_log = time.time()
            time.sleep(5)
        else:
            try:
                srv = admin_api.get_system_info()
                cli = admin_api.get_client_job_status()
            except Exception as e:
                srv = cli = f"<error: {e}>"
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


# ---------------------------------------------------------------------------
# Kit parity test: centralized (`nvflare provision`) vs distributed
# ---------------------------------------------------------------------------

# Participants shared by both workflows. The server name is "localhost" so
# distributed package output aligns with the provision-generated fed_server.json
# target field.
_PARITY_PARTICIPANTS = [
    ("localhost", "server"),
    ("site-1", "client"),
    ("site-2", "client"),
    ("admin@myfl.com", "lead"),
]
_PARITY_PROJECT = "myfl"
_PARITY_FED_PORT = 8002
_PARITY_ADMIN_PORT = 8003

# Files that are cert/key material — content will differ between runs (different CAs).
_BINARY_CERT_FILES = {"server.crt", "client.crt", "server.key", "client.key", "rootCA.pem"}

# Intended design differences between centralized and distributed kits:
#   • signature.json — generated by SignatureBuilder for CC/HE centralized kits;
#     not generated for non-CC/non-HE kits in either workflow (design doc §Change 1).
#     The project dict below omits SignatureBuilder so both produce identical file sets.
_KNOWN_ABSENT_IN_DISTRIBUTED: set = set()


def _run_centralized_provision(workspace: str) -> dict:
    """Run nvflare provision with the standard non-CC/non-HE builder pipeline.

    Returns a dict mapping participant name → kit directory.
    """
    project_dict = {
        "api_version": 3,
        "name": _PARITY_PROJECT,
        "description": "",
        "participants": [
            {
                "name": "localhost",
                "type": "server",
                "org": "myorg",
                "fed_learn_port": _PARITY_FED_PORT,
                "admin_port": _PARITY_ADMIN_PORT,
            },
            {"name": "site-1", "type": "client", "org": "myorg"},
            {"name": "site-2", "type": "client", "org": "myorg"},
            {"name": "admin@myfl.com", "type": "admin", "org": "myorg", "role": "lead"},
        ],
        # Standard non-CC/non-HE builders — no SignatureBuilder so file sets are identical.
        "builders": [
            {"path": "nvflare.lighter.impl.workspace.WorkspaceBuilder"},
            {"path": "nvflare.lighter.impl.cert.CertBuilder"},
            {"path": "nvflare.lighter.impl.static_file.StaticFileBuilder", "args": {"scheme": "grpc"}},
        ],
    }
    args = types.SimpleNamespace(gen_scripts=True)
    ctx = provision(args, project_dict, project_full_path="", workspace_full_path=workspace)
    prod_dir = ctx["current_prod_dir"]
    kit_dirs = {}
    for name, _ in _PARITY_PARTICIPANTS:
        kit_dirs[name] = os.path.join(prod_dir, name)
    return kit_dirs


def _run_distributed_provision(workspace: str) -> dict:
    """Run the full cert init → request → approve → package workflow.

    Returns a dict mapping participant name → kit directory.
    """
    ca_dir = os.path.join(workspace, "ca")
    request_root = os.path.join(workspace, "requests")
    signed_dir = os.path.join(workspace, "signed")
    kits_ws = os.path.join(workspace, "kits")
    profile = os.path.join(workspace, "project_profile.yaml")

    _write_project_profile(profile, _PARITY_PROJECT)
    handle_cert_init(_ns(profile=profile, output_dir=ca_dir))

    request_dirs = {}
    request_zips = {}
    signed_zips = {}
    for name, cert_type in _PARITY_PARTICIPANTS:
        request_dirs[name], request_zips[name] = _run_request(name, cert_type, _PARITY_PROJECT, request_root)

    for name, _ in _PARITY_PARTICIPANTS:
        signed_zips[name] = _run_approve(name, request_zips[name], ca_dir, signed_dir, profile)

    kit_dirs = {}
    for name, _ in _PARITY_PARTICIPANTS:
        result = _run_package(
            signed_zip=signed_zips[name],
            request_dir=request_dirs[name],
            workspace=kits_ws,
        )
        kit_dirs[name] = result["output_dir"]

    return kit_dirs


def _kit_relative_files(kit_dir: str) -> set:
    """Return all file paths relative to kit_dir."""
    result = set()
    for root, _, files in os.walk(kit_dir):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), kit_dir)
            result.add(rel)
    return result


def _json_keys_recursive(obj, prefix=""):
    """Return a flat set of dotted key paths for all leaves in a JSON object."""
    keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            keys.add(path)
            keys |= _json_keys_recursive(v, path)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            keys |= _json_keys_recursive(v, f"{prefix}[{i}]")
    return keys


@pytest.fixture(scope="module")
def parity_kits(tmp_path_factory):
    """Return (centralized_kit_dirs, distributed_kit_dirs)."""
    base = str(tmp_path_factory.mktemp("parity"))
    centralized = _run_centralized_provision(os.path.join(base, "centralized"))
    distributed = _run_distributed_provision(os.path.join(base, "distributed"))
    return centralized, distributed


class TestKitParity:
    """Assert centralized and distributed startup kits are structurally identical.

    Intended differences (none for standard non-CC/non-HE):
      - Cert/key binary content always differs (different CAs and key pairs).
      - signature.json: absent in both when SignatureBuilder is not in the builder list.
    """

    @pytest.mark.parametrize("name,_", _PARITY_PARTICIPANTS)
    def test_file_set_identical(self, name, _, parity_kits):
        """Every file present in the centralized kit must also be in the distributed kit."""
        centralized, distributed = parity_kits
        c_files = _kit_relative_files(centralized[name]) - _BINARY_CERT_FILES
        d_files = _kit_relative_files(distributed[name]) - _BINARY_CERT_FILES
        only_in_centralized = c_files - d_files - _KNOWN_ABSENT_IN_DISTRIBUTED
        only_in_distributed = d_files - c_files
        assert not only_in_centralized, f"{name}: files in centralized but not distributed: {only_in_centralized}"
        assert not only_in_distributed, f"{name}: files in distributed but not centralized: {only_in_distributed}"

    def test_fed_server_json_keys_match(self, parity_kits):
        """fed_server.json must have the same JSON key structure in both kits."""
        centralized, distributed = parity_kits
        with open(os.path.join(centralized["localhost"], "startup", "fed_server.json")) as f:
            c = json.load(f)
        with open(os.path.join(distributed["localhost"], "startup", "fed_server.json")) as f:
            d = json.load(f)
        assert _json_keys_recursive(c) == _json_keys_recursive(d)

    def test_fed_client_json_keys_match(self, parity_kits):
        """fed_client.json must have the same JSON key structure in both kits."""
        centralized, distributed = parity_kits
        with open(os.path.join(centralized["site-1"], "startup", "fed_client.json")) as f:
            c = json.load(f)
        with open(os.path.join(distributed["site-1"], "startup", "fed_client.json")) as f:
            d = json.load(f)
        assert _json_keys_recursive(c) == _json_keys_recursive(d)

    def test_fed_admin_json_keys_match(self, parity_kits):
        """fed_admin.json must have the same JSON key structure in both kits."""
        centralized, distributed = parity_kits
        with open(os.path.join(centralized["admin@myfl.com"], "startup", "fed_admin.json")) as f:
            c = json.load(f)
        with open(os.path.join(distributed["admin@myfl.com"], "startup", "fed_admin.json")) as f:
            d = json.load(f)
        assert _json_keys_recursive(c) == _json_keys_recursive(d)

    def test_fed_server_json_endpoint_values_match(self, parity_kits):
        """fed_server.json target and scheme must be identical."""
        centralized, distributed = parity_kits
        with open(os.path.join(centralized["localhost"], "startup", "fed_server.json")) as f:
            c = json.load(f)
        with open(os.path.join(distributed["localhost"], "startup", "fed_server.json")) as f:
            d = json.load(f)
        assert c["servers"][0]["service"]["target"] == d["servers"][0]["service"]["target"]
        assert c["servers"][0]["service"]["scheme"] == d["servers"][0]["service"]["scheme"]
        assert c["servers"][0]["admin_port"] == d["servers"][0]["admin_port"]

    def test_fed_client_target_matches_server(self, parity_kits):
        """fed_client.json target must match fed_server.json in both workflows."""
        for kit_dirs in parity_kits:
            with open(os.path.join(kit_dirs["localhost"], "startup", "fed_server.json")) as f:
                srv_target = json.load(f)["servers"][0]["service"]["target"]
            with open(os.path.join(kit_dirs["site-1"], "startup", "fed_client.json")) as f:
                cli_target = json.load(f)["servers"][0]["service"]["target"]
            assert cli_target == srv_target

    def test_cert_key_content_differs(self, parity_kits):
        """Cert and key files must differ — each workflow uses its own CA."""
        centralized, distributed = parity_kits
        for cert_file in ("startup/server.crt", "startup/server.key"):
            c_bytes = open(os.path.join(centralized["localhost"], cert_file), "rb").read()
            d_bytes = open(os.path.join(distributed["localhost"], cert_file), "rb").read()
            assert c_bytes != d_bytes, f"{cert_file} is identical — kits may have shared keys"
