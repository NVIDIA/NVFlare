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

"""Unit tests for JobRunner._deploy_job() — focusing on timeout/failure classification
and the min_sites / required_sites abort logic.

The test infrastructure stubs out all engine/fl_ctx interaction so that only
_deploy_job()'s own logic is exercised."""

import os
from unittest.mock import MagicMock, patch

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.x509.oid import NameOID

from nvflare.apis.client import Client
from nvflare.apis.fl_constant import FLContextKey, SecureTrainConst, SiteType
from nvflare.apis.job_def import Job
from nvflare.apis.workspace import Workspace
from nvflare.fuel.f3.drivers.net_utils import get_cert_job_id
from nvflare.lighter.constants import CertExtensionOID, ProvFileName
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key
from nvflare.private.admin_defs import Message, MsgHeader, ReturnCode
from nvflare.private.defs import RequestHeader
from nvflare.private.fed.server.job_runner import JobRunner
from nvflare.private.fed.utils.job_cert_utils import job_cert_paths, read_job_cert, unpack_job_cert_header

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_reply():
    """Simulate a successful deployment ACK."""
    msg = Message(topic="reply", body="ok")
    msg.set_header(MsgHeader.RETURN_CODE, ReturnCode.OK)
    return msg


def _error_reply(body="deploy failed"):
    """Simulate an explicit error ACK."""
    msg = Message(topic="reply", body=body)
    msg.set_header(MsgHeader.RETURN_CODE, ReturnCode.ERROR)
    return msg


def _build_fl_ctx(token_to_reply: dict, job_id="job-1", min_sites=None, required_sites=None):
    """Build a minimal fl_ctx / engine mock for _deploy_job().

    Args:
        token_to_reply: mapping of client_token -> Message|None
                        None simulates a deployment timeout for that client.
        min_sites: job.min_sites value
        required_sites: job.required_sites list (or None)

    Returns:
        (runner, fl_ctx, job, client_sites)
    """
    runner = JobRunner(workspace_root="/tmp")
    runner.log_info = MagicMock()
    runner.log_warning = MagicMock()
    runner.fire_event = MagicMock()

    # Build client objects matching token_to_reply keys
    client_objects = []
    sites = {}
    for i, token in enumerate(token_to_reply):
        client_name = f"site-{i + 1}"
        c = MagicMock(spec=Client)
        c.token = token
        c.name = client_name
        client_objects.append(c)
        sites[client_name] = MagicMock()

    # Engine
    engine = MagicMock()
    engine.validate_targets.return_value = (client_objects, [])
    engine.get_clients.return_value = client_objects

    # AdminServer
    admin_server = MagicMock()
    admin_server.timeout = 10.0
    admin_server.send_requests_and_get_reply_dict.return_value = token_to_reply
    engine.server.admin_server = admin_server

    # fl_ctx
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = engine
    deploy_detail = []
    fl_ctx.get_prop.return_value = deploy_detail
    fl_ctx.set_prop.side_effect = lambda key, val: deploy_detail.__class__  # no-op for other props

    # Job
    job = MagicMock(spec=Job)
    job.job_id = job_id
    job.meta = {}
    job.min_sites = min_sites
    job.required_sites = required_sites or []

    # Simulate a single app deployment to all client sites
    deployment = {"app": list(sites.keys())}
    job.get_deployment.return_value = deployment
    job.get_application.return_value = b"app_data"

    return runner, fl_ctx, engine, job, sites


def test_secure_deploy_fails_without_job_ca(tmp_path):
    runner, fl_ctx, engine, job, sites = _build_fl_ctx({"tok-1": _ok_reply()})
    (tmp_path / "startup").mkdir()
    (tmp_path / "local").mkdir()
    runner.workspace_root = str(tmp_path)
    deploy_detail = fl_ctx.get_prop.return_value
    fl_ctx.get_prop.side_effect = lambda key, default=None: True if key == FLContextKey.SECURE_MODE else deploy_detail

    with pytest.raises(RuntimeError, match="no job CA"):
        runner._deploy_job(job, sites, fl_ctx)

    engine.server.admin_server.send_requests_and_get_reply_dict.assert_not_called()


def _write_server_kit_with_job_ca(startup):
    root_key, root_pub = generate_keys()
    root_cert = generate_cert(Identity("rootCA"), Identity("rootCA"), root_key, root_pub, ca=True)
    server_key, server_pub = generate_keys()
    server_cert = generate_cert(Identity("server-cn"), Identity("rootCA"), root_key, server_pub)
    job_ca_key, job_ca_pub = generate_keys()
    marker = x509.UnrecognizedExtension(x509.ObjectIdentifier(CertExtensionOID.JOB_CA_MARKER), b"job_ca")
    job_ca_cert = generate_cert(
        Identity("job_ca"),
        Identity("rootCA"),
        root_key,
        job_ca_pub,
        ca=True,
        ca_path_length=0,
        extra_extensions=[(marker, False)],
    )
    (startup / "rootCA.pem").write_bytes(serialize_cert(root_cert))
    (startup / "server.crt").write_bytes(serialize_cert(server_cert))
    (startup / "server.key").write_bytes(serialize_pri_key(server_key))
    (startup / ProvFileName.JOB_CA_CERT).write_bytes(serialize_cert(job_ca_cert))
    (startup / ProvFileName.JOB_CA_KEY).write_bytes(serialize_pri_key(job_ca_key))


def _common_name(cert):
    return cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value


def test_secure_deploy_issues_server_and_client_job_credentials(tmp_path):
    startup = tmp_path / "startup"
    startup.mkdir()
    (tmp_path / "local").mkdir()
    _write_server_kit_with_job_ca(startup)
    runner, fl_ctx, engine, job, sites = _build_fl_ctx({"tok-1": _ok_reply()})
    runner.workspace_root = str(tmp_path)
    job.get_deployment.return_value = {"app": [SiteType.SERVER, "site-1"]}
    props = {
        FLContextKey.SECURE_MODE: True,
        FLContextKey.SERVER_CONFIG: [{SecureTrainConst.SSL_CERT: str(startup / "server.crt")}],
    }
    fl_ctx.get_prop.side_effect = lambda key, default=None: props.get(key, default)
    fl_ctx.set_prop.side_effect = lambda key, val, **kw: props.__setitem__(key, val)
    deploy_requests = []

    def make_deploy_message(*_args, **_kwargs):
        request = Message(topic="deploy", body=b"app")
        deploy_requests.append(request)
        return request

    with (
        patch.object(runner, "_make_deploy_message", side_effect=make_deploy_message),
        patch("nvflare.private.fed.server.job_runner.AppDeployer") as deployer_cls,
        patch("nvflare.private.fed.server.job_runner.require_signed_jobs", return_value=False),
    ):
        deployer_cls.return_value.deploy.return_value = ""
        job_id, failed_clients = runner._deploy_job(job, sites, fl_ctx)

    assert (job_id, failed_clients) == ("job-1", [])
    assert "server: OK" in props[FLContextKey.JOB_DEPLOY_DETAIL]

    run_dir = Workspace(root_dir=str(tmp_path), site_name=SiteType.SERVER).get_run_dir("job-1")
    sj_cert_pem, sj_key_pem = read_job_cert(run_dir)
    sj_cert = x509.load_pem_x509_certificate(sj_cert_pem)
    assert _common_name(sj_cert) == "server-cn"
    assert get_cert_job_id(sj_cert) == "job-1"
    assert sj_cert_pem.count(b"BEGIN CERTIFICATE") == 2  # leaf + job CA, chains to the root
    assert oct(os.stat(job_cert_paths(run_dir)[1]).st_mode & 0o777) == "0o600"
    assert serialization.load_pem_private_key(sj_key_pem, None).public_key() == sj_cert.public_key()

    [deploy_request] = deploy_requests
    cj_cert_pem, cj_key_pem = unpack_job_cert_header(deploy_request.get_header(RequestHeader.JOB_CERT))
    cj_cert = x509.load_pem_x509_certificate(cj_cert_pem)
    assert _common_name(cj_cert) == "site-1"
    assert get_cert_job_id(cj_cert) == "job-1"
    assert cj_cert.issuer == x509.load_pem_x509_certificate((startup / ProvFileName.JOB_CA_CERT).read_bytes()).subject
    assert cj_cert.public_key() != sj_cert.public_key()
    assert serialization.load_pem_private_key(cj_key_pem, None).public_key() == cj_cert.public_key()
    engine.server.admin_server.send_requests_and_get_reply_dict.assert_called_once()


# ---------------------------------------------------------------------------
# Deployment timeout classified as failure
# ---------------------------------------------------------------------------

_DEPLOY_PATCHES = [
    "nvflare.private.fed.server.job_runner.Workspace",
    "nvflare.private.fed.server.job_runner.AppDeployer",
    "nvflare.private.fed.server.job_runner.verify_folder_signature",
]


def _run_deploy(runner, job, sites, fl_ctx, *, extra_patches=None):
    """Run _deploy_job with the standard set of external dependencies patched out."""
    patches = list(_DEPLOY_PATCHES)
    if extra_patches:
        patches.extend(extra_patches)
    with patch.object(runner, "_make_deploy_message", return_value=MagicMock()):
        with patch(patches[0]), patch(patches[1]), patch(patches[2], return_value=True):
            return runner._deploy_job(job, sites, fl_ctx)


class TestDeployJobTimeoutClassification:
    def test_timeout_reply_counted_as_failed_client(self):
        """A client that returns None (timeout) must appear in failed_clients."""
        token_to_reply = {"token-1": _ok_reply(), "token-2": None}
        runner, fl_ctx, engine, job, sites = _build_fl_ctx(token_to_reply, min_sites=1)

        _, failed = _run_deploy(runner, job, sites, fl_ctx)

        assert "site-2" in failed

    def test_ok_reply_not_in_failed_clients(self):
        """A client that returns OK must not appear in failed_clients."""
        token_to_reply = {"token-1": _ok_reply()}
        runner, fl_ctx, engine, job, sites = _build_fl_ctx(token_to_reply, min_sites=1)

        _, failed = _run_deploy(runner, job, sites, fl_ctx)

        assert failed == []

    def test_explicit_error_reply_counted_as_failed_client(self):
        """An explicit error reply (non-OK return code) must appear in failed_clients."""
        token_to_reply = {"token-1": _ok_reply(), "token-2": _error_reply("disk full")}
        runner, fl_ctx, engine, job, sites = _build_fl_ctx(token_to_reply, min_sites=1)

        _, failed = _run_deploy(runner, job, sites, fl_ctx)

        assert "site-2" in failed

    def test_timeout_recorded_in_deploy_detail(self):
        """Timed-out clients must produce a 'deployment timeout' entry, not 'unknown'."""
        token_to_reply = {"token-1": None}
        runner, fl_ctx, engine, job, sites = _build_fl_ctx(token_to_reply, min_sites=0)

        # Capture the deploy_detail list set on fl_ctx
        captured = {}

        def capture_set_prop(key, val, **kw):
            captured[key] = val

        fl_ctx.set_prop.side_effect = capture_set_prop

        _run_deploy(runner, job, sites, fl_ctx)

        detail = captured.get(FLContextKey.JOB_DEPLOY_DETAIL, [])
        assert any(
            "deployment timeout" in entry for entry in detail
        ), f"Expected 'deployment timeout' in deploy_detail but got: {detail}"
        assert not any("unknown" in entry for entry in detail), f"Old 'unknown' label should not appear; got: {detail}"

    def test_mixed_outcomes_all_correctly_classified(self):
        """OK + error + timeout in one batch: only error and timeout end up in failed_clients."""
        token_to_reply = {
            "token-1": _ok_reply(),
            "token-2": _error_reply("out of memory"),
            "token-3": None,  # timeout
        }
        runner, fl_ctx, engine, job, sites = _build_fl_ctx(token_to_reply, min_sites=1)

        _, failed = _run_deploy(runner, job, sites, fl_ctx)

        assert "site-1" not in failed  # OK → not failed
        assert "site-2" in failed  # explicit error → failed
        assert "site-3" in failed  # timeout → failed


# ---------------------------------------------------------------------------
# min_sites logic with timeouts treated as failures
# ---------------------------------------------------------------------------


class TestDeployJobMinSites:
    def test_timeout_does_not_abort_when_within_min_sites(self):
        """One timeout but two OK; min_sites=2 → 2 ok ≥ 2 → proceed."""
        token_to_reply = {
            "token-1": _ok_reply(),
            "token-2": None,
            "token-3": _ok_reply(),
        }
        runner, fl_ctx, engine, job, sites = _build_fl_ctx(token_to_reply, min_sites=2)

        job_id, failed = _run_deploy(runner, job, sites, fl_ctx)

        assert "site-2" in failed
        assert job_id == "job-1"

    def test_timeout_aborts_when_below_min_sites(self):
        """All clients time out; min_sites=2 → 0 ok < 2 → RuntimeError."""
        token_to_reply = {"token-1": None, "token-2": None}
        runner, fl_ctx, engine, job, sites = _build_fl_ctx(token_to_reply, min_sites=2)

        with pytest.raises(RuntimeError, match="deploy failure"):
            _run_deploy(runner, job, sites, fl_ctx)

    def test_timeout_aborts_below_min_sites_mixed(self):
        """One OK but two fail (1 error + 1 timeout); min_sites=2 → 1 ok < 2 → abort."""
        token_to_reply = {
            "token-1": _ok_reply(),
            "token-2": None,
            "token-3": _error_reply("refused"),
        }
        runner, fl_ctx, engine, job, sites = _build_fl_ctx(token_to_reply, min_sites=2)

        with pytest.raises(RuntimeError, match="deploy failure"):
            _run_deploy(runner, job, sites, fl_ctx)


# ---------------------------------------------------------------------------
# Full startup sequence integration-style test
# ---------------------------------------------------------------------------


class TestDeployAndStartIntegration:
    """Verify the full deploy → start sequence correctly handles timeouts at both phases."""

    @patch("nvflare.private.fed.server.job_runner.check_client_replies")
    @patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=True)
    def test_deploy_timeout_excluded_from_start_run(self, mock_get_bool, mock_check_replies):
        """Clients that time out at deployment are excluded from _start_run's client_sites
        so the start-job phase never sees them."""
        mock_check_replies.return_value = []  # all start-job replies OK

        runner = JobRunner(workspace_root="/tmp")
        runner.log_info = MagicMock()
        runner.log_warning = MagicMock()
        runner.fire_event = MagicMock()

        # Two clients: site-1 OK, site-2 deployment timeout
        client1 = MagicMock(spec=Client)
        client1.token = "token-1"
        client1.name = "site-1"
        client1.to_dict.return_value = {"name": "site-1"}

        client2 = MagicMock(spec=Client)
        client2.token = "token-2"
        client2.name = "site-2"

        engine = MagicMock()
        engine.validate_targets.return_value = ([client1, client2], [])
        engine.get_job_clients.return_value = {"token-1": client1}
        engine.start_app_on_server.return_value = ""
        engine.start_client_job.return_value = [MagicMock()]

        admin_server = MagicMock()
        admin_server.timeout = 10.0
        admin_server.send_requests_and_get_reply_dict.return_value = {
            "token-1": _ok_reply(),
            "token-2": None,  # deployment timeout
        }
        engine.server.admin_server = admin_server

        fl_ctx = MagicMock()
        fl_ctx.get_engine.return_value = engine
        deploy_detail = []
        fl_ctx.get_prop.return_value = deploy_detail

        job = MagicMock(spec=Job)
        job.job_id = "job-e2e"
        job.meta = {}
        job.min_sites = 1
        job.required_sites = []
        job.get_deployment.return_value = {"app": ["site-1", "site-2"]}
        job.get_application.return_value = b"app_data"

        client_sites = {"site-1": MagicMock(), "site-2": MagicMock()}

        with (
            patch.object(runner, "_make_deploy_message", return_value=MagicMock()),
            patch("nvflare.private.fed.server.job_runner.Workspace"),
            patch("nvflare.private.fed.server.job_runner.AppDeployer"),
            patch("nvflare.private.fed.server.job_runner.verify_folder_signature", return_value=True),
        ):
            job_id, failed = runner._deploy_job(job, client_sites, fl_ctx)

        # site-2 must be in failed (deployment timeout)
        assert "site-2" in failed
        # site-1 must not be in failed
        assert "site-1" not in failed

        # In the real run() loop, deployable_clients = client_sites - failed_clients.
        # Verify _start_run actually uses only deployable clients.
        deployable = {k: v for k, v in client_sites.items() if k not in failed}
        assert "site-1" in deployable
        assert "site-2" not in deployable

        runner._start_run(job_id=job_id, job=job, client_sites=deployable, fl_ctx=fl_ctx)

        engine.start_client_job.assert_called_once_with(job, deployable, fl_ctx)
        assert mock_check_replies.call_args.kwargs["client_sites"] == ["site-1"]
