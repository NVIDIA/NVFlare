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

"""Unit tests for job_runner.py signature verification logic.

Tests cover the _deploy_job server-side signature verification path that replaced
the old secure_train-based logic.
"""

from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.fl_constant import FLContextKey, SiteType
from nvflare.apis.job_def import Job, JobMetaKey
from nvflare.private.admin_defs import Message, MsgHeader, ReturnCode
from nvflare.private.fed.server.job_runner import JobRunner

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _ok_reply():
    msg = Message(topic="reply", body="ok")
    msg.set_header(MsgHeader.RETURN_CODE, ReturnCode.OK)
    return msg


def _build_server_deploy_ctx(job_meta=None, *, app_dir=None, startup_dir=None):
    """Build runner, fl_ctx, engine, job, and workspace for server-only deploy.

    Sets up a job that deploys to the SERVER participant only (no clients).
    """
    runner = JobRunner(workspace_root="/tmp")
    runner.log_info = MagicMock()
    runner.log_warning = MagicMock()
    runner.log_error = MagicMock()
    runner.fire_event = MagicMock()

    if job_meta is None:
        job_meta = {}

    job = MagicMock(spec=Job)
    job.job_id = "test-job-1"
    job.meta = dict(job_meta)
    job.min_sites = None
    job.required_sites = []
    job.get_deployment.return_value = {"app": [SiteType.SERVER]}
    job.get_application.return_value = b"app_data"

    engine = MagicMock()
    engine.get_clients.return_value = []
    engine.validate_targets.return_value = ([], [])
    engine.args.set = []

    admin_server = MagicMock()
    admin_server.timeout = 10.0
    admin_server.send_requests_and_get_reply_dict.return_value = {}
    engine.server.admin_server = admin_server

    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = engine
    deploy_detail = []

    def _get_prop(key, *args, **kwargs):
        if key == FLContextKey.JOB_DEPLOY_DETAIL:
            return deploy_detail
        return None

    def _set_prop(key, val, *args, **kwargs):
        if key == FLContextKey.JOB_DEPLOY_DETAIL:
            deploy_detail.extend(val if isinstance(val, list) else [])

    fl_ctx.get_prop.side_effect = _get_prop
    fl_ctx.set_prop.side_effect = lambda key, val, **kw: None

    # Workspace mock
    workspace_mock = MagicMock()
    workspace_mock.get_app_dir.return_value = app_dir or "/fake/app"
    workspace_mock.get_startup_kit_dir.return_value = startup_dir or "/fake/startup"

    return runner, fl_ctx, engine, job, workspace_mock, deploy_detail


def _run_deploy_with_workspace(runner, job, fl_ctx, workspace_mock):
    """Run _deploy_job with Workspace patched to workspace_mock and AppDeployer returning no error."""
    with (
        patch("nvflare.private.fed.server.job_runner.Workspace", return_value=workspace_mock),
        patch("nvflare.private.fed.server.job_runner.AppDeployer") as mock_deployer_cls,
    ):
        mock_deployer = MagicMock()
        mock_deployer.deploy.return_value = ""  # no error
        mock_deployer_cls.return_value = mock_deployer
        return runner._deploy_job(job, {SiteType.SERVER: MagicMock()}, fl_ctx)


# ---------------------------------------------------------------------------
# Happy path: signed job, signature is valid
# ---------------------------------------------------------------------------


class TestSignedJobValid:
    def test_signed_valid_job_succeeds(self, tmp_path):
        """.__nvfl_sig.json present + verify_folder_signature returns True → deploy succeeds."""
        sig_file = tmp_path / ".__nvfl_sig.json"
        sig_file.write_text('{"sig": "abc"}')

        runner, fl_ctx, engine, job, ws, deploy_detail = _build_server_deploy_ctx(
            app_dir=str(tmp_path), startup_dir=str(tmp_path)
        )

        with patch("nvflare.private.fed.server.job_runner.verify_folder_signature", return_value=True):
            job_id, failed = _run_deploy_with_workspace(runner, job, fl_ctx, ws)

        assert job_id == "test-job-1"
        assert failed == []


# ---------------------------------------------------------------------------
# Signed job, rootCA.pem absent → verify_folder_signature returns False
# ---------------------------------------------------------------------------


class TestSignedJobNoRootCA:
    def test_signed_job_no_root_ca_fails(self, tmp_path):
        """.__nvfl_sig.json present but verify_folder_signature returns False (missing rootCA) → RuntimeError."""
        sig_file = tmp_path / ".__nvfl_sig.json"
        sig_file.write_text('{"sig": "abc"}')

        runner, fl_ctx, engine, job, ws, deploy_detail = _build_server_deploy_ctx(
            app_dir=str(tmp_path), startup_dir=str(tmp_path)
        )

        with patch("nvflare.private.fed.server.job_runner.verify_folder_signature", return_value=False):
            with pytest.raises(RuntimeError, match="Failed to verify app"):
                _run_deploy_with_workspace(runner, job, fl_ctx, ws)


# ---------------------------------------------------------------------------
# Unsigned job, policy on (rootCA.pem present → inferred True)
# ---------------------------------------------------------------------------


class TestUnsignedJobPolicyOn:
    def test_unsigned_job_policy_on_raises(self, tmp_path):
        """No .__nvfl_sig.json, rootCA.pem present → require_signed_jobs inferred True → UNSIGNED_JOB_REJECTED."""
        # Create rootCA.pem in startup dir so _require_signed_jobs returns True
        root_ca = tmp_path / "rootCA.pem"
        root_ca.write_text("FAKECERT")

        runner, fl_ctx, engine, job, ws, deploy_detail = _build_server_deploy_ctx(
            app_dir=str(tmp_path), startup_dir=str(tmp_path)
        )

        with pytest.raises(RuntimeError, match="UNSIGNED_JOB_REJECTED"):
            _run_deploy_with_workspace(runner, job, fl_ctx, ws)

    def test_unsigned_job_error_message_in_deploy_detail(self, tmp_path):
        """The deploy_detail list must contain the rejection reason."""
        root_ca = tmp_path / "rootCA.pem"
        root_ca.write_text("FAKECERT")

        runner, fl_ctx, engine, job, ws, captured_detail = _build_server_deploy_ctx(
            app_dir=str(tmp_path), startup_dir=str(tmp_path)
        )

        # Capture deploy_detail via set_prop
        detail_ref = []
        fl_ctx.set_prop.side_effect = lambda key, val, **kw: None

        # Patch fl_ctx to track the detail list that _deploy_job writes to
        real_detail = []
        fl_ctx.get_prop.side_effect = lambda key, *a, **kw: (
            real_detail if key == FLContextKey.JOB_DEPLOY_DETAIL else None
        )
        fl_ctx.set_prop.side_effect = lambda key, val, *a, **kw: real_detail.__class__  # no-op

        with pytest.raises(RuntimeError, match="UNSIGNED_JOB_REJECTED"):
            _run_deploy_with_workspace(runner, job, fl_ctx, ws)


# ---------------------------------------------------------------------------
# Unsigned job, policy off (explicit config)
# ---------------------------------------------------------------------------


class TestUnsignedJobPolicyOff:
    def test_unsigned_job_policy_off_explicit_succeeds(self, tmp_path):
        """No .__nvfl_sig.json, fed_server.json has require_signed_jobs=false → deploy succeeds."""
        import json

        fed_server = tmp_path / "fed_server.json"
        fed_server.write_text(json.dumps({"require_signed_jobs": False}))

        runner, fl_ctx, engine, job, ws, deploy_detail = _build_server_deploy_ctx(
            app_dir=str(tmp_path), startup_dir=str(tmp_path)
        )

        job_id, failed = _run_deploy_with_workspace(runner, job, fl_ctx, ws)

        assert job_id == "test-job-1"
        assert failed == []


# ---------------------------------------------------------------------------
# Unsigned job, simulator (no rootCA.pem, no explicit config)
# ---------------------------------------------------------------------------


class TestUnsignedJobSimulator:
    def test_unsigned_job_no_root_ca_no_config_succeeds(self, tmp_path):
        """No .__nvfl_sig.json, no rootCA.pem, no config → policy inferred False → deploy succeeds."""
        # tmp_path has neither rootCA.pem nor fed_server.json

        runner, fl_ctx, engine, job, ws, deploy_detail = _build_server_deploy_ctx(
            app_dir=str(tmp_path), startup_dir=str(tmp_path)
        )

        job_id, failed = _run_deploy_with_workspace(runner, job, fl_ctx, ws)

        assert job_id == "test-job-1"
        assert failed == []


# ---------------------------------------------------------------------------
# Tampered signature
# ---------------------------------------------------------------------------


class TestTamperedSignature:
    def test_tampered_signature_raises(self, tmp_path):
        """.__nvfl_sig.json present but content does not match files → verify_folder_signature False → RuntimeError."""
        sig_file = tmp_path / ".__nvfl_sig.json"
        sig_file.write_text('{"sig": "tampered"}')

        runner, fl_ctx, engine, job, ws, deploy_detail = _build_server_deploy_ctx(
            app_dir=str(tmp_path), startup_dir=str(tmp_path)
        )

        with patch("nvflare.private.fed.server.job_runner.verify_folder_signature", return_value=False):
            with pytest.raises(RuntimeError, match="Failed to verify app"):
                _run_deploy_with_workspace(runner, job, fl_ctx, ws)

    def test_tampered_error_message_correct(self, tmp_path):
        """Error message must say 'signature verification failed', not 'UNSIGNED_JOB_REJECTED'."""
        sig_file = tmp_path / ".__nvfl_sig.json"
        sig_file.write_text('{"sig": "tampered"}')

        runner, fl_ctx, engine, job, ws, deploy_detail = _build_server_deploy_ctx(
            app_dir=str(tmp_path), startup_dir=str(tmp_path)
        )

        with patch("nvflare.private.fed.server.job_runner.verify_folder_signature", return_value=False):
            with pytest.raises(RuntimeError) as exc_info:
                _run_deploy_with_workspace(runner, job, fl_ctx, ws)

        assert "signature verification failed" in str(exc_info.value)
        assert "UNSIGNED_JOB_REJECTED" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# from_hub_site bypass
# ---------------------------------------------------------------------------


class TestFromHubSiteBypass:
    def test_from_hub_site_skips_verification_no_sig(self, tmp_path):
        """from_hub_site=True, no .__nvfl_sig.json → verification block skipped → deploy succeeds."""
        job_meta = {JobMetaKey.FROM_HUB_SITE.value: "hub-site-1"}

        runner, fl_ctx, engine, job, ws, deploy_detail = _build_server_deploy_ctx(
            job_meta=job_meta, app_dir=str(tmp_path), startup_dir=str(tmp_path)
        )

        job_id, failed = _run_deploy_with_workspace(runner, job, fl_ctx, ws)

        assert job_id == "test-job-1"

    def test_from_hub_site_skips_verification_even_with_tampered_sig(self, tmp_path):
        """from_hub_site=True with an invalid sig → hub is trusted, block is skipped → deploy succeeds."""
        sig_file = tmp_path / ".__nvfl_sig.json"
        sig_file.write_text('{"sig": "tampered"}')
        job_meta = {JobMetaKey.FROM_HUB_SITE.value: "hub-site-1"}

        runner, fl_ctx, engine, job, ws, deploy_detail = _build_server_deploy_ctx(
            job_meta=job_meta, app_dir=str(tmp_path), startup_dir=str(tmp_path)
        )

        # Even if verify_folder_signature would return False, it should never be called
        with patch("nvflare.private.fed.server.job_runner.verify_folder_signature", return_value=False) as mock_vfs:
            job_id, failed = _run_deploy_with_workspace(runner, job, fl_ctx, ws)

        mock_vfs.assert_not_called()
        assert job_id == "test-job-1"


# ---------------------------------------------------------------------------
# Security regression: bad sig must still fail (not silently accepted)
# ---------------------------------------------------------------------------


class TestSecurityRegression:
    def test_bad_sig_not_silently_accepted(self, tmp_path):
        """Regression: ensure a bad signature never silently passes."""
        sig_file = tmp_path / ".__nvfl_sig.json"
        sig_file.write_text('{"sig": "bad"}')

        runner, fl_ctx, engine, job, ws, deploy_detail = _build_server_deploy_ctx(
            app_dir=str(tmp_path), startup_dir=str(tmp_path)
        )

        with patch("nvflare.private.fed.server.job_runner.verify_folder_signature", return_value=False):
            with pytest.raises(RuntimeError):
                _run_deploy_with_workspace(runner, job, fl_ctx, ws)

    def test_good_sig_not_rejected(self, tmp_path):
        """Regression: a valid signature must not be rejected."""
        sig_file = tmp_path / ".__nvfl_sig.json"
        sig_file.write_text('{"sig": "valid"}')

        runner, fl_ctx, engine, job, ws, deploy_detail = _build_server_deploy_ctx(
            app_dir=str(tmp_path), startup_dir=str(tmp_path)
        )

        with patch("nvflare.private.fed.server.job_runner.verify_folder_signature", return_value=True):
            job_id, _ = _run_deploy_with_workspace(runner, job, fl_ctx, ws)

        assert job_id == "test-job-1"
