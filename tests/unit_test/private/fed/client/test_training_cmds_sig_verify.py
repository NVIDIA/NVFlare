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

"""Unit tests for DeployProcessor.process() signature verification in training_cmds.py.

Tests cover the client-side verification logic that replaced the old secure_train-based check.
"""

import os
import types
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.job_def import JobMetaKey
from nvflare.fuel.sec.job_trust import JOB_AUTHORIZATION_CLOCK_SKEW, JOB_AUTHORIZATION_META_KEY
from nvflare.private.admin_defs import Message
from nvflare.private.defs import RequestHeader
from nvflare.private.fed.client.client_engine_internal_spec import ClientEngineInternalSpec
from nvflare.private.fed.client.training_cmds import DeployProcessor

# ---------------------------------------------------------------------------
# Minimal concrete stub that satisfies ClientEngineInternalSpec isinstance check
# ---------------------------------------------------------------------------


class _StubEngine(ClientEngineInternalSpec):
    """Minimal concrete implementation of ClientEngineInternalSpec for tests."""

    def __init__(self, workspace_dir="/fake/workspace", client_name="site-1", deploy_result=""):
        self._client_name = client_name
        self._workspace_dir = workspace_dir
        self._deploy_result = deploy_result
        self._deploy_calls = []

        # args mock
        self.args = types.SimpleNamespace(workspace=workspace_dir, set=[])

    # --- Required abstract implementations ---

    def get_engine_status(self):
        return {}

    def get_client_name(self) -> str:
        return self._client_name

    def deploy_app(self, app_name, job_id, job_meta, client_name, app_data) -> str:
        self._deploy_calls.append((app_name, job_id, job_meta, client_name, app_data))
        return self._deploy_result

    def start_app(self, job_id, job_meta, allocated_resource=None, token=None, resource_manager=None) -> str:
        return ""

    def notify_job_status(self, job_id, job_status):
        pass

    def abort_app(self, job_id: str) -> str:
        return ""

    def abort_task(self, job_id: str) -> str:
        return ""

    def shutdown(self) -> str:
        return ""

    def restart(self) -> str:
        return ""

    def delete_run(self, job_id: str) -> str:
        return ""

    def get_all_job_ids(self):
        return []

    def add_component(self, component_id, component):
        pass

    def get_component(self, component_id):
        return None

    def get_client_engine(self):
        return self

    def configure_job_log(self, job_id, log_config):
        return ""

    # Optional methods that may be needed by parent classes
    def new_context(self):
        return MagicMock()

    def fire_event(self, event_type, fl_ctx):
        pass

    def get_workspace(self):
        return MagicMock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(job_id="job-1", app_name="test-app", job_meta=None):
    """Build a minimal deploy Message.

    job_meta defaults to a sentinel dict with one key so it passes the "if not job_meta" check.
    """
    req = MagicMock(spec=Message)
    req.topic = "deploy"
    req.body = b"app_data"

    if job_meta is None:
        job_meta = _job_meta_with_server_authorization(app_name)

    def get_header(key, default=None):
        mapping = {
            RequestHeader.JOB_ID: job_id,
            RequestHeader.APP_NAME: app_name,
            RequestHeader.JOB_META: job_meta,
        }
        return mapping.get(key, default)

    req.get_header.side_effect = get_header
    return req


def _job_meta_with_server_authorization(app_name="test-app"):
    return {
        JOB_AUTHORIZATION_META_KEY: {
            app_name: {
                "manifest": {"schema_version": 1},
                "signature": "sig",
                "server_cert": "cert",
            }
        }
    }


def _run_process(req, engine, app_dir, startup_config=None, server_host=None):
    """Run DeployProcessor.process() with Workspace mocked."""
    processor = DeployProcessor()
    workspace_mock = MagicMock()
    workspace_mock.get_app_dir.return_value = app_dir
    workspace_mock.get_startup_kit_dir.return_value = os.path.dirname(app_dir)
    if startup_config is None:
        startup_config = {"servers": [{"identity": "server"}]}

    with (
        patch("nvflare.private.fed.client.training_cmds.Workspace", return_value=workspace_mock),
        patch("nvflare.private.fed.client.training_cmds.ConfigService.get_section", return_value=startup_config),
        patch("nvflare.private.fed.client.training_cmds.get_scope_property", return_value=server_host),
    ):
        return processor.process(req, engine)


# ---------------------------------------------------------------------------
# Signed, valid — should return ok_reply
# ---------------------------------------------------------------------------


class TestSignedValid:
    def test_signed_valid_returns_ok(self, tmp_path):
        """.__nvfl_sig.json present + verify_folder_signature returns True → ok_reply."""
        sig_file = tmp_path / ".__nvfl_sig.json"
        sig_file.write_text('{"sig": "valid"}')

        req = _make_request()
        engine = _StubEngine(workspace_dir=str(tmp_path))

        with (
            patch("nvflare.private.fed.client.training_cmds.verify_job_authorization"),
            patch("nvflare.private.fed.client.training_cmds.verify_folder_signature", return_value=True),
        ):
            reply = _run_process(req, engine, str(tmp_path))

        # deploy_app must have been called (no early return)
        assert len(engine._deploy_calls) == 1
        assert "deployed" in reply.body

    def test_signed_valid_calls_deploy_app(self, tmp_path):
        """Valid signature → engine.deploy_app is invoked."""
        sig_file = tmp_path / ".__nvfl_sig.json"
        sig_file.write_text('{"sig": "valid"}')

        req = _make_request()
        engine = _StubEngine(workspace_dir=str(tmp_path))

        with (
            patch("nvflare.private.fed.client.training_cmds.verify_job_authorization"),
            patch("nvflare.private.fed.client.training_cmds.verify_folder_signature", return_value=True),
        ):
            _run_process(req, engine, str(tmp_path))

        assert len(engine._deploy_calls) == 1


# ---------------------------------------------------------------------------
# Signed, invalid (tampered) — should return error_reply
# ---------------------------------------------------------------------------


class TestSignedInvalid:
    def test_tampered_sig_returns_error(self, tmp_path):
        """.__nvfl_sig.json present + verify_folder_signature returns False → error_reply."""
        sig_file = tmp_path / ".__nvfl_sig.json"
        sig_file.write_text('{"sig": "tampered"}')

        req = _make_request(app_name="my-app")
        engine = _StubEngine(workspace_dir=str(tmp_path))

        with (
            patch("nvflare.private.fed.client.training_cmds.verify_job_authorization"),
            patch("nvflare.private.fed.client.training_cmds.verify_folder_signature", return_value=False),
        ):
            reply = _run_process(req, engine, str(tmp_path))

        assert "does not pass signature verification" in reply.body

    def test_tampered_sig_does_not_call_deploy_app(self, tmp_path):
        """When sig is invalid, engine.deploy_app must NOT be called."""
        sig_file = tmp_path / ".__nvfl_sig.json"
        sig_file.write_text('{"sig": "bad"}')

        req = _make_request()
        engine = _StubEngine(workspace_dir=str(tmp_path))

        with (
            patch("nvflare.private.fed.client.training_cmds.verify_job_authorization"),
            patch("nvflare.private.fed.client.training_cmds.verify_folder_signature", return_value=False),
        ):
            _run_process(req, engine, str(tmp_path))

        assert len(engine._deploy_calls) == 0


# ---------------------------------------------------------------------------
# No legacy admin signature — server authorization is still required
# ---------------------------------------------------------------------------


class TestUnsignedJob:
    def test_unsigned_job_succeeds(self, tmp_path):
        """No .__nvfl_sig.json + valid server authorization -> ok_reply."""
        req = _make_request()
        engine = _StubEngine(workspace_dir=str(tmp_path))

        with patch("nvflare.private.fed.client.training_cmds.verify_job_authorization"):
            reply = _run_process(req, engine, str(tmp_path))

        assert "deployed" in reply.body

    def test_unsigned_job_calls_deploy_app(self, tmp_path):
        """No .__nvfl_sig.json + valid server authorization -> engine.deploy_app is called."""
        req = _make_request()
        engine = _StubEngine(workspace_dir=str(tmp_path))

        with patch("nvflare.private.fed.client.training_cmds.verify_job_authorization"):
            _run_process(req, engine, str(tmp_path))

        assert len(engine._deploy_calls) == 1

    def test_verify_folder_signature_not_called_for_unsigned(self, tmp_path):
        """No .__nvfl_sig.json → verify_folder_signature must not be called."""
        req = _make_request()
        engine = _StubEngine(workspace_dir=str(tmp_path))

        with (
            patch("nvflare.private.fed.client.training_cmds.verify_job_authorization"),
            patch("nvflare.private.fed.client.training_cmds.verify_folder_signature", return_value=True) as mock_vfs,
        ):
            _run_process(req, engine, str(tmp_path))

        mock_vfs.assert_not_called()


# ---------------------------------------------------------------------------
# Server-signed authorization — validates OIDC/private-key-free job trust
# ---------------------------------------------------------------------------


class TestServerJobAuthorization:
    def test_server_authorized_job_verifies_before_deploy(self, tmp_path):
        job_meta = {
            JOB_AUTHORIZATION_META_KEY: {
                "test-app": {
                    "manifest": {"schema_version": 1},
                    "signature": "sig",
                    "server_cert": "cert",
                }
            }
        }
        req = _make_request(job_meta=job_meta)
        engine = _StubEngine(workspace_dir=str(tmp_path), client_name="site-1")

        with patch("nvflare.private.fed.client.training_cmds.verify_job_authorization") as mock_verify:
            reply = _run_process(req, engine, str(tmp_path))

        assert "deployed" in reply.body
        assert len(engine._deploy_calls) == 1
        mock_verify.assert_called_once()
        assert mock_verify.call_args.kwargs["site_name"] == "site-1"
        assert mock_verify.call_args.kwargs["expected_app_name"] == "test-app"
        assert "expected_server_identity" in mock_verify.call_args.kwargs

    def test_bad_server_authorization_rejects_before_deploy(self, tmp_path):
        job_meta = {
            JOB_AUTHORIZATION_META_KEY: {
                "test-app": {
                    "manifest": {"schema_version": 1},
                    "signature": "sig",
                    "server_cert": "cert",
                }
            }
        }
        req = _make_request(job_meta=job_meta)
        engine = _StubEngine(workspace_dir=str(tmp_path))

        with patch("nvflare.private.fed.client.training_cmds.verify_job_authorization", side_effect=ValueError("bad")):
            reply = _run_process(req, engine, str(tmp_path))

        assert "does not pass server authorization verification" in reply.body
        assert len(engine._deploy_calls) == 0

    def test_missing_server_authorization_rejects_before_deploy(self, tmp_path):
        """Server-signed job authorization is required for every non-hub deploy (no knob)."""
        req = _make_request(job_meta={"_test": "stub"})
        engine = _StubEngine(workspace_dir=str(tmp_path))

        with patch("nvflare.private.fed.client.training_cmds.verify_job_authorization") as mock_verify:
            reply = _run_process(req, engine, str(tmp_path))

        assert "missing server job authorization" in reply.body
        assert len(engine._deploy_calls) == 0
        mock_verify.assert_not_called()

    def test_missing_expected_server_identity_rejects_before_deploy(self, tmp_path):
        """No 'identity' in config and no registered server host -> verification cannot proceed."""
        req = _make_request()
        engine = _StubEngine(workspace_dir=str(tmp_path))

        reply = _run_process(req, engine, str(tmp_path), startup_config={})

        assert "expected server identity is not configured" in reply.body
        assert len(engine._deploy_calls) == 0

    def test_expected_server_identity_falls_back_to_server_host(self, tmp_path):
        """Old kits (<2.6) have no 'identity' in the server config; the registered server host is used.

        The 'name' field must NOT be used: it holds the project name in old kits.
        """
        req = _make_request()
        engine = _StubEngine(workspace_dir=str(tmp_path))
        startup_config = {"servers": [{"name": "old-project-name"}]}

        with patch("nvflare.private.fed.client.training_cmds.verify_job_authorization") as mock_verify:
            reply = _run_process(req, engine, str(tmp_path), startup_config=startup_config, server_host="server.org")

        assert "deployed" in reply.body
        mock_verify.assert_called_once()
        assert mock_verify.call_args.kwargs["expected_server_identity"] == "server.org"

    def test_expected_server_identity_prefers_configured_identity(self, tmp_path):
        req = _make_request()
        engine = _StubEngine(workspace_dir=str(tmp_path))
        startup_config = {"servers": [{"name": "project", "identity": "server.example.com"}]}

        with patch("nvflare.private.fed.client.training_cmds.verify_job_authorization") as mock_verify:
            reply = _run_process(req, engine, str(tmp_path), startup_config=startup_config, server_host="other-host")

        assert "deployed" in reply.body
        assert mock_verify.call_args.kwargs["expected_server_identity"] == "server.example.com"


# ---------------------------------------------------------------------------
# from_hub_site — verification block entirely skipped
# ---------------------------------------------------------------------------


class TestFromHubSite:
    def test_from_hub_site_skips_verification(self, tmp_path):
        """from_hub_site=True → verification block skipped entirely → ok_reply."""
        job_meta = {JobMetaKey.FROM_HUB_SITE.value: "hub-1"}
        req = _make_request(job_meta=job_meta)
        engine = _StubEngine(workspace_dir=str(tmp_path))

        # Even if a sig file exists, it should not be checked
        sig_file = tmp_path / ".__nvfl_sig.json"
        sig_file.write_text('{"sig": "doesnotmatter"}')

        with patch("nvflare.private.fed.client.training_cmds.verify_folder_signature", return_value=False) as mock_vfs:
            reply = _run_process(req, engine, str(tmp_path))

        mock_vfs.assert_not_called()
        assert "deployed" in reply.body

    def test_from_hub_site_no_sig_file_succeeds(self, tmp_path):
        """from_hub_site=True, no sig file → deploy succeeds without calling verify."""
        job_meta = {JobMetaKey.FROM_HUB_SITE.value: "hub-1"}
        req = _make_request(job_meta=job_meta)
        engine = _StubEngine(workspace_dir=str(tmp_path))

        with patch("nvflare.private.fed.client.training_cmds.verify_folder_signature") as mock_vfs:
            reply = _run_process(req, engine, str(tmp_path))

        mock_vfs.assert_not_called()
        assert len(engine._deploy_calls) == 1


# ---------------------------------------------------------------------------
# server job authorization is mandatory for non-hub deploys
# ---------------------------------------------------------------------------


class TestServerJobAuthorizationRequired:
    def test_with_manifest_verifies_and_deploys(self, tmp_path):
        """A valid manifest → normal verify + deploy."""
        req = _make_request()
        engine = _StubEngine(workspace_dir=str(tmp_path))

        with patch("nvflare.private.fed.client.training_cmds.verify_job_authorization") as mock_verify:
            reply = _run_process(req, engine, str(tmp_path))

        assert "deployed" in reply.body
        assert len(engine._deploy_calls) == 1
        mock_verify.assert_called_once()

    def test_missing_manifest_rejected_even_if_config_opts_out(self, tmp_path):
        """The requirement is not a config knob: no fed_client.json key can disable it."""
        startup_config = {"servers": [{"identity": "server"}], "require_server_job_authorization": False}
        req = _make_request(job_meta={"_test": "stub"})
        engine = _StubEngine(workspace_dir=str(tmp_path))

        reply = _run_process(req, engine, str(tmp_path), startup_config=startup_config)

        assert "missing server job authorization" in reply.body
        assert len(engine._deploy_calls) == 0

    def test_hub_site_exempt(self, tmp_path):
        """from_hub_site deploys are a separate trust boundary and skip the manifest requirement."""
        job_meta = {JobMetaKey.FROM_HUB_SITE.value: "hub-1"}
        req = _make_request(job_meta=job_meta)
        engine = _StubEngine(workspace_dir=str(tmp_path))

        reply = _run_process(req, engine, str(tmp_path))

        assert "deployed" in reply.body
        assert len(engine._deploy_calls) == 1


# ---------------------------------------------------------------------------
# job_authorization_clock_skew — client-side configurable skew
# ---------------------------------------------------------------------------


class TestJobAuthorizationClockSkew:
    def test_configured_skew_is_passed_to_verify(self, tmp_path):
        startup_config = {"servers": [{"identity": "server"}], "job_authorization_clock_skew": 30}
        req = _make_request()
        engine = _StubEngine(workspace_dir=str(tmp_path))

        with patch("nvflare.private.fed.client.training_cmds.verify_job_authorization") as mock_verify:
            reply = _run_process(req, engine, str(tmp_path), startup_config=startup_config)

        assert "deployed" in reply.body
        assert mock_verify.call_args.kwargs["clock_skew"] == 30.0

    def test_absent_skew_uses_default(self, tmp_path):
        req = _make_request()
        engine = _StubEngine(workspace_dir=str(tmp_path))

        with patch("nvflare.private.fed.client.training_cmds.verify_job_authorization") as mock_verify:
            _run_process(req, engine, str(tmp_path))

        assert mock_verify.call_args.kwargs["clock_skew"] == JOB_AUTHORIZATION_CLOCK_SKEW

    @pytest.mark.parametrize("bad_value", ["abc", -5, 0, None, [10]])
    def test_invalid_skew_warns_and_uses_default(self, tmp_path, bad_value):
        startup_config = {"servers": [{"identity": "server"}], "job_authorization_clock_skew": bad_value}
        req = _make_request()
        engine = _StubEngine(workspace_dir=str(tmp_path))

        with (
            patch("nvflare.private.fed.client.training_cmds.verify_job_authorization") as mock_verify,
            patch("nvflare.private.fed.client.training_cmds.logger.warning") as mock_warning,
        ):
            _run_process(req, engine, str(tmp_path), startup_config=startup_config)

        assert mock_verify.call_args.kwargs["clock_skew"] == JOB_AUTHORIZATION_CLOCK_SKEW
        if bad_value is not None:
            mock_warning.assert_called_once()
            assert "job_authorization_clock_skew" in mock_warning.call_args.args[0]
        else:
            mock_warning.assert_not_called()
