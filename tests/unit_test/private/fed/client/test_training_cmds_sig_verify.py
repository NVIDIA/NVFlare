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

Tests cover client-side verification of the app bytes received for deploy.
"""

import io
import json
import types
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

from nvflare.apis.job_def import JobMetaKey
from nvflare.lighter.tool_consts import NVFLARE_SIG_FILE
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


def _make_app_zip(signed=True) -> bytes:
    zip_bytes = io.BytesIO()
    with ZipFile(zip_bytes, "w") as zf:
        zf.writestr("config/config_fed_client.json", "{}")
        if signed:
            zf.writestr(NVFLARE_SIG_FILE, "{}")
    return zip_bytes.getvalue()


def _make_request(job_id="job-1", app_name="test-app", job_meta=None, body=None):
    """Build a minimal deploy Message.

    job_meta defaults to a sentinel dict with one key so it passes the "if not job_meta" check.
    """
    req = MagicMock(spec=Message)
    req.topic = "deploy"
    req.body = body if body is not None else _make_app_zip()

    if job_meta is None:
        # Non-empty so "if not job_meta" in process() doesn't trigger early return
        job_meta = {"_test": "stub"}

    def get_header(key, default=None):
        mapping = {
            RequestHeader.JOB_ID: job_id,
            RequestHeader.APP_NAME: app_name,
            RequestHeader.JOB_META: job_meta,
        }
        return mapping.get(key, default)

    req.get_header.side_effect = get_header
    return req


def _run_process(req, engine, startup_dir):
    """Run DeployProcessor.process() with Workspace mocked."""
    processor = DeployProcessor()
    workspace_mock = MagicMock()
    workspace_mock.get_startup_kit_dir.return_value = startup_dir

    with patch("nvflare.private.fed.client.training_cmds.Workspace", return_value=workspace_mock):
        return processor.process(req, engine)


def _write_root_ca(startup_dir):
    (startup_dir / "rootCA.pem").write_text("FAKECERT")


def _write_client_policy(startup_dir, require_signed_jobs):
    config = {"require_signed_jobs": require_signed_jobs}
    (startup_dir / "fed_client.json").write_text(json.dumps(config))


# ---------------------------------------------------------------------------
# Signed, valid — should return ok_reply
# ---------------------------------------------------------------------------


class TestSignedValid:
    def test_signed_valid_returns_ok(self, tmp_path):
        """.__nvfl_sig.json present + verify_folder_signature returns True -> ok_reply."""
        req = _make_request()
        engine = _StubEngine(workspace_dir=str(tmp_path))
        _write_root_ca(tmp_path)

        with patch("nvflare.private.fed.client.training_cmds.verify_folder_signature", return_value=True):
            reply = _run_process(req, engine, str(tmp_path))

        # deploy_app must have been called (no early return)
        assert len(engine._deploy_calls) == 1
        assert "deployed" in reply.body

    def test_signed_valid_calls_deploy_app_with_verified_bytes(self, tmp_path):
        """Valid signature -> engine.deploy_app is invoked with the same received bytes."""
        req = _make_request()
        engine = _StubEngine(workspace_dir=str(tmp_path))
        _write_root_ca(tmp_path)

        with patch("nvflare.private.fed.client.training_cmds.verify_folder_signature", return_value=True) as mock_vfs:
            _run_process(req, engine, str(tmp_path))

        mock_vfs.assert_called_once()
        assert len(engine._deploy_calls) == 1
        assert engine._deploy_calls[0][4] == req.body


# ---------------------------------------------------------------------------
# Signed, invalid (tampered) — should return error_reply
# ---------------------------------------------------------------------------


class TestSignedInvalid:
    def test_tampered_sig_returns_error(self, tmp_path):
        """.__nvfl_sig.json present + verify_folder_signature returns False -> error_reply."""
        req = _make_request(app_name="my-app")
        engine = _StubEngine(workspace_dir=str(tmp_path))
        _write_root_ca(tmp_path)

        with patch("nvflare.private.fed.client.training_cmds.verify_folder_signature", return_value=False):
            reply = _run_process(req, engine, str(tmp_path))

        assert "does not pass signature verification" in reply.body

    def test_tampered_sig_does_not_call_deploy_app(self, tmp_path):
        """When sig is invalid, engine.deploy_app must NOT be called."""
        req = _make_request()
        engine = _StubEngine(workspace_dir=str(tmp_path))
        _write_root_ca(tmp_path)

        with patch("nvflare.private.fed.client.training_cmds.verify_folder_signature", return_value=False):
            _run_process(req, engine, str(tmp_path))

        assert len(engine._deploy_calls) == 0

    def test_disabled_policy_does_not_allow_invalid_signature(self, tmp_path):
        """The opt-out permits missing signatures, never invalid signatures."""
        req = _make_request()
        engine = _StubEngine(workspace_dir=str(tmp_path))
        _write_root_ca(tmp_path)
        _write_client_policy(tmp_path, require_signed_jobs=False)

        with patch("nvflare.private.fed.client.training_cmds.verify_folder_signature", return_value=False):
            reply = _run_process(req, engine, str(tmp_path))

        assert "does not pass signature verification" in reply.body
        assert len(engine._deploy_calls) == 0


# ---------------------------------------------------------------------------
# Unsigned — rejected when signing is required, otherwise accepted
# ---------------------------------------------------------------------------


class TestUnsignedJob:
    def test_unsigned_job_calls_deploy_app_when_signing_not_required(self, tmp_path):
        """No .__nvfl_sig.json and no rootCA.pem -> engine.deploy_app is called."""
        req = _make_request(body=_make_app_zip(signed=False))
        engine = _StubEngine(workspace_dir=str(tmp_path))

        _run_process(req, engine, str(tmp_path))

        assert len(engine._deploy_calls) == 1

    def test_unsigned_job_calls_deploy_app_when_client_policy_disabled(self, tmp_path):
        """A PKI client can independently opt in to unsigned deployments."""
        req = _make_request(body=_make_app_zip(signed=False))
        engine = _StubEngine(workspace_dir=str(tmp_path))
        _write_root_ca(tmp_path)
        _write_client_policy(tmp_path, require_signed_jobs=False)

        with patch("nvflare.private.fed.client.training_cmds.verify_folder_signature") as mock_vfs:
            reply = _run_process(req, engine, str(tmp_path))

        assert "deployed" in reply.body
        assert len(engine._deploy_calls) == 1
        mock_vfs.assert_not_called()

    def test_unsigned_job_is_rejected_when_client_policy_enabled(self, tmp_path):
        """An explicit client policy continues to reject unsigned deployments."""
        req = _make_request(body=_make_app_zip(signed=False))
        engine = _StubEngine(workspace_dir=str(tmp_path))
        _write_root_ca(tmp_path)
        _write_client_policy(tmp_path, require_signed_jobs=True)

        reply = _run_process(req, engine, str(tmp_path))

        assert "unsigned job rejected" in reply.body
        assert len(engine._deploy_calls) == 0

    def test_invalid_client_policy_fails_closed(self, tmp_path):
        """A non-boolean client policy must not enable unsigned deployments."""
        req = _make_request(body=_make_app_zip(signed=False))
        engine = _StubEngine(workspace_dir=str(tmp_path))
        _write_root_ca(tmp_path)
        _write_client_policy(tmp_path, require_signed_jobs="false")

        reply = _run_process(req, engine, str(tmp_path))

        assert "unsigned job rejected" in reply.body
        assert len(engine._deploy_calls) == 0

    def test_signed_job_calls_deploy_app_when_root_ca_is_missing(self, tmp_path):
        """No rootCA.pem -> client cannot verify signature and preserves no-rootCA deploy behavior."""
        req = _make_request()
        engine = _StubEngine(workspace_dir=str(tmp_path))

        with patch("nvflare.private.fed.client.training_cmds.verify_folder_signature") as mock_vfs:
            reply = _run_process(req, engine, str(tmp_path))

        assert "deployed" in reply.body
        assert len(engine._deploy_calls) == 1
        mock_vfs.assert_not_called()

    def test_existing_stale_signed_app_does_not_allow_unsigned_new_body(self, tmp_path):
        """Existing signed app dir must not make unsigned received bytes deployable."""
        app_dir = tmp_path / "run_1" / "app"
        app_dir.mkdir(parents=True)
        (app_dir / NVFLARE_SIG_FILE).write_text("{}")
        _write_root_ca(app_dir.parent)
        req = _make_request(body=_make_app_zip(signed=False))
        engine = _StubEngine(workspace_dir=str(tmp_path))

        with patch("nvflare.private.fed.client.training_cmds.verify_folder_signature") as mock_vfs:
            reply = _run_process(req, engine, str(app_dir.parent))

        assert "unsigned job rejected" in reply.body
        assert len(engine._deploy_calls) == 0
        mock_vfs.assert_not_called()

    def test_bad_zip_does_not_call_deploy_app(self, tmp_path):
        """Malformed received app bytes -> error_reply before deploy_app."""
        req = _make_request(body=b"not a zip")
        engine = _StubEngine(workspace_dir=str(tmp_path))

        reply = _run_process(req, engine, str(tmp_path))

        assert "failed to stage app" in reply.body
        assert len(engine._deploy_calls) == 0


# ---------------------------------------------------------------------------
# from_hub_site — verification block entirely skipped
# ---------------------------------------------------------------------------


class TestFromHubSite:
    def test_from_hub_site_skips_verification(self, tmp_path):
        """from_hub_site=True -> verification block skipped entirely -> ok_reply."""
        job_meta = {JobMetaKey.FROM_HUB_SITE.value: "hub-1"}
        req = _make_request(job_meta=job_meta, body=b"hub payload")
        engine = _StubEngine(workspace_dir=str(tmp_path))

        with (
            patch("nvflare.private.fed.client.training_cmds.unzip_all_from_bytes") as mock_unzip,
            patch("nvflare.private.fed.client.training_cmds.verify_folder_signature") as mock_vfs,
        ):
            reply = _run_process(req, engine, str(tmp_path))

        mock_unzip.assert_not_called()
        mock_vfs.assert_not_called()
        assert "deployed" in reply.body
