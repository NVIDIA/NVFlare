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

from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.client import Client
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.job_def import Job
from nvflare.private.admin_defs import Message, MsgHeader, ReturnCode
from nvflare.private.fed.server.job_runner import JobRunner

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

    # Patch server-side app deployment (not exercised here)
    with patch.object(runner, "_make_deploy_message", return_value=MagicMock()):
        pass

    return runner, fl_ctx, engine, job, sites


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
