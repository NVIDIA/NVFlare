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

from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.job_def import JobMetaKey
from nvflare.private.admin_defs import Message, MsgHeader, ReturnCode
from nvflare.private.fed.server.job_runner import JobRunner
from nvflare.private.fed.server.message_send import ClientReply


def _make_runner_inputs(num_clients=1):
    runner = JobRunner(workspace_root="/tmp")
    runner.log_info = MagicMock()
    runner.log_warning = MagicMock()
    runner.fire_event = MagicMock()

    fl_ctx = MagicMock()
    engine = MagicMock()
    fl_ctx.get_engine.return_value = engine

    client_obj = MagicMock()
    client_obj.to_dict.return_value = {"name": "site-1"}
    engine.get_job_clients.return_value = {"token-1": client_obj}
    engine.start_app_on_server.return_value = ""
    engine.start_client_job.return_value = [MagicMock()]

    job = MagicMock()
    job.job_id = "job-1"
    job.meta = {}
    job.min_sites = 0  # no minimum by default
    job.required_sites = None  # no required sites by default

    client_sites = {"site-1": MagicMock()}
    return runner, fl_ctx, engine, job, client_sites


# ---------------------------------------------------------------------------
# strict flag wiring
# ---------------------------------------------------------------------------


@patch("nvflare.private.fed.server.job_runner.check_client_replies")
@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=False)
def test_start_run_passes_strict_false_when_flag_disabled(mock_get_bool, mock_check_replies):
    mock_check_replies.return_value = []  # no timeouts
    runner, fl_ctx, _engine, job, client_sites = _make_runner_inputs()

    runner._start_run(job_id=job.job_id, job=job, client_sites=client_sites, fl_ctx=fl_ctx)

    mock_get_bool.assert_called_once()
    mock_check_replies.assert_called_once()
    assert mock_check_replies.call_args.kwargs["strict"] is False


@patch("nvflare.private.fed.server.job_runner.check_client_replies")
@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=True)
def test_start_run_passes_strict_true_when_flag_enabled(mock_get_bool, mock_check_replies):
    mock_check_replies.return_value = []  # no timeouts
    runner, fl_ctx, _engine, job, client_sites = _make_runner_inputs()

    runner._start_run(job_id=job.job_id, job=job, client_sites=client_sites, fl_ctx=fl_ctx)

    mock_get_bool.assert_called_once()
    mock_check_replies.assert_called_once()
    assert mock_check_replies.call_args.kwargs["strict"] is True


# ---------------------------------------------------------------------------
# timeout exclusion in _start_run
# ---------------------------------------------------------------------------


@patch("nvflare.private.fed.server.job_runner.check_client_replies")
@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=True)
def test_start_run_proceeds_when_timed_out_clients_within_min_sites(mock_get_bool, mock_check_replies):
    """When some clients time out but active count >= min_sites, job proceeds with a warning."""
    mock_check_replies.return_value = ["site-2"]  # site-2 timed out
    runner, fl_ctx, _engine, job, client_sites = _make_runner_inputs()
    client_sites = {"site-1": MagicMock(), "site-2": MagicMock()}
    job.min_sites = 1  # require at least 1; site-1 is still active

    runner._start_run(job_id=job.job_id, job=job, client_sites=client_sites, fl_ctx=fl_ctx)

    runner.log_warning.assert_called_once()
    warning_msg = runner.log_warning.call_args[0][1]
    assert "site-2" in warning_msg
    assert "timed out" in warning_msg


@patch("nvflare.private.fed.server.job_runner.check_client_replies")
@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=True)
def test_start_run_raises_when_timed_out_clients_breach_min_sites(mock_get_bool, mock_check_replies):
    """When timeouts cause active count to fall below min_sites, _start_run raises."""
    mock_check_replies.return_value = ["site-1", "site-2"]  # both timed out
    runner, fl_ctx, _engine, job, client_sites = _make_runner_inputs()
    client_sites = {"site-1": MagicMock(), "site-2": MagicMock()}
    job.min_sites = 2  # need at least 2; 0 active after timeouts

    with pytest.raises(RuntimeError, match="min_sites"):
        runner._start_run(job_id=job.job_id, job=job, client_sites=client_sites, fl_ctx=fl_ctx)


@patch("nvflare.private.fed.server.job_runner.check_client_replies")
@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=True)
def test_start_run_updates_job_clients_meta_after_timeout_exclusion(mock_get_bool, mock_check_replies):
    mock_check_replies.return_value = ["site-2"]
    runner, fl_ctx, engine, job, _client_sites = _make_runner_inputs()

    site1 = MagicMock()
    site1.name = "site-1"
    site1.to_dict.return_value = {"name": "site-1"}

    site2 = MagicMock()
    site2.name = "site-2"
    site2.to_dict.return_value = {"name": "site-2"}

    engine.get_job_clients.return_value = {"token-1": site1, "token-2": site2}
    client_sites = {"site-1": MagicMock(), "site-2": MagicMock()}
    job.min_sites = 1

    runner._start_run(job_id=job.job_id, job=job, client_sites=client_sites, fl_ctx=fl_ctx)

    assert job.meta[JobMetaKey.JOB_CLIENTS] == [{"name": "site-1"}]


@patch("nvflare.private.fed.server.job_runner.check_client_replies")
@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=True)
def test_start_run_keeps_job_clients_meta_when_no_timeouts(mock_get_bool, mock_check_replies):
    mock_check_replies.return_value = []
    runner, fl_ctx, engine, job, _client_sites = _make_runner_inputs()

    site1 = MagicMock()
    site1.name = "site-1"
    site1.to_dict.return_value = {"name": "site-1"}

    site2 = MagicMock()
    site2.name = "site-2"
    site2.to_dict.return_value = {"name": "site-2"}

    engine.get_job_clients.return_value = {"token-1": site1, "token-2": site2}
    client_sites = {"site-1": MagicMock(), "site-2": MagicMock()}

    runner._start_run(job_id=job.job_id, job=job, client_sites=client_sites, fl_ctx=fl_ctx)

    assert job.meta[JobMetaKey.JOB_CLIENTS] == [{"name": "site-1"}, {"name": "site-2"}]


@patch("nvflare.private.fed.server.job_runner.check_client_replies")
@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=True)
def test_start_run_raises_when_required_site_times_out(mock_get_bool, mock_check_replies):
    """A timed-out required site must abort the job even if active_count >= min_sites."""
    mock_check_replies.return_value = ["site-2"]  # site-2 timed out
    runner, fl_ctx, engine, job, _client_sites = _make_runner_inputs()

    site1 = MagicMock()
    site1.name = "site-1"
    site1.to_dict.return_value = {"name": "site-1"}
    site2 = MagicMock()
    site2.name = "site-2"
    site2.to_dict.return_value = {"name": "site-2"}
    engine.get_job_clients.return_value = {"token-1": site1, "token-2": site2}

    client_sites = {"site-1": MagicMock(), "site-2": MagicMock()}
    job.min_sites = 1  # still satisfied after site-2 drops out
    job.required_sites = ["site-2"]  # but site-2 is required

    with pytest.raises(RuntimeError, match="required client site-2 timed out"):
        runner._start_run(job_id=job.job_id, job=job, client_sites=client_sites, fl_ctx=fl_ctx)


@patch("nvflare.private.fed.server.job_runner.check_client_replies")
@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=True)
def test_start_run_proceeds_when_non_required_site_times_out(mock_get_bool, mock_check_replies):
    """A timed-out non-required site proceeds normally when min_sites is still satisfied."""
    mock_check_replies.return_value = ["site-2"]  # site-2 timed out but is not required
    runner, fl_ctx, engine, job, _client_sites = _make_runner_inputs()

    site1 = MagicMock()
    site1.name = "site-1"
    site1.to_dict.return_value = {"name": "site-1"}
    site2 = MagicMock()
    site2.name = "site-2"
    site2.to_dict.return_value = {"name": "site-2"}
    engine.get_job_clients.return_value = {"token-1": site1, "token-2": site2}

    client_sites = {"site-1": MagicMock(), "site-2": MagicMock()}
    job.min_sites = 1
    job.required_sites = ["site-1"]  # site-1 is required, site-2 is not

    runner._start_run(job_id=job.job_id, job=job, client_sites=client_sites, fl_ctx=fl_ctx)

    assert job.meta[JobMetaKey.JOB_CLIENTS] == [{"name": "site-1"}]
    runner.log_warning.assert_called_once()


@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=True)
def test_start_run_integration_real_reply_check_updates_meta(mock_get_bool):
    """Integration-style check: _start_run + real check_client_replies timeout path."""
    runner, fl_ctx, engine, job, _client_sites = _make_runner_inputs()

    site1 = MagicMock()
    site1.name = "site-1"
    site1.to_dict.return_value = {"name": "site-1"}

    site2 = MagicMock()
    site2.name = "site-2"
    site2.to_dict.return_value = {"name": "site-2"}

    engine.get_job_clients.return_value = {"token-1": site1, "token-2": site2}

    ok_reply = Message(topic="reply", body="ok")
    ok_reply.set_header(MsgHeader.RETURN_CODE, ReturnCode.OK)
    req1 = Message(topic="req", body="")
    req2 = Message(topic="req", body="")
    engine.start_client_job.return_value = [
        ClientReply(client_token="token-site-1", client_name="site-1", req=req1, reply=ok_reply),
        ClientReply(client_token="token-site-2", client_name="site-2", req=req2, reply=None),
    ]

    client_sites = {"site-1": MagicMock(), "site-2": MagicMock()}
    job.min_sites = 1

    runner._start_run(job_id=job.job_id, job=job, client_sites=client_sites, fl_ctx=fl_ctx)

    assert job.meta[JobMetaKey.JOB_CLIENTS] == [{"name": "site-1"}]
    runner.log_warning.assert_called_once()
    warning_msg = runner.log_warning.call_args[0][1]
    assert "site-2" in warning_msg
    assert "timed out" in warning_msg
