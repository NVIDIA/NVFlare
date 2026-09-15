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

import json
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.job_def import JobMetaKey, RunStatus
from nvflare.apis.job_scheduler_spec import DispatchInfo
from nvflare.fuel.hci.conn import Connection
from nvflare.private.fed.server import job_runner as runner_module
from nvflare.private.fed.server.job_cmds import JobCommandModule
from nvflare.private.fed.server.job_runner import JobRunner
from nvflare.private.fed.server.server_state import HotState


@pytest.fixture
def admission(tmp_path, monkeypatch):
    runner = JobRunner(str(tmp_path))
    runner._job_complete_process = MagicMock()
    runner.log_debug = MagicMock()
    runner.log_info = MagicMock()
    runner.log_error = MagicMock()
    runner.fire_event = MagicMock()
    runner._fire_job_lifecycle_event = MagicMock()
    engine, ctx, manager = MagicMock(), MagicMock(), MagicMock()
    ctx.get_engine.return_value = engine
    ctx.get_prop.side_effect = lambda key, *args: ["site-1: OK"] if key == FLContextKey.JOB_DEPLOY_DETAIL else None
    engine.new_context.side_effect = lambda: nullcontext(ctx)
    engine.get_component.return_value = manager
    engine.job_def_manager = manager
    engine.job_runner = runner
    engine.server.server_state = HotState()
    job = SimpleNamespace(
        job_id="job-1",
        meta={JobMetaKey.SCHEDULE_COUNT: 1, JobMetaKey.LAST_SCHEDULE_TIME: 0, JobMetaKey.SCHEDULE_HISTORY: []},
    )
    statuses = {job.job_id: RunStatus.SUBMITTED.value, "other-job": RunStatus.SUBMITTED.value}
    manager.get_job.side_effect = lambda jid, _: SimpleNamespace(meta={JobMetaKey.STATUS: statuses[jid]})
    manager.set_status.side_effect = lambda jid, status, _: statuses.__setitem__(jid, status.value)
    runner.scheduler = MagicMock()
    sites = {"site-1": DispatchInfo("app", {"gpu": 1}, "reservation-1")}

    def schedule(**kwargs):
        runner.stop()  # Execute exactly this admission; no polling sleeps or background leaks.
        return job, sites

    runner.scheduler.schedule_job.side_effect = schedule
    runner._deploy_job = MagicMock(return_value=(job.job_id, []))
    runner._start_run = MagicMock()
    runner._stop_run = MagicMock()
    monkeypatch.setattr(runner_module, "time", SimpleNamespace(sleep=lambda _: None))

    def abort(jid=job.job_id):
        conn = Connection(app_ctx=engine)
        conn.set_prop(JobCommandModule.JOB_ID, jid)
        JobCommandModule().abort_job(conn, [])
        return json.loads(conn.close())

    return SimpleNamespace(runner=runner, engine=engine, ctx=ctx, manager=manager, statuses=statuses, abort=abort)


@pytest.mark.parametrize("window", ["control", "selection", "after_check", "deploy", "deploy_error"])
def test_abort_wins_prevents_launch(admission, window):
    a = admission
    responses = []
    if window == "selection":
        responses.append(a.abort())
    if window == "after_check":
        original = a.runner._check_job_status

        def check(*args):
            result = original(*args)
            if not responses:
                responses.append(a.abort())
            return result

        a.runner._check_job_status = check
    if window in ("deploy", "deploy_error"):

        def deploy(*args):
            responses.append(a.abort())
            if window == "deploy_error":
                raise RuntimeError("deployment failed")
            return "job-1", []

        a.runner._deploy_job.side_effect = deploy
    a.runner.run(a.ctx)
    if window == "control":
        assert a.statuses["job-1"] == RunStatus.RUNNING
        a.runner._start_run.assert_called_once()
    else:
        assert responses[0]["meta"]["status"] == "ok"
        assert "before running" in responses[0]["meta"]["info"]
        assert a.statuses["job-1"] == RunStatus.FINISHED_ABORTED
        a.runner._start_run.assert_not_called()
    assert not a.runner._starting_jobs


@pytest.mark.parametrize("stage", ["deploy", "metadata", "start"])
@pytest.mark.parametrize("secure_logging", ["false", "true"])
def test_abort_while_slow_operation_is_outstanding(admission, stage, secure_logging, monkeypatch):
    monkeypatch.setenv("NVFLARE_SECURE_LOGGING", secure_logging)
    a = admission
    entered, release = threading.Event(), threading.Event()

    def gated(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return "job-1", []

    operation = {"deploy": a.runner._deploy_job, "metadata": a.manager.update_meta, "start": a.runner._start_run}[stage]
    operation.side_effect = gated
    thread = threading.Thread(target=a.runner.run, args=(a.ctx,))
    thread.start()
    try:
        assert entered.wait(5)
        # This also proves a slow launch/deployment does not lock out unrelated aborts.
        assert a.abort("other-job")["meta"]["status"] == "ok"
        response = a.abort()
        if stage == "deploy":
            assert response["meta"]["status"] == "ok"
        else:
            assert response["meta"]["status"] == "error"
            assert "is starting; retry abort" in response["meta"]["info"]
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert a.statuses["job-1"] == (RunStatus.FINISHED_ABORTED if stage == "deploy" else RunStatus.RUNNING)
    if stage == "start":
        a.runner.stop_run = MagicMock(return_value="")
        assert a.abort()["meta"]["status"] == "ok"
        a.runner.stop_run.assert_called_once_with("job-1", a.ctx)


@pytest.mark.parametrize("failure", ["deploy", "start", "stop"])
def test_failure_releases_start_claim_and_preserves_abort(admission, failure):
    a = admission
    if failure == "deploy":
        a.runner._deploy_job.side_effect = RuntimeError("deploy failed")
    else:
        a.runner._start_run.side_effect = RuntimeError("start failed")
        if failure == "stop":
            a.runner._stop_run.side_effect = RuntimeError("stop failed")
    a.runner.run(a.ctx)
    assert not a.runner._starting_jobs
    assert a.statuses["job-1"] == RunStatus.FAILED_TO_RUN
    assert a.abort()["meta"]["status"] == "ok"
