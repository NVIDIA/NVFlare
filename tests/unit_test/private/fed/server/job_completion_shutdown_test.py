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

"""Completion/shutdown policy through real handlers and a fake admin transport."""

from contextlib import nullcontext
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.apis.fl_constant import RunProcessKey
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.job_def import JobMetaKey, RunStatus
from nvflare.fuel.common.exit_codes import ProcessExitCode
from nvflare.fuel.flare_api import flare_api
from nvflare.fuel.flare_api.api_spec import InternalError, JobFinishing
from nvflare.fuel.flare_api.flare_api import Session
from nvflare.fuel.hci.client.api import APIStatus, ResultKey
from nvflare.fuel.hci.proto import MetaKey, MetaStatusValue
from nvflare.private.fed.server import job_cmds, job_runner
from nvflare.private.fed.server.job_cmds import JobCommandModule
from nvflare.private.fed.server.job_runner import JobRunner
from nvflare.tool.api_utils import shutdown_system_by_session
from tests.timing_utils import WAIT_TIMEOUT, ManualClock, ThreadGroup

pytestmark = pytest.mark.timeout(10)
JOB_ID = "123e4567-e89b-42d3-a456-426614174000"


@pytest.mark.parametrize(
    "terminal_status",
    [RunStatus.FINISHED_COMPLETED, RunStatus.FINISHED_ABORTED, RunStatus.FINISHED_EXECUTION_EXCEPTION],
)
@pytest.mark.parametrize("stalled_archive", [False, True])
def test_shutdown_waits_for_selected_verdict_publication(monkeypatch, tmp_path, terminal_status, stalled_archive):
    job = SimpleNamespace(
        job_id=JOB_ID,
        run_aborted=terminal_status == RunStatus.FINISHED_ABORTED,
        meta={JobMetaKey.STATUS: RunStatus.RUNNING},
    )
    archive_entered, release_archive, published = Event(), Event(), Event()
    archived = []
    statuses = []

    def publish(job_id, status, _ctx):
        assert archived, "terminal status must follow workspace archival"
        assert status == terminal_status
        statuses.append(status)
        job.meta[JobMetaKey.STATUS] = status
        published.set()

    manager = SimpleNamespace(get_job=lambda *_: job, set_status=publish)
    runner = JobRunner(str(tmp_path))
    runner.running_jobs[JOB_ID] = job
    runner._fire_job_lifecycle_event = MagicMock()
    runner.abort_client_run = MagicMock()
    exception = (
        {JOB_ID: {RunProcessKey.PARTICIPANTS: {}, RunProcessKey.PROCESS_RETURN_CODE: ProcessExitCode.EXCEPTION}}
        if terminal_status == RunStatus.FINISHED_EXECUTION_EXCEPTION
        else {}
    )
    engine = SimpleNamespace(
        lock=Lock(),
        run_processes={},
        exception_run_processes=exception,
        job_runner=runner,
        job_def_manager=manager,
        client_manager=SimpleNamespace(clients={}),
        get_component=lambda _: manager,
        get_cell=lambda: None,
        remove_exception_process=lambda _: None,
    )
    ctx = FLContextManager(engine=engine).new_context()
    engine.new_context = lambda: nullcontext(ctx)
    monkeypatch.setattr(job_runner, "time", SimpleNamespace(sleep=lambda _: setattr(runner, "ask_to_stop", True)))
    monkeypatch.setattr(job_cmds, "record", MagicMock())

    def archive(*_args):
        archive_entered.set()
        release_archive.wait()  # ThreadGroup releases this even if an assertion or retry fails.
        archived.append(True)

    runner._save_workspace = archive
    replies = []

    def dispatch(command, props=None):
        metadata = []

        def append_reply(_message, meta=None):
            metadata.append(meta)

        conn = SimpleNamespace(
            app_ctx=engine,
            get_prop=lambda *_: JOB_ID,
            append_string=lambda *_: None,
            append_success=append_reply,
            append_error=append_reply,
        )
        JobCommandModule().abort_job(conn, command.split())
        replies.append(metadata[-1][MetaKey.STATUS])
        return {ResultKey.STATUS: APIStatus.SUCCESS, ResultKey.META: metadata[-1]}

    session = object.__new__(Session)
    session.api = SimpleNamespace(closed=False, session_expired_reason=None, do_command=dispatch)
    session.list_jobs = lambda: [{"job_id": JOB_ID, "status": job.meta[JobMetaKey.STATUS]}]

    def shutdown(*_args, **_kwargs):
        assert published.is_set()
        assert archived

    session.shutdown = MagicMock(side_effect=shutdown)
    clock = ManualClock(now=0)
    monkeypatch.setattr(flare_api, "time", clock)

    def retry_sleep(delay):
        clock.advance(delay)
        if not stalled_archive:
            release_archive.set()
            assert published.wait(WAIT_TIMEOUT), "completion did not publish after archival was released"

    # Only the Session policy clock is virtual; thread readiness and cleanup use real waits.
    clock.sleep = retry_sleep
    with ThreadGroup(release_archive) as workers:
        workers.start(runner._job_complete_process, engine)
        assert archive_entered.wait(WAIT_TIMEOUT)
        assert job.meta[JobMetaKey.STATUS] == RunStatus.RUNNING
        if stalled_archive:
            with pytest.raises(JobFinishing, match="System shutdown was not sent.*still archiving"):
                shutdown_system_by_session(session, wait=False, verbose=False)
            assert clock.monotonic() == 90
            assert not published.is_set()
            session.shutdown.assert_not_called()
        else:
            shutdown_system_by_session(session, wait=False, verbose=False)
            session.shutdown.assert_called_once()
            assert replies == [MetaStatusValue.JOB_FINISHING, MetaStatusValue.OK]

    assert statuses == [terminal_status]
    assert job.meta[JobMetaKey.STATUS] == terminal_status
    assert JOB_ID not in runner.running_jobs
    first_audit = job_cmds.record.call_args_list[0]
    assert first_audit.args == ("job.abort.retry_required",)
    assert first_audit.kwargs["reason"] == "job_finishing"


@pytest.mark.parametrize("error", ["could not abort", "Job is already finishing."])
def test_shutdown_does_not_swallow_untyped_abort_failures(error):
    # A RUNNING store entry and a failed abort are not evidence of completion,
    # even when the human-readable error happens to mention finishing.
    from nvflare.fuel.hci.proto import make_meta

    session = object.__new__(Session)
    transport = MagicMock(
        return_value={
            ResultKey.STATUS: APIStatus.SUCCESS,
            ResultKey.META: make_meta(MetaStatusValue.INTERNAL_ERROR, error),
        }
    )
    session.api = SimpleNamespace(closed=False, session_expired_reason=None, do_command=transport)
    session.list_jobs = lambda: [{"job_id": JOB_ID, "status": RunStatus.RUNNING}]
    session.shutdown = MagicMock()

    with pytest.raises(InternalError):
        shutdown_system_by_session(session, wait=False, verbose=False)

    transport.assert_called_once()
    session.shutdown.assert_not_called()
