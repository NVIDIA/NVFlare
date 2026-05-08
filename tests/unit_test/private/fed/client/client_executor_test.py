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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, RunProcessKey
from nvflare.apis.job_launcher_spec import JobReturnCode
from nvflare.fuel.common.exit_codes import ProcessExitCode
from nvflare.fuel.f3.cellnet.core_cell import FQCN
from nvflare.private.defs import CellChannel, CellChannelTopic, JobFailureMsgKey
from nvflare.private.fed.client.client_executor import REPORTABLE_JOB_FAILURES, JobExecutor

EXPECTED_REPORTABLE_JOB_FAILURES = {
    ProcessExitCode.EXCEPTION: "exception",
    ProcessExitCode.UNSAFE_COMPONENT: "unsafe component",
    ProcessExitCode.CONFIG_ERROR: "config error",
    JobReturnCode.EXECUTION_ERROR: "execution error",
    JobReturnCode.ABORTED: "aborted",
}


def test_reportable_job_failures_has_expected_codes():
    assert REPORTABLE_JOB_FAILURES == EXPECTED_REPORTABLE_JOB_FAILURES


@pytest.mark.parametrize(
    "return_code, reason",
    EXPECTED_REPORTABLE_JOB_FAILURES.items(),
)
def test_wait_child_process_reports_failure_return_code_to_server(return_code, reason):
    client = MagicMock()
    client.client_name = "site-1"
    job_executor = JobExecutor(client=client, startup="startup")

    job_handle = MagicMock()
    job_executor.run_processes = {"job-1": {RunProcessKey.JOB_HANDLE: job_handle}}

    engine = MagicMock()
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = engine

    with patch("nvflare.private.fed.client.client_executor.get_return_code", return_value=return_code):
        job_executor._wait_child_process_finish(
            client=client,
            job_id="job-1",
            allocated_resource=None,
            token=None,
            resource_manager=MagicMock(),
            workspace="/tmp/workspace",
            fl_ctx=fl_ctx,
        )

    job_handle.wait.assert_called_once()
    client.cell.fire_and_forget.assert_called_once()

    call_kwargs = client.cell.fire_and_forget.call_args.kwargs
    assert call_kwargs["targets"] == [FQCN.ROOT_SERVER]
    assert call_kwargs["channel"] == CellChannel.SERVER_MAIN
    assert call_kwargs["topic"] == CellChannelTopic.REPORT_JOB_FAILURE
    assert call_kwargs["optional"] is True

    payload = call_kwargs["message"].payload
    assert payload[JobFailureMsgKey.JOB_ID] == "job-1"
    assert payload[JobFailureMsgKey.CODE] == return_code
    assert payload[JobFailureMsgKey.REASON] == reason

    assert "job-1" not in job_executor.run_processes
    fl_ctx.set_prop.assert_any_call(FLContextKey.CURRENT_JOB_ID, "job-1", private=True, sticky=False)
    fl_ctx.set_prop.assert_any_call(FLContextKey.CLIENT_NAME, "site-1", private=True, sticky=False)
    engine.fire_event.assert_called_once_with(EventType.JOB_COMPLETED, fl_ctx)


@pytest.mark.parametrize("return_code", [JobReturnCode.SUCCESS, JobReturnCode.UNKNOWN])
def test_wait_child_process_does_not_report_non_failure_return_code(return_code):
    client = MagicMock()
    client.client_name = "site-1"
    job_executor = JobExecutor(client=client, startup="startup")

    job_handle = MagicMock()
    job_executor.run_processes = {"job-1": {RunProcessKey.JOB_HANDLE: job_handle}}

    engine = MagicMock()
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = engine

    with patch("nvflare.private.fed.client.client_executor.get_return_code", return_value=return_code):
        job_executor._wait_child_process_finish(
            client=client,
            job_id="job-1",
            allocated_resource=None,
            token=None,
            resource_manager=MagicMock(),
            workspace="/tmp/workspace",
            fl_ctx=fl_ctx,
        )

    client.cell.fire_and_forget.assert_not_called()
    assert "job-1" not in job_executor.run_processes
    engine.fire_event.assert_called_once_with(EventType.JOB_COMPLETED, fl_ctx)
