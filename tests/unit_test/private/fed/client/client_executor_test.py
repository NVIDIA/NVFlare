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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants, RunProcessKey
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_launcher_spec import JobReturnCode
from nvflare.apis.workspace import Workspace
from nvflare.fuel.common.exit_codes import ProcessExitCode
from nvflare.fuel.f3.cellnet.core_cell import FQCN
from nvflare.private.defs import CellChannel, CellChannelTopic, JobFailureMsgKey
from nvflare.private.fed.client.client_executor import REPORTABLE_JOB_FAILURES, JobExecutor

EXPECTED_REPORTABLE_JOB_FAILURES = {
    ProcessExitCode.EXCEPTION: "exception",
    ProcessExitCode.UNSAFE_COMPONENT: "unsafe component",
    ProcessExitCode.CONFIG_ERROR: "config error",
    JobReturnCode.ABORTED: "aborted",
}


def test_reportable_job_failures_has_expected_codes():
    assert REPORTABLE_JOB_FAILURES == EXPECTED_REPORTABLE_JOB_FAILURES


def _write_deployed_meta(tmp_path, job_id, deployed_meta):
    (tmp_path / "startup").mkdir()
    (tmp_path / "local").mkdir()
    workspace = Workspace(str(tmp_path), site_name="site-1")
    (tmp_path / job_id).mkdir()
    with open(workspace.get_job_meta_path(job_id), "w") as f:
        json.dump(deployed_meta, f)
    return workspace


def test_start_app_allows_scheduler_metadata(tmp_path):
    job_id = "job-1"
    deployed_meta = {
        JobConstants.JOB_ID: job_id,
        JobMetaKey.JOB_LAUNCHER_SPEC.value: {"default": {"docker": {"image": "trusted/image:1"}}},
    }
    workspace = _write_deployed_meta(tmp_path, job_id, deployed_meta)
    start_meta = {
        **deployed_meta,
        JobMetaKey.STATUS.value: "RUNNING",
        JobMetaKey.JOB_CLIENTS.value: [{"name": "site-1", "token": "token-1"}],
    }

    client = MagicMock()
    client.client_name = "site-1"
    client.cell.get_internal_listener_url.return_value = "tcp://parent:8002"
    client.cell.get_internal_listener_params.return_value = {}
    fl_ctx = MagicMock()
    fl_ctx.get_prop.side_effect = lambda key, *args, **kwargs: {
        FLContextKey.WORKSPACE_OBJECT: workspace,
        FLContextKey.SERVER_CONFIG: [{"service": {"scheme": "grpc", "target": "parent:8002"}}],
    }.get(key)
    launcher = MagicMock()
    launcher.launch_job.return_value = MagicMock()

    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        patch.object(threading.Thread, "start", lambda self: None),
    ):
        JobExecutor(client=client, startup=workspace.get_startup_kit_dir()).start_app(
            client,
            job_id,
            start_meta,
            SimpleNamespace(workspace=str(tmp_path), set=[]),
            None,
            None,
            None,
            fl_ctx,
        )

    launch_meta = launcher.launch_job.call_args.args[0]
    assert launch_meta[JobMetaKey.JOB_CLIENTS.value] == [{"name": "site-1", "token": "token-1"}]
    assert launch_meta[JobMetaKey.JOB_LAUNCHER_SPEC.value]["default"]["docker"]["image"] == "trusted/image:1"


@pytest.mark.parametrize(
    "deployed_meta, start_meta, expected_key",
    [
        (
            {
                JobConstants.JOB_ID: "job-1",
                JobMetaKey.JOB_LAUNCHER_SPEC.value: {"default": {"docker": {"image": "trusted/image:1"}}},
            },
            {
                JobConstants.JOB_ID: "job-1",
                JobMetaKey.JOB_LAUNCHER_SPEC.value: {"default": {"docker": {"image": "attacker/image:latest"}}},
            },
            JobMetaKey.JOB_LAUNCHER_SPEC.value,
        ),
        (
            {
                JobConstants.JOB_ID: "job-1",
                JobMetaKey.JOB_LAUNCHER_SPEC.value: {"default": {"k8s": {"image": "trusted/image:1"}}},
            },
            {
                JobConstants.JOB_ID: "job-1",
                JobMetaKey.JOB_LAUNCHER_SPEC.value: {"default": {"k8s": {"image": "attacker/image:latest"}}},
            },
            JobMetaKey.JOB_LAUNCHER_SPEC.value,
        ),
        (
            {
                JobConstants.JOB_ID: "job-1",
                JobMetaKey.RESOURCE_SPEC.value: {"site-1": {"docker": {"image": "trusted/image:1"}}},
            },
            {
                JobConstants.JOB_ID: "job-1",
                JobMetaKey.RESOURCE_SPEC.value: {"site-1": {"docker": {"image": "attacker/image:latest"}}},
            },
            JobMetaKey.RESOURCE_SPEC.value,
        ),
        (
            {JobConstants.JOB_ID: "job-1", JobMetaKey.STUDY.value: "study-a"},
            {JobConstants.JOB_ID: "job-1", JobMetaKey.STUDY.value: "study-b"},
            JobMetaKey.STUDY.value,
        ),
        (
            {JobConstants.JOB_ID: "job-1", JobMetaKey.SCOPE.value: "scope-a"},
            {JobConstants.JOB_ID: "job-1", JobMetaKey.SCOPE.value: "scope-b"},
            JobMetaKey.SCOPE.value,
        ),
        (
            {JobConstants.JOB_ID: "job-1"},
            {JobConstants.JOB_ID: "job-2"},
            JobMetaKey.JOB_ID.value,
        ),
        (
            {JobConstants.JOB_ID: "job-1"},
            {
                JobConstants.JOB_ID: "job-1",
                JobMetaKey.JOB_LAUNCHER_SPEC.value: {"default": {"docker": {"image": "attacker/image:latest"}}},
            },
            JobMetaKey.JOB_LAUNCHER_SPEC.value,
        ),
        (
            {
                JobConstants.JOB_ID: "job-1",
                JobMetaKey.JOB_LAUNCHER_SPEC.value: {"default": {"docker": {"image": "trusted/image:1"}}},
            },
            {JobConstants.JOB_ID: "job-1"},
            JobMetaKey.JOB_LAUNCHER_SPEC.value,
        ),
    ],
)
def test_start_app_rejects_launch_metadata_drift(tmp_path, deployed_meta, start_meta, expected_key):
    job_id = "job-1"
    workspace = _write_deployed_meta(tmp_path, job_id, deployed_meta)
    client = MagicMock()
    client.client_name = "site-1"

    with patch("nvflare.private.fed.client.client_executor.get_job_launcher") as get_job_launcher:
        with pytest.raises(RuntimeError, match=expected_key):
            JobExecutor(client=client, startup=workspace.get_startup_kit_dir()).start_app(
                client=client,
                job_id=job_id,
                job_meta=start_meta,
                args=SimpleNamespace(workspace=str(tmp_path), set=[]),
                allocated_resource=None,
                token=None,
                resource_manager=None,
                fl_ctx=MagicMock(),
            )

    get_job_launcher.assert_not_called()
    with open(workspace.get_job_meta_path(job_id)) as f:
        assert json.load(f) == deployed_meta


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


@pytest.mark.parametrize("return_code", [JobReturnCode.SUCCESS, JobReturnCode.UNKNOWN, JobReturnCode.EXECUTION_ERROR])
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
