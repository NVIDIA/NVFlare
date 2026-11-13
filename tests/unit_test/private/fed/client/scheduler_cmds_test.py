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

from unittest.mock import ANY, MagicMock, patch

import pytest

from nvflare.apis.fl_constant import FLContextKey, SystemComponents
from nvflare.apis.fl_context import FLContext
from nvflare.apis.resource_manager_spec import ResourceManagerSpec
from nvflare.apis.workspace import Workspace
from nvflare.private.admin_defs import Message
from nvflare.private.defs import RequestHeader, TrainingTopic
from nvflare.private.fed.client.client_engine_internal_spec import ClientEngineInternalSpec
from nvflare.private.fed.client.client_executor import JobCleanupPending
from nvflare.private.fed.client.scheduler_cmds import CheckResourceProcessor, StartJobProcessor
from nvflare.private.scheduler_constants import ShareableHeader


def test_check_resource_processor_keeps_resource_manager_error_local():
    resource_manager = MagicMock(spec=ResourceManagerSpec)
    resource_manager.check_resources.side_effect = ValueError("resource_requirement is missing num_gpu_key num_of_gpus")
    engine = MagicMock(spec=ClientEngineInternalSpec)
    engine.get_component.return_value = resource_manager
    engine.new_context.return_value.__enter__.return_value = FLContext()
    request = Message(topic=TrainingTopic.CHECK_RESOURCE, body={"license": 2})
    request.set_header(RequestHeader.JOB_ID, "job-1")

    with (
        patch("nvflare.private.fed.client.scheduler_cmds.logger") as logger,
        patch("nvflare.private.fed.client.scheduler_cmds.secure_format_exception", return_value="sensitive detail"),
    ):
        reply = CheckResourceProcessor().process(request, engine)

    assert not reply.body.get_header(ShareableHeader.IS_RESOURCE_ENOUGH)
    assert (
        reply.body.get_header(ShareableHeader.RESOURCE_RESERVE_TOKEN)
        == "resource manager raised an exception; see site log"
    )
    engine.get_component.assert_called_once_with(SystemComponents.RESOURCE_MANAGER)
    logger.error.assert_called_once_with("Job job-1: resource check failed: sensitive detail")


@pytest.mark.parametrize(
    "runtime_file, job_meta, reason",
    [
        pytest.param("studies:\n  study-a:\n", {"study": "study-a"}, None, id="listed-study"),
        pytest.param(
            "studies:\n  study-a:\n", {"study": "study-b"}, "study 'study-b' is not configured", id="unlisted"
        ),
        pytest.param("studies:\n  study-a:\n", {}, "study 'default' is not configured", id="unlisted-default"),
        pytest.param("studies: {}\n", {"study": "study-b"}, None, id="empty-studies"),
        pytest.param(None, {"study": "study-b"}, None, id="no-file"),
    ],
)
def test_client_runs_only_studies_listed_in_study_runtime(tmp_path, runtime_file, job_meta, reason):
    (tmp_path / "startup").mkdir()
    (tmp_path / "local").mkdir()
    if runtime_file is not None:
        (tmp_path / "local" / "study_runtime.yaml").write_text(f"format_version: 2\n{runtime_file}")
    fl_ctx = FLContext()
    fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, Workspace(str(tmp_path)), private=True)
    resource_manager = MagicMock(spec=ResourceManagerSpec)
    resource_manager.check_resources.return_value = (True, "token-1")
    resource_manager.allocate_resources.return_value = None
    engine = MagicMock(spec=ClientEngineInternalSpec)
    engine.get_component.return_value = resource_manager
    engine.new_context.return_value.__enter__.return_value = fl_ctx
    engine.start_app.return_value = ""

    replies = []
    for topic, processor in [
        (TrainingTopic.CHECK_RESOURCE, CheckResourceProcessor),
        (TrainingTopic.START_JOB, StartJobProcessor),
    ]:
        request = Message(topic=topic, body={})
        request.set_header(RequestHeader.JOB_ID, "job-1")
        request.set_header(RequestHeader.JOB_META, job_meta)
        request.set_header(ShareableHeader.RESOURCE_RESERVE_TOKEN, "token-1")
        replies.append(processor().process(request, engine).body)
    check_reply, start_reply = replies

    assert check_reply.get_header(ShareableHeader.IS_RESOURCE_ENOUGH) is (reason is None)
    assert engine.start_app.called is (reason is None)
    if reason:
        assert reason in check_reply.get_header(ShareableHeader.RESOURCE_RESERVE_TOKEN)
        assert reason in start_reply


@pytest.mark.parametrize("retained", [False, True])
def test_start_failure_frees_only_resources_not_owned_by_executor(retained):
    resource_manager = MagicMock(spec=ResourceManagerSpec)
    resource_manager.allocate_resources.return_value = {"gpu": 1}
    engine = MagicMock(spec=ClientEngineInternalSpec)
    engine.new_context.return_value.__enter__.return_value = FLContext()
    error = JobCleanupPending("cleanup remains owned by CP") if retained else RuntimeError("no runtime created")
    engine.start_app.side_effect = error
    request = Message(topic=TrainingTopic.START_JOB, body={"gpu": 1})
    request.set_header(RequestHeader.JOB_ID, "job-1")
    request.set_header(RequestHeader.JOB_META, {})
    request.set_header(ShareableHeader.RESOURCE_RESERVE_TOKEN, "token-1")
    with (
        patch("nvflare.private.fed.client.scheduler_cmds._get_resource_manager", return_value=resource_manager),
        patch("nvflare.private.fed.client.scheduler_cmds._get_resource_consumer"),
    ):
        reply = StartJobProcessor().process(request, engine)
    assert str(error) in reply.body
    if retained:
        resource_manager.free_resources.assert_not_called()
    else:
        resource_manager.free_resources.assert_called_once_with(resources={"gpu": 1}, token="token-1", fl_ctx=ANY)
