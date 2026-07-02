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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.edge.constants import EdgeApiStatus, EdgeConfigFile, EdgeContextKey, EdgeMsgTopic, JobDataKey
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.edge.widgets.etd import EdgeTaskDispatcher
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.message import Message


def _context(engine=None):
    fl_ctx = FLContext()
    if engine:
        fl_ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=False)
    return fl_ctx


def _job_meta(job_id="job-1", name="edge-job"):
    return {JobMetaKey.JOB_ID: job_id, JobMetaKey.JOB_NAME: name, JobMetaKey.EDGE_METHOD: "method"}


def test_init_registers_event_handlers():
    with patch.object(EdgeTaskDispatcher, "register_event_handler") as register:
        dispatcher = EdgeTaskDispatcher(request_timeout=3.0)
    handler_names = {call.args[1].__name__ for call in register.call_args_list}
    assert {"_handle_job_launched", "_handle_job_done", "_handle_edge_job_request", "_handle_edge_request"}.issubset(
        handler_names
    )
    assert dispatcher.request_timeout == 3.0


def test_add_match_exists_and_remove_job(tmp_path):
    dispatcher = EdgeTaskDispatcher()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    device_config = {"batch_size": 4}
    (config_dir / EdgeConfigFile.DEVICE_CONFIG).write_text(json.dumps(device_config))
    workspace = MagicMock()
    workspace.get_app_config_dir.return_value = str(config_dir)
    fl_ctx = _context()
    fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace, private=True, sticky=False)

    dispatcher._add_job({}, fl_ctx)
    dispatcher._add_job(_job_meta(), fl_ctx)
    dispatcher._add_job(_job_meta(), fl_ctx)

    assert dispatcher.edge_jobs == {"edge-job": ["job-1"]}
    assert dispatcher._job_exists("job-1")
    with patch("nvflare.edge.widgets.etd.randrange", return_value=0):
        assert dispatcher._match_job("edge-job") == ("job-1", device_config)
    assert dispatcher._match_job("missing") == (None, None)

    dispatcher._remove_job("job-1")
    assert not dispatcher._job_exists("job-1")
    assert dispatcher.edge_jobs == {}
    dispatcher._remove_job("missing")


def test_job_lifecycle_handlers_validate_context(tmp_path):
    dispatcher = EdgeTaskDispatcher()
    workspace = MagicMock()
    workspace.get_app_config_dir.return_value = str(tmp_path)
    fl_ctx = _context()
    fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace, private=True, sticky=False)
    dispatcher.logger.error = MagicMock()

    dispatcher._handle_job_launched("launched", fl_ctx)
    dispatcher.logger.error.assert_called_once()

    fl_ctx.set_prop(FLContextKey.JOB_META, _job_meta(), private=True, sticky=False)
    dispatcher._handle_job_launched("launched", fl_ctx)
    assert dispatcher._job_exists("job-1")

    dispatcher._handle_job_done("done", _context())
    fl_ctx.set_prop(FLContextKey.CURRENT_JOB_ID, "job-1", private=True, sticky=False)
    dispatcher._handle_job_done("done", fl_ctx)
    assert not dispatcher._job_exists("job-1")


def test_edge_job_request_returns_invalid_no_job_and_match():
    dispatcher = EdgeTaskDispatcher()
    dispatcher.logger.error = MagicMock()
    fl_ctx = _context()
    fl_ctx.set_prop(
        EdgeContextKey.REQUEST_FROM_EDGE,
        JobRequest("", None, None),
        private=True,
        sticky=False,
    )
    dispatcher._handle_edge_job_request("request", fl_ctx)
    assert fl_ctx.get_prop(EdgeContextKey.REPLY_TO_EDGE).status == EdgeApiStatus.INVALID_REQUEST

    fl_ctx.set_prop(EdgeContextKey.REQUEST_FROM_EDGE, JobRequest("missing", None, None), private=True, sticky=False)
    dispatcher._handle_edge_job_request("request", fl_ctx)
    assert fl_ctx.get_prop(EdgeContextKey.REPLY_TO_EDGE).status == EdgeApiStatus.NO_JOB

    dispatcher.edge_jobs = {"edge-job": ["job-1"]}
    dispatcher.job_metas = {"job-1": _job_meta()}
    dispatcher.job_device_config = {"job-1": {"batch_size": 2}}
    fl_ctx.set_prop(EdgeContextKey.REQUEST_FROM_EDGE, JobRequest("edge-job", None, None), private=True, sticky=False)
    with patch("nvflare.edge.widgets.etd.randrange", return_value=0):
        dispatcher._handle_edge_job_request("request", fl_ctx)
    reply = fl_ctx.get_prop(EdgeContextKey.REPLY_TO_EDGE)
    assert reply.status == EdgeApiStatus.OK
    assert reply.job_id == "job-1"
    assert reply.job_data[JobDataKey.CONFIG] == {"batch_size": 2}
    assert fl_ctx.get_prop(FLContextKey.JOB_META) == _job_meta()


def test_edge_request_validates_job_and_handles_cell_reply():
    dispatcher = EdgeTaskDispatcher(request_timeout=2.0)
    engine = MagicMock()
    fl_ctx = _context(engine)
    bad_reply = TaskResponse(EdgeApiStatus.INVALID_REQUEST)
    no_job_reply = TaskResponse(EdgeApiStatus.NO_JOB)
    comm_reply = TaskResponse(EdgeApiStatus.RETRY)

    fl_ctx.set_prop(EdgeContextKey.REQUEST_FROM_EDGE, SimpleNamespace(job_id=""), private=True, sticky=False)
    dispatcher._handle_edge_request("event", fl_ctx, EdgeMsgTopic.TASK_REQUEST, bad_reply, no_job_reply, comm_reply)
    assert fl_ctx.get_prop(EdgeContextKey.REPLY_TO_EDGE) is bad_reply

    fl_ctx.set_prop(EdgeContextKey.REQUEST_FROM_EDGE, SimpleNamespace(job_id="missing"), private=True, sticky=False)
    dispatcher._handle_edge_request("event", fl_ctx, EdgeMsgTopic.TASK_REQUEST, bad_reply, no_job_reply, comm_reply)
    assert fl_ctx.get_prop(EdgeContextKey.REPLY_TO_EDGE) is no_job_reply

    dispatcher.edge_jobs = {"edge-job": ["job-1"]}
    response = JobResponse(EdgeApiStatus.OK)
    engine.send_to_job.return_value = Message(
        headers={MessageHeaderKey.RETURN_CODE: CellReturnCode.OK}, payload=response
    )
    fl_ctx.set_prop(EdgeContextKey.REQUEST_FROM_EDGE, SimpleNamespace(job_id="job-1"), private=True, sticky=False)
    dispatcher._handle_edge_request("event", fl_ctx, EdgeMsgTopic.TASK_REQUEST, bad_reply, no_job_reply, comm_reply)
    assert fl_ctx.get_prop(EdgeContextKey.REPLY_TO_EDGE) is response
    assert engine.send_to_job.call_args.kwargs["optional"] is True

    engine.send_to_job.return_value = Message(headers={MessageHeaderKey.RETURN_CODE: CellReturnCode.TIMEOUT})
    dispatcher._handle_edge_request("event", fl_ctx, EdgeMsgTopic.TASK_REQUEST, bad_reply, no_job_reply, comm_reply)
    assert fl_ctx.get_prop(EdgeContextKey.REPLY_TO_EDGE) is comm_reply
