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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import Task
from nvflare.apis.filter import Filter
from nvflare.apis.fl_constant import FilterKey, FLContextKey, ReservedKey, ReservedTopic, ReturnCode
from nvflare.apis.impl.task_controller import TaskController
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.private.privacy_manager import Scope
from tests.unit_test.fl_context_helper import make_fl_context


class _ReplacingFilter(Filter):
    def __init__(self, marker, ttl):
        super().__init__()
        self.marker = marker
        self.ttl = ttl
        self.input = None
        self.output = None

    def process(self, shareable, fl_ctx):
        self.input = shareable
        self.output = Shareable({"payload": f"{shareable['payload']}:{self.marker}"})
        if self.ttl is not None:
            self.output.set_header(ReservedHeaderKey.MSG_ROOT_TTL, self.ttl)
        return self.output


def _make_controller_context(job_filters=None, scope=None):
    engine = MagicMock()
    engine.all_clients = [Client("site-1", "token-1")]
    engine.validate_targets.return_value = (engine.all_clients, [])
    engine.send_aux_request.return_value = {"site-1": make_reply(ReturnCode.OK)}
    fl_ctx = make_fl_context(engine=engine)
    runner = SimpleNamespace(task_data_filters=job_filters or {}, task_result_filters={})
    fl_ctx.set_prop(FLContextKey.RUNNER, runner, private=True, sticky=False)
    if scope is not None:
        fl_ctx.set_prop(FLContextKey.SCOPE_OBJECT, scope, private=True, sticky=False)
    controller = TaskController()
    controller.start_controller(fl_ctx)
    return controller, fl_ctx, engine


@pytest.mark.parametrize("sources", [("job",), ("site",), ("site", "job")], ids=["job", "site", "site_and_job"])
@pytest.mark.parametrize("filtered_ttl", [None, 3600.0, 0])
def test_broadcast_and_wait_sends_filter_chain_output(sources, filtered_ttl):
    scope = Scope()
    job_filters = {}
    filters = []
    for source in sources:
        filter_ = _ReplacingFilter(source, filtered_ttl)
        filters.append(filter_)
        if source == "site":
            scope.add_task_data_filter(filter_, FilterKey.OUT)
        else:
            job_filters["train/out"] = [filter_]
    controller, fl_ctx, engine = _make_controller_context(job_filters, scope)
    original = Shareable({"payload": "raw"})
    original.set_header(ReservedHeaderKey.MSG_ROOT_TTL, 10)
    task = Task(name="train", data=original, timeout=30)

    replies = controller.broadcast_and_wait(task, fl_ctx, targets=["site-1"])

    previous = original
    for filter_ in filters:
        assert filter_.input is previous
        previous = filter_.output
    assert task.data is previous
    assert task.data["payload"] == ":".join(("raw", *sources))
    assert original["payload"] == "raw"
    engine.send_aux_request.assert_called_once()
    request = engine.send_aux_request.call_args.kwargs["request"]
    assert request is task.data
    assert request is not original
    assert request.get_header(ReservedKey.TASK_NAME) == task.name
    assert request.get_header(ReservedHeaderKey.MSG_ROOT_ID) == task.msg_root_id
    assert request.get_header(ReservedHeaderKey.MSG_ROOT_TTL) == (30 if filtered_ttl is None else filtered_ttl)
    engine.send_aux_request.assert_called_once_with(
        targets=["site-1"],
        topic=ReservedTopic.DO_TASK,
        request=request,
        timeout=30,
        fl_ctx=fl_ctx,
        secure=task.secure,
    )
    assert replies["site-1"].get_return_code() == ReturnCode.OK


@pytest.mark.parametrize("preset_ttl", [None, 3600.0, 0])
def test_broadcast_and_wait_without_filters_sends_original(preset_ttl):
    controller, fl_ctx, engine = _make_controller_context()
    original = Shareable({"payload": "raw"})
    if preset_ttl is not None:
        original.set_header(ReservedHeaderKey.MSG_ROOT_TTL, preset_ttl)
    task = Task(name="train", data=original, timeout=30)

    controller.broadcast_and_wait(task, fl_ctx, targets=["site-1"])

    engine.send_aux_request.assert_called_once()
    request = engine.send_aux_request.call_args.kwargs["request"]
    assert request is original
    assert task.data is original
    assert request["payload"] == "raw"
    assert request.get_header(ReservedKey.TASK_NAME) == task.name
    assert request.get_header(ReservedHeaderKey.MSG_ROOT_ID) == task.msg_root_id
    assert request.get_header(ReservedHeaderKey.MSG_ROOT_TTL) == (30 if preset_ttl is None else preset_ttl)
