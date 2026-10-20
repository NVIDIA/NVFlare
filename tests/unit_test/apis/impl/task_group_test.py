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

import threading
from unittest.mock import Mock

import pytest

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import Task, TaskGroup
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.impl.controller import Controller
from nvflare.apis.impl.wf_comm_server import WFCommServer
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.shareable import ReservedHeaderKey, Shareable


class _Controller(Controller):
    def control_flow(self, abort_signal, fl_ctx):
        pass

    def start_controller(self, fl_ctx):
        pass

    def stop_controller(self, fl_ctx):
        pass

    def process_result_of_unknown_task(self, client, task_name, client_task_id, result, fl_ctx):
        pass


@pytest.fixture
def system():
    clients = [Client(f"client-{index}", f"token-{index}") for index in range(2)]
    engine = Mock(spec=ServerEngineSpec)
    manager = FLContextManager(engine=engine, identity_name="server", job_id="job-1")
    engine.new_context.side_effect = manager.new_context
    engine.get_clients.return_value = clients
    controller = _Controller()
    communicator = WFCommServer()
    controller.set_communicator(communicator)
    fl_ctx = manager.new_context()
    controller.initialize(fl_ctx)
    communicator.initialize_run(fl_ctx)
    yield controller, communicator, manager, clients
    communicator.finalize_run(fl_ctx)


def _request(communicator, manager, client):
    fl_ctx = manager.new_context()
    name, task_id, data = communicator.process_task_request(client, fl_ctx)
    return name, data, fl_ctx.get_prop(FLContextKey.TASK_GROUPS_CLOSED)


def test_tasks_in_block_carry_group_and_closure_reaches_touched_clients_twice(system):
    controller, communicator, manager, clients = system
    fl_ctx = manager.new_context()
    with controller.task_group(fl_ctx) as group:
        task = Task("train", Shareable())
        controller.broadcast(task, fl_ctx, targets=[clients[0]], min_responses=1)
        assert task.group is group
        name, data, closed = _request(communicator, manager, clients[0])
        assert name == "train"
        assert data.get_header(ReservedHeaderKey.TASK_GROUP_ID) == group.id
        assert data.get_cookie(ReservedHeaderKey.TASK_GROUP_ID) == group.id
        assert closed is None
        assert not group.closed
    assert group.closed

    assert _request(communicator, manager, clients[1])[2] is None
    assert _request(communicator, manager, clients[0])[2] == [group.id]
    assert _request(communicator, manager, clients[0])[2] == [group.id]
    assert _request(communicator, manager, clients[0])[2] is None


def test_outside_block_tasks_have_no_group_and_closed_or_nested_groups_are_rejected(system):
    controller, communicator, manager, clients = system
    fl_ctx = manager.new_context()
    task = Task("train", Shareable())
    controller.broadcast(task, fl_ctx, targets=[clients[0]], min_responses=1)
    assert task.group is None
    name, data, _ = _request(communicator, manager, clients[0])
    assert data.get_header(ReservedHeaderKey.TASK_GROUP_ID) is None

    with controller.task_group(fl_ctx) as group:
        with pytest.raises(RuntimeError, match="do not nest"):
            with controller.task_group(fl_ctx):
                pass
    with pytest.raises(RuntimeError, match="closed task group"):
        controller.broadcast(Task("late", Shareable(), group=group), fl_ctx, targets=[clients[0]])


def test_explicit_group_wins_and_other_threads_do_not_inherit_the_block(system):
    controller, communicator, manager, clients = system
    fl_ctx = manager.new_context()
    explicit = TaskGroup()
    seen = {}

    def issue_from_thread():
        task = Task("other-thread", Shareable())
        controller.broadcast(task, fl_ctx, targets=[clients[1]], min_responses=1)
        seen["group"] = task.group

    with controller.task_group(fl_ctx) as ambient:
        task = Task("explicit", Shareable(), group=explicit)
        controller.broadcast(task, fl_ctx, targets=[clients[0]], min_responses=1)
        assert task.group is explicit
        thread = threading.Thread(target=issue_from_thread)
        thread.start()
        thread.join()
    assert seen["group"] is None
    assert ambient.closed and not explicit.closed


def test_block_closes_on_exception(system):
    controller, communicator, manager, clients = system
    fl_ctx = manager.new_context()
    with pytest.raises(ValueError):
        with controller.task_group(fl_ctx) as group:
            controller.broadcast(Task("train", Shareable()), fl_ctx, targets=[clients[0]], min_responses=1)
            _request(communicator, manager, clients[0])
            raise ValueError("workflow failed")
    assert group.closed
    assert _request(communicator, manager, clients[0])[2] == [group.id]
