# Copyright (c) 2021, NVIDIA CORPORATION.
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
import time

import pytest

from nvflare.apis.controller_spec import SendOrder, TaskCompletionStatus
from nvflare.apis.shareable import Shareable

from .controller_test import TestController, create_client, create_task, get_ready, launch_task


def _assert_other_clients_get_no_task(controller, fl_ctx, client_idx: int, clients):
    """Assert clients get no task."""
    assert client_idx < len(clients)
    for i, client in enumerate(clients):
        if i == client_idx:
            continue
        _task_name_out, _client_task_id, data = controller.process_task_request(client, fl_ctx)
        assert _task_name_out == ""
        assert _client_task_id == ""


def _get_process_task_request_test_cases():
    """Returns a lit of

    targets, send_order, request_client_idx
    """
    num_clients = 3
    clients = [create_client(name=f"__test_client{i}") for i in range(num_clients)]

    return [
        [clients, SendOrder.ANY, 0],
        [clients, SendOrder.ANY, 1],
        [clients, SendOrder.ANY, 2],
        [clients, SendOrder.SEQUENTIAL, 0],
    ]


def _get_process_task_request_with_task_assignment_timeout_test_cases():
    """Returns a list of
    targets, send_order, task_assignment_timeout, time_before_first_request, request_order, expected_client_to_get_task
    """
    num_clients = 3
    clients = [create_client(name=f"__test_client{i}") for i in range(num_clients)]
    clients_120 = [clients[1], clients[2], clients[0]]
    clients_201 = [clients[2], clients[0], clients[1]]
    return [
        [clients, SendOrder.SEQUENTIAL, 2, 1, clients, clients[0].name],
        [clients, SendOrder.SEQUENTIAL, 2, 1, clients_120, clients[0].name],
        [clients, SendOrder.SEQUENTIAL, 2, 1, clients_201, clients[0].name],
        [clients, SendOrder.SEQUENTIAL, 2, 3, clients, clients[0].name],
        [clients, SendOrder.SEQUENTIAL, 2, 3, clients_120, clients[1].name],
        [clients, SendOrder.SEQUENTIAL, 2, 3, clients_201, clients[0].name],
        [clients, SendOrder.SEQUENTIAL, 2, 5, clients, clients[0].name],
        [clients, SendOrder.SEQUENTIAL, 2, 5, clients_120, clients[1].name],
        [clients, SendOrder.SEQUENTIAL, 2, 5, clients_201, clients[2].name],
        [clients, SendOrder.SEQUENTIAL, 2, 3, [clients[2], clients[1], clients[0]], clients[1].name],
        [clients, SendOrder.ANY, 2, 1, clients, clients[0].name],
        [clients, SendOrder.ANY, 2, 1, clients_120, clients[1].name],
        [clients, SendOrder.ANY, 2, 1, clients_201, clients[2].name],
    ]


@pytest.mark.parametrize("method", ["send", "send_and_wait"])
class TestSendBehavior(TestController):
    @pytest.mark.parametrize("send_order", [SendOrder.ANY, SendOrder.SEQUENTIAL])
    def test_process_task_request_client_not_in_target_get_nothing(self, method, send_order):
        controller, fl_ctx = self.start_controller()
        client = create_client(f"__test_client")
        targets = [create_client(f"__target_client")]
        task = create_task("__test_task")
        launch_thread = threading.Thread(
            target=launch_task,
            kwargs={
                "controller": controller,
                "task": task,
                "method": method,
                "fl_ctx": fl_ctx,
                "kwargs": {"targets": targets, "send_order": send_order},
            },
        )
        get_ready(launch_thread)
        assert controller.get_num_standing_tasks() == 1

        # this client not in target so should get nothing
        _task_name_out, _client_task_id, data = controller.process_task_request(client, fl_ctx)
        assert _task_name_out == ""
        assert _client_task_id == ""

        controller.cancel_task(task)
        controller._check_tasks()
        assert controller.get_num_standing_tasks() == 0
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("targets,send_order,client_idx", _get_process_task_request_test_cases())
    def test_process_task_request_expected_client_get_task_and_unexpected_clients_get_nothing(
        self, method, targets, send_order, client_idx
    ):
        controller, fl_ctx = self.start_controller()
        input_data = Shareable()
        input_data["hello"] = "world"
        task = create_task("__test_task", data=input_data)
        launch_thread = threading.Thread(
            target=launch_task,
            kwargs={
                "controller": controller,
                "task": task,
                "method": method,
                "fl_ctx": fl_ctx,
                "kwargs": {"targets": targets, "send_order": SendOrder.ANY},
            },
        )
        get_ready(launch_thread)
        assert controller.get_num_standing_tasks() == 1

        # first client
        task_name_out = ""
        data = None
        while task_name_out == "":
            task_name_out, client_task_id, data = controller.process_task_request(targets[client_idx], fl_ctx)
            time.sleep(0.1)
        assert task_name_out == "__test_task"
        assert data == input_data
        assert task.last_client_task_map[targets[client_idx].name].task_send_count == 1

        # other clients
        _assert_other_clients_get_no_task(controller=controller, fl_ctx=fl_ctx, client_idx=client_idx, clients=targets)

        controller.cancel_task(task)
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        controller._check_tasks()
        assert controller.get_num_standing_tasks() == 0
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize(
        "targets,send_order,task_assignment_timeout,time_before_first_request,request_order,expected_client_to_get_task",
        _get_process_task_request_with_task_assignment_timeout_test_cases(),
    )
    def test_process_task_request_with_task_assignment_timeout_expected_client_get_task(
        self,
        method,
        targets,
        send_order,
        task_assignment_timeout,
        time_before_first_request,
        request_order,
        expected_client_to_get_task,
    ):
        controller, fl_ctx = self.start_controller()
        input_data = Shareable()
        input_data["hello"] = "world"
        task = create_task("__test_task", data=input_data)
        launch_thread = threading.Thread(
            target=launch_task,
            kwargs={
                "controller": controller,
                "task": task,
                "method": method,
                "fl_ctx": fl_ctx,
                "kwargs": {
                    "targets": targets,
                    "send_order": send_order,
                    "task_assignment_timeout": task_assignment_timeout,
                },
            },
        )
        get_ready(launch_thread)
        assert controller.get_num_standing_tasks() == 1

        time.sleep(time_before_first_request)

        for client in request_order:
            data = None
            if client.name == expected_client_to_get_task:
                task_name_out = ""
                while task_name_out == "":
                    task_name_out, client_task_id, data = controller.process_task_request(client, fl_ctx)
                    time.sleep(0.1)
                assert task_name_out == "__test_task"
                assert data == input_data
                assert task.last_client_task_map[client.name].task_send_count == 1
            else:
                task_name_out, client_task_id, data = controller.process_task_request(client, fl_ctx)
                assert task_name_out == ""
                assert client_task_id == ""

        controller.cancel_task(task)
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("num_of_clients", [1, 2, 3])
    def test_send_only_one_task_and_exit_when_client_task_done(self, method, num_of_clients):
        controller, fl_ctx = self.start_controller()
        clients = [create_client(f"__test_client{i}") for i in range(num_of_clients)]
        input_data = Shareable()
        input_data["hello"] = "world"
        task = create_task("__test_task", data=input_data)
        launch_thread = threading.Thread(
            target=launch_task,
            kwargs={
                "controller": controller,
                "task": task,
                "method": method,
                "fl_ctx": fl_ctx,
                "kwargs": {"targets": clients, "send_order": SendOrder.SEQUENTIAL},
            },
        )
        get_ready(launch_thread)
        assert controller.get_num_standing_tasks() == 1

        # first client
        task_name_out = ""
        client_task_id = ""
        data = None
        while task_name_out == "":
            task_name_out, client_task_id, data = controller.process_task_request(clients[0], fl_ctx)
            time.sleep(0.1)
        assert task_name_out == "__test_task"
        assert data == input_data
        assert task.last_client_task_map[clients[0].name].task_send_count == 1

        # once a client gets a task, other clients should not get task
        _assert_other_clients_get_no_task(controller=controller, fl_ctx=fl_ctx, client_idx=0, clients=clients)

        controller._check_tasks()
        assert controller.get_num_standing_tasks() == 1

        controller.process_submission(
            client=clients[0], task_name="__test_task", task_id=client_task_id, fl_ctx=fl_ctx, result=data
        )

        controller._check_tasks()
        assert controller.get_num_standing_tasks() == 0
        assert task.completion_status == TaskCompletionStatus.OK
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)
