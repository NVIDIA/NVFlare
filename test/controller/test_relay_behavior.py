# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
from itertools import permutations

import pytest

from nvflare.apis.controller_spec import SendOrder, TaskCompletionStatus
from nvflare.apis.shareable import ReservedHeaderKey, Shareable

from ..utils import skip_if_quick
from .controller_test import TestController, create_client, create_task, get_ready, launch_task


def _process_task_request_test_cases():
    """Returns a list of
    targets, request_client, dynamic_targets, task_assignment_timeout, time_before_first_request,
    expected_to_get_task, expected_targets
    """
    clients = [create_client(f"__test_client{i}") for i in range(3)]
    client_names = [c.name for c in clients]

    dynamic_targets_cases = [
        [clients[1:], clients[0], True, 1, 0, False, [clients[1].name, clients[2].name, clients[0].name]],
        [clients[1:], clients[1], True, 1, 0, True, client_names[1:]],
        [clients[1:], clients[2], True, 1, 0, False, client_names[1:]],
        [[clients[0]], clients[1], True, 1, 0, False, [clients[0].name, clients[1].name]],
        [[clients[0]], clients[1], True, 1, 2, False, [clients[0].name]],
        [[clients[0], clients[0]], clients[0], True, 1, 0, True, [clients[0].name, clients[0].name]],
        [None, clients[0], True, 1, 0, True, [clients[0].name]],
    ]

    static_targets_cases = [
        [clients[1:], clients[0], False, 1, 0, False, client_names[1:]],
        [clients[1:], clients[1], False, 1, 0, True, client_names[1:]],
        [clients[1:], clients[2], False, 1, 0, False, client_names[1:]],
        [clients[1:], clients[0], False, 1, 2, False, client_names[1:]],
        [clients[1:], clients[1], False, 1, 2, True, client_names[1:]],
        [clients[1:], clients[2], False, 1, 2, True, client_names[1:]],
    ]

    return dynamic_targets_cases + static_targets_cases


def _get_sequential_sequence_test_cases():
    """Returns a list of list of clients"""
    clients = [create_client(f"__test_client{i}") for i in range(3)]
    normal_cases = [list(x) for x in permutations(clients)]
    duplicate_clients = [[clients[0], clients[0]], [clients[0], clients[1], clients[0], clients[2]]]
    return normal_cases + duplicate_clients


def _get_order_with_task_assignment_timeout_test_cases():
    """Returns a list of
    send_order, targets, task_assignment_timeout,
    time_before_first_request, request_orders, expected_clients_to_get_task

    Each item in request_orders is a request_order,
    In reality, this request order should be random, because each client side does not sync with other clients.
    """
    num_clients = 3
    clients = [create_client(name=f"__test_client{i}") for i in range(num_clients)]

    # these are just helpful orders
    clients_120 = [clients[1], clients[2], clients[0]]
    clients_201 = [clients[2], clients[0], clients[1]]
    clients_210 = [clients[2], clients[1], clients[0]]
    return [
        [
            SendOrder.SEQUENTIAL,
            clients,
            2,
            1,
            [clients, clients, clients],
            clients,
        ],
        [
            SendOrder.SEQUENTIAL,
            clients,
            2,
            1,
            [clients_120, clients_120, clients_120],
            clients,
        ],
        [
            SendOrder.SEQUENTIAL,
            clients,
            2,
            1,
            [clients_201, clients_201, clients_201],
            clients,
        ],
        [
            SendOrder.SEQUENTIAL,
            clients,
            2,
            3,
            [clients, clients, clients],
            clients,
        ],
        [
            SendOrder.SEQUENTIAL,
            clients,
            2,
            3,
            [clients_120, clients_120, clients_120],
            [clients[1], clients[2], None],
        ],
        [
            SendOrder.SEQUENTIAL,
            clients,
            2,
            3,
            [clients_201, clients_201, clients_201],
            clients,
        ],
        [
            SendOrder.SEQUENTIAL,
            clients,
            2,
            3,
            [clients_210, clients_210, clients_210],
            [clients[1], clients[2], None],
        ],
        [
            SendOrder.SEQUENTIAL,
            clients,
            2,
            3,
            [clients_120, clients, clients_120],
            [clients[1], clients[2], None],
        ],
        [
            SendOrder.SEQUENTIAL,
            clients,
            2,
            5,
            [clients, clients, clients],
            clients,
        ],
        [
            SendOrder.SEQUENTIAL,
            clients,
            2,
            5,
            [clients_120, clients_120, clients_120],
            [clients[1], clients[2], None],
        ],
        [SendOrder.SEQUENTIAL, clients, 2, 5, [clients_201, clients_201, clients_201], [clients[2], None, None]],
        [SendOrder.SEQUENTIAL, clients, 2, 5, [clients_201, clients, clients_120], [clients[2], None, None]],
        [
            SendOrder.SEQUENTIAL,
            [clients[0], clients[1], clients[2], clients[1], clients[0], clients[0]],
            2,
            5,
            [clients, clients, clients, clients, clients, clients],
            [clients[0], clients[1], clients[2], clients[1], clients[0], clients[0]],
        ],
        [
            SendOrder.SEQUENTIAL,
            [clients[0], clients[1], clients[2], clients[1], clients[0], clients[0]],
            2,
            5,
            [clients_201, clients_201, clients_201, clients_201, clients_201, clients_201],
            [clients[2], clients[1], clients[0], clients[0], None, None],
        ],
        [
            SendOrder.ANY,
            clients,
            2,
            1,
            [clients, clients, clients],
            clients,
        ],
        [
            SendOrder.ANY,
            clients,
            2,
            1,
            [clients_120, clients_120, clients_120],
            clients_120,
        ],
        [
            SendOrder.ANY,
            clients,
            2,
            1,
            [clients_201, clients_201, clients_201],
            clients_201,
        ],
        [
            SendOrder.ANY,
            clients,
            2,
            3,
            [clients, clients, clients],
            clients,
        ],
        [
            SendOrder.ANY,
            clients,
            2,
            3,
            [clients_120, clients_120, clients_120],
            clients_120,
        ],
        [
            SendOrder.ANY,
            clients,
            2,
            3,
            [clients_201, clients_201, clients_201],
            clients_201,
        ],
        [
            SendOrder.ANY,
            clients,
            2,
            3,
            [clients_210, clients_210, clients_210],
            clients_210,
        ],
        [
            SendOrder.ANY,
            clients,
            2,
            3,
            [clients_120, clients, clients_120],
            [clients[1], clients[0], clients[2]],
        ],
        [
            SendOrder.ANY,
            clients,
            2,
            5,
            [clients, clients, clients],
            clients,
        ],
        [
            SendOrder.ANY,
            clients,
            2,
            5,
            [clients_120, clients_120, clients_120],
            clients_120,
        ],
        [
            SendOrder.ANY,
            clients,
            2,
            5,
            [clients_201, clients_201, clients_201],
            clients_201,
        ],
        [
            SendOrder.ANY,
            clients,
            2,
            5,
            [clients_201, clients, clients_120],
            clients_201,
        ],
        [
            SendOrder.ANY,
            [clients[0], clients[1], clients[2], clients[1], clients[0], clients[0]],
            2,
            5,
            [clients, clients, clients, clients, clients, clients],
            [clients[0], clients[0], clients[0], clients[1], clients[1], clients[2]],
        ],
        [
            SendOrder.ANY,
            [clients[0], clients[1], clients[2], clients[1], clients[0], clients[0]],
            2,
            5,
            [clients_201, clients_201, clients_201, clients_201, clients_201, clients_201],
            [clients[2], clients[0], clients[0], clients[0], clients[1], clients[1]],
        ],
        [
            SendOrder.ANY,
            [clients[0], clients[1], clients[2], clients[1], clients[0], clients[0]],
            2,
            5,
            [clients_210, clients_210, clients_210, clients_210, clients_201, clients_201, clients],
            [clients[2], clients[1], clients[1], clients[0], clients[0], clients[0], None],
        ],
    ]


@skip_if_quick
@pytest.mark.parametrize("method", ["relay", "relay_and_wait"])
class TestRelayBehavior(TestController):
    @pytest.mark.parametrize("send_order", [SendOrder.ANY, SendOrder.SEQUENTIAL])
    def test_only_client_in_target_will_get_task(self, method, send_order):
        controller, fl_ctx = self.start_controller()
        clients = [create_client(f"__test_client{i}") for i in range(4)]
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
                "kwargs": {"targets": [clients[0]], "send_order": send_order},
            },
        )
        get_ready(launch_thread)
        assert controller.get_num_standing_tasks() == 1

        task_name_out = ""
        data = None
        while task_name_out == "":
            task_name_out, client_task_id, data = controller.process_task_request(clients[0], fl_ctx)
            time.sleep(0.1)
        assert task_name_out == "__test_task"
        assert data == input_data
        assert task.last_client_task_map[clients[0].name].task_send_count == 1
        assert controller.get_num_standing_tasks() == 1

        for client in clients[1:]:
            task_name_out, client_task_id, data = controller.process_task_request(client, fl_ctx)
            assert task_name_out == ""
            assert client_task_id == ""

        controller.cancel_task(task)
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)

    def test_task_assignment_timeout_sequential_order_only_client_in_target_will_get_task(self, method):
        task_assignment_timeout = 3
        task_result_timeout = 3
        controller, fl_ctx = self.start_controller()
        clients = [create_client(f"__test_client{i}") for i in range(4)]
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
                    "targets": [clients[0]],
                    "send_order": SendOrder.SEQUENTIAL,
                    "task_assignment_timeout": task_assignment_timeout,
                    "task_result_timeout": task_result_timeout,
                    "dynamic_targets": False,
                },
            },
        )
        get_ready(launch_thread)
        assert controller.get_num_standing_tasks() == 1

        time.sleep(task_assignment_timeout + 1)

        for client in clients[1:]:
            task_name_out, client_task_id, data = controller.process_task_request(client, fl_ctx)
            assert task_name_out == ""
            assert client_task_id == ""

        controller.cancel_task(task)
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize(
        "targets,request_client,dynamic_targets,task_assignment_timeout,time_before_first_request,"
        "expected_to_get_task,expected_targets",
        _process_task_request_test_cases(),
    )
    def test_process_task_request(
        self,
        method,
        targets,
        request_client,
        dynamic_targets,
        task_assignment_timeout,
        time_before_first_request,
        expected_to_get_task,
        expected_targets,
    ):
        controller, fl_ctx = self.start_controller()
        task = create_task("__test_task")
        launch_thread = threading.Thread(
            target=launch_task,
            kwargs={
                "controller": controller,
                "task": task,
                "method": method,
                "fl_ctx": fl_ctx,
                "kwargs": {
                    "targets": targets,
                    "dynamic_targets": dynamic_targets,
                    "task_assignment_timeout": task_assignment_timeout,
                },
            },
        )
        get_ready(launch_thread)
        assert controller.get_num_standing_tasks() == 1
        time.sleep(time_before_first_request)

        task_name, task_id, data = controller.process_task_request(client=request_client, fl_ctx=fl_ctx)
        client_get_a_task = True if task_name == "__test_task" else False

        assert client_get_a_task == expected_to_get_task
        assert task.targets == expected_targets

        controller.cancel_task(task)
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("targets", _get_sequential_sequence_test_cases())
    def test_sequential_sequence(self, method, targets):
        controller, fl_ctx = self.start_controller()
        input_data = Shareable()
        input_data["result"] = "start_"
        task = create_task("__test_task", data=input_data)
        launch_thread = threading.Thread(
            target=launch_task,
            kwargs={
                "controller": controller,
                "task": task,
                "method": method,
                "fl_ctx": fl_ctx,
                "kwargs": {"targets": targets, "send_order": SendOrder.SEQUENTIAL},
            },
        )
        get_ready(launch_thread)
        assert controller.get_num_standing_tasks() == 1

        expected_client_index = 0
        while controller.get_num_standing_tasks() != 0:
            client_tasks_and_results = {}

            for c in targets:
                task_name, task_id, data = controller.process_task_request(client=c, fl_ctx=fl_ctx)
                if task_name != "":
                    client_result = Shareable()
                    client_result["result"] = f"{c.name}"
                    if task_id not in client_tasks_and_results:
                        client_tasks_and_results[task_id] = (c, task_name, client_result)
                    assert c == targets[expected_client_index]

            for task_id in client_tasks_and_results.keys():
                c, task_name, client_result = client_tasks_and_results[task_id]
                task.data["result"] += client_result["result"]
                controller.process_submission(
                    client=c, task_name=task_name, task_id=task_id, result=client_result, fl_ctx=fl_ctx
                )
                assert task.last_client_task_map[c.name].result == client_result
            expected_client_index += 1

        launch_thread.join()
        assert task.data["result"] == "start_" + "".join([c.name for c in targets])
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize(
        "send_order,targets,task_assignment_timeout,time_before_first_request,request_orders,"
        "expected_clients_to_get_task",
        _get_order_with_task_assignment_timeout_test_cases(),
    )
    def test_process_request_and_submission_with_task_assignment_timeout(
        self,
        method,
        send_order,
        targets,
        request_orders,
        task_assignment_timeout,
        time_before_first_request,
        expected_clients_to_get_task,
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

        for request_order, expected_client_to_get_task in zip(request_orders, expected_clients_to_get_task):
            task_name_out = ""
            client_task_id = ""
            # processing task request
            for client in request_order:
                if expected_client_to_get_task and client.name == expected_client_to_get_task.name:
                    data = None
                    task_name_out = ""
                    while task_name_out == "":
                        task_name_out, client_task_id, data = controller.process_task_request(client, fl_ctx)
                        time.sleep(0.1)
                    assert task_name_out == "__test_task"
                    assert data == input_data
                    assert task.last_client_task_map[client.name].task_send_count == 1
                else:
                    _task_name_out, _client_task_id, _ = controller.process_task_request(client, fl_ctx)
                    assert _task_name_out == ""
                    assert _client_task_id == ""

            # client side running some logic to generate result
            if expected_client_to_get_task:
                controller._check_tasks()
                assert controller.get_num_standing_tasks() == 1
                result = Shareable()
                controller.process_submission(
                    client=expected_client_to_get_task,
                    task_name=task_name_out,
                    task_id=client_task_id,
                    fl_ctx=fl_ctx,
                    result=result,
                )

        launch_thread.join()
        controller._check_tasks()
        assert controller.get_num_standing_tasks() == 0
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("send_order", [SendOrder.ANY, SendOrder.SEQUENTIAL])
    def test_process_submission_after_first_client_task_result_timeout(self, method, send_order):
        clients = [create_client(name=f"__test_client{i}") for i in range(2)]
        task_assignment_timeout = 1
        task_result_timeout = 2
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
                    "targets": clients,
                    "send_order": send_order,
                    "task_assignment_timeout": task_assignment_timeout,
                    "task_result_timeout": task_result_timeout,
                },
            },
        )
        get_ready(launch_thread)
        assert controller.get_num_standing_tasks() == 1

        # first client get a task
        data = None
        task_name_out = ""
        old_client_task_id = ""
        while task_name_out == "":
            task_name_out, old_client_task_id, data = controller.process_task_request(clients[0], fl_ctx)
            time.sleep(0.1)
        assert task_name_out == "__test_task"
        assert data == input_data
        assert task.last_client_task_map[clients[0].name].task_send_count == 1

        time.sleep(task_result_timeout + 1)

        # same client ask should get the same task
        task_name_out, client_task_id, data = controller.process_task_request(clients[0], fl_ctx)
        assert client_task_id == old_client_task_id
        assert task.last_client_task_map[clients[0].name].task_send_count == 2

        time.sleep(task_result_timeout + 1)

        # second client ask should get a task since task_result_timeout passed
        task_name_out, client_task_id_1, data = controller.process_task_request(clients[1], fl_ctx)
        assert task_name_out == "__test_task"
        assert data == input_data
        assert task.last_client_task_map[clients[1].name].task_send_count == 1

        # then we get back first client's result
        result = Shareable()
        controller.process_submission(
            client=clients[0],
            task_name=task_name_out,
            task_id=client_task_id,
            fl_ctx=fl_ctx,
            result=result,
        )

        # need to make sure the header is set
        assert result.get_header(ReservedHeaderKey.REPLY_IS_LATE)
        controller._check_tasks()
        assert controller.get_num_standing_tasks() == 1
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("send_order", [SendOrder.ANY, SendOrder.SEQUENTIAL])
    def test_process_submission_all_client_task_result_timeout(self, method, send_order):
        clients = [create_client(name=f"__test_client{i}") for i in range(2)]
        task_assignment_timeout = 1
        task_result_timeout = 2
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
                    "targets": clients,
                    "send_order": send_order,
                    "task_assignment_timeout": task_assignment_timeout,
                    "task_result_timeout": task_result_timeout,
                },
            },
        )
        get_ready(launch_thread)
        assert controller.get_num_standing_tasks() == 1

        # each client get a client task then time out
        for client in clients:
            data = None
            task_name_out = ""

            while task_name_out == "":
                task_name_out, old_client_task_id, data = controller.process_task_request(client, fl_ctx)
                time.sleep(0.1)
            assert task_name_out == "__test_task"
            assert data == input_data
            assert task.last_client_task_map[client.name].task_send_count == 1

            time.sleep(task_result_timeout + 1)

        if send_order == SendOrder.SEQUENTIAL:
            assert task.completion_status == TaskCompletionStatus.TIMEOUT
            assert controller.get_num_standing_tasks() == 0
        elif send_order == SendOrder.ANY:
            assert controller.get_num_standing_tasks() == 1
            controller.cancel_task(task)
            assert task.completion_status == TaskCompletionStatus.CANCELLED

        launch_thread.join()
        self.stop_controller(controller, fl_ctx)
