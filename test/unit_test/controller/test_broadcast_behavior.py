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

import pytest

from nvflare.apis.controller_spec import TaskCompletionStatus
from nvflare.apis.shareable import Shareable

from .controller_test import TestController, create_client, create_task, get_ready, launch_task


@pytest.mark.parametrize("method", ["broadcast", "broadcast_and_wait"])
class TestBroadcastBehavior(TestController):
    @pytest.mark.parametrize("num_of_clients", [1, 2, 3, 4])
    def test_client_receive_only_one_task(self, method, num_of_clients):
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
                "kwargs": {"targets": None, "min_responses": num_of_clients},
            },
        )
        get_ready(launch_thread)
        assert controller.get_num_standing_tasks() == 1

        for client in clients:
            task_name_out = ""
            client_task_id = ""
            data = None
            while task_name_out == "":
                task_name_out, client_task_id, data = controller.process_task_request(client, fl_ctx)
                time.sleep(0.1)
            assert task_name_out == "__test_task"
            assert data == input_data
            assert task.last_client_task_map[client.name].task_send_count == 1
            assert controller.get_num_standing_tasks() == 1
            _, next_client_task_id, _ = controller.process_task_request(client, fl_ctx)
            assert next_client_task_id == client_task_id
            assert task.last_client_task_map[client.name].task_send_count == 2

            result = Shareable()
            result["result"] = "result"
            controller.process_submission(
                client=client,
                task_name="__test_task",
                task_id=client_task_id,
                fl_ctx=fl_ctx,
                result=result,
            )
            assert task.last_client_task_map[client.name].result == result

        controller._check_tasks()
        assert task.completion_status == TaskCompletionStatus.OK
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("num_of_clients", [1, 2, 3, 4])
    def test_only_client_in_target_will_get_task(self, method, num_of_clients):
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
                "kwargs": {"targets": [clients[0]], "min_responses": 0},
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

    @pytest.mark.parametrize("min_responses", [1, 2, 3, 4])
    def test_task_only_exit_when_min_responses_received(self, method, min_responses):
        controller, fl_ctx = self.start_controller()
        clients = [create_client(f"__test_client{i}") for i in range(min_responses)]
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
                "kwargs": {"targets": None, "min_responses": min_responses},
            },
        )
        get_ready(launch_thread)
        assert controller.get_num_standing_tasks() == 1

        client_task_ids = []
        for client in clients:
            task_name_out, client_task_id, data = controller.process_task_request(client, fl_ctx)
            client_task_ids.append(client_task_id)
            assert task_name_out == "__test_task"

        for client, client_task_id in zip(clients, client_task_ids):
            result = Shareable()
            controller._check_tasks()
            assert controller.get_num_standing_tasks() == 1
            controller.process_submission(
                client=client, task_name="__test_task", task_id=client_task_id, result=result, fl_ctx=fl_ctx
            )

        controller._check_tasks()
        assert controller.get_num_standing_tasks() == 0
        assert task.completion_status == TaskCompletionStatus.OK
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("min_responses", [1, 2, 3, 4])
    @pytest.mark.parametrize("wait_time_after_min_received", [1, 2])
    def test_task_only_exit_when_wait_time_after_min_received(
        self, method, min_responses, wait_time_after_min_received
    ):
        controller, fl_ctx = self.start_controller()
        clients = [create_client(f"__test_client{i}") for i in range(min_responses)]
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
                    "targets": None,
                    "min_responses": min_responses,
                    "wait_time_after_min_received": wait_time_after_min_received,
                },
            },
        )
        get_ready(launch_thread)
        assert controller.get_num_standing_tasks() == 1

        client_task_ids = []
        for client in clients:
            task_name_out, client_task_id, data = controller.process_task_request(client, fl_ctx)
            client_task_ids.append(client_task_id)
            assert task_name_out == "__test_task"

        for client, client_task_id in zip(clients, client_task_ids):
            result = Shareable()
            controller.process_submission(
                client=client, task_name="__test_task", task_id=client_task_id, result=result, fl_ctx=fl_ctx
            )

        wait_time = 0
        while wait_time <= wait_time_after_min_received:
            assert controller.get_num_standing_tasks() == 1
            for client in clients:
                task_name_out, client_task_id, data = controller.process_task_request(client, fl_ctx)
                assert task_name_out == ""
                assert client_task_id == ""
            time.sleep(1)
            wait_time += 1

        assert controller.get_num_standing_tasks() == 0
        assert task.completion_status == TaskCompletionStatus.OK
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("num_clients", [1, 2, 3, 4])
    def test_min_resp_is_zero_task_only_exit_when_all_client_task_done(self, method, num_clients):
        controller, fl_ctx = self.start_controller()
        clients = [create_client(f"__test_client{i}") for i in range(num_clients)]
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
                    "targets": None,
                    "min_responses": 0,
                },
            },
        )
        get_ready(launch_thread)
        assert controller.get_num_standing_tasks() == 1

        client_task_ids = []
        for client in clients:
            task_name_out, client_task_id, data = controller.process_task_request(client, fl_ctx)
            client_task_ids.append(client_task_id)
            assert task_name_out == "__test_task"
            controller._check_tasks()
            assert controller.get_num_standing_tasks() == 1

        for client, client_task_id in zip(clients, client_task_ids):
            controller._check_tasks()
            assert controller.get_num_standing_tasks() == 1
            result = Shareable()
            controller.process_submission(
                client=client, task_name="__test_task", task_id=client_task_id, result=result, fl_ctx=fl_ctx
            )

        controller._check_tasks()
        assert controller.get_num_standing_tasks() == 0
        assert task.completion_status == TaskCompletionStatus.OK
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)
