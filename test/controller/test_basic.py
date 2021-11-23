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
import uuid

import pytest

from nvflare.apis.controller_spec import TaskCompletionStatus
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable

from .controller_test import TestController, create_client, create_task, get_ready, launch_task


class TestBasic(TestController):
    @pytest.mark.parametrize("task_name,client_name", [["__test_task", "__test_client"]])
    def test_process_submission_invalid_task(self, task_name, client_name):
        controller, fl_ctx = self.start_controller()
        client = create_client(name="__test_client")
        with pytest.raises(RuntimeError, match=f"Unknown task: {task_name} from client {client_name}."):
            controller.process_submission(
                client=client, task_name=task_name, task_id=str(uuid.uuid4()), fl_ctx=FLContext(), result=Shareable()
            )
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    @pytest.mark.parametrize("num_client_requests", [1, 2, 3, 4])
    def test_process_task_request_client_request_multiple_times(self, method, num_client_requests):
        controller, fl_ctx = self.start_controller()
        client = create_client(name="__test_client")
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
                "kwargs": {"targets": [client]},
            },
        )
        get_ready(launch_thread)

        for i in range(num_client_requests):
            task_name_out, _, data = controller.process_task_request(client, fl_ctx)
            assert task_name_out == "__test_task"
            assert data == input_data
        assert task.last_client_task_map["__test_client"].task_send_count == num_client_requests
        controller.cancel_task(task)
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    def test_process_submission(self, method):
        controller, fl_ctx = self.start_controller()
        client = create_client(name="__test_client")
        task = create_task("__test_task")
        kwargs = {"targets": [client]}
        launch_thread = threading.Thread(
            target=launch_task,
            kwargs={
                "controller": controller,
                "task": task,
                "method": method,
                "fl_ctx": fl_ctx,
                "kwargs": kwargs,
            },
        )
        get_ready(launch_thread)

        task_name_out, client_task_id, data = controller.process_task_request(client, fl_ctx)
        # in here we make up client results:
        result = Shareable()
        result["result"] = "result"

        controller.process_submission(
            client=client, task_name="__test_task", task_id=client_task_id, fl_ctx=fl_ctx, result=result
        )
        assert task.last_client_task_map["__test_client"].result == result
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    @pytest.mark.parametrize("timeout", [1, 2])
    def test_task_timeout(self, method, timeout):
        controller, fl_ctx = self.start_controller()
        client = create_client(name="__test_client")
        task = create_task(name="__test_task", data=Shareable(), timeout=timeout)

        launch_thread = threading.Thread(
            target=launch_task,
            kwargs={
                "controller": controller,
                "task": task,
                "method": method,
                "fl_ctx": fl_ctx,
                "kwargs": {"targets": [client]},
            },
        )
        get_ready(launch_thread)

        assert controller.get_num_standing_tasks() == 1
        time.sleep(timeout + 1)
        assert controller.get_num_standing_tasks() == 0
        assert task.completion_status == TaskCompletionStatus.TIMEOUT
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    def test_cancel_task(self, method):
        controller, fl_ctx = self.start_controller()
        client = create_client(name="__test_client")
        task = create_task(name="__test_task")
        launch_thread = threading.Thread(
            target=launch_task,
            kwargs={
                "controller": controller,
                "task": task,
                "method": method,
                "fl_ctx": fl_ctx,
                "kwargs": {"targets": [client]},
            },
        )
        get_ready(launch_thread)
        assert controller.get_num_standing_tasks() == 1

        controller.cancel_task(task=task)
        controller._check_tasks()
        assert controller.get_num_standing_tasks() == 0
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    def test_cancel_all_tasks(self, method):
        controller, fl_ctx = self.start_controller()
        client = create_client(name="__test_client")
        task = create_task("__test_task")
        launch_thread = threading.Thread(
            target=launch_task,
            kwargs={
                "controller": controller,
                "task": task,
                "method": method,
                "fl_ctx": fl_ctx,
                "kwargs": {"targets": [client]},
            },
        )
        get_ready(launch_thread)
        task1 = create_task("__test_task1")
        launch_thread1 = threading.Thread(
            target=launch_task,
            kwargs={
                "controller": controller,
                "task": task1,
                "method": method,
                "fl_ctx": fl_ctx,
                "kwargs": {"targets": [client]},
            },
        )
        get_ready(launch_thread1)
        assert controller.get_num_standing_tasks() == 2

        controller.cancel_all_tasks()
        controller._check_tasks()
        assert controller.get_num_standing_tasks() == 0
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        assert task1.completion_status == TaskCompletionStatus.CANCELLED
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)
