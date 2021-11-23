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

from nvflare.apis.controller_spec import TaskCompletionStatus
from nvflare.apis.shareable import Shareable

from .controller_test import TestController, create_client, create_task, get_ready, launch_task


class TestTaskManagement(TestController):
    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    @pytest.mark.parametrize("num_of_tasks", [2, 3, 4])
    def test_add_task(self, method, num_of_tasks):
        controller, fl_ctx = self.start_controller()
        client = create_client(name="__test_client")

        all_threads = []
        all_tasks = []
        for i in range(num_of_tasks):
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
            all_threads.append(launch_thread)
            all_tasks.append(task)
        assert controller.get_num_standing_tasks() == num_of_tasks
        controller.cancel_all_tasks()
        for task in all_tasks:
            assert task.completion_status == TaskCompletionStatus.CANCELLED
        for thread in all_threads:
            thread.join()
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("method1", TestController.ALL_APIS)
    @pytest.mark.parametrize("method2", TestController.ALL_APIS)
    def test_reuse_same_task(self, method1, method2):
        controller, fl_ctx = self.start_controller()
        client = create_client(name="__test_client")
        task = create_task(name="__test_task")
        targets = [client]

        launch_thread = threading.Thread(
            target=launch_task,
            kwargs={
                "controller": controller,
                "task": task,
                "method": method1,
                "fl_ctx": fl_ctx,
                "kwargs": {"targets": targets},
            },
        )
        get_ready(launch_thread)

        with pytest.raises(ValueError, match="Task was already used. Please create a new task object."):
            launch_task(controller=controller, method=method2, task=task, fl_ctx=fl_ctx, kwargs={"targets": targets})

        controller.cancel_task(task)
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    @pytest.mark.parametrize("num_of_start_tasks", [2, 3, 4])
    @pytest.mark.parametrize("num_of_cancel_tasks", [1, 2])
    def test_check_task_remove_cancelled_tasks(self, method, num_of_start_tasks, num_of_cancel_tasks):
        controller, fl_ctx = self.start_controller()
        client = create_client(name="__test_client")

        all_threads = []
        all_tasks = []
        for i in range(num_of_start_tasks):
            task = create_task(name=f"__test_task{i}")
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
            all_threads.append(launch_thread)
            all_tasks.append(task)

        for i in range(num_of_cancel_tasks):
            controller.cancel_task(task=all_tasks[i], fl_ctx=fl_ctx)
            assert all_tasks[i].completion_status == TaskCompletionStatus.CANCELLED
        controller._check_tasks()
        assert controller.get_num_standing_tasks() == (num_of_start_tasks - num_of_cancel_tasks)
        controller.cancel_all_tasks()
        for thread in all_threads:
            thread.join()
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    @pytest.mark.parametrize("num_client_requests", [1, 2, 3, 4])
    def test_client_request_after_cancel_task(self, method, num_client_requests):
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
        controller.cancel_task(task)
        for i in range(num_client_requests):
            _, task_id, data = controller.process_task_request(client, fl_ctx)
            # check if task_id is empty means this task is not assigned
            assert task_id == ""
            assert data is None
        controller._check_tasks()
        assert controller.get_num_standing_tasks() == 0
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    def test_client_submit_result_after_cancel_task(self, method):
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
        task_name_out, client_task_id, data = controller.process_task_request(client, fl_ctx)
        controller.cancel_task(task)
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        time.sleep(1)

        # in here we make up client results:
        result = Shareable()
        result["result"] = "result"

        with pytest.raises(RuntimeError, match=f"Unknown task: __test_task from client __test_client."):
            controller.process_submission(
                client=client, task_name="__test_task", task_id=client_task_id, fl_ctx=fl_ctx, result=result
            )

        assert task.last_client_task_map["__test_client"].result is None
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)
