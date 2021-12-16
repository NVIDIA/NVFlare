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

from nvflare.apis.controller_spec import ClientTask, Task, TaskCompletionStatus
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable

from .controller_test import TestController, create_client, create_task, get_ready, launch_task


def _get_task_done_callback_test_cases():
    task_name = "__test_task"

    def task_done_cb(task: Task, fl_ctx: FLContext):
        client_names = [x.client.name for x in task.client_tasks]
        expected_str = "_".join(client_names)
        task.props[task_name] = expected_str

    input_data = Shareable()
    test_cases = [
        [
            "broadcast",
            [create_client(f"__test_client{i}") for i in range(10)],
            task_name,
            input_data,
            task_done_cb,
            "_".join([f"__test_client{i}" for i in range(10)]),
        ],
        [
            "broadcast_and_wait",
            [create_client(f"__test_client{i}") for i in range(10)],
            task_name,
            input_data,
            task_done_cb,
            "_".join([f"__test_client{i}" for i in range(10)]),
        ],
        ["send", [create_client("__test_client")], task_name, input_data, task_done_cb, "__test_client"],
        ["send_and_wait", [create_client("__test_client")], task_name, input_data, task_done_cb, "__test_client"],
        ["relay", [create_client("__test_client")], task_name, input_data, task_done_cb, "__test_client"],
        ["relay_and_wait", [create_client("__test_client")], task_name, input_data, task_done_cb, "__test_client"],
    ]
    return test_cases


class TestCallback(TestController):
    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    def test_before_task_sent_cb(self, method):
        def before_task_sent_cb(client_task: ClientTask, fl_ctx: FLContext):
            client_task.task.data["_test_data"] = client_task.client.name

        client_name = "_test_client"
        controller, fl_ctx = self.start_controller()
        client = create_client(name=client_name)
        task = create_task("__test_task", before_task_sent_cb=before_task_sent_cb)
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

        task_name_out, _, data = controller.process_task_request(client, fl_ctx)

        expected = Shareable()
        expected["_test_data"] = client_name
        assert data == expected
        controller.cancel_task(task)
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    def test_result_received_cb(self, method):
        def result_received_cb(client_task: ClientTask, fl_ctx: FLContext):
            client_task.result["_test_data"] = client_task.client.name

        client_name = "_test_client"
        input_data = Shareable()
        input_data["_test_data"] = "_old_data"
        controller, fl_ctx = self.start_controller()
        client = create_client(name=client_name)
        task = create_task("__test_task", data=input_data, result_received_cb=result_received_cb)
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
        controller.process_submission(
            client=client, task_name="__test_task", task_id=client_task_id, fl_ctx=fl_ctx, result=data
        )

        expected = Shareable()
        expected["_test_data"] = client_name
        assert task.last_client_task_map[client_name].result == expected
        controller._check_tasks()
        assert task.completion_status == TaskCompletionStatus.OK
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("task_complete", ["normal", "timeout", "cancel"])
    @pytest.mark.parametrize("method,clients,task_name,input_data,cb,expected", _get_task_done_callback_test_cases())
    def test_task_done_cb(self, method, clients, task_name, input_data, cb, expected, task_complete):
        controller, fl_ctx = self.start_controller()

        timeout = 0 if task_complete != "timeout" else 1
        task = create_task("__test_task", data=input_data, task_done_cb=cb, timeout=timeout)
        kwargs = {"targets": clients}
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

        client_task_ids = len(clients) * [None]
        for i, client in enumerate(clients):
            task_name_out, client_task_ids[i], _ = controller.process_task_request(client, fl_ctx)

            if task_name_out == "":
                client_task_ids[i] = None

        # in here we make up client results:
        result = Shareable()
        result["result"] = "result"

        for client, client_task_id in zip(clients, client_task_ids):
            if client_task_id is not None:
                if task_complete == "normal":
                    controller.process_submission(
                        client=client, task_name="__test_task", task_id=client_task_id, fl_ctx=fl_ctx, result=result
                    )
        if task_complete == "timeout":
            time.sleep(timeout)
            assert task.completion_status == TaskCompletionStatus.TIMEOUT
        elif task_complete == "cancel":
            controller.cancel_task(task)
            assert task.completion_status == TaskCompletionStatus.CANCELLED
        controller._check_tasks()
        assert task.props[task_name] == expected
        assert controller.get_num_standing_tasks() == 0
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    def test_cancel_task_before_send_cb(self, method):
        def before_task_sent_cb(client_task: ClientTask, fl_ctx: FLContext):
            client_task.task.completion_status = TaskCompletionStatus.CANCELLED

        controller, fl_ctx = self.start_controller()
        client = create_client(name="__test_client")
        task = create_task("__test_task", before_task_sent_cb=before_task_sent_cb)
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
        assert task_name_out == ""
        assert client_task_id == ""

        launch_thread.join()
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    def test_cancel_task_result_received_cb(self, method):
        def result_received_cb(client_task: ClientTask, fl_ctx: FLContext):
            client_task.task.completion_status = TaskCompletionStatus.CANCELLED

        controller, fl_ctx = self.start_controller()
        client1 = create_client(name="__test_client")
        client2 = create_client(name="__another_client")
        task = create_task("__test_task", result_received_cb=result_received_cb)
        launch_thread = threading.Thread(
            target=launch_task,
            kwargs={
                "controller": controller,
                "task": task,
                "method": method,
                "fl_ctx": fl_ctx,
                "kwargs": {"targets": [client1, client2]},
            },
        )
        get_ready(launch_thread)

        task_name_out, client_task_id, data = controller.process_task_request(client1, fl_ctx)

        result = Shareable()
        result["__result"] = "__test_result"
        controller.process_submission(
            client=client1, task_name="__test_task", task_id=client_task_id, fl_ctx=fl_ctx, result=result
        )
        assert task.last_client_task_map["__test_client"].result == result

        task_name_out, client_task_id, data = controller.process_task_request(client2, fl_ctx)
        assert task_name_out == ""
        assert client_task_id == ""

        launch_thread.join()
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    @pytest.mark.parametrize("method2", ["broadcast", "send", "relay"])
    def test_schedule_task_before_send_cb(self, method, method2):
        def before_task_sent_cb(client_task: ClientTask, fl_ctx: FLContext):
            controller = fl_ctx.get_prop(key="controller")
            new_task = create_task("__new_test_task")
            inner_launch_thread = threading.Thread(
                target=launch_task,
                kwargs={
                    "controller": controller,
                    "task": new_task,
                    "method": method2,
                    "fl_ctx": fl_ctx,
                    "kwargs": {"targets": [client_task.client]},
                },
            )
            inner_launch_thread.start()
            inner_launch_thread.join()

        controller, fl_ctx = self.start_controller()
        fl_ctx.set_prop("controller", controller)
        client = create_client(name="__test_client")
        task = create_task("__test_task", before_task_sent_cb=before_task_sent_cb)
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
        launch_thread.start()

        task_name_out = ""
        while task_name_out == "":
            task_name_out, _, _ = controller.process_task_request(client, fl_ctx)
            time.sleep(0.1)
        assert task_name_out == "__test_task"
        new_task_name_out = ""
        while new_task_name_out == "":
            new_task_name_out, _, _ = controller.process_task_request(client, fl_ctx)
            time.sleep(0.1)
        assert new_task_name_out == "__new_test_task"

        controller.cancel_task(task)
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    @pytest.mark.parametrize("method2", ["broadcast", "send", "relay"])
    def test_schedule_task_result_received_cb(self, method, method2):
        def result_received_cb(client_task: ClientTask, fl_ctx: FLContext):
            controller = fl_ctx.get_prop(key="controller")
            new_task = create_task("__new_test_task")
            inner_launch_thread = threading.Thread(
                target=launch_task,
                kwargs={
                    "controller": controller,
                    "task": new_task,
                    "method": method2,
                    "fl_ctx": fl_ctx,
                    "kwargs": {"targets": [client_task.client]},
                },
            )
            get_ready(inner_launch_thread)
            inner_launch_thread.join()

        controller, fl_ctx = self.start_controller()
        fl_ctx.set_prop("controller", controller)
        client = create_client(name="__test_client")
        task = create_task("__test_task", result_received_cb=result_received_cb)
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
        launch_thread.start()

        task_name_out = ""
        client_task_id = ""
        data = None
        while task_name_out == "":
            task_name_out, client_task_id, data = controller.process_task_request(client, fl_ctx)
            time.sleep(0.1)
        assert task_name_out == "__test_task"

        controller.process_submission(
            client=client, task_name="__test_task", task_id=client_task_id, fl_ctx=fl_ctx, result=data
        )
        controller._check_tasks()
        assert controller.get_num_standing_tasks() == 1
        new_task_name_out = ""
        while new_task_name_out == "":
            new_task_name_out, _, _ = controller.process_task_request(client, fl_ctx)
            time.sleep(0.1)
        assert new_task_name_out == "__new_test_task"
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)
