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

import logging
import re
import threading
import time
import uuid
from itertools import permutations

import pytest

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, SendOrder, Task, TaskCompletionStatus
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.apis.signal import Signal
from tests.unit_test.utils import skip_if_quick

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def create_task(name, data=None, timeout=0, before_task_sent_cb=None, result_received_cb=None, task_done_cb=None):
    data = Shareable() if data is None else data
    task = Task(
        name=name,
        data=data,
        timeout=timeout,
        before_task_sent_cb=before_task_sent_cb,
        result_received_cb=result_received_cb,
        task_done_cb=task_done_cb,
    )
    return task


def create_client(name, token=None):
    token = str(uuid.uuid4()) if token is None else token
    return Client(name=name, token=token)


# TODO:
#   - provide a easy way for researchers to test their own Controller / their own control loop?
#   - how can they write their own test cases, simulating different client in diff. scenario...


class DummyController(Controller):
    def __init__(self):
        super().__init__(task_check_period=0.1)

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        print(f"Entering control loop of {self.__class__.__name__}")

    def start_controller(self, fl_ctx: FLContext):
        print("Start controller")

    def stop_controller(self, fl_ctx: FLContext):
        print("Stop controller")

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        raise RuntimeError(f"Unknown task: {task_name} from client {client.name}.")


class MockEngine:
    def __init__(self, run_name="exp1"):
        self.fl_ctx_mgr = FLContextManager(
            engine=self,
            identity_name="__mock_engine",
            job_id=run_name,
            public_stickers={},
            private_stickers={},
        )

    def new_context(self):
        return self.fl_ctx_mgr.new_context()

    def fire_event(self, event_type: str, fl_ctx: FLContext):
        pass


def launch_task(controller, method, task, fl_ctx, kwargs):
    if method == "broadcast":
        controller.broadcast(task=task, fl_ctx=fl_ctx, **kwargs)
    elif method == "broadcast_and_wait":
        controller.broadcast_and_wait(task=task, fl_ctx=fl_ctx, **kwargs)
    elif method == "send":
        controller.send(task=task, fl_ctx=fl_ctx, **kwargs)
    elif method == "send_and_wait":
        controller.send_and_wait(task=task, fl_ctx=fl_ctx, **kwargs)
    elif method == "relay":
        controller.relay(task=task, fl_ctx=fl_ctx, **kwargs)
    elif method == "relay_and_wait":
        controller.relay_and_wait(task=task, fl_ctx=fl_ctx, **kwargs)


def get_ready(thread, sleep_time=0.1):
    thread.start()
    time.sleep(sleep_time)


def get_controller_and_engine():
    return DummyController(), MockEngine()


class TestController:
    NO_RELAY = ["broadcast", "broadcast_and_wait", "send", "send_and_wait"]
    RELAY = ["relay", "relay_and_wait"]
    ALL_APIS = NO_RELAY + RELAY

    @staticmethod
    def start_controller():
        """starts the controller"""
        controller, engine = get_controller_and_engine()
        fl_ctx = engine.fl_ctx_mgr.new_context()
        controller.initialize_run(fl_ctx=fl_ctx)
        return controller, fl_ctx

    @staticmethod
    def stop_controller(controller, fl_ctx):
        """stops the controller"""
        controller.finalize_run(fl_ctx=fl_ctx)


def _get_create_task_cases():
    test_cases = [
        (
            {"timeout": -1},
            ValueError,
            "timeout must be >= 0, but got -1.",
        ),
        (
            {"timeout": 1.1},
            TypeError,
            "timeout must be an int, but got <class 'float'>.",
        ),
        (
            {"before_task_sent_cb": list()},
            TypeError,
            "before_task_sent must be a callable function.",
        ),
        (
            {"result_received_cb": list()},
            TypeError,
            "result_received must be a callable function.",
        ),
        (
            {"task_done_cb": list()},
            TypeError,
            "task_done must be a callable function.",
        ),
    ]
    return test_cases


class TestTask:
    @pytest.mark.parametrize("kwargs,error,msg", _get_create_task_cases())
    def test_create_task_with_invalid_input(self, kwargs, error, msg):
        with pytest.raises(error, match=msg):
            _ = create_task(name="__test_task", **kwargs)

    def test_set_task_prop(self):
        task = create_task(name="__test_task")
        task.set_prop("hello", "world")
        assert task.props["hello"] == "world"

    def test_get_task_prop(self):
        task = create_task(name="__test_task")
        task.props["hello"] = "world"
        assert task.get_prop("hello") == "world"

    def test_set_task_prop_invalid_key(self):
        task = create_task(name="__test_task")
        with pytest.raises(ValueError, match="Keys start with __ is reserved. Please use other key."):
            task.set_prop("__test", "world")

    def test_get_task_prop_invalid_key(self):
        task = create_task(name="__test_task")
        with pytest.raises(ValueError, match="Keys start with __ is reserved. Please use other key."):
            task.get_prop("__test")


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

        with pytest.raises(RuntimeError, match="Unknown task: __test_task from client __test_client."):
            controller.process_submission(
                client=client, task_name="__test_task", task_id=client_task_id, fl_ctx=fl_ctx, result=result
            )

        assert task.last_client_task_map["__test_client"].result is None
        launch_thread.join()
        self.stop_controller(controller, fl_ctx)


def _get_common_test_cases():
    test_cases = [
        ({"task": list(), "fl_ctx": FLContext()}, TypeError, "task must be an instance of Task."),
        (
            {"task": create_task("__test"), "fl_ctx": list()},
            TypeError,
            "fl_ctx must be an instance of FLContext.",
        ),
        (
            {"task": create_task("__test"), "fl_ctx": FLContext(), "targets": dict()},
            TypeError,
            "targets must be a list of Client or string.",
        ),
        (
            {"task": create_task("__test"), "fl_ctx": FLContext(), "targets": [1, 2, 3]},
            TypeError,
            "targets must be a list of Client or string.",
        ),
    ]
    return test_cases


def _get_broadcast_test_cases():
    test_cases = [
        (
            {"task": create_task("__test"), "fl_ctx": FLContext(), "min_responses": -1},
            ValueError,
            "min_responses must >= 0.",
        ),
        (
            {"task": create_task("__test"), "fl_ctx": FLContext(), "min_responses": 1.1},
            TypeError,
            "min_responses must be an instance of int.",
        ),
    ]
    return test_cases


def _get_send_test_cases():
    test_cases = [
        (
            {"task": create_task("__test"), "fl_ctx": FLContext(), "task_assignment_timeout": -1},
            ValueError,
            "task_assignment_timeout must >= 0.",
        ),
        (
            {"task": create_task("__test"), "fl_ctx": FLContext(), "task_assignment_timeout": 1.1},
            TypeError,
            "task_assignment_timeout must be an instance of int.",
        ),
        (
            {
                "task": create_task("__test"),
                "fl_ctx": FLContext(),
                "send_order": SendOrder.SEQUENTIAL,
                "targets": [],
            },
            ValueError,
            "Targets must be provided for send",
        ),
        (
            {"task": create_task("__test"), "fl_ctx": FLContext(), "send_order": "hello"},
            TypeError,
            "send_order must be in Enum SendOrder.",
        ),
        (
            {
                "task": create_task("__test", timeout=2),
                "fl_ctx": FLContext(),
                "task_assignment_timeout": 3,
            },
            ValueError,
            re.escape("task_assignment_timeout (3) needs to be less than or equal to task.timeout (2)."),
        ),
    ]
    return test_cases


def _get_relay_test_cases():
    test_cases = [
        (
            {"task": create_task("__test"), "fl_ctx": FLContext(), "task_assignment_timeout": -1},
            ValueError,
            "task_assignment_timeout must >= 0.",
        ),
        (
            {"task": create_task("__test"), "fl_ctx": FLContext(), "task_assignment_timeout": 1.1},
            TypeError,
            "task_assignment_timeout must be an instance of int.",
        ),
        (
            {"task": create_task("__test"), "fl_ctx": FLContext(), "task_result_timeout": -1},
            ValueError,
            "task_result_timeout must >= 0.",
        ),
        (
            {"task": create_task("__test"), "fl_ctx": FLContext(), "task_result_timeout": 1.1},
            TypeError,
            "task_result_timeout must be an instance of int.",
        ),
        (
            {"task": create_task("__test"), "fl_ctx": FLContext(), "send_order": "hello"},
            TypeError,
            "send_order must be in Enum SendOrder.",
        ),
        (
            {
                "task": create_task("__test", timeout=2),
                "fl_ctx": FLContext(),
                "task_assignment_timeout": 3,
            },
            ValueError,
            re.escape("task_assignment_timeout (3) needs to be less than or equal to task.timeout (2)."),
        ),
        (
            {
                "task": create_task("__test", timeout=2),
                "fl_ctx": FLContext(),
                "task_result_timeout": 3,
            },
            ValueError,
            re.escape("task_result_timeout (3) needs to be less than or equal to task.timeout (2)."),
        ),
        (
            {"task": create_task("__test"), "fl_ctx": FLContext(), "dynamic_targets": False},
            ValueError,
            "Need to provide targets when dynamic_targets is set to False.",
        ),
    ]
    return test_cases


def _get_process_submission_test_cases():
    return [
        (
            {
                "client": None,
                "task_name": "__test_task",
                "fl_ctx": FLContext(),
                "task_id": "abc",
                "result": Shareable(),
            },
            TypeError,
            "client must be an instance of Client.",
        ),
        (
            {
                "client": create_client("__test"),
                "task_name": "__test_task",
                "fl_ctx": None,
                "task_id": "abc",
                "result": Shareable(),
            },
            TypeError,
            "fl_ctx must be an instance of FLContext.",
        ),
        (
            {
                "client": create_client("__test"),
                "task_name": "__test_task",
                "fl_ctx": FLContext(),
                "task_id": "abc",
                "result": "abc",
            },
            TypeError,
            "result must be an instance of Shareable.",
        ),
    ]


class TestInvalidInput(TestController):
    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    @pytest.mark.parametrize("kwargs,error,msg", _get_common_test_cases())
    def test_invalid_input(self, method, kwargs, error, msg):
        controller, fl_ctx = self.start_controller()
        with pytest.raises(error, match=msg):
            if method == "broadcast":
                controller.broadcast(**kwargs)
            elif method == "broadcast_and_wait":
                controller.broadcast_and_wait(**kwargs)
            elif method == "send":
                controller.send(**kwargs)
            elif method == "send_and_wait":
                controller.send_and_wait(**kwargs)
            elif method == "relay":
                controller.relay(**kwargs)
            elif method == "relay_and_wait":
                controller.relay_and_wait(**kwargs)
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("method", ["broadcast", "broadcast_and_wait"])
    @pytest.mark.parametrize("kwargs,error,msg", _get_broadcast_test_cases())
    def test_broadcast_invalid_input(self, method, kwargs, error, msg):
        controller, fl_ctx = self.start_controller()
        with pytest.raises(error, match=msg):
            if method == "broadcast":
                controller.broadcast(**kwargs)
            else:
                controller.broadcast_and_wait(**kwargs)
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("method", ["send", "send_and_wait"])
    @pytest.mark.parametrize("kwargs,error,msg", _get_send_test_cases())
    def test_send_invalid_input(self, method, kwargs, error, msg):
        controller, fl_ctx = self.start_controller()
        with pytest.raises(error, match=msg):
            if method == "send":
                controller.send(**kwargs)
            else:
                controller.send_and_wait(**kwargs)
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("method", ["relay", "relay_and_wait"])
    @pytest.mark.parametrize("kwargs,error,msg", _get_relay_test_cases())
    def test_relay_invalid_input(self, method, kwargs, error, msg):
        controller, fl_ctx = self.start_controller()
        with pytest.raises(error, match=msg):
            if method == "relay":
                controller.relay(**kwargs)
            else:
                controller.relay_and_wait(**kwargs)
        self.stop_controller(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    @pytest.mark.parametrize("kwargs,error,msg", _get_process_submission_test_cases())
    def test_process_submission_invalid_input(self, method, kwargs, error, msg):
        controller, fl_ctx = self.start_controller()

        with pytest.raises(error, match=msg):
            controller.process_submission(**kwargs)
        self.stop_controller(controller, fl_ctx)


def _get_task_done_callback_test_cases():
    task_name = "__test_task"

    def task_done_cb(task: Task, **kwargs):
        client_names = [x.client.name for x in task.client_tasks]
        expected_str = "_".join(client_names)
        task.props[task_name] = expected_str

    input_data = Shareable()
    test_cases = [
        (
            "broadcast",
            [create_client(f"__test_client{i}") for i in range(10)],
            task_name,
            input_data,
            task_done_cb,
            "_".join([f"__test_client{i}" for i in range(10)]),
        ),
        (
            "broadcast_and_wait",
            [create_client(f"__test_client{i}") for i in range(10)],
            task_name,
            input_data,
            task_done_cb,
            "_".join([f"__test_client{i}" for i in range(10)]),
        ),
        ("send", [create_client("__test_client")], task_name, input_data, task_done_cb, "__test_client"),
        ("send_and_wait", [create_client("__test_client")], task_name, input_data, task_done_cb, "__test_client"),
        ("relay", [create_client("__test_client")], task_name, input_data, task_done_cb, "__test_client"),
        ("relay_and_wait", [create_client("__test_client")], task_name, input_data, task_done_cb, "__test_client"),
    ]
    return test_cases


class TestCallback(TestController):
    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    def test_before_task_sent_cb(self, method):
        def before_task_sent_cb(client_task: ClientTask, **kwargs):
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
        def result_received_cb(client_task: ClientTask, **kwargs):
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
        def before_task_sent_cb(client_task: ClientTask, **kwargs):
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
        def result_received_cb(client_task: ClientTask, **kwargs):
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
            inner_controller = fl_ctx.get_prop(key="controller")
            new_task = create_task("__new_test_task")
            inner_launch_thread = threading.Thread(
                target=launch_task,
                kwargs={
                    "controller": inner_controller,
                    "task": new_task,
                    "method": method2,
                    "fl_ctx": fl_ctx,
                    "kwargs": {"targets": [client_task.client]},
                },
            )
            inner_launch_thread.start()
            inner_launch_thread.join()

        controller, ctx = self.start_controller()
        ctx.set_prop("controller", controller)
        client = create_client(name="__test_client")
        task = create_task("__test_task", before_task_sent_cb=before_task_sent_cb)
        launch_thread = threading.Thread(
            target=launch_task,
            kwargs={
                "controller": controller,
                "task": task,
                "method": method,
                "fl_ctx": ctx,
                "kwargs": {"targets": [client]},
            },
        )
        launch_thread.start()

        task_name_out = ""
        while task_name_out == "":
            task_name_out, _, _ = controller.process_task_request(client, ctx)
            time.sleep(0.1)
        assert task_name_out == "__test_task"
        new_task_name_out = ""
        while new_task_name_out == "":
            new_task_name_out, _, _ = controller.process_task_request(client, ctx)
            time.sleep(0.1)
        assert new_task_name_out == "__new_test_task"

        controller.cancel_task(task)
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        launch_thread.join()
        self.stop_controller(controller, ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    @pytest.mark.parametrize("method2", ["broadcast", "send", "relay"])
    def test_schedule_task_result_received_cb(self, method, method2):
        def result_received_cb(client_task: ClientTask, fl_ctx: FLContext):
            inner_controller = fl_ctx.get_prop(key="controller")
            new_task = create_task("__new_test_task")
            inner_launch_thread = threading.Thread(
                target=launch_task,
                kwargs={
                    "controller": inner_controller,
                    "task": new_task,
                    "method": method2,
                    "fl_ctx": fl_ctx,
                    "kwargs": {"targets": [client_task.client]},
                },
            )
            get_ready(inner_launch_thread)
            inner_launch_thread.join()

        controller, ctx = self.start_controller()
        ctx.set_prop("controller", controller)
        client = create_client(name="__test_client")
        task = create_task("__test_task", result_received_cb=result_received_cb)
        launch_thread = threading.Thread(
            target=launch_task,
            kwargs={
                "controller": controller,
                "task": task,
                "method": method,
                "fl_ctx": ctx,
                "kwargs": {"targets": [client]},
            },
        )
        launch_thread.start()

        task_name_out = ""
        client_task_id = ""
        data = None
        while task_name_out == "":
            task_name_out, client_task_id, data = controller.process_task_request(client, ctx)
            time.sleep(0.1)
        assert task_name_out == "__test_task"

        controller.process_submission(
            client=client, task_name="__test_task", task_id=client_task_id, fl_ctx=ctx, result=data
        )
        controller._check_tasks()
        assert controller.get_num_standing_tasks() == 1
        new_task_name_out = ""
        while new_task_name_out == "":
            new_task_name_out, _, _ = controller.process_task_request(client, ctx)
            time.sleep(0.1)
        assert new_task_name_out == "__new_test_task"
        launch_thread.join()
        self.stop_controller(controller, ctx)


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


def _process_task_request_test_cases():
    """Returns a list of
    targets, request_client, dynamic_targets, task_assignment_timeout, time_before_first_request,
    expected_to_get_task, expected_targets
    """
    clients = [create_client(f"__test_client{i}") for i in range(3)]
    client_names = [c.name for c in clients]

    dynamic_targets_cases = [
        (clients[1:], clients[0], True, 1, 0, False, [clients[1].name, clients[2].name, clients[0].name]),
        (clients[1:], clients[1], True, 1, 0, True, client_names[1:]),
        (clients[1:], clients[2], True, 1, 0, False, client_names[1:]),
        ([clients[0]], clients[1], True, 1, 0, False, [clients[0].name, clients[1].name]),
        ([clients[0]], clients[1], True, 1, 2, False, [clients[0].name]),
        ([clients[0], clients[0]], clients[0], True, 1, 0, True, [clients[0].name, clients[0].name]),
        (None, clients[0], True, 1, 0, True, [clients[0].name]),
    ]

    static_targets_cases = [
        (clients[1:], clients[0], False, 1, 0, False, client_names[1:]),
        (clients[1:], clients[1], False, 1, 0, True, client_names[1:]),
        (clients[1:], clients[2], False, 1, 0, False, client_names[1:]),
        (clients[1:], clients[0], False, 1, 2, False, client_names[1:]),
        (clients[1:], clients[1], False, 1, 2, True, client_names[1:]),
        (clients[1:], clients[2], False, 1, 2, True, client_names[1:]),
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

    Each item in request_orders is a request_order.
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
        (clients, SendOrder.SEQUENTIAL, 2, 1, clients, clients[0].name),
        (clients, SendOrder.SEQUENTIAL, 2, 1, clients_120, clients[0].name),
        (clients, SendOrder.SEQUENTIAL, 2, 1, clients_201, clients[0].name),
        (clients, SendOrder.SEQUENTIAL, 2, 3, clients, clients[0].name),
        (clients, SendOrder.SEQUENTIAL, 2, 3, clients_120, clients[1].name),
        (clients, SendOrder.SEQUENTIAL, 2, 3, clients_201, clients[0].name),
        (clients, SendOrder.SEQUENTIAL, 2, 5, clients, clients[0].name),
        (clients, SendOrder.SEQUENTIAL, 2, 5, clients_120, clients[1].name),
        (clients, SendOrder.SEQUENTIAL, 2, 5, clients_201, clients[2].name),
        (clients, SendOrder.SEQUENTIAL, 2, 3, [clients[2], clients[1], clients[0]], clients[1].name),
        (clients, SendOrder.ANY, 2, 1, clients, clients[0].name),
        (clients, SendOrder.ANY, 2, 1, clients_120, clients[1].name),
        (clients, SendOrder.ANY, 2, 1, clients_201, clients[2].name),
    ]


@pytest.mark.parametrize("method", ["send", "send_and_wait"])
class TestSendBehavior(TestController):
    @pytest.mark.parametrize("send_order", [SendOrder.ANY, SendOrder.SEQUENTIAL])
    def test_process_task_request_client_not_in_target_get_nothing(self, method, send_order):
        controller, fl_ctx = self.start_controller()
        client = create_client("__test_client")
        targets = [create_client("__target_client")]
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
        "targets,send_order,task_assignment_timeout,"
        "time_before_first_request,request_order,expected_client_to_get_task",
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
