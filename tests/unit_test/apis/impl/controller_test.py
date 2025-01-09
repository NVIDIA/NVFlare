# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from unittest.mock import Mock

import pytest

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, SendOrder, Task, TaskCompletionStatus
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.impl.controller import Controller
from nvflare.apis.impl.wf_comm_server import WFCommServer
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.apis.signal import Signal

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


def launch_task(controller, method, task, fl_ctx, kwargs):
    if method == "broadcast":
        if "min_responses" in kwargs:
            min_responses = kwargs.pop("min_responses")
        elif "targets" in kwargs:
            min_responses = len(kwargs["targets"])
        else:
            min_responses = 1
        controller.broadcast(task=task, fl_ctx=fl_ctx, min_responses=min_responses, **kwargs)
    elif method == "broadcast_and_wait":
        if "min_responses" in kwargs:
            min_responses = kwargs.pop("min_responses")
        elif "targets" in kwargs:
            min_responses = len(kwargs["targets"])
        else:
            min_responses = 1
        controller.broadcast_and_wait(task=task, fl_ctx=fl_ctx, min_responses=min_responses, **kwargs)
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


def _setup_system(num_clients=1):
    clients_list = [create_client(f"__test_client{i}") for i in range(num_clients)]
    mock_server_engine = Mock(spec=ServerEngineSpec)
    context_manager = FLContextManager(
        engine=mock_server_engine,
        identity_name="__mock_server_engine",
        job_id="job_1",
        public_stickers={},
        private_stickers={},
    )
    mock_server_engine.new_context.return_value = context_manager.new_context()
    mock_server_engine.get_clients.return_value = clients_list

    controller = DummyController()
    fl_ctx = mock_server_engine.new_context()
    communicator = WFCommServer()
    controller.set_communicator(communicator)
    controller.initialize(fl_ctx)
    controller.communicator.initialize_run(fl_ctx=fl_ctx)
    return controller, mock_server_engine, fl_ctx, clients_list


class TestController:
    NO_RELAY = ["broadcast", "broadcast_and_wait", "send", "send_and_wait"]
    RELAY = ["relay", "relay_and_wait"]
    ALL_APIS = NO_RELAY + RELAY

    @staticmethod
    def setup_system(num_of_clients=1):
        controller, server_engine, fl_ctx, clients_list = _setup_system(num_clients=num_of_clients)
        return controller, fl_ctx, clients_list

    @staticmethod
    def teardown_system(controller, fl_ctx):
        controller.communicator.finalize_run(fl_ctx=fl_ctx)


class TestTaskManagement(TestController):
    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    @pytest.mark.parametrize("num_of_tasks", [2, 3, 4])
    def test_add_task(self, method, num_of_tasks):
        controller, fl_ctx, clients = self.setup_system()
        client = clients[0]

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
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("method1", TestController.ALL_APIS)
    @pytest.mark.parametrize("method2", TestController.ALL_APIS)
    def test_reuse_same_task(self, method1, method2):
        controller, fl_ctx, clients = self.setup_system()
        client = clients[0]
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
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    @pytest.mark.parametrize("num_of_start_tasks", [2, 3, 4])
    @pytest.mark.parametrize("num_of_cancel_tasks", [1, 2])
    def test_check_task_remove_cancelled_tasks(self, method, num_of_start_tasks, num_of_cancel_tasks):
        controller, fl_ctx, clients = self.setup_system()
        client = clients[0]

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
        controller.communicator.check_tasks()
        assert controller.get_num_standing_tasks() == (num_of_start_tasks - num_of_cancel_tasks)
        controller.cancel_all_tasks()
        for thread in all_threads:
            thread.join()
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    @pytest.mark.parametrize("num_client_requests", [1, 2, 3, 4])
    def test_client_request_after_cancel_task(self, method, num_client_requests):
        controller, fl_ctx, clients = self.setup_system()
        client = clients[0]
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
            _, task_id, data = controller.communicator.process_task_request(client, fl_ctx)
            # check if task_id is empty means this task is not assigned
            assert task_id == ""
            assert data is None
        controller.communicator.check_tasks()
        assert controller.get_num_standing_tasks() == 0
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        launch_thread.join()
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    def test_client_submit_result_after_cancel_task(self, method):
        controller, fl_ctx, clients = self.setup_system()
        client = clients[0]
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
        task_name_out, client_task_id, data = controller.communicator.process_task_request(client, fl_ctx)
        controller.cancel_task(task)
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        time.sleep(1)
        print(controller.communicator._tasks)

        # in here we make up client results:
        result = Shareable()
        result["result"] = "result"

        with pytest.raises(RuntimeError, match="Unknown task: __test_task from client __test_client0."):
            controller.communicator.process_submission(
                client=client, task_name="__test_task", task_id=client_task_id, fl_ctx=fl_ctx, result=result
            )

        assert task.last_client_task_map["__test_client0"].result is None
        launch_thread.join()
        self.teardown_system(controller, fl_ctx)


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
        controller, fl_ctx, clients = self.setup_system()
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
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("method", ["broadcast", "broadcast_and_wait"])
    @pytest.mark.parametrize("kwargs,error,msg", _get_broadcast_test_cases())
    def test_broadcast_invalid_input(self, method, kwargs, error, msg):
        controller, fl_ctx, clients = self.setup_system()
        with pytest.raises(error, match=msg):
            if method == "broadcast":
                controller.broadcast(**kwargs)
            else:
                controller.broadcast_and_wait(**kwargs)
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("method", ["send", "send_and_wait"])
    @pytest.mark.parametrize("kwargs,error,msg", _get_send_test_cases())
    def test_send_invalid_input(self, method, kwargs, error, msg):
        controller, fl_ctx, clients = self.setup_system()
        with pytest.raises(error, match=msg):
            if method == "send":
                controller.send(**kwargs)
            else:
                controller.send_and_wait(**kwargs)
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("method", ["relay", "relay_and_wait"])
    @pytest.mark.parametrize("kwargs,error,msg", _get_relay_test_cases())
    def test_relay_invalid_input(self, method, kwargs, error, msg):
        controller, fl_ctx, clients = self.setup_system()
        with pytest.raises(error, match=msg):
            if method == "relay":
                controller.relay(**kwargs)
            else:
                controller.relay_and_wait(**kwargs)
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    @pytest.mark.parametrize("kwargs,error,msg", _get_process_submission_test_cases())
    def test_process_submission_invalid_input(self, method, kwargs, error, msg):
        controller, fl_ctx, clients = self.setup_system()

        with pytest.raises(error, match=msg):
            controller.communicator.process_submission(**kwargs)
        self.teardown_system(controller, fl_ctx)


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
            10,
            task_name,
            input_data,
            task_done_cb,
            "_".join([f"__test_client{i}" for i in range(10)]),
        ),
        (
            "broadcast_and_wait",
            10,
            task_name,
            input_data,
            task_done_cb,
            "_".join([f"__test_client{i}" for i in range(10)]),
        ),
        ("send", 1, task_name, input_data, task_done_cb, "__test_client0"),
        ("send_and_wait", 1, task_name, input_data, task_done_cb, "__test_client0"),
        ("relay", 1, task_name, input_data, task_done_cb, "__test_client0"),
        ("relay_and_wait", 1, task_name, input_data, task_done_cb, "__test_client0"),
    ]
    return test_cases


def clients_pull_and_submit_result(controller, ctx, clients, task_name):
    client_task_ids = []
    num_of_clients = len(clients)
    for i in range(num_of_clients):
        task_name_out, client_task_id, data = controller.communicator.process_task_request(clients[i], ctx)
        assert task_name_out == task_name
        client_task_ids.append(client_task_id)

    for client, client_task_id in zip(clients, client_task_ids):
        data = Shareable()
        controller.communicator.process_submission(
            client=client, task_name=task_name, task_id=client_task_id, fl_ctx=ctx, result=data
        )


class TestCallback(TestController):
    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    def test_before_task_sent_cb(self, method):
        def before_task_sent_cb(client_task: ClientTask, **kwargs):
            client_task.task.data["_test_data"] = client_task.client.name

        client_name = "__test_client0"
        controller, fl_ctx, clients = self.setup_system()
        client = clients[0]
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

        task_name_out, _, data = controller.communicator.process_task_request(client, fl_ctx)

        assert data["_test_data"] == client_name
        controller.cancel_task(task)
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        launch_thread.join()
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    def test_result_received_cb(self, method):
        def result_received_cb(client_task: ClientTask, **kwargs):
            client_task.result["_test_data"] = client_task.client.name

        client_name = "__test_client0"
        input_data = Shareable()
        input_data["_test_data"] = "_old_data"
        controller, fl_ctx, clients = self.setup_system()
        client = clients[0]
        task = create_task("__test_task", data=input_data, result_received_cb=result_received_cb)
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
        task_name_out, client_task_id, data = controller.communicator.process_task_request(client, fl_ctx)
        controller.communicator.process_submission(
            client=client, task_name="__test_task", task_id=client_task_id, fl_ctx=fl_ctx, result=data
        )

        assert task.last_client_task_map[client_name].result["_test_data"] == client_name
        controller.communicator.check_tasks()
        assert task.completion_status == TaskCompletionStatus.OK
        launch_thread.join()
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("task_complete", ["normal", "timeout", "cancel"])
    @pytest.mark.parametrize(
        "method,num_clients,task_name,input_data,cb,expected", _get_task_done_callback_test_cases()
    )
    def test_task_done_cb(self, method, num_clients, task_name, input_data, cb, expected, task_complete):
        controller, fl_ctx, clients = self.setup_system(num_clients)

        timeout = 0 if task_complete != "timeout" else 1
        task = create_task("__test_task", data=input_data, task_done_cb=cb, timeout=timeout)
        launch_thread = threading.Thread(
            target=launch_task,
            kwargs={
                "controller": controller,
                "task": task,
                "method": method,
                "fl_ctx": fl_ctx,
                "kwargs": {"targets": clients},
            },
        )
        get_ready(launch_thread)

        client_task_ids = len(clients) * [None]
        for i, client in enumerate(clients):
            task_name_out, client_task_ids[i], _ = controller.communicator.process_task_request(client, fl_ctx)

            if task_name_out == "":
                client_task_ids[i] = None

        # in here we make up client results:
        result = Shareable()
        result["result"] = "result"

        for client, client_task_id in zip(clients, client_task_ids):
            if client_task_id is not None:
                if task_complete == "normal":
                    controller.communicator.process_submission(
                        client=client, task_name="__test_task", task_id=client_task_id, fl_ctx=fl_ctx, result=result
                    )
        if task_complete == "timeout":
            time.sleep(timeout)
            controller.communicator.check_tasks()
            assert task.completion_status == TaskCompletionStatus.TIMEOUT
        elif task_complete == "cancel":
            controller.cancel_task(task)
            assert task.completion_status == TaskCompletionStatus.CANCELLED
        controller.communicator.check_tasks()
        assert task.props[task_name] == expected
        assert controller.get_num_standing_tasks() == 0
        launch_thread.join()
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    def test_cancel_task_before_send_cb(self, method):
        def before_task_sent_cb(client_task: ClientTask, **kwargs):
            client_task.task.completion_status = TaskCompletionStatus.CANCELLED

        controller, fl_ctx, clients = self.setup_system()
        client = clients[0]
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

        task_name_out, client_task_id, data = controller.communicator.process_task_request(client, fl_ctx)
        assert task_name_out == ""
        assert client_task_id == ""

        launch_thread.join()
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    def test_cancel_task_result_received_cb(self, method):
        # callback needs to have args name client_task and fl_ctx
        def result_received_cb(client_task: ClientTask, **kwargs):
            client_task.task.completion_status = TaskCompletionStatus.CANCELLED

        controller, fl_ctx, clients = self.setup_system()
        client1 = clients[0]
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

        task_name_out, client_task_id, data = controller.communicator.process_task_request(client1, fl_ctx)

        result = Shareable()
        result["__result"] = "__test_result"
        controller.communicator.process_submission(
            client=client1, task_name="__test_task", task_id=client_task_id, fl_ctx=fl_ctx, result=result
        )
        assert task.last_client_task_map["__test_client0"].result == result

        task_name_out, client_task_id, data = controller.communicator.process_task_request(client2, fl_ctx)
        assert task_name_out == ""
        assert client_task_id == ""

        launch_thread.join()
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    @pytest.mark.parametrize("method2", ["broadcast", "send", "relay"])
    def test_schedule_task_before_send_cb(self, method, method2):
        # callback needs to have args name client_task and fl_ctx
        def before_task_sent_cb(client_task: ClientTask, fl_ctx: FLContext):
            inner_controller = ctx.get_prop(key="controller")
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

        controller, ctx, clients = self.setup_system()
        ctx.set_prop("controller", controller)
        client = clients[0]
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
            task_name_out, _, _ = controller.communicator.process_task_request(client, ctx)
            time.sleep(0.1)
        assert task_name_out == "__test_task"
        new_task_name_out = ""
        while new_task_name_out == "":
            new_task_name_out, _, _ = controller.communicator.process_task_request(client, ctx)
            time.sleep(0.1)
        assert new_task_name_out == "__new_test_task"

        controller.cancel_task(task)
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        launch_thread.join()
        self.teardown_system(controller, ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    @pytest.mark.parametrize("method2", ["broadcast", "send", "relay"])
    def test_schedule_task_result_received_cb(self, method, method2):
        # callback needs to have args name client_task and fl_ctx
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

        controller, ctx, clients = self.setup_system()
        ctx.set_prop("controller", controller)
        client = clients[0]
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
            task_name_out, client_task_id, data = controller.communicator.process_task_request(client, ctx)
            time.sleep(0.1)
        assert task_name_out == "__test_task"

        controller.communicator.process_submission(
            client=client, task_name="__test_task", task_id=client_task_id, fl_ctx=ctx, result=data
        )
        controller.communicator.check_tasks()
        assert controller.get_num_standing_tasks() == 1
        new_task_name_out = ""
        while new_task_name_out == "":
            new_task_name_out, _, _ = controller.communicator.process_task_request(client, ctx)
            time.sleep(0.1)
        assert new_task_name_out == "__new_test_task"
        launch_thread.join()
        self.teardown_system(controller, ctx)

    def test_broadcast_schedule_task_in_result_received_cb(self):
        num_of_clients = 100
        controller, ctx, clients = self.setup_system(num_of_clients=num_of_clients)

        # callback needs to have args name client_task and fl_ctx
        def result_received_cb(client_task: ClientTask, fl_ctx: FLContext):
            inner_controller = fl_ctx.get_prop(key="controller")
            client = client_task.client
            new_task = create_task(f"__new_test_task_{client.name}")
            inner_launch_thread = threading.Thread(
                target=launch_task,
                kwargs={
                    "controller": inner_controller,
                    "task": new_task,
                    "method": "broadcast",
                    "fl_ctx": fl_ctx,
                    "kwargs": {"targets": clients},
                },
            )
            get_ready(inner_launch_thread)
            inner_launch_thread.join()

        ctx.set_prop("controller", controller)
        task = create_task("__test_task", result_received_cb=result_received_cb)
        launch_thread = threading.Thread(
            target=launch_task,
            kwargs={
                "controller": controller,
                "task": task,
                "method": "broadcast",
                "fl_ctx": ctx,
                "kwargs": {"targets": clients},
            },
        )
        launch_thread.start()

        clients_pull_and_submit_result(controller=controller, ctx=ctx, clients=clients, task_name="__test_task")
        controller.communicator.check_tasks()
        assert controller.get_num_standing_tasks() == num_of_clients

        for i in range(num_of_clients):
            clients_pull_and_submit_result(
                controller=controller, ctx=ctx, clients=clients, task_name=f"__new_test_task_{clients[i].name}"
            )
            controller.communicator.check_tasks()
            assert controller.get_num_standing_tasks() == num_of_clients - (i + 1)

        launch_thread.join()
        self.teardown_system(controller, ctx)


class TestBasic(TestController):
    @pytest.mark.parametrize("task_name,client_name", [["__test_task", "__test_client0"]])
    def test_process_submission_invalid_task(self, task_name, client_name):
        controller, fl_ctx, clients = self.setup_system()
        client = clients[0]
        with pytest.raises(RuntimeError, match=f"Unknown task: {task_name} from client {client_name}."):
            controller.communicator.process_submission(
                client=client, task_name=task_name, task_id=str(uuid.uuid4()), fl_ctx=FLContext(), result=Shareable()
            )
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    @pytest.mark.parametrize("num_client_requests", [1, 2, 3, 4])
    def test_process_task_request_client_request_multiple_times(self, method, num_client_requests):
        controller, fl_ctx, clients = self.setup_system()
        client = clients[0]
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
            task_name_out, _, data = controller.communicator.process_task_request(client, fl_ctx)
            assert task_name_out == "__test_task"
            assert data == input_data
        assert task.last_client_task_map["__test_client0"].task_send_count == num_client_requests
        controller.cancel_task(task)
        launch_thread.join()
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    def test_process_submission(self, method):
        controller, fl_ctx, clients = self.setup_system()
        client = clients[0]
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

        task_name_out, client_task_id, data = controller.communicator.process_task_request(client, fl_ctx)
        # in here we make up client results:
        result = Shareable()
        result["result"] = "result"

        controller.communicator.process_submission(
            client=client, task_name="__test_task", task_id=client_task_id, fl_ctx=fl_ctx, result=result
        )
        assert task.last_client_task_map["__test_client0"].result == result
        launch_thread.join()
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    @pytest.mark.parametrize("timeout", [1, 2])
    def test_task_timeout(self, method, timeout):
        controller, fl_ctx, clients = self.setup_system()
        client = clients[0]
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
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    def test_cancel_task(self, method):
        controller, fl_ctx, clients = self.setup_system()
        client = clients[0]
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
        controller.communicator.check_tasks()
        assert controller.get_num_standing_tasks() == 0
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        launch_thread.join()
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("method", TestController.ALL_APIS)
    def test_cancel_all_tasks(self, method):
        controller, fl_ctx, clients = self.setup_system()
        client = clients[0]
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
        controller.communicator.check_tasks()
        assert controller.get_num_standing_tasks() == 0
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        assert task1.completion_status == TaskCompletionStatus.CANCELLED
        launch_thread.join()
        self.teardown_system(controller, fl_ctx)


@pytest.mark.parametrize("method", ["broadcast", "broadcast_and_wait"])
class TestBroadcastBehavior(TestController):
    @pytest.mark.parametrize("num_of_clients", [1, 2, 3, 4])
    def test_client_receive_only_one_task(self, method, num_of_clients):
        controller, fl_ctx, clients = self.setup_system(num_of_clients=num_of_clients)

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
                task_name_out, client_task_id, data = controller.communicator.process_task_request(client, fl_ctx)
                time.sleep(0.1)
            assert task_name_out == "__test_task"
            assert data == input_data
            assert task.last_client_task_map[client.name].task_send_count == 1
            assert controller.get_num_standing_tasks() == 1
            _, next_client_task_id, _ = controller.communicator.process_task_request(client, fl_ctx)
            assert next_client_task_id == client_task_id
            assert task.last_client_task_map[client.name].task_send_count == 2

            result = Shareable()
            result["result"] = "result"
            controller.communicator.process_submission(
                client=client,
                task_name="__test_task",
                task_id=client_task_id,
                fl_ctx=fl_ctx,
                result=result,
            )
            assert task.last_client_task_map[client.name].result == result

        controller.communicator.check_tasks()
        assert task.completion_status == TaskCompletionStatus.OK
        launch_thread.join()
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("num_of_clients", [1, 2, 3, 4])
    def test_only_client_in_target_will_get_task(self, method, num_of_clients):
        controller, fl_ctx, clients = self.setup_system(num_of_clients=num_of_clients)

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
            task_name_out, client_task_id, data = controller.communicator.process_task_request(clients[0], fl_ctx)
            time.sleep(0.1)
        assert task_name_out == "__test_task"
        assert data == input_data
        assert task.last_client_task_map[clients[0].name].task_send_count == 1
        assert controller.get_num_standing_tasks() == 1

        for client in clients[1:]:
            task_name_out, client_task_id, data = controller.communicator.process_task_request(client, fl_ctx)
            assert task_name_out == ""
            assert client_task_id == ""

        controller.cancel_task(task)
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        launch_thread.join()
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("min_responses", [1, 2, 3, 4])
    def test_task_only_exit_when_min_responses_received(self, method, min_responses):
        controller, fl_ctx, clients = self.setup_system(num_of_clients=min_responses)

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
            task_name_out, client_task_id, data = controller.communicator.process_task_request(client, fl_ctx)
            client_task_ids.append(client_task_id)
            assert task_name_out == "__test_task"

        for client, client_task_id in zip(clients, client_task_ids):
            result = Shareable()
            controller.communicator.check_tasks()
            assert controller.get_num_standing_tasks() == 1
            controller.communicator.process_submission(
                client=client, task_name="__test_task", task_id=client_task_id, result=result, fl_ctx=fl_ctx
            )

        controller.communicator.check_tasks()
        assert controller.get_num_standing_tasks() == 0
        assert task.completion_status == TaskCompletionStatus.OK
        launch_thread.join()
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("min_responses", [1, 2, 3, 4])
    @pytest.mark.parametrize("wait_time_after_min_received", [1, 2])
    def test_task_exit_quickly_when_all_responses_received(self, method, min_responses, wait_time_after_min_received):
        controller, fl_ctx, clients = self.setup_system(num_of_clients=min_responses)

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
            task_name_out, client_task_id, data = controller.communicator.process_task_request(client, fl_ctx)
            client_task_ids.append(client_task_id)
            assert task_name_out == "__test_task"

        for client, client_task_id in zip(clients, client_task_ids):
            result = Shareable()
            controller.communicator.process_submission(
                client=client, task_name="__test_task", task_id=client_task_id, result=result, fl_ctx=fl_ctx
            )

        controller.communicator.check_tasks()
        assert controller.get_num_standing_tasks() == 0
        assert task.completion_status == TaskCompletionStatus.OK
        launch_thread.join()
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("num_clients", [1, 2, 3, 4])
    def test_min_resp_is_zero_task_only_exit_when_all_client_task_done(self, method, num_clients):
        controller, fl_ctx, clients = self.setup_system(num_clients)
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
            task_name_out, client_task_id, data = controller.communicator.process_task_request(client, fl_ctx)
            client_task_ids.append(client_task_id)
            assert task_name_out == "__test_task"
            controller.communicator.check_tasks()
            assert controller.get_num_standing_tasks() == 1

        for client, client_task_id in zip(clients, client_task_ids):
            controller.communicator.check_tasks()
            assert controller.get_num_standing_tasks() == 1
            result = Shareable()
            controller.communicator.process_submission(
                client=client, task_name="__test_task", task_id=client_task_id, result=result, fl_ctx=fl_ctx
            )

        controller.communicator.check_tasks()
        assert controller.get_num_standing_tasks() == 0
        assert task.completion_status == TaskCompletionStatus.OK
        launch_thread.join()
        self.teardown_system(controller, fl_ctx)


def _process_task_request_test_cases():
    """Returns a list of
    targets, request_client, dynamic_targets, task_assignment_timeout, time_before_first_request,
    expected_to_get_task, expected_targets
    """
    clients = [create_client(f"__test_client{i}") for i in range(3)]
    client_names = [c.name for c in clients]

    # For each task_assignment_timeout pass by, the "window" grow by one
    #   all the targets within the window can receive a task
    dynamic_targets_cases = [
        # For dynamic window, the new target will be appended to the end of the list.
        (clients[1:], clients[0], True, 2, 0, False, [clients[1].name, clients[2].name, clients[0].name]),
        # test_client1 is the first, is in target and is in window so it gets task
        (clients[1:], clients[1], True, 2, 0, True, client_names[1:]),
        # test_client2 is the second, is in target BUT not in window so it does not get task
        (clients[1:], clients[2], True, 2, 0, False, client_names[1:]),
        # test_client1 is not in target AND not in window so it does not get task
        ([clients[0]], clients[1], True, 2, 0, False, [clients[0].name, clients[1].name]),
        # the "time_before_first_request" takes too long the SequentialRelayTaskManager times out
        ([clients[0]], clients[1], True, 1, 5, False, [clients[0].name]),
        # the "time_before_first_request" > task_assignment_timeout (1 second), so the window will move to second
        ([clients[0]], clients[1], True, 1, 1.5, True, [clients[0].name, clients[1].name]),
        ([clients[0], clients[0]], clients[0], True, 1, 0, True, [clients[0].name, clients[0].name]),
        (None, clients[0], True, 1, 0, True, [clients[0].name]),
    ]

    # For static target, if the requesting site is not in the targets list,
    #  it will not get any tasks
    static_targets_cases = [
        (clients[1:], clients[0], False, 2, 0, False, client_names[1:]),
        (clients[1:], clients[1], False, 2, 0, True, client_names[1:]),
        (clients[1:], clients[2], False, 2, 0, False, client_names[1:]),
        (clients[1:], clients[0], False, 1, 1.5, False, client_names[1:]),
        (clients[1:], clients[1], False, 1, 1.5, True, client_names[1:]),
        (clients[1:], clients[2], False, 1, 1.5, True, client_names[1:]),
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


@pytest.mark.parametrize("method", ["relay", "relay_and_wait"])
class TestRelayBehavior(TestController):
    @pytest.mark.parametrize("send_order", [SendOrder.ANY, SendOrder.SEQUENTIAL])
    def test_only_client_in_target_will_get_task(self, method, send_order):
        controller, fl_ctx, clients = self.setup_system(4)
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
            task_name_out, client_task_id, data = controller.communicator.process_task_request(clients[0], fl_ctx)
            time.sleep(0.1)
        assert task_name_out == "__test_task"
        assert data == input_data
        assert task.last_client_task_map[clients[0].name].task_send_count == 1
        assert controller.get_num_standing_tasks() == 1

        for client in clients[1:]:
            task_name_out, client_task_id, data = controller.communicator.process_task_request(client, fl_ctx)
            assert task_name_out == ""
            assert client_task_id == ""

        controller.cancel_task(task)
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        launch_thread.join()
        self.teardown_system(controller, fl_ctx)

    def test_task_assignment_timeout_sequential_order_only_client_in_target_will_get_task(self, method):
        task_assignment_timeout = 3
        task_result_timeout = 3
        controller, fl_ctx, clients = self.setup_system(4)
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
            task_name_out, client_task_id, data = controller.communicator.process_task_request(client, fl_ctx)
            assert task_name_out == ""
            assert client_task_id == ""

        controller.cancel_task(task)
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        launch_thread.join()
        self.teardown_system(controller, fl_ctx)

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
        controller, fl_ctx, clients = self.setup_system()
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

        task_name, task_id, data = controller.communicator.process_task_request(client=request_client, fl_ctx=fl_ctx)
        client_get_a_task = True if task_name == "__test_task" else False

        assert client_get_a_task == expected_to_get_task
        assert task.targets == expected_targets

        controller.cancel_task(task)
        launch_thread.join()
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("targets", _get_sequential_sequence_test_cases())
    def test_sequential_sequence(self, method, targets):
        controller, fl_ctx, clients = self.setup_system()
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
                task_name, task_id, data = controller.communicator.process_task_request(client=c, fl_ctx=fl_ctx)
                if task_name != "":
                    client_result = Shareable()
                    client_result["result"] = f"{c.name}"
                    if task_id not in client_tasks_and_results:
                        client_tasks_and_results[task_id] = (c, task_name, client_result)
                    assert c == targets[expected_client_index]

            for task_id in client_tasks_and_results.keys():
                c, task_name, client_result = client_tasks_and_results[task_id]
                task.data["result"] += client_result["result"]
                controller.communicator.process_submission(
                    client=c, task_name=task_name, task_id=task_id, result=client_result, fl_ctx=fl_ctx
                )
                assert task.last_client_task_map[c.name].result == client_result
            expected_client_index += 1

        launch_thread.join()
        assert task.data["result"] == "start_" + "".join([c.name for c in targets])
        self.teardown_system(controller, fl_ctx)

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
        controller, fl_ctx, clients = self.setup_system()
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
                        task_name_out, client_task_id, data = controller.communicator.process_task_request(
                            client, fl_ctx
                        )
                        time.sleep(0.1)
                    assert task_name_out == "__test_task"
                    assert data == input_data
                    assert task.last_client_task_map[client.name].task_send_count == 1
                else:
                    _task_name_out, _client_task_id, _ = controller.communicator.process_task_request(client, fl_ctx)
                    assert _task_name_out == ""
                    assert _client_task_id == ""

            # client side running some logic to generate result
            if expected_client_to_get_task:
                controller.communicator.check_tasks()
                assert controller.get_num_standing_tasks() == 1
                result = Shareable()
                controller.communicator.process_submission(
                    client=expected_client_to_get_task,
                    task_name=task_name_out,
                    task_id=client_task_id,
                    fl_ctx=fl_ctx,
                    result=result,
                )

        launch_thread.join()
        controller.communicator.check_tasks()
        assert controller.get_num_standing_tasks() == 0
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("send_order", [SendOrder.ANY, SendOrder.SEQUENTIAL])
    def test_process_submission_after_first_client_task_result_timeout(self, method, send_order):
        task_assignment_timeout = 1
        task_result_timeout = 2
        controller, fl_ctx, clients = self.setup_system(2)
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
            task_name_out, old_client_task_id, data = controller.communicator.process_task_request(clients[0], fl_ctx)
            time.sleep(0.1)
        assert task_name_out == "__test_task"
        assert data == input_data
        assert task.last_client_task_map[clients[0].name].task_send_count == 1

        time.sleep(task_result_timeout + 1)

        # same client ask should get the same task
        task_name_out, client_task_id, data = controller.communicator.process_task_request(clients[0], fl_ctx)
        assert client_task_id == old_client_task_id
        assert task.last_client_task_map[clients[0].name].task_send_count == 2

        time.sleep(task_result_timeout + 1)

        # second client ask should get a task since task_result_timeout passed
        task_name_out, client_task_id_1, data = controller.communicator.process_task_request(clients[1], fl_ctx)
        assert task_name_out == "__test_task"
        assert data == input_data
        assert task.last_client_task_map[clients[1].name].task_send_count == 1

        # then we get back first client's result
        result = Shareable()
        controller.communicator.process_submission(
            client=clients[0],
            task_name=task_name_out,
            task_id=client_task_id,
            fl_ctx=fl_ctx,
            result=result,
        )

        # need to make sure the header is set
        assert result.get_header(ReservedHeaderKey.REPLY_IS_LATE)
        controller.communicator.check_tasks()
        assert controller.get_num_standing_tasks() == 1
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("send_order", [SendOrder.ANY, SendOrder.SEQUENTIAL])
    def test_process_submission_all_client_task_result_timeout(self, method, send_order):
        task_assignment_timeout = 1
        task_result_timeout = 2
        controller, fl_ctx, clients = self.setup_system(2)
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
                task_name_out, old_client_task_id, data = controller.communicator.process_task_request(client, fl_ctx)
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
        self.teardown_system(controller, fl_ctx)


def _assert_other_clients_get_no_task(controller, fl_ctx, client_idx: int, clients):
    """Assert clients get no task."""
    assert client_idx < len(clients)
    for i, client in enumerate(clients):
        if i == client_idx:
            continue
        _task_name_out, _client_task_id, data = controller.communicator.process_task_request(client, fl_ctx)
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
        controller, fl_ctx, clients = self.setup_system()
        client = clients[0]
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
        _task_name_out, _client_task_id, data = controller.communicator.process_task_request(client, fl_ctx)
        assert _task_name_out == ""
        assert _client_task_id == ""

        controller.cancel_task(task)
        controller.communicator.check_tasks()
        assert controller.get_num_standing_tasks() == 0
        launch_thread.join()
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("targets,send_order,client_idx", _get_process_task_request_test_cases())
    def test_process_task_request_expected_client_get_task_and_unexpected_clients_get_nothing(
        self, method, targets, send_order, client_idx
    ):
        controller, fl_ctx, clients = self.setup_system()
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
            task_name_out, client_task_id, data = controller.communicator.process_task_request(
                targets[client_idx], fl_ctx
            )
            time.sleep(0.1)
        assert task_name_out == "__test_task"
        assert data == input_data
        assert task.last_client_task_map[targets[client_idx].name].task_send_count == 1

        # other clients
        _assert_other_clients_get_no_task(controller=controller, fl_ctx=fl_ctx, client_idx=client_idx, clients=targets)

        controller.cancel_task(task)
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        controller.communicator.check_tasks()
        assert controller.get_num_standing_tasks() == 0
        launch_thread.join()
        self.teardown_system(controller, fl_ctx)

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
        controller, fl_ctx, clients = self.setup_system()
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
                    task_name_out, client_task_id, data = controller.communicator.process_task_request(client, fl_ctx)
                    time.sleep(0.1)
                assert task_name_out == "__test_task"
                assert data == input_data
                assert task.last_client_task_map[client.name].task_send_count == 1
            else:
                task_name_out, client_task_id, data = controller.communicator.process_task_request(client, fl_ctx)
                assert task_name_out == ""
                assert client_task_id == ""

        controller.cancel_task(task)
        assert task.completion_status == TaskCompletionStatus.CANCELLED
        launch_thread.join()
        self.teardown_system(controller, fl_ctx)

    @pytest.mark.parametrize("num_of_clients", [1, 2, 3])
    def test_send_only_one_task_and_exit_when_client_task_done(self, method, num_of_clients):
        controller, fl_ctx, clients = self.setup_system()

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
            task_name_out, client_task_id, data = controller.communicator.process_task_request(clients[0], fl_ctx)
            time.sleep(0.1)
        assert task_name_out == "__test_task"
        assert data == input_data
        assert task.last_client_task_map[clients[0].name].task_send_count == 1

        # once a client gets a task, other clients should not get task
        _assert_other_clients_get_no_task(controller=controller, fl_ctx=fl_ctx, client_idx=0, clients=clients)

        controller.communicator.check_tasks()
        assert controller.get_num_standing_tasks() == 1

        controller.communicator.process_submission(
            client=clients[0], task_name="__test_task", task_id=client_task_id, fl_ctx=fl_ctx, result=data
        )

        controller.communicator.check_tasks()
        assert controller.get_num_standing_tasks() == 0
        assert task.completion_status == TaskCompletionStatus.OK
        launch_thread.join()
        self.teardown_system(controller, fl_ctx)
