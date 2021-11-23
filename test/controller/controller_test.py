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

import logging
import time
import uuid

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import Task
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable
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


# - provide a easy way for researchers to test their own Controller / their own control loop?
# - how can they write their own test cases, simulating different client in diff. scenario...


class DummyController(Controller):
    def __init__(self):
        super().__init__(task_check_period=0.1)

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        print(f"Entering control loop of TestController")

    def start_controller(self, fl_ctx: FLContext):
        print("Start controller")

    def stop_controller(self, fl_ctx: FLContext):
        print("Stop controller")

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        raise RuntimeError(f"Unknown task: {task_name} from client {client.name}.")


class MockEngine:
    def __init__(self, run_num=0):
        self.fl_ctx_mgr = FLContextManager(
            engine=self,
            identity_name="__mock_engine",
            run_num=run_num,
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
