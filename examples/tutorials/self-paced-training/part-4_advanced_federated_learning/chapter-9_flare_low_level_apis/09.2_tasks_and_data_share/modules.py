# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.apis.controller_spec import Task
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal


class HelloController(Controller):
    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        # Create the task with name "hello"
        task = Task(name="hello", data=Shareable())

        # Broadcast the task to all clients and wait for all to respond
        self.broadcast_and_wait(
            task=task,
            targets=None,  # meaning all clients, determined dynamically
            min_responses=0,  # meaning all clients
            fl_ctx=fl_ctx,
        )

    def start_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Starting the controller...")

    def stop_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Stopping the controller...")


class HelloExecutor(Executor):
    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ):
        if task_name == "hello":
            self.log_info(fl_ctx, f"Received task with name {task_name} and data {shareable}")
            return make_reply(ReturnCode.OK)


class HelloDataController(Controller):
    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        # Prepare any extra parameters to send to the clients
        data = DXO(
            data_kind=DataKind.APP_DEFINED,
            data={"message": "howdy, I'm the controller"},
        ).to_shareable()

        # Create the task with name "hello"
        task = Task(name="hello", data=data)

        # Broadcast the task to all clients and wait for all to respond
        self.broadcast_and_wait(
            task=task,
            targets=None,  # meaning all clients
            min_responses=0,
            fl_ctx=fl_ctx,
        )

    def start_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Starting the controller...")

    def stop_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Stopping the controller...")


class HelloDataExecutor(Executor):
    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ):
        if task_name == "hello":
            received_dxo = from_shareable(shareable)
            message = received_dxo.data["message"]
            self.log_info(fl_ctx, f"Received message from server: {message}")
            return make_reply(ReturnCode.OK)


class HelloResponseController(Controller):
    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        # Prepare any extra parameters to send to the clients
        dxo = DXO(
            data_kind=DataKind.APP_DEFINED,
            data={"message": "howdy, I'm the controller"},
        )
        shareable = dxo.to_shareable()

        # Create the task with name "hello"
        task = Task(name="hello", data=shareable, result_received_cb=self._process_client_response)

        # Broadcast the task to all clients and wait for all to respond
        self.broadcast_and_wait(
            task=task,
            targets=None,  # meaning all clients
            min_responses=0,
            fl_ctx=fl_ctx,
        )

    def start_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Starting the controller...")

    def stop_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Stopping the controller...")

    def _process_client_response(self, client_task, fl_ctx: FLContext) -> None:
        task = client_task.task
        client = client_task.client
        response = client_task.result
        received_msg = from_shareable(response).data["message"]

        self.log_info(fl_ctx, f"Received message {received_msg} from client {client.name} for task {task.name}")


class HelloResponseExecutor(Executor):
    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ):
        if task_name == "hello":
            received_dxo = from_shareable(shareable)
            message = received_dxo.data["message"]
            self.log_info(fl_ctx, f"Received message: {message}")
            self.log_info(fl_ctx, "Sending response to server...")
            response = DXO(
                data_kind=DataKind.APP_DEFINED,
                data={"message": "howdy, I'm a client"},
            ).to_shareable()
            return response
