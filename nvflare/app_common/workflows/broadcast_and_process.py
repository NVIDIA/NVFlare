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

from typing import Union

from nvflare.apis.client import Client
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import ClientTask, Controller, Task
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.response_processor import ResponseProcessor


class BroadcastAndProcess(Controller):
    def __init__(
        self,
        processor: Union[str, ResponseProcessor],
        task_name: str,
        min_responses_required: int = 0,
        wait_time_after_min_received: int = 10,
        timeout: int = 0,
        clients=None,
    ):
        """This controller broadcast a task to specified clients to collect responses, and uses the
        ResponseProcessor object to process the client responses.

        Args:
            processor: the processor that implements logic for client responses and final check.
            It must be a component id (str), or a ResponseProcessor object.
            task_name: name of the task to be sent to client to collect responses
            min_responses_required: min number responses required from clients. 0 means all.
            wait_time_after_min_received: how long to wait after min responses are received from clients
            timeout: timeout of the task. 0 means never time out
            clients: list of clients to send the task to. None means all clients.
        """
        Controller.__init__(self)
        if not (isinstance(processor, str) or isinstance(processor, ResponseProcessor)):
            raise TypeError(f"value of processor must be a str or ResponseProcessor but got {type(processor)}")

        self.processor = processor
        self.task_name = task_name
        self.min_responses_required = min_responses_required
        self.wait_time_after_min_received = wait_time_after_min_received
        self.timeout = timeout
        self.clients = clients

    def start_controller(self, fl_ctx: FLContext) -> None:
        self.log_info(fl_ctx, "Initializing BroadcastAndProcess.")
        if isinstance(self.processor, str):
            checker_id = self.processor

            # the processor is a component id - get the processor component
            engine = fl_ctx.get_engine()
            if not engine:
                self.system_panic("Engine not found. BroadcastAndProcess exiting.", fl_ctx)
                return

            self.processor = engine.get_component(checker_id)
            if not isinstance(self.processor, ResponseProcessor):
                self.system_panic(
                    f"component {checker_id} must be a ResponseProcessor type object but got {type(self.processor)}",
                    fl_ctx,
                )

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        task_data = self.processor.create_task_data(self.task_name, fl_ctx)
        if not isinstance(task_data, Shareable):
            self.system_panic(
                f"ResponseProcessor {type(self.processor)} failed to return valid task data: "
                f"expect Shareable but got {type(task_data)}",
                fl_ctx,
            )
            return

        task = Task(
            name=self.task_name,
            data=task_data,
            timeout=self.timeout,
            result_received_cb=self._process_client_response,
        )

        self.broadcast_and_wait(
            task=task,
            wait_time_after_min_received=self.wait_time_after_min_received,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
            targets=self.clients,
            min_responses=self.min_responses_required,
        )

        success = self.processor.final_process(fl_ctx)
        if not success:
            self.system_panic(reason=f"ResponseProcessor {type(self.processor)} failed final check!", fl_ctx=fl_ctx)

    def _process_client_response(self, client_task: ClientTask, fl_ctx: FLContext) -> None:
        task = client_task.task
        response = client_task.result
        client = client_task.client

        ok = self.processor.process_client_response(
            client=client, task_name=task.name, response=response, fl_ctx=fl_ctx
        )

        # Cleanup task result
        client_task.result = None

        if not ok:
            self.system_panic(
                reason=f"ResponseProcessor {type(self.processor)} failed to check client {client.name}", fl_ctx=fl_ctx
            )

    def stop_controller(self, fl_ctx: FLContext):
        pass

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        pass
