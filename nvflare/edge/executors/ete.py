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
import time
from abc import abstractmethod
from typing import Any

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.edge.constants import EventType as EdgeEventType
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.security.logging import secure_format_exception


class EdgeTaskExecutor(Executor):
    """This is the base class for executors to handling requests from edge devices.
    Subclasses must implement the required abstract methods defined here.
    """

    def __init__(self):
        """Constructor of EdgeTaskExecutor"""
        Executor.__init__(self)
        self.current_task = None

        self.register_event_handler(EdgeEventType.EDGE_REQUEST_RECEIVED, self._handle_edge_request)

    @abstractmethod
    def process_edge_request(self, request: Any, fl_ctx: FLContext) -> Any:
        """This is called to process an edge request sent from the edge device.

        Args:
            request: the request from edge device
            fl_ctx: FLContext object

        Returns: reply to the edge device

        """
        pass

    def task_received(self, task_name: str, task_data: Shareable, fl_ctx: FLContext):
        """This method is called when a task assignment is received from the controller.
        Subclass can implement this method to prepare for task processing.

        Args:
            task_name: name of the task
            task_data: task data
            fl_ctx: FLContext object

        Returns: None

        """
        pass

    @abstractmethod
    def is_task_done(self, fl_ctx: FLContext) -> bool:
        """This is called by the base class to determine whether the task processing is done.
        Subclass must implement this method.

        Args:
            fl_ctx: FLContext object

        Returns: whether task is done.

        """
        pass

    @abstractmethod
    def get_task_result(self, fl_ctx: FLContext) -> Shareable:
        """This is called by the base class to get the final result of the task.
        Base class will send the result to the controller.

        Args:
            fl_ctx: FLContext object

        Returns: a Shareable object that is the task result

        """
        pass

    def _handle_edge_request(self, event_type: str, fl_ctx: FLContext):
        if not self.current_task:
            self.logger.debug(f"received edge event {event_type} but I don't have pending task")
            return

        try:
            msg = fl_ctx.get_prop(FLContextKey.CELL_MESSAGE)
            assert isinstance(msg, CellMessage)
            self.log_debug(fl_ctx, f"received edge request: {msg.payload}")
            reply = self.process_edge_request(request=msg.payload, fl_ctx=fl_ctx)
            fl_ctx.set_prop(FLContextKey.TASK_RESULT, reply, private=True, sticky=False)
        except Exception as ex:
            self.logger.error(f"exception from self.process_edge_request: {secure_format_exception(ex)}")
            fl_ctx.set_prop(FLContextKey.EXCEPTIONS, ex, private=True, sticky=False)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.current_task = shareable
        result = self._execute(task_name, shareable, fl_ctx, abort_signal)
        self.current_task = None
        return result

    def _execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        try:
            self.task_received(task_name, shareable, fl_ctx)
        except Exception as ex:
            self.log_error(fl_ctx, f"exception from self.task_received: {secure_format_exception(ex)}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        start_time = time.time()
        while True:
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            try:
                task_done = self.is_task_done(fl_ctx)
            except Exception as ex:
                self.log_error(fl_ctx, f"exception from self.is_task_done: {secure_format_exception(ex)}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            if task_done:
                break

            time.sleep(0.2)

        self.log_debug(fl_ctx, f"task done after {time.time() - start_time} seconds")
        try:
            result = self.get_task_result(fl_ctx)

            if not isinstance(result, Shareable):
                self.log_error(
                    fl_ctx,
                    f"bad result from self.get_task_result: expect Shareable but got {type(result)}",
                )
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        except Exception as ex:
            self.log_error(fl_ctx, f"exception from self.get_task_result: {secure_format_exception(ex)}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        return result
