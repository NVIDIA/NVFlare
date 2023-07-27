# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Union, List

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ControllerSpec, Task, SendOrder, ClientTask, TaskCompletionStatus
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, ReservedTopic, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.private.privacy_manager import Scope
from nvflare.security.logging import secure_format_exception


class ClientController(FLComponent, ControllerSpec):
    def __init__(self,
                 ) -> None:
        super().__init__()
        self.task_data_filters = None
        self.task_result_filters = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.start_controller(fl_ctx)
        elif event_type == EventType.END_RUN:
            self.stop_controller(fl_ctx)

    def start_controller(self, fl_ctx: FLContext):
        client_runner = fl_ctx.get_prop(FLContextKey.RUNNER)
        self.task_data_filters = client_runner.task_data_filters
        self.task_result_filters = client_runner.task_result_filters

    def stop_controller(self, fl_ctx: FLContext):
        pass

    def process_result_of_unknown_task(self, client: Client, task_name: str, client_task_id: str, result: Shareable,
                                       fl_ctx: FLContext):
        pass

    def broadcast(self, task: Task, fl_ctx: FLContext, targets: Union[List[Client], List[str], None] = None,
                  min_responses: int = 0, wait_time_after_min_received: int = 0):

        return self.broadcast_and_wait(task, fl_ctx, targets, min_responses, wait_time_after_min_received)

    def broadcast_and_wait(self, task: Task, fl_ctx: FLContext, targets: Union[List[Client], List[str], None] = None,
                           min_responses: int = 0, wait_time_after_min_received: int = 0, abort_signal: Signal = None):
        engine = fl_ctx.get_engine()
        request = task.data
        # apply task filters
        self.log_debug(fl_ctx, "firing event EventType.BEFORE_TASK_DATA_FILTER")
        self.fire_event(EventType.BEFORE_TASK_DATA_FILTER, fl_ctx)

        # first apply privacy-defined filters
        scope_object = fl_ctx.get_prop(FLContextKey.SCOPE_OBJECT)
        filter_list = []
        if scope_object:
            assert isinstance(scope_object, Scope)
            if scope_object.task_data_filters:
                filter_list.extend(scope_object.task_data_filters)

        task_filter_list = self.task_data_filters.get(task.name)
        if task_filter_list:
            filter_list.extend(task_filter_list)

        if filter_list:
            for f in filter_list:
                filter_name = f.__class__.__name__
                try:
                    request = f.process(request, fl_ctx)
                except Exception as e:
                    self.log_exception(
                        fl_ctx, f"Processing error from Task Data Filter {filter_name}: {secure_format_exception(e)}"
                    )
                    return make_reply(ReturnCode.TASK_DATA_FILTER_ERROR)

        for client in targets:
            self._call_tasK_cb(task.before_task_sent_cb, client, task, fl_ctx)

        request.set_header(ReservedKey.TASK_NAME, task.name)
        replies = engine.send_aux_request(
            targets=targets, topic=ReservedTopic.CLIENT_CONTROLLER_TASK, request=request,
            timeout=task.timeout, fl_ctx=fl_ctx
        )

        # apply result filters
        self.log_debug(fl_ctx, "firing event EventType.AFTER_TASK_EXECUTION")
        self.fire_event(EventType.AFTER_TASK_EXECUTION, fl_ctx)

        self.log_debug(fl_ctx, "firing event EventType.BEFORE_TASK_RESULT_FILTER")
        self.fire_event(EventType.BEFORE_TASK_RESULT_FILTER, fl_ctx)

        filter_list = []
        if scope_object and scope_object.task_result_filters:
            filter_list.extend(scope_object.task_result_filters)

        task_filter_list = self.task_result_filters.get(task.name)
        if task_filter_list:
            filter_list.extend(task_filter_list)

        for _, reply in replies.items():
            if filter_list:
                for f in filter_list:
                    filter_name = f.__class__.__name__
                    try:
                        reply = f.process(reply, fl_ctx)
                    except Exception as e:
                        self.log_exception(
                            fl_ctx, f"Processing error in Task Result Filter {filter_name}: {secure_format_exception(e)}"
                        )
                        return make_reply(ReturnCode.TASK_RESULT_FILTER_ERROR)

        task.set_result(replies)
        for client in targets:
            self._call_tasK_cb(task.result_received_cb, client, task, fl_ctx)

        for client in targets:
            self._call_tasK_cb(task.task_done_cb, client, task, fl_ctx)

        return replies

    def _call_tasK_cb(self, task_cb, client, task, fl_ctx):
        with task.cb_lock:
            # client_task = ClientTask(task=task, client=client)
            client_task = task.get_client_task(client)
            if task_cb is not None:
                try:
                    task_cb(client_task=client_task, fl_ctx=fl_ctx)
                except Exception as e:
                    self.log_exception(
                        fl_ctx,
                        f"processing error in {task_cb} on task {client_task.task.name} "
                        f"({client_task.id}): {secure_format_exception(e)}"
                        )
                    # this task cannot proceed anymore
                    task.completion_status = TaskCompletionStatus.ERROR
                    task.exception = e

            self.logger.debug(f"{task_cb} done on client_task: {client_task}")
            self.logger.debug(f"task completion status is {task.completion_status}")

    def send(self, task: Task, fl_ctx: FLContext, targets: Union[List[Client], List[str], None] = None,
             send_order: SendOrder = SendOrder.SEQUENTIAL, task_assignment_timeout: int = 0):
        return self.send_and_wait(task, fl_ctx, targets, send_order, task_assignment_timeout)

    def send_and_wait(self, task: Task, fl_ctx: FLContext, targets: Union[List[Client], List[str], None] = None,
                      send_order: SendOrder = SendOrder.SEQUENTIAL, task_assignment_timeout: int = 0,
                      abort_signal: Signal = None):
        replies = {}
        for target in targets:
            reply = self.broadcast_and_wait(task, fl_ctx, [target], abort_signal=abort_signal)
            replies.update(reply)
        return replies
