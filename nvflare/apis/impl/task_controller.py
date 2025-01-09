# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from typing import List, Union

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, ControllerSpec, SendOrder, Task, TaskCompletionStatus
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FilterKey, FLContextKey, ReservedKey, ReservedTopic, ReturnCode, SiteType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.utils.task_utils import apply_filters
from nvflare.private.fed.utils.fed_utils import get_target_names
from nvflare.private.privacy_manager import Scope
from nvflare.security.logging import secure_format_exception


class TaskController(FLComponent, ControllerSpec):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.task_data_filters = {}
        self.task_result_filters = {}

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.start_controller(fl_ctx)
        elif event_type == EventType.END_RUN:
            self.stop_controller(fl_ctx)

    def start_controller(self, fl_ctx: FLContext):
        client_runner = fl_ctx.get_prop(FLContextKey.RUNNER)
        self.task_data_filters = client_runner.task_data_filters
        if not self.task_data_filters:
            self.task_data_filters = {}

        self.task_result_filters = client_runner.task_result_filters
        if not self.task_result_filters:
            self.task_result_filters = {}

    def control_flow(self, fl_ctx: FLContext):
        pass

    def stop_controller(self, fl_ctx: FLContext):
        pass

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        pass

    def broadcast(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        min_responses: int = 0,
        wait_time_after_min_received: int = 0,
    ):

        return self.broadcast_and_wait(task, fl_ctx, targets, min_responses, wait_time_after_min_received)

    def broadcast_and_wait(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        min_responses: int = 0,
        wait_time_after_min_received: int = 0,
        abort_signal: Signal = None,
    ):
        engine = fl_ctx.get_engine()
        request = task.data
        # apply task filters
        self.log_debug(fl_ctx, "firing event EventType.BEFORE_TASK_DATA_FILTER")
        fl_ctx.set_prop(FLContextKey.TASK_DATA, task.data, sticky=False, private=True)
        self.fire_event(EventType.BEFORE_TASK_DATA_FILTER, fl_ctx)

        # # first apply privacy-defined filters
        try:
            filter_name = Scope.TASK_DATA_FILTERS_NAME
            task.data = apply_filters(filter_name, request, fl_ctx, self.task_data_filters, task.name, FilterKey.OUT)
        except Exception as e:
            self.log_exception(
                fl_ctx,
                "processing error in task data filter {}; "
                "asked client to try again later".format(secure_format_exception(e)),
            )
            replies = self._make_error_reply(ReturnCode.TASK_DATA_FILTER_ERROR, targets)
            return replies

        self.log_debug(fl_ctx, "firing event EventType.AFTER_TASK_DATA_FILTER")
        fl_ctx.set_prop(FLContextKey.TASK_DATA, task.data, sticky=False, private=True)
        self.fire_event(EventType.AFTER_TASK_DATA_FILTER, fl_ctx)

        target_names = get_target_names(targets)
        _, invalid_names = engine.validate_targets(target_names)
        if invalid_names:
            raise ValueError(f"invalid target(s): {invalid_names}")

        # set up ClientTask for each client
        for target in targets:
            client: Client = self._get_client(target, engine)
            client_task = ClientTask(task=task, client=client)
            task.client_tasks.append(client_task)
            task.last_client_task_map[client_task.id] = client_task

            # task_cb_error = self._call_task_cb(task.before_task_sent_cb, client, task, fl_ctx)
            # if task_cb_error:
            #     return self._make_error_reply(ReturnCode.ERROR, targets)

        if task.timeout <= 0:
            raise ValueError(f"The task timeout must > 0. But got {task.timeout}")

        request.set_header(ReservedKey.TASK_NAME, task.name)
        replies = engine.send_aux_request(
            targets=targets,
            topic=ReservedTopic.DO_TASK,
            request=request,
            timeout=task.timeout,
            fl_ctx=fl_ctx,
            secure=task.secure,
        )

        self.log_debug(fl_ctx, "firing event EventType.BEFORE_TASK_RESULT_FILTER")
        self.fire_event(EventType.BEFORE_TASK_RESULT_FILTER, fl_ctx)

        for target, reply in replies.items():
            # get the client task for the target
            for client_task in task.client_tasks:
                if client_task.client.name == target:
                    rc = reply.get_return_code()
                    if rc and rc == ReturnCode.OK:
                        # apply result filters
                        try:
                            filter_name = Scope.TASK_RESULT_FILTERS_NAME
                            reply = apply_filters(
                                filter_name, reply, fl_ctx, self.task_result_filters, task.name, FilterKey.IN
                            )
                        except Exception as e:
                            self.log_exception(
                                fl_ctx,
                                "processing error in task result filter {}; ".format(secure_format_exception(e)),
                            )
                            error_reply = make_reply(ReturnCode.TASK_RESULT_FILTER_ERROR)
                            client_task.result = error_reply
                            break

                        # assign replies to client task, prepare for the result_received_cb
                        client_task.result = reply

                        client: Client = self._get_client(target, engine)
                        task_cb_error = self._call_task_cb(task.result_received_cb, client, task, fl_ctx)
                        if task_cb_error:
                            client_task.result = make_reply(ReturnCode.ERROR)
                            break
                    else:
                        client_task.result = make_reply(ReturnCode.ERROR)

                    break

        # apply task_done_cb
        if task.task_done_cb is not None:
            try:
                task.task_done_cb(task=task, fl_ctx=fl_ctx)
            except Exception as e:
                self.log_exception(
                    fl_ctx, f"processing error in task_done_cb error on task {task.name}: {secure_format_exception(e)}"
                ),
                task.completion_status = TaskCompletionStatus.ERROR
                task.exception = e
                return self._make_error_reply(ReturnCode.ERROR, targets)

        replies = {}
        for client_task in task.client_tasks:
            replies[client_task.client.name] = client_task.result
        return replies

    def _make_error_reply(self, error_type, targets):
        error_reply = make_reply(error_type)
        replies = {}
        for target in targets:
            replies[target] = error_reply
        return replies

    def _get_client(self, client, engine) -> Client:
        if isinstance(client, Client):
            return client

        if client == SiteType.SERVER:
            return Client(SiteType.SERVER, None)

        client_obj = None
        for _, c in engine.all_clients.items():
            if client == c.name:
                client_obj = c
        return client_obj

    def _call_task_cb(self, task_cb, client, task, fl_ctx):
        task_cb_error = False
        with task.cb_lock:
            client_task = self._get_client_task(client, task)

            if task_cb is not None:
                try:
                    task_cb(client_task=client_task, fl_ctx=fl_ctx)
                except Exception as e:
                    self.log_exception(
                        fl_ctx,
                        f"processing error in {task_cb} on task {client_task.task.name} "
                        f"({client_task.id}): {secure_format_exception(e)}",
                    )
                    # this task cannot proceed anymore
                    task.completion_status = TaskCompletionStatus.ERROR
                    task.exception = e
                    task_cb_error = True

            self.logger.debug(f"{task_cb} done on client_task: {client_task}")
            self.logger.debug(f"task completion status is {task.completion_status}")
        return task_cb_error

    def _get_client_task(self, client, task):
        client_task = None
        for t in task.client_tasks:
            if t.client.name == client.name:
                client_task = t
        return client_task

    def send(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        send_order: SendOrder = SendOrder.SEQUENTIAL,
        task_assignment_timeout: int = 0,
    ):
        engine = fl_ctx.get_engine()

        self._validate_target(engine, targets)

        return self.send_and_wait(task, fl_ctx, targets, send_order, task_assignment_timeout)

    def send_and_wait(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        send_order: SendOrder = SendOrder.SEQUENTIAL,
        task_assignment_timeout: int = 0,
        abort_signal: Signal = None,
    ):
        engine = fl_ctx.get_engine()

        self._validate_target(engine, targets)

        replies = {}
        for target in targets:
            reply = self.broadcast_and_wait(task, fl_ctx, [target], abort_signal=abort_signal)
            replies.update(reply)
        return replies

    def _validate_target(self, engine, targets):
        if len(targets) == 0:
            raise ValueError("Must provide a target to send.")
        if len(targets) != 1:
            raise ValueError("send_and_wait can only send to a single target.")
        target_names = get_target_names(targets)
        _, invalid_names = engine.validate_targets(target_names)
        if invalid_names:
            raise ValueError(f"invalid target(s): {invalid_names}")
