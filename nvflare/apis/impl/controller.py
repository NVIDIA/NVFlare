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

import threading
import time
from abc import ABC
from threading import Lock
from typing import List, Optional, Tuple, Union

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, ControllerSpec, SendOrder, Task, TaskCompletionStatus
from nvflare.apis.fl_constant import FLContextKey, ReservedTopic
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import WorkflowError
from nvflare.apis.responder import Responder
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.widgets.info_collector import GroupInfoCollector, InfoCollector

from .any_relay_manager import AnyRelayTaskManager
from .bcast_manager import BcastForeverTaskManager, BcastTaskManager
from .send_manager import SendTaskManager
from .seq_relay_manager import SequentialRelayTaskManager
from .task_manager import TaskCheckStatus, TaskManager

_TASK_KEY_ENGINE = "___engine"
_TASK_KEY_MANAGER = "___mgr"
_TASK_KEY_DONE = "___done"


def _check_positive_int(name, value):
    if not isinstance(value, int):
        raise TypeError("{} must be an instance of int, but got {}.".format(name, type(name)))
    if value < 0:
        raise ValueError("{} must >= 0.".format(name))


def _check_inputs(task: Task, fl_ctx: FLContext, targets: Union[List[Client], List[str], None]):
    if not isinstance(task, Task):
        raise TypeError("task must be an instance of Task, but got {}".format(type(task)))

    if not isinstance(fl_ctx, FLContext):
        raise TypeError("fl_ctx must be an instance of FLContext, but got {}".format(type(fl_ctx)))

    if targets is not None:
        if not isinstance(targets, list):
            raise TypeError("targets must be a list of Client or string, but got {}".format(type(targets)))

        for t in targets:
            if not isinstance(t, (Client, str)):
                raise TypeError(
                    "targets must be a list of Client or string, but got element of type {}".format(type(t))
                )


class Controller(Responder, ControllerSpec, ABC):
    def __init__(self, task_check_period=0.5):
        """Manage life cycles of tasks and their destinations.

        Args:
            task_check_period (float, optional): interval for checking status of tasks. Defaults to 0.5.
        """
        Responder.__init__(self)
        self._engine = None
        self._tasks = []  # list of standing tasks
        self._client_task_map = {}  # client_task_id => client_task
        self._all_done = False
        self._task_lock = Lock()
        self._task_monitor = threading.Thread(target=self._monitor_tasks, args=())
        self._task_check_period = task_check_period

    def initialize_run(self, fl_ctx: FLContext):
        """Called by runners to initialize controller with information in fl_ctx.

        Note: Controller subclasses must not overwrite this method.

        Args:
            fl_ctx (FLContext): FLContext information
        """
        self._engine = fl_ctx.get_engine()
        self.start_controller(fl_ctx)
        self._task_monitor.start()

    def _try_again(self) -> Tuple[str, str, Shareable]:
        # TODO: how to tell client no shareable available now?
        return "", "", None

    def _set_stats(self, fl_ctx: FLContext):
        """Called to set stats into InfoCollector.

        Args:
            fl_ctx (FLContext): info collector is retrieved from fl_ctx with InfoCollector.CTX_KEY_STATS_COLLECTOR key
        """
        collector = fl_ctx.get_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR, None)
        if collector:
            if not isinstance(collector, GroupInfoCollector):
                raise TypeError(
                    "collector must be an instance of GroupInfoCollector, but got {}".format(type(collector))
                )
            collector.set_info(
                group_name=self._name,
                info={
                    "tasks": {t.name: [ct.client.name for ct in t.client_tasks] for t in self._tasks},
                },
            )

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """Called when events are fired.

        Args:
            event_type (str): all event types, including AppEventType and EventType
            fl_ctx (FLContext): FLContext information with current event type
        """
        if event_type == InfoCollector.EVENT_TYPE_GET_STATS:
            self._set_stats(fl_ctx)

    def process_task_request(self, client: Client, fl_ctx: FLContext) -> Tuple[str, str, Shareable]:
        """Called by runner when a client asks for a task.

        Note: this is called in a separate thread.

        Args:
            client (Client): The record of one client requesting tasks
            fl_ctx (FLContext): The FLContext associated with this request

        Raises:
            TypeError: when client is not an instance of Client
            TypeError: when fl_ctx is not an instance of FLContext
            TypeError: when any standing task containing an invalid client_task

        Returns:
            Tuple[str, str, Shareable]: task_name, an id for the client_task, and the data for this requst
        """
        if not isinstance(client, Client):
            raise TypeError("client must be an instance of Client, but got {}".format(type(client)))
        if not isinstance(fl_ctx, FLContext):
            raise TypeError("fl_ctx must be an instance of FLContext, but got {}".format(type(fl_ctx)))

        client_task_to_send = None
        with self._task_lock:
            self.logger.debug("self._tasks: {}".format(self._tasks))
            for task in self._tasks:
                if task.completion_status is not None:
                    # this task is finished (and waiting for the monitor to exit it)
                    continue

                # do we need to send this task to this client?
                # note: the task could be sent to a client multiple times (e.g. in relay)
                # we only check the last ClientTask sent to the client
                client_task_to_check = task.last_client_task_map.get(client.name, None)
                self.logger.debug("client_task_to_check: {}".format(client_task_to_check))
                resend_task = False

                if client_task_to_check is not None:
                    # this client has been sent the task already
                    if not isinstance(client_task_to_check, ClientTask):
                        raise TypeError(
                            "client_task_to_check must be an instance of ClientTask, but got {}".format(
                                type(client_task_to_check)
                            )
                        )
                    if client_task_to_check.result_received_time is None:
                        # controller has not received result from client
                        # something wrong happens when client working on this task, so resend the task
                        resend_task = True
                        client_task_to_send = client_task_to_check
                        fl_ctx.set_prop(FLContextKey.IS_CLIENT_TASK_RESEND, True, sticky=False)

                if not resend_task:
                    # check with the task manager whether to send
                    manager = task.props[_TASK_KEY_MANAGER]
                    if client_task_to_check is None:
                        client_task_to_check = ClientTask(task=task, client=client)
                    check_status = manager.check_task_send(client_task_to_check, fl_ctx)
                    self.logger.debug(
                        "Checking client task: {}, task.client.name: {}".format(
                            client_task_to_check, client_task_to_check.client.name
                        )
                    )
                    self.logger.debug("Check task send get check_status: {}".format(check_status))
                    if check_status == TaskCheckStatus.BLOCK:
                        # do not send this task, and do not check other tasks
                        return self._try_again()
                    elif check_status == TaskCheckStatus.NO_BLOCK:
                        # do not send this task, but continue to check next task
                        continue
                    else:
                        # send the task and remember the client_task
                        client_task_to_send = ClientTask(client, task)
                        task.last_client_task_map[client.name] = client_task_to_send
                        task.client_tasks.append(client_task_to_send)
                        self._client_task_map[client_task_to_send.id] = client_task_to_send
                        break

        # NOTE: move task sending process outside the lock
        # This is to minimize the locking time and to avoid potential deadlock:
        # the CB could schedule another task, which requires lock
        self.logger.debug("Determining based on client_task_to_send: {}".format(client_task_to_send))
        if client_task_to_send is None:
            # no task available for this client
            return self._try_again()

        # try to send the task
        can_send_task = True
        task = client_task_to_send.task
        with task.cb_lock:
            # Note: must guarantee the after_task_sent_cb is always called
            # regardless whether the task is sent successfully.
            # This is so that the app could clear up things in after_task_sent_cb.
            if task.before_task_sent_cb is not None:
                try:
                    task.before_task_sent_cb(client_task=client_task_to_send, fl_ctx=fl_ctx)
                except WorkflowError as ex:
                    self._engine.ask_to_stop()
                    self.log_exception(
                        fl_ctx,
                        "processing error in before_task_sent_cb on task {} ({}): {}".format(
                            client_task_to_send.task.name, client_task_to_send.id, ex
                        ),
                    )
                    task.completion_status = TaskCompletionStatus.ERROR
                    task.exception = ex
                except BaseException as ex:
                    self.log_exception(
                        fl_ctx,
                        "processing error in before_task_sent_cb on task {} ({}): {}".format(
                            client_task_to_send.task.name, client_task_to_send.id, ex
                        ),
                    )
                    # this task cannot proceed anymore
                    task.completion_status = TaskCompletionStatus.ERROR
                    task.exception = ex

            self.logger.debug("before_task_sent_cb done on client_task_to_send: {}".format(client_task_to_send))
            self.logger.debug(f"task completion status is {task.completion_status}")

            if task.completion_status is not None:
                can_send_task = False

            # remember the task name and data to be sent to the client
            # since task.data could be reset by the after_task_sent_cb
            task_name = task.name
            task_data = task.data

            if task.after_task_sent_cb is not None:
                try:
                    task.after_task_sent_cb(client_task=client_task_to_send, fl_ctx=fl_ctx)
                except WorkflowError as ex:
                    self._engine.ask_to_stop()
                    self.log_exception(
                        fl_ctx,
                        "processing error in after_task_sent_cb on task {} ({}): {}".format(
                            client_task_to_send.task.name, client_task_to_send.id, ex
                        ),
                    )
                    task.completion_status = TaskCompletionStatus.ERROR
                    task.exception = ex
                except BaseException as ex:
                    self.log_exception(
                        fl_ctx,
                        "processing error in after_task_sent_cb on task {} ({}): {}".format(
                            client_task_to_send.task.name, client_task_to_send.id, ex
                        ),
                    )
                    task.completion_status = TaskCompletionStatus.ERROR
                    task.exception = ex

            if task.completion_status is not None:
                # NOTE: the CB could cancel the task
                can_send_task = False

            if not can_send_task:
                return self._try_again()

            self.logger.debug("after_task_sent_cb done on client_task_to_send: {}".format(client_task_to_send))

            client_task_to_send.task_sent_time = time.time()
            client_task_to_send.task_send_count += 1
            return task_name, client_task_to_send.id, task_data

    def handle_exception(self, task_id: str, fl_ctx: FLContext) -> None:
        """Called to cancel one task as its client_task is causing exception at upper level.

        Args:
            task_id (str): an id to the failing client_task
            fl_ctx (FLContext): FLContext associated with this client_task
        """
        with self._task_lock:
            # task_id is the uuid associated with the client_task
            client_task = self._client_task_map.get(task_id, None)
            self.logger.debug("Handle exception on client_task {} with id {}".format(client_task, task_id))

        if client_task is None:
            # cannot find a standing task on the exception
            return

        task = client_task.task
        self.cancel_task(task=task, fl_ctx=fl_ctx)
        self.log_error(fl_ctx, "task {} is cancelled due to exception".format(task.name))

    def process_submission(self, client: Client, task_name: str, task_id: str, result: Shareable, fl_ctx: FLContext):
        """Called to process a submission from one client.

        Note: this method is called by a separate thread.

        Args:
            client (Client): the client that submitted this task
            task_name (str): the task name associated this submission
            task_id (str): the id associated with the client_task
            result (Shareable): the actual submitted data from the client
            fl_ctx (FLContext): the FLContext associated with this submission

        Raises:
            TypeError: when client is not an instance of Client
            TypeError: when fl_ctx is not an instance of FLContext
            TypeError: when result is not an instance of Shareable
            ValueError: task_name is not found in the client_task
        """
        if not isinstance(client, Client):
            raise TypeError("client must be an instance of Client, but got {}".format(type(client)))
        if not isinstance(fl_ctx, FLContext):
            raise TypeError("fl_ctx must be an instance of FLContext, but got {}".format(type(fl_ctx)))
        if not isinstance(result, Shareable):
            raise TypeError("result must be an instance of Shareable, but got {}".format(type(result)))

        with self._task_lock:
            # task_id is the uuid associated with the client_task
            client_task = self._client_task_map.get(task_id, None)
            self.logger.debug("Get submission={} from client task={} id={}".format(result, client_task, task_id))

        if client_task is None:
            # cannot find a standing task for the submission
            self.log_info(fl_ctx, "no standing task found for {}:{}".format(task_name, task_id))
            self.process_result_of_unknown_task(client, task_name, task_id, result, fl_ctx)
            return

        task = client_task.task
        with task.cb_lock:
            if task.name != task_name:
                raise ValueError("client specified task name {} doesn't match {}".format(task_name, task.name))

            if task.completion_status is not None:
                # the task is already finished - drop the result
                self.log_info(fl_ctx, "task is already finished - submission dropped")
                return

            # do client task CB processing outside the lock
            # this is because the CB could schedule another task, which requires the lock
            client_task.result = result
            client_task.result_received_time = time.time()

            manager = task.props[_TASK_KEY_MANAGER]
            manager.check_task_result(result, client_task, fl_ctx)

            if task.result_received_cb is not None:
                try:
                    self.log_info(fl_ctx, "invoking result_received_cb ...")
                    task.result_received_cb(client_task=client_task, fl_ctx=fl_ctx)
                except WorkflowError as ex:
                    self._engine.ask_to_stop()
                    self.log_exception(
                        fl_ctx,
                        "processing error in result_received_cb on task {}({}): {}".format(task_name, task_id, ex),
                    )
                    task.completion_status = TaskCompletionStatus.ERROR
                    task.exception = ex
                    return
                except BaseException as ex:
                    # this task cannot proceed anymore
                    self.log_exception(
                        fl_ctx,
                        "processing error in result_received_cb on task {}({}): {}".format(task_name, task_id, ex),
                    )
                    task.completion_status = TaskCompletionStatus.ERROR
                    task.exception = ex
                    return
            else:
                self.log_info(fl_ctx, "no result_received_cb")

    def _schedule_task(
        self,
        task: Task,
        fl_ctx: FLContext,
        manager: TaskManager,
        targets: Union[List[Client], List[str], None],
        allow_dup_targets: bool = False,
    ):
        if task.schedule_time is not None:
            # this task was scheduled before
            # we do not allow a task object to be reused
            self.logger.debug("task.schedule_time: {}".format(task.schedule_time))
            raise ValueError("Task was already used. Please create a new task object.")

        task.targets = targets
        if targets is not None:
            target_names = list()
            if not isinstance(targets, list):
                raise ValueError("task targets must be a list, but got {}".format(type(targets)))
            for t in targets:
                if isinstance(t, str):
                    name = t
                elif isinstance(t, Client):
                    name = t.name
                else:
                    raise ValueError("element in targets must be string or Client type, but got {}".format(type(t)))

                if allow_dup_targets or (name not in target_names):
                    target_names.append(name)
            task.targets = target_names

        task.props[_TASK_KEY_MANAGER] = manager
        task.props[_TASK_KEY_ENGINE] = fl_ctx.get_engine()
        task.is_standing = True
        task.schedule_time = time.time()

        with self._task_lock:
            self._tasks.append(task)
            self.log_info(fl_ctx, "scheduled task {}".format(task.name))

    def broadcast(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        min_responses: int = 1,
        wait_time_after_min_received: int = 0,
    ):
        """Schedule a broadcast task.  This is a non-blocking call.

        The task is scheduled into a task list.  Clients can request tasks and controller will dispatch the task to eligible clients.

        Args:
            task (Task): the task to be scheduled
            fl_ctx (FLContext): FLContext associated with this task
            targets (Union[List[Client], List[str], None], optional): the list of eligible clients or client names or None (all clients). Defaults to None.
            min_responses (int, optional): the condition to mark this task as completed because enough clients respond with submission. Defaults to 1.
            wait_time_after_min_received (int, optional): a grace period for late clients to contribute their submission.  0 means no grace period.
              Submission of late clients in the grace period are still collected as valid submission. Defaults to 0.

        Raises:
            ValueError: min_responses is greater than the length of targets since this condition will make the task, if allowed to be scheduled, never exit.
        """
        _check_inputs(task=task, fl_ctx=fl_ctx, targets=targets)
        _check_positive_int("min_responses", min_responses)
        _check_positive_int("wait_time_after_min_received", wait_time_after_min_received)
        if targets and min_responses > len(targets):
            raise ValueError(
                "min_responses ({}) must be less than length of targets ({}).".format(min_responses, len(targets))
            )

        manager = BcastTaskManager(
            task=task, min_responses=min_responses, wait_time_after_min_received=wait_time_after_min_received
        )
        self._schedule_task(task=task, fl_ctx=fl_ctx, manager=manager, targets=targets)

    def broadcast_and_wait(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        min_responses: int = 1,
        wait_time_after_min_received: int = 0,
        abort_signal: Optional[Signal] = None,
    ):
        """Schedule a broadcast task.  This is a blocking call.

        The task is scheduled into a task list.  Clients can request tasks and controller will dispatch the task to eligible clients.

        Args:
            task (Task): the task to be scheduled
            fl_ctx (FLContext): FLContext associated with this task
            targets (Union[List[Client], List[str], None], optional): the list of eligible clients or client names or None (all clients). Defaults to None.
            min_responses (int, optional): the condition to mark this task as completed because enough clients respond with submission. Defaults to 1.
            wait_time_after_min_received (int, optional): a grace period for late clients to contribute their submission.  0 means no grace period.
            Submission of late clients in the grace period are still collected as valid submission. Defaults to 0.
            abort_signal (Optional[Signal], optional): as this is a blocking call, this abort_signal informs this method to return. Defaults to None.
        """
        self.broadcast(
            task=task,
            fl_ctx=fl_ctx,
            targets=targets,
            min_responses=min_responses,
            wait_time_after_min_received=wait_time_after_min_received,
        )
        self._wait_for_task(task, abort_signal)

    def broadcast_forever(self, task: Task, fl_ctx: FLContext, targets: Union[List[Client], List[str], None] = None):
        """Schedule a broadcast task.  This is a non-blocking call.

        The task is scheduled into a task list.  Clients can request tasks and controller will dispatch the task to eligible clients.
        This broadcast will not end.

        Args:
            task (Task): the task to be scheduled
            fl_ctx (FLContext): FLContext associated with this task
            targets (Union[List[Client], List[str], None], optional): the list of eligible clients or client names or None (all clients). Defaults to None.
        """
        _check_inputs(task=task, fl_ctx=fl_ctx, targets=targets)
        manager = BcastForeverTaskManager()
        self._schedule_task(task=task, fl_ctx=fl_ctx, manager=manager, targets=targets)

    def send(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        send_order: SendOrder = SendOrder.SEQUENTIAL,
        task_assignment_timeout: int = 0,
    ):
        """Schedule a single task to targets.  This is a non-blocking call.

        The task is scheduled into a task list.  Clients can request tasks and controller will dispatch the task to eligible clients based on the send_order.

        Args:
            task (Task): the task to be scheduled
            fl_ctx (FLContext): FLContext associated with this task
            targets (Union[List[Client], List[str], None], optional): the list of eligible clients or client names or None (all clients). Defaults to None.
            send_order (SendOrder, optional): the order for clients to become eligible.  SEQUENTIAL means the order in targets is enforced.  ANY means
            clients in targets and haven't received task are eligible for task. Defaults to SendOrder.SEQUENTIAL.
            task_assignment_timeout (int, optional): how long to wait for one client to pick the task. Defaults to 0.

        Raises:
            ValueError: when task_assignment_timeout is greater than task's timeout.
            TypeError: send_order is not defined in SendOrder
            ValueError: targets is None or an empty list
        """
        _check_inputs(
            task=task,
            fl_ctx=fl_ctx,
            targets=targets,
        )
        _check_positive_int("task_assignment_timeout", task_assignment_timeout)
        if task.timeout and task_assignment_timeout and task_assignment_timeout > task.timeout:
            raise ValueError(
                "task_assignment_timeout ({}) needs to be less than or equal to task.timeout ({}).".format(
                    task_assignment_timeout, task.timeout
                )
            )
        if not isinstance(send_order, SendOrder):
            raise TypeError("send_order must be in Enum SendOrder, but got {}".format(type(send_order)))

        # targets must be provided
        if targets is None or len(targets) == 0:
            raise ValueError("Targets must be provided for send.")

        manager = SendTaskManager(task, send_order, task_assignment_timeout)
        self._schedule_task(
            task=task,
            fl_ctx=fl_ctx,
            manager=manager,
            targets=targets,
        )

    def send_and_wait(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        send_order: SendOrder = SendOrder.SEQUENTIAL,
        task_assignment_timeout: int = 0,
        abort_signal: Signal = None,
    ):
        """Schedule a single task to targets.  This is a blocking call.

        The task is scheduled into a task list.  Clients can request tasks and controller will dispatch the task to eligible clients based on the send_order.

        Args:
            task (Task): the task to be scheduled
            fl_ctx (FLContext): FLContext associated with this task
            targets (Union[List[Client], List[str], None], optional): the list of eligible clients or client names or None (all clients). Defaults to None.
            send_order (SendOrder, optional): the order for clients to become eligible.  SEQUENTIAL means the order in targets is enforced.  ANY means
            clients in targets and haven't received task are eligible for task. Defaults to SendOrder.SEQUENTIAL.
            task_assignment_timeout (int, optional): how long to wait for one client to pick the task. Defaults to 0.
            abort_signal (Optional[Signal], optional): as this is a blocking call, this abort_signal informs this method to return. Defaults to None.

        """
        self.send(
            task=task,
            fl_ctx=fl_ctx,
            targets=targets,
            send_order=send_order,
            task_assignment_timeout=task_assignment_timeout,
        )
        self._wait_for_task(task, abort_signal)

    def get_num_standing_tasks(self) -> int:
        """Get the number of tasks that are currently standing.

        Returns:
            int: length of the list of standing tasks
        """
        return len(self._tasks)

    def cancel_task(
        self, task: Task, completion_status=TaskCompletionStatus.CANCELLED, fl_ctx: Optional[FLContext] = None
    ):
        """Cancel the specified task.

        Change the task completion_status, which will inform task monitor to clean up this task
        NOTE: we only mark the task as completed and leave it to the task monitor to clean up
        This is to avoid potential dead lock of task_lock

        Args:
            task (Task): the task to be cancelled
            completion_status (str, optional): the completion status for this cancellation. Defaults to TaskCompletionStatus.CANCELLED.
            fl_ctx (Optional[FLContext], optional): FLContext associated with this cancellation. Defaults to None.
        """
        task.completion_status = completion_status

    def cancel_all_tasks(self, completion_status=TaskCompletionStatus.CANCELLED, fl_ctx: Optional[FLContext] = None):
        """Cancel all standing tasks in this controller.

        Args:
            completion_status (str, optional): the completion status for this cancellation. Defaults to TaskCompletionStatus.CANCELLED.
            fl_ctx (Optional[FLContext], optional): FLContext associated with this cancellation. Defaults to None.
        """
        with self._task_lock:
            for t in self._tasks:
                t.completion_status = completion_status

    def abort_task(self, task, fl_ctx: FLContext):
        """Ask all clients to abort the execution of the specified task.

        Args:
            task (str): the task to be aborted
            fl_ctx (FLContext): FLContext associated with this action
        """
        self.log_info(fl_ctx, "asked all clients to abort task {}".format(task.name))
        self._end_task([task.name], fl_ctx)

    def _end_task(self, task_names, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        if not isinstance(engine, ServerEngineSpec):
            raise TypeError("engine should be an instance of ServerEngineSpec, but got {}".format(type(engine)))
        request = Shareable()
        request["task_names"] = task_names
        engine.send_aux_request(targets=None, topic=ReservedTopic.ABORT_ASK, request=request, timeout=0, fl_ctx=fl_ctx)

    def abort_all_tasks(self, fl_ctx: FLContext):
        """Ask clients to abort the execution of all tasks.

        NOTE: the server should send a notification to all clients, regardless of whether the server
        has any standing tasks.

        Args:
            fl_ctx (FLContext): FLContext associated with this action
        """
        self._end_task([], fl_ctx)

    def finalize_run(self, fl_ctx: FLContext):
        """Do cleanup of the coordinator implementation.

        NOTE: subclass controllers should not overwrite finalize_run.

        Args:
            fl_ctx (FLContext): FLContext associated with this action
        """
        self.cancel_all_tasks()  # unconditionally cancel all tasks
        self._all_done = True
        try:
            if self._task_monitor.is_alive():
                self._task_monitor.join()
        except RuntimeError:
            self.log_debug(fl_ctx, "unable to join monitor thread (not started?)")
        self.stop_controller(fl_ctx)

    def relay(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        send_order: SendOrder = SendOrder.SEQUENTIAL,
        task_assignment_timeout: int = 0,
        task_result_timeout: int = 0,
        dynamic_targets: bool = True,
    ):
        """Schedule a single task to targets in one-after-another style.  This is a non-blocking call.

        The task is scheduled into a task list.  Clients can request tasks and controller will dispatch the task to eligible clients based on the send_order.

        Args:
            task (Task): the task to be scheduled
            fl_ctx (FLContext): FLContext associated with this task
            targets (Union[List[Client], List[str], None], optional): the list of eligible clients or client names or None (all clients). Defaults to None.
            send_order (SendOrder, optional): the order for clients to become eligible.
              SEQUENTIAL means the order in targets is enforced.
              ANY means any clients that are inside the targets and haven't received the task are eligible. Defaults to SendOrder.SEQUENTIAL.
            task_assignment_timeout (int, optional): how long to wait for one client to pick the task. Defaults to 0.
            task_result_timeout (int, optional): how long to wait for current working client to reply its result. Defaults to 0.
            dynamic_targets (bool, optional): allow clients not in targets to join at the end of targets list. Defaults to True.

        Raises:
            ValueError: when task_assignment_timeout is greater than task's timeout
            ValueError: when task_result_timeout is greater than task's timeout
            TypeError: send_order is not defined in SendOrder
            TypeError: when dynamic_targets is not a boolean variable
            ValueError: targets is None or an empty list but dynamic_targets is False
        """
        _check_inputs(
            task=task,
            fl_ctx=fl_ctx,
            targets=targets,
        )
        _check_positive_int("task_assignment_timeout", task_assignment_timeout)
        _check_positive_int("task_result_timeout", task_result_timeout)
        if task.timeout and task_assignment_timeout and task_assignment_timeout > task.timeout:
            raise ValueError(
                "task_assignment_timeout ({}) needs to be less than or equal to task.timeout ({}).".format(
                    task_assignment_timeout, task.timeout
                )
            )
        if task.timeout and task_result_timeout and task_result_timeout > task.timeout:
            raise ValueError(
                "task_result_timeout ({}) needs to be less than or equal to task.timeout ({}).".format(
                    task_result_timeout, task.timeout
                )
            )
        if not isinstance(send_order, SendOrder):
            raise TypeError("send_order must be in Enum SendOrder, but got {}".format(type(send_order)))
        if not isinstance(dynamic_targets, bool):
            raise TypeError("dynamic_targets must be an instance of bool, but got {}".format(type(dynamic_targets)))
        if targets is None and dynamic_targets is False:
            raise ValueError("Need to provide targets when dynamic_targets is set to False.")

        if send_order == SendOrder.SEQUENTIAL:
            manager = SequentialRelayTaskManager(
                task=task,
                task_assignment_timeout=task_assignment_timeout,
                task_result_timeout=task_result_timeout,
                dynamic_targets=dynamic_targets,
            )
        else:
            manager = AnyRelayTaskManager(
                task=task, task_result_timeout=task_result_timeout, dynamic_targets=dynamic_targets
            )

        self._schedule_task(
            task=task,
            fl_ctx=fl_ctx,
            manager=manager,
            targets=targets,
            allow_dup_targets=True,
        )

    def relay_and_wait(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        send_order=SendOrder.SEQUENTIAL,
        task_assignment_timeout: int = 0,
        task_result_timeout: int = 0,
        dynamic_targets: bool = True,
        abort_signal: Optional[Signal] = None,
    ):
        """Schedule a single task to targets in one-after-another style.  This is a blocking call.

        The task is scheduled into a task list.  Clients can request tasks and controller will dispatch the task to eligible clients based on the send_order.

        Args:
            task (Task): the task to be scheduled
            fl_ctx (FLContext): FLContext associated with this task
            targets (Union[List[Client], List[str], None], optional): the list of eligible clients or client names or None (all clients). Defaults to None.
            send_order (SendOrder, optional): the order for clients to become eligible.  SEQUENTIAL means the order in targets is enforced.  ANY means
            clients in targets and haven't received task are eligible for task. Defaults to SendOrder.SEQUENTIAL.
            task_assignment_timeout (int, optional): how long to wait for one client to pick the task. Defaults to 0.
            task_result_timeout (int, optional): how long to wait for current working client to reply its result. Defaults to 0.
            dynamic_targets (bool, optional): allow clients not in targets to join at the end of targets list. Defaults to True.
            abort_signal (Optional[Signal], optional): as this is a blocking call, this abort_signal informs this method to return. Defaults to None.
        """
        self.relay(
            task=task,
            fl_ctx=fl_ctx,
            targets=targets,
            send_order=send_order,
            task_assignment_timeout=task_assignment_timeout,
            task_result_timeout=task_result_timeout,
            dynamic_targets=dynamic_targets,
        )
        self._wait_for_task(task, abort_signal)

    def _monitor_tasks(self):
        while not self._all_done:
            self._check_tasks()
            time.sleep(self._task_check_period)

    def _check_tasks(self):
        exit_tasks = []
        with self._task_lock:
            for task in self._tasks:
                if task.completion_status is not None:
                    exit_tasks.append(task)
                    continue

                # check the task-specific exit condition
                manager = task.props[_TASK_KEY_MANAGER]
                if manager is not None:
                    if not isinstance(manager, TaskManager):
                        raise TypeError(
                            "manager in task must be an instance of TaskManager, but got {}".format(manager)
                        )
                    should_exit, exit_status = manager.check_task_exit(task)
                    self.logger.debug("should_exit: {}, exit_status: {}".format(should_exit, exit_status))
                    if should_exit:
                        task.completion_status = exit_status
                        exit_tasks.append(task)
                        continue

                # check if task timeout
                if task.timeout and time.time() - task.schedule_time >= task.timeout:
                    task.completion_status = TaskCompletionStatus.TIMEOUT
                    exit_tasks.append(task)
                    continue

            for exit_task in exit_tasks:
                exit_task.is_standing = False
                self.logger.debug(
                    "Removing task={}, completion_status={}".format(exit_task, exit_task.completion_status)
                )
                self._tasks.remove(exit_task)
                for client_task in exit_task.client_tasks:
                    self.logger.debug("Removing client_task with id={}".format(client_task.id))
                    self._client_task_map.pop(client_task.id)

        # do the task exit processing outside the lock to minimize the locking time
        # and to avoid potential deadlock since the CB could schedule another task
        if len(exit_tasks) <= 0:
            return

        with self._engine.new_context() as fl_ctx:
            for exit_task in exit_tasks:
                with exit_task.cb_lock:
                    self.log_info(
                        fl_ctx, "task {} exit with status {}".format(exit_task.name, exit_task.completion_status)
                    )

                    if exit_task.task_done_cb is not None:
                        try:
                            exit_task.task_done_cb(task=exit_task, fl_ctx=fl_ctx)
                        except WorkflowError as ex:
                            self._engine.ask_to_stop()
                            self.log_exception(
                                fl_ctx,
                                "processing error in task_done_cb error on task {}: {}".format(exit_task.name, ex),
                            )
                            task.completion_status = TaskCompletionStatus.ERROR
                            task.exception = ex
                        except BaseException as ex:
                            self.log_exception(
                                fl_ctx,
                                "processing error in task_done_cb error on task {}: {}".format(exit_task.name, ex),
                            )
                            exit_task.completion_status = TaskCompletionStatus.ERROR
                            exit_task.exception = ex

    @staticmethod
    def _process_finished_task(task, func):
        def wrap(*args, **kwargs):
            if func:
                func(*args, **kwargs)
            task.props[_TASK_KEY_DONE] = True

        return wrap

    def _wait_for_task(self, task: Task, abort_signal: Signal):
        task.props[_TASK_KEY_DONE] = False
        task.task_done_cb = self._process_finished_task(task=task, func=task.task_done_cb)
        while True:
            if task.completion_status is not None:
                break

            if abort_signal and abort_signal.triggered:
                self.cancel_task(task, fl_ctx=None, completion_status=TaskCompletionStatus.ABORTED)
                break

            task_done = task.props[_TASK_KEY_DONE]
            if task_done:
                break
            time.sleep(self._task_check_period)
