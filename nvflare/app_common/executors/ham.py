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
import random
import threading
import time
from typing import Optional

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.app_constant import AppConstants
from nvflare.edge.constants import EdgeTaskHeaderKey
from nvflare.edge.utils import message_topic_for_task_end, message_topic_for_task_report, process_aggr_result_from_child
from nvflare.fuel.utils.tree_utils import Forest, Node
from nvflare.fuel.utils.validation_utils import check_positive_number, check_str
from nvflare.fuel.utils.waiter_utils import WaiterRC, conditional_wait
from nvflare.security.logging import secure_format_exception


class TaskInfo:
    def __init__(self, task: Shareable):
        self.task = task
        self.round = task.get_header(AppConstants.CURRENT_ROUND)
        self.id = task.get_header(ReservedKey.TASK_ID)
        self.name = task.get_header(ReservedKey.TASK_NAME)
        self.seq = task.get_header(EdgeTaskHeaderKey.TASK_SEQ)
        self.aggr_interval = task.get_header(EdgeTaskHeaderKey.AGGR_INTERVAL, 1.0)


class HierarchicalAggregationManager(Executor):
    def __init__(
        self,
        learner_id: str,
        aggregator_id: str,
        aggr_report_timeout,
    ):
        Executor.__init__(self)

        check_str("learner_id", learner_id)
        check_str("aggregator_id", aggregator_id)
        check_positive_number("aggr_report_timeout", aggr_report_timeout)
        self.learner_id = learner_id
        self.aggregator_id = aggregator_id
        self.aggr_report_timeout = aggr_report_timeout

        self._pending_task = None
        self._pending_clients = {}
        self._aggregator = None
        self._learner = None
        self._status_lock = threading.Lock()
        self._aggr_lock = threading.Lock()
        self._process_error = None
        self._task_start_time = None
        self._num_aggrs = 0
        self._children = None
        self._num_children = 0
        self._num_children_done = 0
        self._parent_name = None
        self._task_done = False

        self._msg_handler_registered = {}  # topic => bool
        self.register_event_handler(EventType.START_RUN, self._handle_start_run)
        self.register_event_handler(EventType.POST_TASK_ASSIGNMENT_SENT, self._handle_task_sent)
        self.register_event_handler(EventType.POST_TASK_RESULT_RECEIVED, self._handle_result_received)

    def _handle_start_run(self, event_type: str, fl_ctx: FLContext):
        self.log_debug(fl_ctx, f"handling event {event_type}")
        engine = fl_ctx.get_engine()

        aggr = engine.get_component(self.aggregator_id)
        if not isinstance(aggr, Aggregator):
            self.log_error(fl_ctx, f"component '{self.aggregator_id}' must be Aggregator but got {type(aggr)}")
        self._aggregator = aggr

        if self.learner_id:
            learner = engine.get_component(self.learner_id)
            if not isinstance(learner, Executor):
                self.log_error(fl_ctx, f"component '{self.learner_id}' must be Executor but got {type(learner)}")
            self._learner = learner

        client_hierarchy = fl_ctx.get_prop(FLContextKey.CLIENT_HIERARCHY)
        if not isinstance(client_hierarchy, Forest):
            self.system_panic(
                f"cannot get client hierarchy from fl-ctx: expect Forest but got {type(client_hierarchy)}", fl_ctx
            )
            return

        my_name = fl_ctx.get_identity_name()
        my_node = client_hierarchy.nodes.get(my_name)
        if not isinstance(my_node, Node):
            self.system_panic(f"cannot get my node from client hierarchy: expect Noe but got {type(my_node)}", fl_ctx)
            return

        self._children = [n.obj.name for n in my_node.children]
        self._num_children = len(self._children)
        self.log_info(fl_ctx, f"got {self._num_children} child clients: {self._children}")

        parent_node = my_node.parent
        if not parent_node:
            self._parent_name = None  # for server
        else:
            self._parent_name = parent_node.obj.name
        self.log_info(fl_ctx, f"my parent is: {self._parent_name}")

    def _handle_task_sent(self, event_type: str, fl_ctx: FLContext):
        # the task was sent to a child client
        self.log_debug(fl_ctx, f"handling event {event_type}")
        fl_ctx.set_prop(FLContextKey.EVENT_PROCESSED, True, private=True, sticky=False)

        task_info = self._pending_task
        if not task_info:
            # I don't have a pending task
            return

        assert isinstance(task_info, TaskInfo)
        sent_task_id = fl_ctx.get_prop(FLContextKey.TASK_ID)
        if sent_task_id != task_info.id:
            # task sent is not the same as what I have
            self.log_warning(fl_ctx, f"task sent {sent_task_id} is not the same as what I have {task_info.id}")
            return

        child_client_ctx = fl_ctx.get_peer_context()
        assert isinstance(child_client_ctx, FLContext)
        child_client_name = child_client_ctx.get_identity_name()
        self._update_client_status(child_client_name, None)
        self.log_info(fl_ctx, f"sent task {sent_task_id} to child {child_client_name}")

    def _handle_result_received(self, event_type: str, fl_ctx: FLContext):
        # received results from a child client
        self.log_debug(fl_ctx, f"handling event {event_type}")

        # indicate that this event has been processed by me
        fl_ctx.set_prop(FLContextKey.EVENT_PROCESSED, True, private=True, sticky=False)

        task_info = self._pending_task
        if not task_info:
            # I don't have a pending task
            return

        assert isinstance(task_info, TaskInfo)
        result = fl_ctx.get_prop(FLContextKey.TASK_RESULT)
        assert isinstance(result, Shareable)
        peer_ctx = fl_ctx.get_peer_context()
        assert isinstance(peer_ctx, FLContext)
        child_client_name = peer_ctx.get_identity_name()

        rc = result.get_return_code(ReturnCode.OK)
        self.log_info(fl_ctx, f"received task submission from child {child_client_name}: {rc}")

        result_task_id = result.get_header(ReservedKey.TASK_ID)
        if result_task_id != task_info.id:
            self.log_info(
                fl_ctx,
                f"dropped task submission from child {child_client_name} for task {result_task_id}: "
                f"we are working on task {task_info.id}",
            )
            return

        self._update_client_status(child_client_name, time.time())

        has_aggr_data = result.get_header(EdgeTaskHeaderKey.HAS_AGGR_DATA, False)
        if has_aggr_data:
            accepted = self._accept_result(result, fl_ctx, task_info.round)
            self.log_debug(fl_ctx, f"processed aggr result from task submission: {accepted=}")

    def _pending_clients_status(self):
        with self._status_lock:
            if not self._pending_clients:
                return 0, 0

            received = 0
            for received_time in self._pending_clients.values():
                if received_time:
                    received += 1

            return received, len(self._pending_clients)

    def _update_client_status(self, client_name, status):
        with self._status_lock:
            self._pending_clients[client_name] = status

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """Execute the assigned task.
        If we are a leaf node in client hierarchy, we'll execute the task by using the configured executor for the
        task name "exec_<task_name>". This way different tasks can be handled by different executors.

        If we are not leaf node, we'll wait for results from child clients and then aggregate their results using
        the configured aggregator.

        Args:
            task_name: name of the assigned task
            shareable: task data
            fl_ctx: FLContext object
            abort_signal: signal to notify abort

        Returns: task result

        """
        is_leaf = fl_ctx.get_prop(ReservedKey.IS_LEAF)
        if is_leaf and self._learner:
            try:
                return self._learner.execute(task_name, shareable, fl_ctx, abort_signal)
            except Exception as ex:
                self.log_error(fl_ctx, f"exception from {type(self._learner)}: {secure_format_exception(ex)}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # register msg handler for aggr reports from children
        if not self._msg_handler_registered.get(task_name):
            engine = fl_ctx.get_engine()
            engine.register_aux_message_handler(message_topic_for_task_report(task_name), self._process_aggr_result)
            engine.register_aux_message_handler(message_topic_for_task_end(task_name), self._process_task_end)
            self._msg_handler_registered[task_name] = True

        self._pending_task = TaskInfo(shareable)
        self.task_started(self._pending_task, fl_ctx)  # give the CB a separate copy of TaskInfo
        self.log_info(fl_ctx, f"got current_round: {self._pending_task.round}")

        # Set header to indicate that we are ready to manage child clients
        # Note: when a child comes to pull task, the communicator only sends it after the task is ready.
        # This is to avoid the potential race condition that the client gets the task and then quickly submits
        # result before we are even ready.
        shareable.set_header(ReservedKey.TASK_IS_READY, True)
        self._task_start_time = time.time()

        result = self._do_task(fl_ctx, abort_signal)

        # reset state
        self.task_ended(self._pending_task, fl_ctx)
        self._task_done = False
        self._task_start_time = None
        self._pending_task = None
        self._pending_clients = {}
        self._aggregator.reset(fl_ctx)
        self._process_error = False
        return result

    def _do_task(self, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        task_info = self._pending_task
        assert isinstance(task_info, TaskInfo)
        self.log_info(fl_ctx, f"Starting task seq {task_info.seq} and waiting for results from children ...")

        aggr_interval = task_info.aggr_interval
        while True:
            # has the task_done been set?
            if self._task_done:
                self.log_info(fl_ctx, f"task {task_info.seq} is done: task was set to done")
                break

            if self._process_error:
                # we bail out when any processing error encountered
                self.log_info(fl_ctx, f"task seq {task_info.seq} is done: processing error occurred")
                break

            # send aggr results periodically
            report = self._make_aggr_report(task_info, fl_ctx)

            engine = fl_ctx.get_engine()
            replies = engine.send_aux_request(
                targets=self._parent_name,
                topic=message_topic_for_task_report(task_info.name),
                request=report,
                timeout=self.aggr_report_timeout,
                fl_ctx=fl_ctx,
            )

            assert isinstance(replies, dict)
            if len(replies) != 1:
                # this should never happen since the engine should always return a reply
                self.log_error(fl_ctx, f"no reply from parent {self._parent_name}")
                self._process_error = True
                break

            reply = list(replies.values())[0]
            if not isinstance(reply, Shareable):
                self.log_error(
                    fl_ctx,
                    f"bad reply from parent {self._parent_name}: expect reply to be Shareable but got {type(reply)}",
                )
                self._process_error = True
                break

            rc = reply.get_return_code()
            if rc == ReturnCode.TASK_ABORTED:
                self.log_info(fl_ctx, f"task {task_info.seq} is done: parent task is gone")
                break

            if rc != ReturnCode.OK:
                self.log_error(fl_ctx, f"error sending aggr result to parent {self._parent_name}: {rc}")

            parent_task_seq = reply.get_header(EdgeTaskHeaderKey.TASK_SEQ, task_info.seq)
            if parent_task_seq != task_info.seq:
                # this task is done
                self.log_info(fl_ctx, f"task {task_info.seq} is done: parent moved to task {parent_task_seq}")
                break

            # have I received all possible responses from my children?
            if self._num_children > 0:
                received, _ = self._pending_clients_status()
                if received >= self._num_children:
                    self.log_info(fl_ctx, f"task {task_info.seq} is done: all {received} child clients are done!")
                    break

            wrc = conditional_wait(
                waiter=None,
                timeout=aggr_interval + random.uniform(0.0, 0.5),
                abort_signal=abort_signal,
                condition_cb=self._check_task_done,
            )
            if wrc == WaiterRC.ABORTED:
                return make_reply(ReturnCode.TASK_ABORTED)
            elif wrc == WaiterRC.IS_SET:
                break

        received, total = self._pending_clients_status()
        self.log_info(fl_ctx, f"task done after {time.time() - self._task_start_time} secs: {received=} {total=}")

        if self._process_error:
            self.log_error(fl_ctx, "there is process error")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # still anything to be aggregated?
        return self._make_aggr_report(task_info, fl_ctx)

    def _check_task_done(self):
        if self._task_done:
            # force the conditional wait to stop
            return WaiterRC.IS_SET

    def _make_aggr_report(self, task_info: TaskInfo, fl_ctx: FLContext):
        task_data = task_info.task
        cookie_jar = task_data.get_cookie_jar()
        with self._aggr_lock:
            has_aggr_data = True
            report = self._do_aggr(fl_ctx)
            if not report:
                has_aggr_data = False
                report = Shareable()

            self.log_debug(fl_ctx, f"making aggr report to parent for task {task_info.seq}: {has_aggr_data=}")
            report.set_header(EdgeTaskHeaderKey.TASK_SEQ, task_info.seq)
            report.set_header(EdgeTaskHeaderKey.HAS_AGGR_DATA, has_aggr_data)
            report.set_return_code(ReturnCode.OK)
            report.set_cookie_jar(cookie_jar)
            return report

    def _do_aggr(self, fl_ctx: FLContext):
        if self._num_aggrs <= 0:
            # nothing to aggregate
            return None

        try:
            result = self._aggregator.aggregate(fl_ctx)
            self._aggregator.reset(fl_ctx)
            self._num_aggrs = 0
        except Exception as ex:
            self.log_error(
                fl_ctx, f"exception 'aggregate' from {type(self._aggregator)}: {secure_format_exception(ex)}"
            )
            self._process_error = True
            result = None
        return result

    def _process_task_end(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        # process notification from parent that task is ended
        task_seq = request.get_header(EdgeTaskHeaderKey.TASK_SEQ)

        if self._num_children > 0:
            # fire-and-forget notification to all my children
            req = Shareable()
            req.set_header(EdgeTaskHeaderKey.TASK_SEQ, task_seq)
            engine = fl_ctx.get_engine()
            engine.send_aux_request(
                targets=self._children,
                topic=topic,
                request=req,
                timeout=0,  # fire and forget
                fl_ctx=fl_ctx,
                optional=True,
            )

        task_info = self._pending_task
        if task_info:
            if task_info.seq <= task_seq:
                # my current task is before the ended task - end my task
                self.log_info(
                    fl_ctx, f"ended current task seq {task_info.seq}: got end_task from parent for task {task_seq}"
                )
                self._task_done = True
            else:
                self.log_info(
                    fl_ctx, f"ignored end_task from parent for task {task_seq} since it's < my task seq {task_info.seq}"
                )
        else:
            self.log_info(fl_ctx, f"ignored end_task from parent for task {task_seq} since I have no current task")

        return make_reply(ReturnCode.OK)

    def _process_aggr_result(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        self.log_debug(fl_ctx, f"processing aggregation result report: {topic}")

        task_info = self._pending_task
        if task_info:
            assert isinstance(task_info, TaskInfo)
            seq = task_info.seq
            current_round = task_info.round
        else:
            seq = 0
            current_round = -1

        accepted, reply = process_aggr_result_from_child(
            processor=self,
            request=request,
            current_task_seq=seq,
            fl_ctx=fl_ctx,
            accept_f=self._accept_result,
            current_round=current_round,
        )

        peer_ctx = fl_ctx.get_peer_context()
        assert isinstance(peer_ctx, FLContext)
        client_name = peer_ctx.get_identity_name()
        self.log_debug(fl_ctx, f"processed aggr result report from {client_name} at round {current_round}: {accepted=}")
        return reply

    def _accept_result(self, result: Shareable, fl_ctx: FLContext, current_round) -> bool:
        fl_ctx.set_prop(AppConstants.CURRENT_ROUND, current_round, private=True, sticky=False)
        with self._aggr_lock:
            try:
                accepted = self._aggregator.accept(result, fl_ctx)
            except Exception as ex:
                self.log_error(fl_ctx, f"exception accepting result: {secure_format_exception(ex)}")
                accepted = False

            if accepted:
                self._num_aggrs += 1

            return accepted

    def accept_contribution(self, task_id: str, contribution: Shareable, fl_ctx: FLContext) -> bool:
        """This is to be called by subclass to accept a specified contribution to aggregation

        Args:
            task_id: ID of the task
            contribution: the contribution to be accepted.
            fl_ctx: FLContext object

        Returns: whether the contribution is accepted

        """
        task_info = self._pending_task
        if not task_info:
            self.log_warning(fl_ctx, f"contribution dropped for task_id {task_id}: no current task")
            return False

        if task_id != task_info.id:
            self.log_warning(
                fl_ctx, f"contribution dropped for task_id {task_id}: it does not match current task {task_info.id}"
            )
            return False
        return self._accept_result(contribution, fl_ctx, task_info.round)

    def set_task_done(self, task_id: str, fl_ctx: FLContext) -> bool:
        """This method is to be called by subclass to forcefully end the specified task

        Args:
            task_id: ID of the task to be ended
            fl_ctx: FLContext object

        Returns: whether this request is accepted

        """
        task_info = self._pending_task
        if not task_info:
            self.log_info(fl_ctx, f"ignored set_task_done for task_id {task_id}: no current task.")
            return False

        if task_id != task_info.id:
            self.log_info(
                fl_ctx, f"ignored set_task_done for task_id {task_id}: it does not match current task {task_info.id}"
            )
            return False

        self._task_done = True
        self.log_info(fl_ctx, f"accepted set_task_done for task_id {task_id}")
        return True

    def get_current_task(self) -> Optional[TaskInfo]:
        """Get the info of current task

        Returns: TaskInfo of current task or None if no current task

        """
        return self._pending_task

    def task_started(self, task: TaskInfo, fl_ctx: FLContext):
        """This method is called when a task assignment is received from the controller.
        Subclass can implement this method to prepare for task processing.

        Args:
            task: info of the received task
            fl_ctx: FLContext object

        Returns: None

        """
        pass

    def task_ended(self, task: TaskInfo, fl_ctx: FLContext):
        """This method is called when the current task is ended.
        Subclass can implement this method to finish task processing.

        Args:
            task: info of the task that is ended
            fl_ctx: FLContext object

        Returns: None

        """
        pass
