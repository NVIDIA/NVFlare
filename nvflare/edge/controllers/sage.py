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

import gc
import time
from enum import Enum

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.edge.assessor import Assessment, Assessor
from nvflare.edge.constants import EdgeTaskHeaderKey
from nvflare.edge.utils import message_topic_for_task_end, message_topic_for_task_update, process_update_from_child
from nvflare.fuel.utils.validation_utils import check_positive_number, check_str
from nvflare.fuel.utils.waiter_utils import WaiterRC, conditional_wait
from nvflare.security.logging import secure_format_exception
from nvflare.widgets.info_collector import GroupInfoCollector, InfoCollector


class TaskDoneReason(Enum):
    ALL_CHILDREN_DONE = "all_children_done"
    ABORTED = "aborted"
    ASSESSED_TASK_DONE = "assessed_task_done"
    ASSESSED_WORKFLOW_DONE = "assessed_workflow_done"


class ScatterAndGatherForEdge(Controller):

    next_task_seq = 0

    def __init__(
        self,
        num_rounds: int = 5,
        assessor_id: str = "assessor",
        task_name=AppConstants.TASK_TRAIN,
        task_check_period: float = 0.5,
        assess_interval: float = 0.5,
        update_interval: float = 1.0,
    ):
        """ScatterAndGatherForEdge Workflow.

        The ScatterAndGatherForEdge workflow is a Fed Average algorithm for hierarchically organized edge devices.

        During the execution of a task, the assessor (specified by assessor_id) is invoked periodically to assess
        the quality of training results to determine whether the task should be continued.

        Args:
            num_rounds (int, optional): The total number of training rounds. Defaults to 5.
            assessor_id (str): ID of the assessor component.
            task_name (str): Name of the train task. Defaults to "train".
            task_check_period (float, optional): interval for checking status of tasks. Defaults to 0.5.
            assess_interval: how often to invoke the assessor during task execution
            update_interval: how often for children to send updates

        Raises:
            TypeError: when any of input arguments does not have correct type
            ValueError: when any of input arguments is out of range
        """
        super().__init__(task_check_period=task_check_period)

        # Check arguments
        check_str("assessor_id", assessor_id)
        check_str("task_name", task_name)
        check_positive_number("task_check_period", task_check_period)
        check_positive_number("assess_interval", assess_interval)
        check_positive_number("update_interval", update_interval)

        self.assessor_id = assessor_id
        self.task_name = task_name
        self.assessor = None

        # config data
        self._num_rounds = num_rounds
        self._assess_interval = assess_interval
        self._update_interval = update_interval

        # workflow phases: init, train, validate
        self._current_round = None
        self._current_task_seq = 0
        self._num_children = 0
        self._children = None
        self._end_task_topic = message_topic_for_task_end(self.task_name)
        self._wf_done = False

    @classmethod
    def get_next_task_seq(cls):
        cls.next_task_seq += 1
        return cls.next_task_seq

    def start_controller(self, fl_ctx: FLContext) -> None:
        self.log_info(fl_ctx, f"Initializing {self._name} workflow.")

        engine = fl_ctx.get_engine()
        self.assessor = engine.get_component(self.assessor_id)
        if not isinstance(self.assessor, Assessor):
            self.system_panic(
                f"Assessor {self.assessor_id} must be an Assessor but got {type(self.assessor)}",
                fl_ctx,
            )
            return

        # register aux message handler for receiving aggr results from children
        engine.register_aux_message_handler(message_topic_for_task_update(self.task_name), self._process_update_report)

        # get children clients
        client_hierarchy = fl_ctx.get_prop(FLContextKey.CLIENT_HIERARCHY)
        self._children = client_hierarchy.roots
        self._num_children = len(self._children)
        self.log_info(fl_ctx, f"my child clients: {self._children}")

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        try:
            self.log_info(fl_ctx, f"Starting {self._name}")
            fl_ctx.set_prop(AppConstants.NUM_ROUNDS, self._num_rounds, private=True, sticky=False)
            self.fire_event(AppEventType.TRAINING_STARTED, fl_ctx)

            for r in range(self._num_rounds):
                round_start = time.time()
                self._current_round = r

                if self._check_abort_signal(fl_ctx, abort_signal):
                    break

                self.log_info(fl_ctx, f"Round {r} started.")
                self._current_task_seq = self.get_next_task_seq()

                # Create train_task
                fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self._current_round, private=True, sticky=True)

                try:
                    task_data = self.assessor.start_task(fl_ctx)
                except Exception as ex:
                    self.log_exception(fl_ctx, f"exception in 'start_task' from {type(self.assessor)}")
                    self.system_panic(
                        f"Task execution encountered exception: {secure_format_exception(ex)}",
                        fl_ctx,
                    )
                    break

                task_data.set_header(AppConstants.CURRENT_ROUND, self._current_round)
                task_data.set_header(AppConstants.NUM_ROUNDS, self._num_rounds)
                task_data.set_header(EdgeTaskHeaderKey.TASK_SEQ, self._current_task_seq)
                task_data.set_header(EdgeTaskHeaderKey.UPDATE_INTERVAL, self._update_interval)
                task_data.add_cookie(AppConstants.CONTRIBUTION_ROUND, self._current_round)

                self.fire_event_with_data(AppEventType.ROUND_STARTED, fl_ctx, FLContextKey.TASK_DATA, task_data)

                task = Task(
                    name=self.task_name,
                    data=task_data,
                    before_task_sent_cb=self._prepare_train_task_data,
                    result_received_cb=self._process_train_result,
                )

                self.broadcast(
                    task=task,
                    fl_ctx=fl_ctx,
                    targets=self._children,
                    min_responses=self._num_children,
                    wait_time_after_min_received=0,
                )

                # monitor the task until it's done
                seq = self._current_task_seq
                try:
                    task_done_reason = self._monitor_task(task, fl_ctx, abort_signal)
                except Exception as ex:
                    self.system_panic(
                        f"Task {seq} execution encountered exception: {secure_format_exception(ex)}",
                        fl_ctx,
                    )
                    self.log_exception(fl_ctx, "exception in monitor_task")
                    return

                try:
                    self.assessor.end_task(fl_ctx)
                except Exception as ex:
                    self.log_exception(fl_ctx, f"exception in 'end_task' from {type(self.assessor)}")
                    self.system_panic(
                        f"Task execution encountered exception: {secure_format_exception(ex)}",
                        fl_ctx,
                    )
                    return

                self._current_task_seq = 0
                if not task.completion_status:
                    self.cancel_task(task, fl_ctx=fl_ctx)

                if task_done_reason in [TaskDoneReason.ABORTED, TaskDoneReason.ASSESSED_WORKFLOW_DONE]:
                    break

                self.fire_event(AppEventType.ROUND_DONE, fl_ctx)
                self.log_info(fl_ctx, f"Round {self._current_round} finished in {time.time() - round_start} seconds")
                gc.collect()

            self._wf_done = True
            self._current_task_seq = 0
            self.log_info(fl_ctx, f"Finished {self._name}")

            # give some time for clients to end gracefully when sync task seq
            time.sleep(self._update_interval + 1.0)
        except Exception as e:
            error_msg = f"Exception in {self._name} workflow: {secure_format_exception(e)}"
            self.log_exception(fl_ctx, error_msg)
            self.system_panic(error_msg, fl_ctx)

    def _monitor_task(self, task: Task, fl_ctx: FLContext, abort_signal: Signal) -> TaskDoneReason:
        seq = self._current_task_seq
        while True:
            if task.completion_status:
                # all children are done with their current task
                self.log_info(fl_ctx, f"Task seq {seq} is completed: {task.completion_status=}")
                return TaskDoneReason.ALL_CHILDREN_DONE

            assessment = self.assessor.assess(fl_ctx)
            if assessment != Assessment.CONTINUE:
                self.log_info(fl_ctx, f"Task seq {seq} is done: {assessment=}")

                # notify children to end task
                req = Shareable()
                req.set_header(EdgeTaskHeaderKey.TASK_SEQ, seq)
                engine = fl_ctx.get_engine()
                engine.send_aux_request(
                    targets=self._children,
                    topic=self._end_task_topic,
                    request=req,
                    timeout=0,  # fire and forget
                    fl_ctx=fl_ctx,
                    optional=True,
                )
                if assessment == Assessment.WORKFLOW_DONE:
                    return TaskDoneReason.ASSESSED_WORKFLOW_DONE
                else:
                    return TaskDoneReason.ASSESSED_TASK_DONE

            wrc = conditional_wait(
                waiter=None,
                timeout=self._assess_interval,
                abort_signal=abort_signal,
            )
            if wrc == WaiterRC.ABORTED:
                self.log_info(fl_ctx, f"Task seq {seq} is done: ABORTED")
                return TaskDoneReason.ABORTED

    def stop_controller(self, fl_ctx: FLContext):
        try:
            self.assessor.finalize(fl_ctx)
        except Exception as e:
            self.log_exception(fl_ctx, f"error finalizing assessor: {secure_format_exception(e)}")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        super().handle_event(event_type, fl_ctx)
        if event_type == InfoCollector.EVENT_TYPE_GET_STATS:
            collector = fl_ctx.get_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR, None)
            if collector:
                if not isinstance(collector, GroupInfoCollector):
                    raise TypeError("collector must be GroupInfoCollector but got {}".format(type(collector)))

                collector.add_info(
                    group_name=self._name,
                    info={"current_round": self._current_round, "num_rounds": self._num_rounds},
                )

    def _prepare_train_task_data(self, client_task: ClientTask, fl_ctx: FLContext) -> None:
        self.fire_event_with_data(
            AppEventType.BEFORE_TRAIN_TASK, fl_ctx, AppConstants.TRAIN_SHAREABLE, client_task.task.data
        )

    def _process_train_result(self, client_task: ClientTask, fl_ctx: FLContext) -> None:
        result = client_task.result
        client_task.result = None
        client_name = client_task.client.name

        assert isinstance(result, Shareable)
        rc = result.get_return_code()

        # Raise errors if bad peer context or execution exception.
        if rc and rc != ReturnCode.OK:
            self.system_panic(
                f"Result from {client_name} is bad, error code: {rc}. "
                f"{self.__class__.__name__} exiting at round {self._current_round}.",
                fl_ctx=fl_ctx,
            )
            return

        has_update_data = result.get_header(EdgeTaskHeaderKey.HAS_UPDATE_DATA, False)
        if has_update_data:
            accepted = self._accept_update(result, fl_ctx)
            self.log_debug(fl_ctx, f"processed update from task submission: {accepted=}")

    def process_result_of_unknown_task(
        self, client: Client, task_name, client_task_id, result: Shareable, fl_ctx: FLContext
    ) -> None:
        if not self._wf_done:
            self.log_warning(fl_ctx, f"Ignoring late result from {client.name} for task '{task_name}' {client_task_id}")

    def _process_update_report(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        accepted, reply = process_update_from_child(
            processor=self,
            update=request,
            update_f=self._accept_update,
            current_task_seq=self._current_task_seq,
            fl_ctx=fl_ctx,
        )
        self.log_debug(fl_ctx, f"processed update from report: {accepted=}")
        return reply

    def _check_abort_signal(self, fl_ctx, abort_signal: Signal):
        if abort_signal.triggered:
            self.log_info(fl_ctx, f"Abort signal received. Exiting at round {self._current_round}.")
            return True
        return False

    def _accept_update(self, update: Shareable, fl_ctx: FLContext):
        return self.assessor.process_child_update(update, fl_ctx)
