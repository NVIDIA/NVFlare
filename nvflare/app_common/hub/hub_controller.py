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

import json
import time
from typing import Union

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import (
    ClientTask,
    ControllerSpec,
    OperatorConfigKey,
    OperatorMethod,
    Task,
    TaskOperatorKey,
)
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.operator_spec import OperatorSpec
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.fuel.utils.constants import PipeChannelName
from nvflare.fuel.utils.pipe.pipe import Message, Pipe
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler, Topic
from nvflare.fuel.utils.validation_utils import check_object_type, check_positive_number, check_str


class BcastOperator(OperatorSpec, FLComponent):

    _PROP_AGGR = "aggr"

    def __init__(self):
        OperatorSpec.__init__(self)
        FLComponent.__init__(self)
        self.current_aggregator = None

    @staticmethod
    def _get_aggregator(op_description: dict, fl_ctx: FLContext):
        aggr_id = op_description.get(TaskOperatorKey.AGGREGATOR, "")
        if not aggr_id:
            raise RuntimeError("missing aggregator component id")

        engine = fl_ctx.get_engine()
        aggr = engine.get_component(aggr_id)
        if not aggr:
            raise RuntimeError(f"no aggregator defined for component id {aggr_id}")

        if not isinstance(aggr, Aggregator):
            raise RuntimeError(f"component {aggr_id} must be Aggregator but got {type(aggr)}")

        return aggr

    def operate(
        self,
        op_description: dict,
        controller: ControllerSpec,
        task_name: str,
        task_data: Shareable,
        abort_signal: Signal,
        fl_ctx: FLContext,
    ) -> Union[Shareable, None]:
        aggr = self._get_aggregator(op_description, fl_ctx)

        # reset the internal state of the aggregator for next round of aggregation
        self.current_aggregator = aggr
        aggr.reset(fl_ctx)

        engine = fl_ctx.get_engine()
        total_num_clients = len(engine.get_clients())
        timeout = op_description.get(TaskOperatorKey.TIMEOUT, 0)
        wait_time_after_min_resps = op_description.get(TaskOperatorKey.WAIT_TIME_AFTER_MIN_RESPS, 5)
        min_clients = op_description.get(TaskOperatorKey.MIN_TARGETS, 0)
        if min_clients > total_num_clients:
            min_clients = total_num_clients
            wait_time_after_min_resps = 0
        targets = op_description.get(TaskOperatorKey.TARGETS, None)

        # data is from T1
        train_task = Task(
            name=task_name,
            data=task_data,
            props={self._PROP_AGGR: aggr},
            timeout=timeout,
            result_received_cb=self._process_bcast_result,
        )

        controller.broadcast_and_wait(
            task=train_task,
            targets=targets,
            min_responses=min_clients,
            wait_time_after_min_received=wait_time_after_min_resps,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )

        aggr_result = aggr.aggregate(fl_ctx)
        self.current_aggregator = None
        return aggr_result

    def _process_bcast_result(self, client_task: ClientTask, fl_ctx: FLContext) -> None:
        result = client_task.result
        aggr = client_task.task.get_prop(self._PROP_AGGR)
        aggr.accept(result, fl_ctx)

        # Cleanup task result
        client_task.result = None

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        aggr = self.current_aggregator
        if aggr:
            aggr.accept(result, fl_ctx)


class RelayOperator(OperatorSpec, FLComponent):

    _PROP_LAST_RESULT = "last_result"
    _PROP_SHAREABLE_GEN = "shareable_generator"

    def __init__(self):
        OperatorSpec.__init__(self)
        FLComponent.__init__(self)

    @staticmethod
    def _get_shareable_generator(op_description: dict, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        comp_id = op_description.get(TaskOperatorKey.SHAREABLE_GENERATOR, "")
        if not comp_id:
            return None

        shareable_generator = engine.get_component(comp_id)
        if not shareable_generator:
            raise RuntimeError(f"no shareable generator defined for component id {comp_id}")

        if not isinstance(shareable_generator, ShareableGenerator):
            raise RuntimeError(f"component {comp_id} must be ShareableGenerator but got {type(shareable_generator)}")
        return shareable_generator

    @staticmethod
    def _get_persistor(op_description: dict, fl_ctx: FLContext):
        persistor_id = op_description.get(TaskOperatorKey.PERSISTOR, "")
        if not persistor_id:
            return None

        engine = fl_ctx.get_engine()
        persistor = engine.get_component(persistor_id)
        if not persistor:
            raise RuntimeError(f"no persistor defined for component id {persistor_id}")

        if not isinstance(persistor, LearnablePersistor):
            raise RuntimeError(f"component {persistor_id} must be LearnablePersistor but got {type(persistor)}")
        return persistor

    def operate(
        self,
        op_description: dict,
        controller: ControllerSpec,
        task_name: str,
        task_data: Shareable,
        abort_signal: Signal,
        fl_ctx: FLContext,
    ) -> Union[None, Shareable]:
        current_round = task_data.get_header(AppConstants.CURRENT_ROUND, None)
        shareable_generator = self._get_shareable_generator(op_description, fl_ctx)
        persistor = self._get_persistor(op_description, fl_ctx)
        if persistor:
            # The persistor should convert the TASK_DATA in the fl_ctx into a learnable
            # This learnable is the base for the relay
            learnable_base = persistor.load(fl_ctx)
            fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, learnable_base, private=True, sticky=False)

        task = Task(
            name=task_name,
            data=task_data,
            props={
                AppConstants.CURRENT_ROUND: current_round,
                self._PROP_LAST_RESULT: None,
                self._PROP_SHAREABLE_GEN: shareable_generator,
            },
            result_received_cb=self._process_relay_result,
        )

        targets = op_description.get(TaskOperatorKey.TARGETS, None)
        task_assignment_timeout = op_description.get(TaskOperatorKey.TASK_ASSIGNMENT_TIMEOUT, 0)

        controller.relay_and_wait(
            task=task,
            targets=targets,
            task_assignment_timeout=task_assignment_timeout,
            fl_ctx=fl_ctx,
            dynamic_targets=True,
            abort_signal=abort_signal,
        )
        if abort_signal.triggered:
            return None

        return task.get_prop(self._PROP_LAST_RESULT)

    def _process_relay_result(self, client_task: ClientTask, fl_ctx: FLContext):
        # submitted shareable is stored in client_task.result
        # we need to update task.data with that shareable so the next target
        # will get the updated shareable
        task = client_task.task
        current_round = task.get_prop(AppConstants.CURRENT_ROUND)
        task.set_prop(self._PROP_LAST_RESULT, client_task.result)

        task_data = client_task.result
        shareable_generator = task.get_prop(self._PROP_SHAREABLE_GEN)
        if shareable_generator:
            # turn received result (a Shareable) to learnable (i.e. weight diff => weight)
            learnable = shareable_generator.shareable_to_learnable(client_task.result, fl_ctx)

            # turn the learnable to task data for the next leg (i.e. weight Learnable to weight Shareable)
            task_data = shareable_generator.learnable_to_shareable(learnable, fl_ctx)

        if current_round:
            task_data.set_header(AppConstants.CURRENT_ROUND, current_round)
        task.data = task_data
        client_task.result = None


class HubController(Controller):
    def __init__(
        self,
        pipe_id: str,
        task_wait_time=None,
        task_data_poll_interval: float = 0.1,
    ):
        Controller.__init__(self)

        check_positive_number("task_data_poll_interval", task_data_poll_interval)
        check_str("pipe_id", pipe_id)
        if task_wait_time is not None:
            check_positive_number("task_wait_time", task_wait_time)

        self.pipe_id = pipe_id
        self.operator_descs = None
        self.task_wait_time = task_wait_time
        self.task_data_poll_interval = task_data_poll_interval
        self.pipe = None
        self.pipe_handler = None
        self.run_ended = False
        self.task_abort_signal = None
        self.current_task_name = None
        self.current_task_id = None
        self.current_operator = None
        self.builtin_operators = {OperatorMethod.BROADCAST: BcastOperator(), OperatorMethod.RELAY: RelayOperator()}
        self.project_name = ""

    def start_controller(self, fl_ctx: FLContext) -> None:
        self.project_name = fl_ctx.get_identity_name()

        # get operators
        engine = fl_ctx.get_engine()
        job_id = fl_ctx.get_job_id()
        workspace = engine.get_workspace()
        app_config_file = workspace.get_server_app_config_file_path(job_id)
        with open(app_config_file) as file:
            app_config = json.load(file)
            self.operator_descs = app_config.get(OperatorConfigKey.OPERATORS, {})
            self.log_debug(fl_ctx, f"Got operator descriptions: {self.operator_descs}")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        if event_type == EventType.START_RUN:
            pipe = engine.get_component(self.pipe_id)
            check_object_type("pipe", pipe, Pipe)
            pipe.open(name=PipeChannelName.TASK)
            self.pipe_handler = PipeHandler(pipe)
        elif event_type == EventType.END_RUN:
            self.run_ended = True

    def _abort(self, reason: str, abort_signal: Signal, fl_ctx):
        self.pipe_handler.notify_abort(reason)
        if reason:
            self.log_error(fl_ctx, reason)
        if abort_signal:
            abort_signal.trigger(True)

    def _get_operator(self, task_name: str, op_desc: dict, fl_ctx: FLContext):
        method_name = op_desc.get(TaskOperatorKey.METHOD)
        if not method_name:
            return None, f"bad operator in task '{task_name}' from T1 - missing method name"

        # see whether an Operator is defined for the method
        engine = fl_ctx.get_engine()
        operator = engine.get_component(method_name)
        if not operator:
            operator = self.builtin_operators.get(method_name, None)

        if not operator:
            return None, f"bad task '{task_name}' from T1 - no operator for '{method_name}'"

        if not isinstance(operator, OperatorSpec):
            return None, f"operator for '{method_name}' must be OperatorSpec but got {type(operator)}"
        return operator, ""

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        try:
            self.pipe_handler.start()
            self._control_flow(abort_signal, fl_ctx)
            self.pipe_handler.stop()
        except Exception as ex:
            self.log_exception(fl_ctx, "control flow exception")
            self._abort(f"control_flow exception {ex}", abort_signal, fl_ctx)

    def _control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        control_flow_start = time.time()
        task_start = control_flow_start

        while True:
            if self.run_ended:
                # tell T1 to end the run
                self._abort(reason="", abort_signal=abort_signal, fl_ctx=fl_ctx)
                return

            if abort_signal.triggered:
                # tell T1 to end the run
                self._abort(reason="", abort_signal=abort_signal, fl_ctx=fl_ctx)
                return

            msg = self.pipe_handler.get_next()
            if not msg:
                if self.task_wait_time and time.time() - task_start > self.task_wait_time:
                    # timed out - tell T1 to end the RUN
                    self._abort(
                        reason=f"task data timeout after {self.task_wait_time} secs",
                        abort_signal=abort_signal,
                        fl_ctx=fl_ctx,
                    )
                    return
            else:
                if msg.topic in [Topic.ABORT, Topic.END, Topic.PEER_GONE]:
                    # the T1 peer is gone
                    self.log_info(fl_ctx, f"T1 stopped: '{msg.topic}'")
                    return

                if msg.msg_type != Message.REQUEST:
                    self.log_info(fl_ctx, f"ignored '{msg.topic}' from T1 - not a request!")
                    continue

                self.log_info(fl_ctx, f"got data for task '{msg.topic}' from T1")
                if not isinstance(msg.data, Shareable):
                    self._abort(
                        reason=f"bad data for task '{msg.topic}' from T1 - must be Shareable but got {type(msg.data)}",
                        abort_signal=abort_signal,
                        fl_ctx=fl_ctx,
                    )
                    return

                task_data = msg.data
                task_name = task_data.get_header(ReservedHeaderKey.TASK_NAME)
                if not task_name:
                    self._abort(
                        reason=f"bad data for task '{msg.topic}' from T1 - missing task name",
                        abort_signal=abort_signal,
                        fl_ctx=fl_ctx,
                    )
                    return

                task_id = task_data.get_header(ReservedHeaderKey.TASK_ID)
                if not task_id:
                    self._abort(
                        reason=f"bad data for task '{msg.topic}' from T1 - missing task id",
                        abort_signal=abort_signal,
                        fl_ctx=fl_ctx,
                    )
                    return

                op_desc = task_data.get_header(ReservedHeaderKey.TASK_OPERATOR, {})
                op_id = op_desc.get(TaskOperatorKey.OP_ID)
                if not op_id:
                    # use task_name as the operation id
                    op_desc[TaskOperatorKey.OP_ID] = task_name

                self._resolve_op_desc(op_desc, fl_ctx)
                operator, err = self._get_operator(task_name, op_desc, fl_ctx)
                if not operator:
                    self._abort(reason=err, abort_signal=abort_signal, fl_ctx=fl_ctx)
                    return

                operator_name = operator.__class__.__name__
                self.log_info(fl_ctx, f"Invoking Operator {operator_name} for task {task_name}")
                try:
                    current_round = task_data.get_header(AppConstants.CURRENT_ROUND, 0)
                    fl_ctx.set_prop(AppConstants.CURRENT_ROUND, current_round, private=True, sticky=True)

                    contrib_round = task_data.get_cookie(AppConstants.CONTRIBUTION_ROUND)
                    if contrib_round is None:
                        self.log_warning(fl_ctx, "CONTRIBUTION_ROUND Not Set!")

                    self.fire_event(AppEventType.ROUND_STARTED, fl_ctx)
                    fl_ctx.set_prop(key=FLContextKey.TASK_DATA, value=task_data, private=True, sticky=False)
                    self.current_task_name = task_name
                    self.current_task_id = task_id
                    self.task_abort_signal = abort_signal
                    self.current_operator = operator
                    result = operator.operate(
                        task_name=task_name,
                        task_data=task_data,
                        op_description=op_desc,
                        controller=self,
                        abort_signal=abort_signal,
                        fl_ctx=fl_ctx,
                    )
                except:
                    self.log_exception(fl_ctx, f"exception processing '{task_name}' from operator '{operator_name}'")
                    result = None
                finally:
                    self.task_abort_signal = None
                    self.current_task_id = None
                    self.current_operator = None
                    self.fire_event(AppEventType.ROUND_DONE, fl_ctx)

                if not result:
                    self.log_error(fl_ctx, f"no result from operator '{operator_name}'")
                    result = make_reply(ReturnCode.EXECUTION_EXCEPTION)
                elif not isinstance(result, Shareable):
                    self.log_error(
                        fl_ctx, f"bad result from operator '{operator_name}': expect Shareable but got {type(result)}"
                    )
                    result = make_reply(ReturnCode.EXECUTION_EXCEPTION)

                reply = Message.new_reply(topic=msg.topic, data=result, req_msg_id=msg.msg_id)
                self.pipe_handler.send_to_peer(reply)
                task_start = time.time()

            time.sleep(self.task_data_poll_interval)

    def _resolve_op_desc(self, op_desc: dict, fl_ctx: FLContext):
        """
        Determine the correct operation description.

        There may be "operators" in job's config_fed_server.json.
        If present, it describes the operations for tasks, and its descriptions override op_desc that comes from task!
        It may specify a different method than the one in op_desc!
        For example, the op_desc may specify the method 'bcast', but the config could specify 'relay'.
        In this case, the 'relay' method will be used.

        Args:
            op_desc: the op description that comes from the task data

        Returns: None

        """
        op_id = op_desc.get(TaskOperatorKey.OP_ID, None)
        if op_id:
            # see whether config is set up for this op
            # if so, the info in the config overrides op_desc!
            # first try to find project-specific definition
            op_config = self.operator_descs.get(f"{self.project_name}.{op_id}", None)
            if op_config:
                self.log_debug(fl_ctx, f"Use CONFIGURED OPERATORS for {self.project_name}.{op_id}")
            else:
                # try to find general definition
                op_config = self.operator_descs.get(op_id, None)
                if op_config:
                    self.log_debug(fl_ctx, f"Use CONFIGURED OPERATORS for {op_id}")

            if op_config:
                op_desc.update(op_config)
            else:
                self.log_debug(fl_ctx, "OPERATORS NOT CONFIGURED")

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        # A late reply is received from client.
        # We'll include the late reply into the aggregation only if it's for the same type of tasks (i.e.
        # same task name). Note that the same task name could be used many times (rounds).
        self.log_info(fl_ctx, f"Late response received from client {client.name} for task '{task_name}'")
        operator = self.current_operator
        if task_name == self.current_task_name and operator:
            operator.process_result_of_unknown_task(
                client=client, task_name=task_name, client_task_id=client_task_id, result=result, fl_ctx=fl_ctx
            )
        else:
            self.log_warning(fl_ctx, f"Dropped late response received from client {client.name} for task '{task_name}'")

    def stop_controller(self, fl_ctx: FLContext):
        pass
