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

import json
import time
from typing import Union

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, ControllerSpec, Task, TaskOperatorKey
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.operator_spec import OperatorSpec
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.fuel.utils.pipe.pipe import Pipe
from nvflare.fuel.utils.pipe.pipe_monitor import PipeMonitor, Topic, send_to_pipe


class BcastOperator(OperatorSpec, FLComponent):
    def __init__(self):
        OperatorSpec.__init__(self)
        FLComponent.__init__(self)
        self.current_aggregator = None

    def operate(
        self,
        op_description: dict,
        controller: ControllerSpec,
        task_name: str,
        task_data: Shareable,
        abort_signal: Signal,
        fl_ctx: FLContext,
    ) -> Union[Shareable, None]:
        aggr_id = op_description.get(TaskOperatorKey.AGGREGATOR, "")
        if not aggr_id:
            self.log_error(fl_ctx, "missing aggregator component id")
            return None

        engine = fl_ctx.get_engine()
        assert isinstance(engine, ServerEngineSpec)
        aggr = engine.get_component(aggr_id)
        if not aggr:
            self.log_error(fl_ctx, f"no aggregator defined for component id {aggr_id}")
            return None

        if not isinstance(aggr, Aggregator):
            self.log_error(fl_ctx, f"component {aggr_id} must be Aggregator but got {type(aggr)}")
            return None

        # reset the internal state of the aggregator for next round of aggregation
        self.current_aggregator = aggr
        aggr.reset(fl_ctx)

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
            props={"aggr": aggr},
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
        aggr = client_task.task.get_prop("aggr")
        assert isinstance(aggr, Aggregator)
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
    def __init__(self):
        OperatorSpec.__init__(self)
        FLComponent.__init__(self)

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
        engine = fl_ctx.get_engine()

        shareable_generator = None
        shareable_generator_id = op_description.get(TaskOperatorKey.SHAREABLE_GENERATOR, "")
        if shareable_generator_id:
            shareable_generator = engine.get_component(shareable_generator_id)
            if not shareable_generator:
                self.log_error(fl_ctx, f"no shareable generator defined for component id {shareable_generator_id}")
                return None

            if not isinstance(shareable_generator, ShareableGenerator):
                self.log_error(
                    fl_ctx,
                    f"component {shareable_generator_id} must be ShareableGenerator but got {type(shareable_generator)}",
                )
                return None

        persistor_id = op_description.get(TaskOperatorKey.PERSISTOR, "")
        if persistor_id:
            persistor = engine.get_component(persistor_id)
            if not persistor:
                self.log_error(fl_ctx, f"no persistor defined for component id {persistor_id}")
                return None

            if not isinstance(persistor, LearnablePersistor):
                self.log_error(fl_ctx, f"component {persistor_id} must be LearnablePersistor but got {type(persistor)}")
                return None

            # The persistor should convert the TASK_DATA in the fl_ctx into a learnable
            # This learnable is the base for the relay
            learnable_base = persistor.load(fl_ctx)
            fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, learnable_base, private=True, sticky=False)

        task = Task(
            name=task_name,
            data=task_data,
            props={
                AppConstants.CURRENT_ROUND: current_round,
                "last_result": None,
                "shareable_generator": shareable_generator,
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
            dynamic_targets=False,
            abort_signal=abort_signal,
        )
        if abort_signal.triggered:
            return None

        return task.get_prop("last_result")

    def _process_relay_result(self, client_task: ClientTask, fl_ctx: FLContext):
        # submitted shareable is stored in client_task.result
        # we need to update task.data with that shareable so the next target
        # will get the updated shareable
        task = client_task.task
        current_round = task.get_prop(AppConstants.CURRENT_ROUND)
        task.set_prop("last_result", client_task.result)

        task_data = client_task.result
        shareable_generator = task.get_prop("shareable_generator")
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
        task_data_poll_interval: float = 0.5,
    ):
        Controller.__init__(self)
        self.pipe_id = pipe_id
        self.operator_descs = None
        self.task_wait_time = task_wait_time
        self.task_data_poll_interval = task_data_poll_interval
        self.pipe = None
        self.pipe_monitor = None
        self.run_ended = False
        self.task_abort_signal = None
        self.current_task_name = None
        self.current_task_id = None
        self.current_operator = None
        self.t1_run_ended = False
        self.builtin_operators = {"bcast": BcastOperator(), "relay": RelayOperator()}

    def start_controller(self, fl_ctx: FLContext) -> None:
        # get operators
        engine = fl_ctx.get_engine()
        assert isinstance(engine, ServerEngineSpec)
        job_id = fl_ctx.get_job_id()
        workspace = engine.get_workspace()
        app_config_file = workspace.get_server_app_config_file_path(job_id)
        with open(app_config_file) as file:
            app_config = json.load(file)
            self.operator_descs = app_config.get("operators", {})
            self.log_info(fl_ctx, f"Got operator descriptions: {self.operator_descs}")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        if event_type == EventType.START_RUN:
            job_id = fl_ctx.get_job_id()
            self.pipe = engine.get_component(self.pipe_id)
            if not isinstance(self.pipe, Pipe):
                raise TypeError(f"pipe must be Pipe type. Got: {type(self.pipe)}")
            self.pipe.open(name=job_id, me="y")
        elif event_type == EventType.END_RUN:
            self.run_ended = True

    def _abort(self, reason: str, abort_signal: Signal, fl_ctx):
        send_to_pipe(self.pipe, topic=Topic.END_RUN, data=reason)
        if reason:
            self.log_error(fl_ctx, reason)
        if abort_signal:
            abort_signal.trigger(True)

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        try:
            self.pipe_monitor = PipeMonitor(self.pipe, self._run_status_changed)
            self.pipe_monitor.start()
            self._control_flow(abort_signal, fl_ctx)
            self.pipe_monitor.stop()
        except BaseException as ex:
            self.log_exception(fl_ctx, "control flow exception")
            self._abort(f"control_flow exception {ex}", abort_signal, fl_ctx)

    def _control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        start = time.time()
        while True:
            if self.run_ended:
                # tell T1 to end the run
                self._abort(reason="", abort_signal=abort_signal, fl_ctx=fl_ctx)
                return

            if self.t1_run_ended:
                return

            if abort_signal.triggered:
                # tell T1 to end the run
                self._abort(reason="", abort_signal=abort_signal, fl_ctx=fl_ctx)
                return

            topic, data = self.pipe_monitor.get_next()
            if not topic:
                if self.task_wait_time and time.time() - start > self.task_wait_time:
                    # timed out - tell T1 to end the RUN
                    self._abort(
                        reason=f"task data timeout after {self.task_wait_time} secs",
                        abort_signal=abort_signal,
                        fl_ctx=fl_ctx,
                    )
                    return
            else:
                self.log_info(fl_ctx, f"got data for task '{topic}' from T1")
                if not isinstance(data, Shareable):
                    self._abort(
                        reason=f"bad data for task '{topic}' from T1 - must be Shareable but got {type(data)}",
                        abort_signal=abort_signal,
                        fl_ctx=fl_ctx,
                    )
                    return

                task_name = data.get_header(ReservedHeaderKey.TASK_NAME)
                if not task_name:
                    self._abort(
                        reason=f"bad data for task '{topic}' from T1 - missing task name",
                        abort_signal=abort_signal,
                        fl_ctx=fl_ctx,
                    )
                    return

                task_id = data.get_header(ReservedHeaderKey.TASK_ID)
                if not task_id:
                    self._abort(
                        reason=f"bad data for task '{topic}' from T1 - missing task id",
                        abort_signal=abort_signal,
                        fl_ctx=fl_ctx,
                    )
                    return

                op_desc = data.get_header(ReservedHeaderKey.TASK_OPERATOR, {})
                op_id = op_desc.get(TaskOperatorKey.OP_ID)
                if not op_id:
                    op_desc[TaskOperatorKey.OP_ID] = task_name

                self._resolve_op_desc(op_desc)
                op = op_desc.get(TaskOperatorKey.OP, "bcast")
                if not op:
                    self._abort(
                        reason=f"bad operator in task '{topic}' from T1 - missing op",
                        abort_signal=abort_signal,
                        fl_ctx=fl_ctx,
                    )
                    return

                # see whether an Operator is defined
                engine = fl_ctx.get_engine()
                operator = engine.get_component(op)
                if not operator:
                    operator = self.builtin_operators.get(op, None)

                if not operator:
                    self._abort(
                        reason=f"bad task '{topic}' from T1 - no operator for '{op}'",
                        abort_signal=abort_signal,
                        fl_ctx=fl_ctx,
                    )
                    return

                if not isinstance(operator, OperatorSpec):
                    self._abort(
                        reason=f"bad operator for task '{topic}' from T1 - "
                        f"operator for '{op}' must be OperatorSpec but got {type(operator)}",
                        abort_signal=abort_signal,
                        fl_ctx=fl_ctx,
                    )
                    return

                try:
                    self.fire_event(AppEventType.ROUND_STARTED, fl_ctx)
                    fl_ctx.set_prop(key=FLContextKey.TASK_DATA, value=data, private=True, sticky=False)
                    self.current_task_name = task_name
                    self.current_task_id = task_id
                    self.task_abort_signal = abort_signal
                    self.current_operator = operator
                    result = operator.operate(
                        task_name=task_name,
                        task_data=data,
                        op_description=op_desc,
                        controller=self,
                        abort_signal=abort_signal,
                        fl_ctx=fl_ctx,
                    )
                except BaseException as ex:
                    self.log_exception(fl_ctx, f"error processing {task_name}: {ex}")
                    result = None
                finally:
                    self.task_abort_signal = None
                    self.current_task_id = None
                    self.current_operator = None
                    self.fire_event(AppEventType.ROUND_DONE, fl_ctx)

                if not result:
                    self.log_error(fl_ctx, f"no result from '{op}' handler")
                    result = make_reply(ReturnCode.EXECUTION_EXCEPTION)
                elif not isinstance(result, Shareable):
                    self.log_error(fl_ctx, f"bad result from '{op}' handler: expect Shareable but got {type(result)}")
                    result = make_reply(ReturnCode.EXECUTION_EXCEPTION)

                send_to_pipe(self.pipe, topic, result)
                start = time.time()

            time.sleep(self.task_data_poll_interval)

    def _run_status_changed(self, topic: str, data):
        if topic == Topic.ABORT_TASK:
            self.logger.info("ABORT_TASK received from T1")
            # get the signal first since self.task_abort_signal could be set to None at any moment
            # by another thread!
            s = self.task_abort_signal
            if s and data == self.current_task_id:
                s.trigger(True)
            return

        if topic == Topic.END_RUN:
            self.logger.info("END_RUN received from T1")
            # get the signal first since self.task_abort_signal could be set to None at any moment!
            s = self.task_abort_signal
            if s:
                s.trigger(True)
            self.t1_run_ended = True

    def _resolve_op_desc(self, op_desc: dict):
        op_id = op_desc.get(TaskOperatorKey.OP_ID, None)
        if op_id:
            # see whether config is set up for this op
            op_config = self.operator_descs.get(op_id, None)
            if op_config:
                for k, v in op_config.items():
                    if k not in op_desc:
                        op_desc[k] = v

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        # A late reply is received from client.
        # We'll include the late reply into the aggregation only if it's for the same type of tasks (i.e.
        # same task name). Note that the same task name could be used many times (rounds).
        operator = self.current_operator
        if task_name == self.current_task_name and operator:
            assert isinstance(operator, OperatorSpec)
            operator.process_result_of_unknown_task(
                client=client, task_name=task_name, client_task_id=client_task_id, result=result, fl_ctx=fl_ctx
            )

    def stop_controller(self, fl_ctx: FLContext):
        pass
