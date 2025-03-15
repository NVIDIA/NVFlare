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
import threading
import time

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.model import ModelLearnable, make_model_learnable
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.edge.assessor import Assessor, AssessResult
from nvflare.edge.constants import EdgeTaskHeaderKey
from nvflare.edge.utils import message_topic_for_task, process_aggr_result_from_child
from nvflare.fuel.utils.validation_utils import (
    check_non_negative_int,
    check_positive_int,
    check_positive_number,
    check_str,
)
from nvflare.fuel.utils.waiter_utils import WaiterRC, conditional_wait
from nvflare.security.logging import secure_format_exception
from nvflare.widgets.info_collector import GroupInfoCollector, InfoCollector


class _DummyShareableGenerator(ShareableGenerator):
    def shareable_to_learnable(self, shareable: Shareable, fl_ctx: FLContext) -> Learnable:
        result = Learnable()
        for k, v in shareable.items():
            result[k] = v
        return result

    def learnable_to_shareable(self, model: Learnable, fl_ctx: FLContext) -> Shareable:
        result = Shareable()
        for k, v in model.items():
            result[k] = v
        return result


class ScatterAndGatherForEdge(Controller):

    next_task_seq = 0

    def __init__(
        self,
        num_rounds: int = 5,
        aggregator_id=AppConstants.DEFAULT_AGGREGATOR_ID,
        persistor_id: str = "",
        shareable_generator_id=AppConstants.DEFAULT_SHAREABLE_GENERATOR_ID,
        assessor_id: str = "assessor",
        task_name=AppConstants.TASK_TRAIN,
        allow_empty_global_weights: bool = False,
        task_check_period: float = 0.5,
        persist_every_n_rounds: int = 1,
        assess_interval: float = 1.0,
        aggregation_interval: float = 2.0,
    ):
        """ScatterAndGatherForEdge Workflow.

        The ScatterAndGather workflow defines FederatedAveraging on all clients.
        The model persistor (persistor_id) is used to load the initial global model which is sent to all clients.
        Each client sends it's updated weights after local training which is aggregated (aggregator_id). The
        shareable generator is used to convert the aggregated weights to shareable and shareable back to weight.
        The model_persistor also saves the model after training.

        Args:
            num_rounds (int, optional): The total number of training rounds. Defaults to 5.
            aggregator_id (str, optional): ID of the aggregator component. Defaults to "aggregator".
            persistor_id (str, optional): ID of the persistor component. Defaults to "".
            shareable_generator_id (str, optional): ID of the shareable generator. Defaults to "shareable_generator".
            task_name (str, optional): Name of the train task. Defaults to "train".
            train_timeout (int, optional): Time to wait for clients to do local training.
            allow_empty_global_weights (bool, optional): whether to allow empty global weights. Some pipelines can have
                empty global weights at first round, such that clients start training from scratch without any
                global info. Defaults to False.
            task_check_period (float, optional): interval for checking status of tasks. Defaults to 0.5.
            persist_every_n_rounds (int, optional): persist the global model every n rounds. Defaults to 1.
                If n is 0 then no persist.

        Raises:
            TypeError: when any of input arguments does not have correct type
            ValueError: when any of input arguments is out of range
        """
        super().__init__(task_check_period=task_check_period)

        # Check arguments
        check_positive_int("persist_every_n_rounds", persist_every_n_rounds)
        check_str("aggregator_id", aggregator_id)
        check_str("persistor_id", persistor_id)
        check_str("assessor_id", assessor_id)
        check_str("shareable_generator_id", shareable_generator_id)
        check_str("task_name", task_name)
        check_positive_number("task_check_period", task_check_period)
        check_positive_number("assess_interval", assess_interval)
        check_positive_number("aggregation_interval", aggregation_interval)

        self.aggregator_id = aggregator_id
        self.persistor_id = persistor_id
        self.assessor_id = assessor_id
        self.shareable_generator_id = shareable_generator_id
        self.task_name = task_name
        self.aggregator = None
        self.persistor = None
        self.assessor = None
        self.shareable_gen = None

        # config data
        self._num_rounds = num_rounds
        self._persist_every_n_rounds = persist_every_n_rounds
        self.allow_empty_global_weights = allow_empty_global_weights
        self._aggr_interval = aggregation_interval
        self._assess_interval = assess_interval

        # workflow phases: init, train, validate
        self._phase = AppConstants.PHASE_INIT
        self._global_weights = make_model_learnable({}, {})
        self._current_round = None
        self._current_task_seq = 0
        self._num_children = 0
        self._children = None
        self._num_children_done = 0
        self._aggr_lock = threading.Lock()

    @classmethod
    def get_next_task_seq(cls):
        cls.next_task_seq += 1
        return cls.next_task_seq

    def start_controller(self, fl_ctx: FLContext) -> None:
        self.log_info(fl_ctx, "Initializing ScatterAndGatherForEdge workflow.")
        self._phase = AppConstants.PHASE_INIT

        engine = fl_ctx.get_engine()
        self.aggregator = engine.get_component(self.aggregator_id)
        if not isinstance(self.aggregator, Aggregator):
            self.system_panic(
                f"aggregator {self.aggregator_id} must be an Aggregator type object but got {type(self.aggregator)}",
                fl_ctx,
            )
            return

        if self.shareable_generator_id:
            self.shareable_gen = engine.get_component(self.shareable_generator_id)
            if not isinstance(self.shareable_gen, ShareableGenerator):
                self.system_panic(
                    f"Shareable generator {self.shareable_generator_id} must be a ShareableGenerator type object, "
                    f"but got {type(self.shareable_gen)}",
                    fl_ctx,
                )
                return
        else:
            self.shareable_gen = _DummyShareableGenerator()

        self.assessor = engine.get_component(self.assessor_id)
        if not isinstance(self.assessor, Assessor):
            self.system_panic(
                f"Assessor {self.assessor_id} must be an Assessor but got {type(self.assessor)}",
                fl_ctx,
            )
            return

        self.assessor.initialize(self.aggregator, fl_ctx)

        if self.persistor_id:
            self.persistor = engine.get_component(self.persistor_id)
            if not isinstance(self.persistor, LearnablePersistor):
                self.system_panic(
                    f"Model Persistor {self.persistor_id} must be a LearnablePersistor type object, "
                    f"but got {type(self.persistor)}",
                    fl_ctx,
                )
                return

        # initialize global model
        fl_ctx.set_prop(AppConstants.START_ROUND, 0, private=True, sticky=True)
        fl_ctx.set_prop(AppConstants.NUM_ROUNDS, self._num_rounds, private=True, sticky=False)
        if self.persistor:
            model = self.persistor.load(fl_ctx)

            if not isinstance(model, ModelLearnable):
                self.system_panic(
                    reason=f"Expected model loaded by persistor to be `ModelLearnable` but received {type(model)}",
                    fl_ctx=fl_ctx,
                )
                return

            if model.is_empty():
                if not self.allow_empty_global_weights:
                    # if empty not allowed, further check whether it is available from fl_ctx
                    model = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
                    if not isinstance(model, ModelLearnable):
                        self.system_panic(
                            reason=f"Expected model from fl-ctx to be `ModelLearnable` but received {type(model)}",
                            fl_ctx=fl_ctx,
                        )
                        return

            self._global_weights = model
            fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, model, private=True, sticky=True)
            self.fire_event(AppEventType.INITIAL_MODEL_LOADED, fl_ctx)

        # register aux message handler for receiving aggr results from children
        engine.register_aux_message_handler(message_topic_for_task(self.task_name), self._process_aggr_result)

        # get children clients
        client_hierarchy = fl_ctx.get_prop(FLContextKey.CLIENT_HIERARCHY)
        self._children = client_hierarchy.roots
        self._num_children = len(self._children)
        self.log_info(fl_ctx, f"my child clients: {self._children}")

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        try:
            self.log_info(fl_ctx, "Beginning ScatterAndGatherForEdge training phase.")
            self._phase = AppConstants.PHASE_TRAIN

            fl_ctx.set_prop(AppConstants.PHASE, self._phase, private=True, sticky=False)
            fl_ctx.set_prop(AppConstants.NUM_ROUNDS, self._num_rounds, private=True, sticky=False)
            self.fire_event(AppEventType.TRAINING_STARTED, fl_ctx)

            for r in range(self._num_rounds):
                self._current_round = r

                if self._check_abort_signal(fl_ctx, abort_signal):
                    break

                self.log_info(fl_ctx, f"Round {self._current_round} started.")
                self._current_task_seq = self.get_next_task_seq()
                fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, self._global_weights, private=True, sticky=True)
                fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self._current_round, private=True, sticky=True)
                self.fire_event(AppEventType.ROUND_STARTED, fl_ctx)

                # Create train_task
                task_data = self.shareable_gen.learnable_to_shareable(self._global_weights, fl_ctx)
                task_data.set_header(AppConstants.CURRENT_ROUND, self._current_round)
                task_data.set_header(AppConstants.NUM_ROUNDS, self._num_rounds)
                task_data.set_header(EdgeTaskHeaderKey.TASK_SEQ, self._current_task_seq)
                task_data.set_header(EdgeTaskHeaderKey.AGGR_INTERVAL, self._aggr_interval)
                task_data.add_cookie(AppConstants.CONTRIBUTION_ROUND, self._current_round)

                train_task = Task(
                    name=self.task_name,
                    data=task_data,
                    before_task_sent_cb=self._prepare_train_task_data,
                    result_received_cb=self._process_train_result,
                )

                self.broadcast(
                    task=train_task,
                    fl_ctx=fl_ctx,
                    targets=self._children,
                    min_responses=self._num_children,
                    wait_time_after_min_received=0,
                )

                # wait for the task to finish
                self.assessor.start(fl_ctx)
                assess_result = AssessResult.TASK_DONE
                seq = self._current_task_seq
                while True:
                    if self._num_children_done >= self._num_children:
                        # all children are done with their current task
                        self.log_info(fl_ctx, f"Task is done ({seq=}): all children are done with their task")
                        break

                    assess_result = self.assessor.assess(fl_ctx)
                    if assess_result != AssessResult.CONTINUE:
                        self.log_info(fl_ctx, f"Task is done ({seq=}): {assess_result=}")
                        break

                    wrc = conditional_wait(
                        waiter=None,
                        timeout=self._assess_interval,
                        abort_signal=abort_signal,
                    )
                    if wrc == WaiterRC.ABORTED:
                        self.log_info(fl_ctx, f"Task is done ({seq=}): ABORTED")
                        break

                self._current_task_seq = 0
                self._num_children_done = 0
                self.cancel_task(train_task, fl_ctx=fl_ctx)

                if self._check_abort_signal(fl_ctx, abort_signal):
                    break

                self.log_info(fl_ctx, f"Start aggregation for task seq {seq}")
                self.fire_event(AppEventType.BEFORE_AGGREGATION, fl_ctx)
                with self._aggr_lock:
                    try:
                        aggr_result = self.aggregator.aggregate(fl_ctx)
                        self.aggregator.reset(fl_ctx)
                        self.assessor.reset(fl_ctx)
                    except Exception as ex:
                        self.log_error(fl_ctx, f"aggregation error: {secure_format_exception(ex)}")
                        self.system_panic(f"aggregation error: {secure_format_exception(ex)}", fl_ctx)
                        return

                fl_ctx.set_prop(AppConstants.AGGREGATION_RESULT, aggr_result, private=True, sticky=False)
                self.fire_event(AppEventType.AFTER_AGGREGATION, fl_ctx)
                self.log_info(fl_ctx, f"End aggregation for task seq {seq}.")

                self.fire_event(AppEventType.BEFORE_SHAREABLE_TO_LEARNABLE, fl_ctx)
                self._global_weights = self.shareable_gen.shareable_to_learnable(aggr_result, fl_ctx)
                fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, self._global_weights, private=True, sticky=True)
                self.fire_event(AppEventType.AFTER_SHAREABLE_TO_LEARNABLE, fl_ctx)

                if self.persistor:
                    if (
                        self._persist_every_n_rounds != 0
                        and (self._current_round + 1) % self._persist_every_n_rounds == 0
                    ) or self._current_round == self._num_rounds - 1:
                        self.log_info(fl_ctx, "Start persist model on server.")
                        self.fire_event(AppEventType.BEFORE_LEARNABLE_PERSIST, fl_ctx)
                        self.persistor.save(self._global_weights, fl_ctx)
                        self.fire_event(AppEventType.AFTER_LEARNABLE_PERSIST, fl_ctx)
                        self.log_info(fl_ctx, "End persist model on server.")

                self.fire_event(AppEventType.ROUND_DONE, fl_ctx)
                self.log_info(fl_ctx, f"Round {self._current_round} finished.")
                gc.collect()

                if assess_result == AssessResult.WORKFLOW_DONE:
                    break

            self._phase = AppConstants.PHASE_FINISHED
            self._current_task_seq = 0
            self.log_info(fl_ctx, "Finished ScatterAndGatherForEdge Training.")

            # give some time for clients to end gracefully when sync task seq
            time.sleep(self._aggr_interval + 1.0)
        except Exception as e:
            error_msg = f"Exception in ScatterAndGather control_flow: {secure_format_exception(e)}"
            self.log_exception(fl_ctx, error_msg)
            self.system_panic(error_msg, fl_ctx)

    def stop_controller(self, fl_ctx: FLContext):
        self._phase = AppConstants.PHASE_FINISHED
        self.assessor.finalize(fl_ctx)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        super().handle_event(event_type, fl_ctx)
        if event_type == InfoCollector.EVENT_TYPE_GET_STATS:
            collector = fl_ctx.get_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR, None)
            if collector:
                if not isinstance(collector, GroupInfoCollector):
                    raise TypeError("collector must be GroupInfoCollector but got {}".format(type(collector)))

                collector.add_info(
                    group_name=self._name,
                    info={"phase": self._phase, "current_round": self._current_round, "num_rounds": self._num_rounds},
                )

    def _prepare_train_task_data(self, client_task: ClientTask, fl_ctx: FLContext) -> None:
        fl_ctx.set_prop(AppConstants.TRAIN_SHAREABLE, client_task.task.data, private=True, sticky=False)
        self.fire_event(AppEventType.BEFORE_TRAIN_TASK, fl_ctx)

    def _process_train_result(self, client_task: ClientTask, fl_ctx: FLContext) -> None:
        self._num_children_done += 1
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

        has_aggr_data = result.get_header(EdgeTaskHeaderKey.HAS_AGGR_DATA, False)
        if has_aggr_data:
            accepted = self._accept_result(result, fl_ctx)
            self.log_info(fl_ctx, f"processed aggr data from task submission: {accepted=}")

    def process_result_of_unknown_task(
        self, client: Client, task_name, client_task_id, result: Shareable, fl_ctx: FLContext
    ) -> None:
        self.log_error(fl_ctx, f"Ignoring result from {client.name} for unknown task '{task_name}' {client_task_id}")

    def _process_aggr_result(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        accepted, reply = process_aggr_result_from_child(
            processor=self,
            request=request,
            accept_f=self._accept_result,
            current_task_seq=self._current_task_seq,
            fl_ctx=fl_ctx,
        )
        self.log_info(fl_ctx, f"processed aggr data from result report: {accepted=}")
        return reply

    def _check_abort_signal(self, fl_ctx, abort_signal: Signal):
        if abort_signal.triggered:
            self._phase = AppConstants.PHASE_FINISHED
            self.log_info(fl_ctx, f"Abort signal received. Exiting at round {self._current_round}.")
            return True
        return False

    def _accept_result(self, result: Shareable, fl_ctx: FLContext) -> bool:
        self.log_info(fl_ctx, "trying to accept result ...")
        with self._aggr_lock:
            try:
                accepted = self.aggregator.accept(result, fl_ctx)
            except Exception as ex:
                self.log_error(fl_ctx, f"exception accepting result: {secure_format_exception(ex)}")
                accepted = False

        self.log_info(fl_ctx, f"done trying to accept result: {accepted=}")

        peer_ctx = fl_ctx.get_peer_context()
        assert isinstance(peer_ctx, FLContext)
        client_name = peer_ctx.get_identity_name()

        accepted_msg = "ACCEPTED" if accepted else "REJECTED"
        self.log_info(
            fl_ctx, f"Contribution from {client_name} {accepted_msg} by the aggregator at round {self._current_round}."
        )
        fl_ctx.set_prop(AppConstants.AGGREGATION_ACCEPTED, accepted, private=True, sticky=False)
        self.fire_event(AppEventType.AFTER_CONTRIBUTION_ACCEPT, fl_ctx)
        return accepted
