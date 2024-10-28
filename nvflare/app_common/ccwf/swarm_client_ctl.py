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
import copy
import random
import threading
import time

from nvflare.apis.controller_spec import Task
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.abstract.metric_comparator import MetricComparator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.ccwf.client_ctl import ClientSideController
from nvflare.app_common.ccwf.common import Constant, NumberMetricComparator, ResultType, make_task_name
from nvflare.fuel.utils.validation_utils import check_non_empty_str, check_positive_int, check_positive_number
from nvflare.security.logging import secure_format_traceback


class _TrainerStatus:
    def __init__(self, name: str):
        self.name = name
        self.reply_time = None


class Gatherer(FLComponent):
    def __init__(
        self,
        task_data: Shareable,
        fl_ctx: FLContext,
        for_round: int,
        executor: ClientSideController,
        aggregator: Aggregator,
        metric_comparator: MetricComparator,
        all_clients: list,
        trainers: list,
        min_responses_required: int,
        wait_time_after_min_resps_received: float,
        timeout,
    ):
        FLComponent.__init__(self)
        self.fl_ctx = fl_ctx
        self.executor = executor
        self.aggregator = aggregator
        self.metric_comparator = metric_comparator
        self.all_clients = all_clients
        self.trainers = trainers
        self.for_round = for_round
        self.trainer_statuses = {}
        self.start_time = time.time()
        self.timeout = timeout

        for t in trainers:
            self.trainer_statuses[t] = _TrainerStatus(t)
        if min_responses_required <= 0 or min_responses_required >= len(trainers):
            min_responses_required = len(trainers)
        self.min_responses_required = min_responses_required
        self.wait_time_after_min_resps_received = wait_time_after_min_resps_received
        self.min_resps_received_time = None
        self.lock = threading.Lock()
        self.current_best_client = task_data.get_header(Constant.CLIENT)
        self.current_best_global_metric = task_data.get_header(Constant.METRIC)
        self.current_best_round = task_data.get_header(Constant.ROUND)
        if not self.current_best_client:
            self.log_info(fl_ctx, "gatherer starting from scratch")
        else:
            self.log_info(
                fl_ctx,
                f"gatherer starting with previous best result from client {self.current_best_client} "
                f"with metric {self.current_best_global_metric} "
                f"at round {self.current_best_round}",
            )

    def gather(self, client_name: str, result: Shareable, fl_ctx: FLContext) -> Shareable:
        with self.lock:
            try:
                return self._do_gather(client_name, result, fl_ctx)
            except:
                self.log_error(fl_ctx, f"exception gathering: {secure_format_traceback()}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _do_gather(self, client_name: str, result: Shareable, fl_ctx: FLContext) -> Shareable:
        result_round = result.get_header(AppConstants.CURRENT_ROUND)
        ts = self.trainer_statuses.get(client_name)
        if not ts:
            self.log_error(
                fl_ctx, f"received result from {client_name} for round {result_round}, but it is not a trainer"
            )
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        if result_round > self.for_round:
            # this should never happen!
            # otherwise it means that the client is sending me result for a round that I couldn't possibly schedule!
            self.log_error(
                fl_ctx,
                f"logic error: received result from {client_name} for round {result_round}, "
                f"which is > gatherer's current round {self.for_round}",
            )
            self.executor.update_status(action="gather", error=ReturnCode.EXECUTION_EXCEPTION)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        if result_round < self.for_round:
            # this is a late result for a round that I scheduled in the past.
            # Note: we still accept it!
            self.log_warning(
                fl_ctx,
                f"received late result from {client_name} for round {result_round}, "
                f"which is < gatherer's current round {self.for_round}",
            )

        if result_round == self.for_round:
            # this is the result that I'm waiting for.
            now = time.time()
            ts.reply_time = now
            if not self.min_resps_received_time:
                # see how many responses I have received
                num_resps_received = 0
                for _, ts in self.trainer_statuses.items():
                    if ts.reply_time:
                        num_resps_received += 1
                if num_resps_received >= self.min_responses_required:
                    self.min_resps_received_time = now

        rc = result.get_return_code(ReturnCode.OK)
        if rc != ReturnCode.OK:
            self.log_error(fl_ctx, f"Bad result from {client_name} for round {result_round}: {rc}.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self.for_round, private=True, sticky=True)
        fl_ctx.set_prop(AppConstants.TRAINING_RESULT, result, private=True, sticky=False)
        self.fire_event(AppEventType.BEFORE_CONTRIBUTION_ACCEPT, fl_ctx)

        accepted = self.aggregator.accept(result, fl_ctx)
        accepted_msg = "ACCEPTED" if accepted else "REJECTED"
        self.log_info(
            fl_ctx, f"Contribution from {client_name} {accepted_msg} by the aggregator at round {result_round}."
        )

        fl_ctx.set_prop(AppConstants.AGGREGATION_ACCEPTED, accepted, private=True, sticky=False)
        self.fire_event(AppEventType.AFTER_CONTRIBUTION_ACCEPT, fl_ctx)
        return make_reply(ReturnCode.OK)

    def aggregate(self):
        fl_ctx = self.fl_ctx
        self.log_info(fl_ctx, f"Start aggregation for round {self.for_round}")
        self.fire_event(AppEventType.BEFORE_AGGREGATION, fl_ctx)
        aggr_result = self.aggregator.aggregate(fl_ctx)
        fl_ctx.set_prop(AppConstants.AGGREGATION_RESULT, aggr_result, private=True, sticky=False)
        self.fire_event(AppEventType.AFTER_AGGREGATION, fl_ctx)
        self.log_info(fl_ctx, f"Finished aggregation for round {self.for_round}")

        mine_is_better = False
        if self.current_best_global_metric is not None:
            if (
                self.executor.best_metric is not None
                and self.metric_comparator.compare(self.executor.best_metric, self.current_best_global_metric) > 0
            ):
                mine_is_better = True
        elif self.executor.best_metric is not None:
            mine_is_better = True

        if mine_is_better:
            self.log_info(
                fl_ctx, f"I got better metric {self.executor.best_metric} at round {self.executor.best_round}"
            )
            best_round = self.executor.best_round
            best_metric = self.executor.best_metric
            best_client = self.executor.me
        else:
            best_round = self.current_best_round
            best_metric = self.current_best_global_metric
            best_client = self.current_best_client

        self.log_info(fl_ctx, f"global best metric is {best_metric} from client {best_client} at round {best_round}")

        aggr_result.set_header(Constant.ROUND, best_round)
        aggr_result.set_header(Constant.METRIC, best_metric)
        aggr_result.set_header(Constant.CLIENT, best_client)

        return aggr_result

    def is_done(self):
        unfinished = 0
        for c, s in self.trainer_statuses.items():
            if not s.reply_time:
                unfinished += 1
        if unfinished == 0:
            return True

        # timeout?
        now = time.time()
        if self.timeout and now - self.start_time > self.timeout:
            self.log_warning(self.fl_ctx, f"gatherer for round {self.for_round} timed out after {self.timeout} seconds")
            return True

        if (
            self.min_resps_received_time
            and now - self.min_resps_received_time > self.wait_time_after_min_resps_received
        ):
            # received min responses required and waited for long time
            self.log_info(
                self.fl_ctx,
                f"gatherer for round {self.for_round} exit after {self.wait_time_after_min_resps_received} seconds "
                f"since received minimum responses",
            )
            return True


class SwarmClientController(ClientSideController):
    def __init__(
        self,
        task_name_prefix=Constant.TN_PREFIX_SWARM,
        learn_task_name=AppConstants.TASK_TRAIN,
        persistor_id=AppConstants.DEFAULT_PERSISTOR_ID,
        shareable_generator_id=AppConstants.DEFAULT_SHAREABLE_GENERATOR_ID,
        aggregator_id=AppConstants.DEFAULT_AGGREGATOR_ID,
        metric_comparator_id=None,
        learn_task_check_interval=Constant.LEARN_TASK_CHECK_INTERVAL,
        learn_task_abort_timeout=Constant.LEARN_TASK_ABORT_TIMEOUT,
        learn_task_ack_timeout=Constant.LEARN_TASK_ACK_TIMEOUT,
        learn_task_timeout=None,
        final_result_ack_timeout=Constant.FINAL_RESULT_ACK_TIMEOUT,
        min_responses_required: int = 1,
        wait_time_after_min_resps_received: float = 10.0,
    ):
        check_non_empty_str("learn_task_name", learn_task_name)
        check_non_empty_str("persistor_id", persistor_id)
        check_non_empty_str("shareable_generator_id", shareable_generator_id)
        check_non_empty_str("aggregator_id", aggregator_id)

        if metric_comparator_id:
            check_non_empty_str("metric_comparator_id", metric_comparator_id)

        if learn_task_timeout:
            check_positive_number("learn_task_timeout", learn_task_timeout)

        check_positive_int("min_responses_required", min_responses_required)
        check_positive_number("wait_time_after_min_resps_received", wait_time_after_min_resps_received)

        super().__init__(
            task_name_prefix=task_name_prefix,
            learn_task_name=learn_task_name,
            persistor_id=persistor_id,
            shareable_generator_id=shareable_generator_id,
            learn_task_check_interval=learn_task_check_interval,
            learn_task_ack_timeout=learn_task_ack_timeout,
            learn_task_abort_timeout=learn_task_abort_timeout,
            final_result_ack_timeout=final_result_ack_timeout,
            allow_busy_task=True,
        )
        self.metric_comparator_id = metric_comparator_id
        self.metric_comparator = None
        self.report_learn_result_task_name = make_task_name(task_name_prefix, Constant.BASENAME_REPORT_LEARN_RESULT)
        self.learn_task_timeout = learn_task_timeout
        self.min_responses_required = min_responses_required
        self.wait_time_after_min_resps_received = wait_time_after_min_resps_received
        self.aggregator_id = aggregator_id
        self.aggregator = None
        self.gatherer = None
        self.gatherer_waiter = threading.Event()
        self.trainers = None
        self.aggrs = None
        self.is_trainer = False
        self.is_aggr = False
        self.last_aggr_round_done = -1

    def process_config(self, fl_ctx: FLContext):
        all_clients = self.get_config_prop(Constant.CLIENTS)

        self.trainers = self.get_config_prop(Constant.TRAIN_CLIENTS)
        if not self.trainers:
            self.trainers = all_clients
        self.is_trainer = self.me in self.trainers

        self.aggrs = self.get_config_prop(Constant.AGGR_CLIENTS)
        if not self.aggrs:
            self.aggrs = all_clients
        self.is_aggr = self.me in self.aggrs

        self.engine.register_aux_message_handler(
            topic=self.topic_for_my_workflow(Constant.TOPIC_SHARE_RESULT),
            message_handle_func=self._process_share_result,
        )

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name == self.report_learn_result_task_name:
            return self._process_learn_result(shareable, fl_ctx, abort_signal)
        return super().execute(task_name, shareable, fl_ctx, abort_signal)

    def start_run(self, fl_ctx: FLContext):
        super().start_run(fl_ctx)
        self.aggregator = self.engine.get_component(self.aggregator_id)
        if not isinstance(self.aggregator, Aggregator):
            self.system_panic(
                f"aggregator {self.aggregator_id} must be an Aggregator but got {type(self.aggregator)}",
                fl_ctx,
            )
            return

        if self.metric_comparator_id:
            self.metric_comparator = self.engine.get_component(self.metric_comparator_id)
            if not isinstance(self.metric_comparator, MetricComparator):
                self.system_panic(
                    f"metric comparator {self.metric_comparator_id} must be a MetricComparator "
                    f"but got {type(self.metric_comparator)}",
                    fl_ctx,
                )
                return
        else:
            # use default comparator
            self.metric_comparator = NumberMetricComparator()

        aggr_thread = threading.Thread(target=self._monitor_gather)
        aggr_thread.daemon = True
        aggr_thread.start()
        self.log_info(fl_ctx, "started aggregator thread")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == AppEventType.GLOBAL_BEST_MODEL_AVAILABLE:
            client = fl_ctx.get_prop(Constant.CLIENT)
            if client and client != self.me:
                # this global best model is from other client
                # we got here because this event is fired when I receive the best model shared from another
                # client at the end of the workflow.
                return

            # we got here because the best model selector fired this event: it found the "local best global"
            self.best_metric = fl_ctx.get_prop(AppConstants.VALIDATION_RESULT)
            self.best_result = copy.deepcopy(fl_ctx.get_prop(AppConstants.GLOBAL_MODEL))
            self.log_info(fl_ctx, f"got GLOBAL_BEST_MODEL_AVAILABLE: best metric={self.best_metric}")
            current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
            self.best_round = current_round
            self.update_status(last_round=current_round, action="better_aggregation")
        else:
            super().handle_event(event_type, fl_ctx)

    def start_workflow(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        clients = self.get_config_prop(Constant.CLIENTS)
        aggr_clients = self.get_config_prop(Constant.AGGR_CLIENTS, [])
        train_clients = self.get_config_prop(Constant.TRAIN_CLIENTS, [])

        self.log_info(
            fl_ctx, f"Starting Swarm Workflow on clients {clients}, aggrs {aggr_clients}, trainers {train_clients}"
        )

        if not self._scatter(
            task_data=shareable, for_round=self.get_config_prop(Constant.START_ROUND, 0), fl_ctx=fl_ctx
        ):
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        self.log_info(fl_ctx, "Started Swarm Workflow")
        return make_reply(ReturnCode.OK)

    def _scatter(self, task_data: Shareable, for_round: int, fl_ctx: FLContext) -> bool:
        clients = self.get_config_prop(Constant.TRAIN_CLIENTS)
        aggr_clients = self.get_config_prop(Constant.AGGR_CLIENTS)

        # determine aggr client
        aggr = random.choice(aggr_clients)

        task_data.set_header(AppConstants.CURRENT_ROUND, for_round)
        task_data.add_cookie(AppConstants.CONTRIBUTION_ROUND, for_round)
        task_data.set_header(Constant.AGGREGATOR, aggr)

        targets = copy.copy(clients)
        if aggr not in targets:
            targets.append(aggr)

        self.log_info(fl_ctx, f"broadcasting learn task of round {for_round} to {targets}; aggr client is {aggr}")
        return self.send_learn_task(targets=targets, request=task_data, fl_ctx=fl_ctx)

    def _monitor_gather(self):
        while True:
            if self.asked_to_stop:
                return

            gatherer = self.gatherer
            if gatherer:
                assert isinstance(gatherer, Gatherer)
                if gatherer.is_done():
                    self.last_aggr_round_done = gatherer.for_round
                    self.gatherer = None
                    self.gatherer_waiter.clear()
                    try:
                        self._end_gather(gatherer)
                    except:
                        self.logger.error(f"exception ending gatherer: {secure_format_traceback()}")
                        self.update_status(action="aggregate", error=ReturnCode.EXECUTION_EXCEPTION)
            time.sleep(0.2)

    def _end_gather(self, gatherer: Gatherer):
        fl_ctx = gatherer.fl_ctx
        try:
            aggr_result = gatherer.aggregate()
        except:
            self.log_error(fl_ctx, f"exception in aggregation: {secure_format_traceback()}")
            self.update_status(action="aggregate", error=ReturnCode.EXECUTION_EXCEPTION)
            return

        # aggr_result could be just weight diffs, not full weights!
        # need to call shareable_to_learnable to get full weights.
        self.log_debug(fl_ctx, f"aggr result: {aggr_result}")
        global_weights = self.shareable_generator.shareable_to_learnable(aggr_result, fl_ctx)
        self.record_last_result(fl_ctx, gatherer.for_round, global_weights)

        # are we done with training?
        num_rounds_done = gatherer.for_round - self.get_config_prop(Constant.START_ROUND, 0) + 1
        if num_rounds_done >= self.get_config_prop(AppConstants.NUM_ROUNDS):
            self.log_info(fl_ctx, f"Swarm Learning Done: number of rounds completed {num_rounds_done}")

            # determine the best global result
            self._distribute_final_results(aggr_result, fl_ctx)
            return

        # continue next round
        next_round_data = self.shareable_generator.learnable_to_shareable(global_weights, fl_ctx)
        assert isinstance(next_round_data, Shareable)

        best_round = aggr_result.get_header(Constant.ROUND)
        best_metric = aggr_result.get_header(Constant.METRIC)
        best_client = aggr_result.get_header(Constant.CLIENT)

        if best_client:
            next_round_data.set_header(Constant.ROUND, best_round)
            next_round_data.set_header(Constant.CLIENT, best_client)
            next_round_data.set_header(Constant.METRIC, best_metric)

        self._scatter(next_round_data, gatherer.for_round + 1, gatherer.fl_ctx)

    def _ask_to_share_best_result(self, client: str, metric, fl_ctx: FLContext):
        # other client has best model - ask it to distribute its result
        self.log_info(fl_ctx, f"client {client} has the best metric {metric} - ask it to share result")
        resp = self.engine.send_aux_request(
            targets=[client],
            topic=self.topic_for_my_workflow(Constant.TOPIC_SHARE_RESULT),
            request=Shareable(),
            timeout=self.final_result_ack_timeout,
            fl_ctx=fl_ctx,
            secure=False,
        )

        assert isinstance(resp, dict)
        reply = resp.get(client)
        if not reply:
            self.log_error(fl_ctx, f"failed to ask client {client} to share final result")
            return

        if not isinstance(reply, Shareable):
            self.log_error(fl_ctx, f"client {client} failed to respond to share final result request")
            return

        rc = reply.get_return_code()
        if rc != ReturnCode.OK:
            self.log_error(fl_ctx, f"client {client} failed to respond to share final result request: {rc}")

    def _distribute_final_results(self, aggr_result: Shareable, fl_ctx: FLContext):
        best_client = aggr_result.get_header(Constant.CLIENT)
        best_metric = aggr_result.get_header(Constant.METRIC)

        if best_client:
            if best_client == self.me:
                # I have the best model
                self.log_info(fl_ctx, f"I have global best metric {best_metric}")
                self.broadcast_final_result(
                    fl_ctx, ResultType.BEST, self.best_result, self.best_metric, self.best_round
                )
            else:
                try:
                    self._ask_to_share_best_result(best_client, best_metric, fl_ctx)
                except:
                    self.log_error(
                        fl_ctx, f"error asking client {best_client} to share best result {secure_format_traceback()}"
                    )
        else:
            self.log_info(fl_ctx, "No global best result!")

        self.log_info(fl_ctx, "distributing last result")
        self.broadcast_final_result(fl_ctx, ResultType.LAST, self.last_result, round_num=self.last_round)

    def _process_learn_result(self, request: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        try:
            peer_ctx = fl_ctx.get_peer_context()
            assert isinstance(peer_ctx, FLContext)
            client_name = peer_ctx.get_identity_name()
            current_round = request.get_header(AppConstants.CURRENT_ROUND)
            self.log_info(fl_ctx, f"got training result from {client_name} for round {current_round}")

            # to be compatible with some widgets that rely on peer_ctx to get result
            peer_ctx.set_prop(FLContextKey.SHAREABLE, request)

            gatherer = self.gatherer
            if not gatherer:
                # this could be from a fast client before I even create the waiter;
                # or from a late client after I already finished gathering.
                if current_round <= self.last_aggr_round_done:
                    # late client case - drop the result
                    self.log_info(fl_ctx, f"dropped result from late {client_name} for round {current_round}")
                    return make_reply(ReturnCode.OK)

                # case of fast client
                # wait until the gatherer is set up.
                self.log_info(fl_ctx, f"got result from {client_name} for round {current_round} before gatherer setup")
                self.gatherer_waiter.wait(self.learn_task_abort_timeout)

            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            gatherer = self.gatherer
            if not gatherer:
                self.log_error(fl_ctx, f"Still no gatherer after {self.learn_task_abort_timeout} seconds")
                self.log_error(fl_ctx, f"Ignored result from {client_name} for round {current_round} since no gatherer")
                self.update_status(action="wait_for_gatherer", error=ReturnCode.EXECUTION_EXCEPTION)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            assert isinstance(gatherer, Gatherer)
            if gatherer.for_round != current_round:
                self.log_warning(
                    fl_ctx,
                    f"Got result from {client_name} for round {current_round}, "
                    f"but I'm waiting for round {gatherer.for_round}",
                )
            return gatherer.gather(client_name, request, fl_ctx)
        except:
            self.log_exception(fl_ctx, f"exception processing learn result: {secure_format_traceback()}")
            self.update_status(action="process_learn_result", error=ReturnCode.EXECUTION_EXCEPTION)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def do_learn_task(self, name: str, task_data: Shareable, fl_ctx: FLContext, abort_signal: Signal):
        # set status report of starting task
        current_round = task_data.get_header(AppConstants.CURRENT_ROUND)
        self.update_status(last_round=current_round, action="start_learn_task")

        aggr = task_data.get_header(Constant.AGGREGATOR)
        if not aggr:
            self.log_error(fl_ctx, f"missing aggregation client for round {current_round}")
            self.update_status(action="do_learn_task", error=ReturnCode.EXECUTION_EXCEPTION)
            return

        self.log_info(fl_ctx, f"Round {current_round} started.")

        task_data.set_header(FLContextKey.TASK_NAME, name)

        # Some shareable generators assume the base model (GLOBAL_MODEL) is always available, which is true for
        # server-controlled fed-avg. But this is not true for swarm learning.
        # To make these generators happy, we create an empty global model here if not present.
        base_model = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
        if not base_model:
            base_model = Learnable()
            fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, base_model, private=True, sticky=True)
        global_weights = self.shareable_generator.shareable_to_learnable(task_data, fl_ctx)

        self.log_debug(fl_ctx, f"current global model: {global_weights}")

        fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, global_weights, private=True, sticky=True)
        fl_ctx.set_prop(AppConstants.CURRENT_ROUND, current_round, private=True, sticky=True)
        self.fire_event(AppEventType.ROUND_STARTED, fl_ctx)

        if self.me == aggr:
            # set up the aggr waiter
            gatherer = self.gatherer
            if gatherer:
                # already waiting for aggregation - should never happen
                self.log_error(
                    fl_ctx,
                    f"logic error: got task for round {current_round} while gathering for round {gatherer.for_round}",
                )
                self.update_status(action="do_learn_task", error=ReturnCode.EXECUTION_EXCEPTION)
                return

            self.log_info(fl_ctx, f"setting up the gatherer for round {current_round}")

            self.gatherer = Gatherer(
                fl_ctx=fl_ctx,
                all_clients=self.get_config_prop(Constant.CLIENTS),
                metric_comparator=self.metric_comparator,
                trainers=self.trainers,
                for_round=current_round,
                timeout=self.learn_task_timeout,
                min_responses_required=self.min_responses_required,
                wait_time_after_min_resps_received=self.wait_time_after_min_resps_received,
                aggregator=self.aggregator,
                executor=self,
                task_data=task_data,
            )
            self.gatherer_waiter.set()

        # execute the task
        if self.is_trainer:
            # update status
            result = self.execute_learn_task(task_data, fl_ctx, abort_signal)

            rc = result.get_return_code(ReturnCode.OK)
            if rc != ReturnCode.OK:
                self.log_error(fl_ctx, f"learn executor failed: {rc}")
                self.update_status(action="learner_execution", error=rc)
                return

            # send the result to the aggr
            self.log_info(fl_ctx, f"sending training result to aggregation client {aggr}")

            task = Task(
                name=self.report_learn_result_task_name,
                data=result,
                timeout=int(self.learn_task_ack_timeout),
                secure=self.is_task_secure(fl_ctx),
            )

            resp = self.broadcast_and_wait(
                task=task,
                targets=[aggr],
                min_responses=1,
                fl_ctx=fl_ctx,
            )

            reply = resp.get(aggr)
            if not reply:
                self.log_error(fl_ctx, f"failed to receive reply from aggregation client: {aggr}")
                self.update_status(action="receive_learn_result_reply", error=ReturnCode.EXECUTION_EXCEPTION)
                return

            if not isinstance(reply, Shareable):
                self.log_error(
                    fl_ctx, f"bad reply from aggregation client {aggr}: expect Shareable but got {type(reply)}"
                )
                self.update_status(action="receive_learn_result_reply", error=ReturnCode.EXECUTION_EXCEPTION)
                return

            rc = reply.get_return_code(ReturnCode.OK)
            if rc != ReturnCode.OK:
                self.log_error(fl_ctx, f"bad return code from aggregation client {aggr}: {rc}")
                self.update_status(action="receive_learn_result_reply", error=ReturnCode.EXECUTION_EXCEPTION)
                return

            self.log_info(fl_ctx, f"Finished round {current_round}")

            # update status
            self.update_status(last_round=current_round, action="finished_learn_task")

    def _process_share_result(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        peer_ctx = fl_ctx.get_peer_context()
        assert isinstance(peer_ctx, FLContext)
        client_name = peer_ctx.get_identity_name()
        if not self.best_result:
            self.log_error(
                fl_ctx, f"got request from {client_name} to share my best result, but I don't have best result"
            )
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        self.update_status(action="start_share_result_request_process")
        self.broadcast_final_result(
            fl_ctx, ResultType.BEST, self.best_result, metric=self.best_metric, round_num=self.best_round
        )
        return make_reply(ReturnCode.OK)
