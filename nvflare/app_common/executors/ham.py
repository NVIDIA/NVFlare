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
import threading
import time

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.app_constant import AppConstants
from nvflare.fuel.utils.validation_utils import check_non_negative_int, check_positive_number, check_str
from nvflare.security.logging import secure_format_exception


class HierarchicalAggregationManager(Executor):
    def __init__(
        self,
        learner_id: str,
        aggregator_id: str,
        aggr_timeout: float,
        min_responses: int,
        wait_time_after_min_resps_received: float,
    ):
        Executor.__init__(self)

        check_str("learner_id", learner_id)
        self.learner_id = learner_id

        check_str("aggregator_id", aggregator_id)
        check_positive_number("aggr_timeout", aggr_timeout)
        check_non_negative_int("min_responses", min_responses)
        check_positive_number("wait_time_after_min_resps_received", wait_time_after_min_resps_received)

        self.aggregator_id = aggregator_id
        self.aggr_timeout = aggr_timeout
        self.pending_task_id = None
        self.current_round = None
        self.pending_clients = {}
        self.aggregator = None
        self.learner = None
        self.min_responses = min_responses
        self.wait_time_after_min_resps_received = wait_time_after_min_resps_received
        self._status_lock = threading.Lock()
        self._aggr_lock = threading.Lock()
        self._process_error = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            engine = fl_ctx.get_engine()
            aggr = engine.get_component(self.aggregator_id)
            if not isinstance(aggr, Aggregator):
                self.log_error(fl_ctx, f"component '{self.aggregator_id}' must be Aggregator but got {type(aggr)}")
            self.aggregator = aggr

            learner = engine.get_component(self.learner_id)
            if not isinstance(learner, Executor):
                self.log_error(fl_ctx, f"component '{self.learner_id}' must be Executor but got {type(learner)}")
            self.learner = learner

        elif event_type == EventType.TASK_ASSIGNMENT_SENT:
            # sent the task to a child client
            child_client_ctx = fl_ctx.get_peer_context()
            assert isinstance(child_client_ctx, FLContext)
            child_client_name = child_client_ctx.get_identity_name()
            self._update_client_status(child_client_name, None)
            task_id = fl_ctx.get_prop(FLContextKey.TASK_ID)
            self.log_info(fl_ctx, f"sent task {task_id} to child {child_client_name}")
        elif event_type == EventType.TASK_RESULT_RECEIVED:
            # received results from a child client
            result = fl_ctx.get_prop(FLContextKey.TASK_RESULT)
            assert isinstance(result, Shareable)
            task_id = result.get_header(ReservedKey.TASK_ID)
            peer_ctx = fl_ctx.get_peer_context()
            assert isinstance(peer_ctx, FLContext)
            child_client_name = peer_ctx.get_identity_name()
            self.log_info(fl_ctx, f"received result for task {task_id} from child {child_client_name}")

            if task_id != self.pending_task_id:
                self.log_warning(
                    fl_ctx,
                    f"dropped the received result from child {child_client_name} "
                    f"for task {task_id} while waiting for task {self.pending_task_id}",
                )
                return

            rc = result.get_return_code(ReturnCode.OK)
            if rc == ReturnCode.OK:
                self.log_info(fl_ctx, f"accepting result from client {child_client_name}")
                self._do_aggregation(result, fl_ctx)
            else:
                self.log_error(fl_ctx, f"Received bad result from client {child_client_name}: {rc=}")
            self.log_info(fl_ctx, f"received result from child {child_client_name}")
            self._update_client_status(child_client_name, time.time())

    def _do_aggregation(self, result: Shareable, fl_ctx: FLContext):
        with self._aggr_lock:
            try:
                # some aggregators expect current_round to be in the fl_ctx!
                fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self.current_round, private=True, sticky=False)
                self.aggregator.accept(result, fl_ctx)
            except Exception as ex:
                self.log_error(
                    fl_ctx,
                    f"exception when 'accept' from aggregator {type(self.aggregator)}: {secure_format_exception(ex)}",
                )
                self._process_error = True

    def _pending_clients_status(self):
        with self._status_lock:
            if not self.pending_clients:
                return 0, 0

            received = 0
            for received_time in self.pending_clients.values():
                if received_time:
                    received += 1

            return received, len(self.pending_clients)

    def _update_client_status(self, client_name, status):
        with self._status_lock:
            self.pending_clients[client_name] = status

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        is_leaf = fl_ctx.get_prop(ReservedKey.IS_LEAF)
        if is_leaf:
            self.log_info(fl_ctx, f"I'm leaf - invoking {type(self.learner)}")
            return self.learner.execute(task_name, shareable, fl_ctx, abort_signal)

        self.log_info(fl_ctx, "waiting for results from children ...")
        self.current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        self.log_info(fl_ctx, f"got current_round: {self.current_round}")
        self.pending_task_id = shareable.get_header(ReservedKey.TASK_ID)

        result = self._do_execute(fl_ctx, abort_signal)

        self.pending_task_id = None
        self.pending_fl_ctx = None
        self.pending_clients = {}
        self.aggregator.reset(fl_ctx)
        self._process_error = False
        return result

    def _do_execute(self, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        start_time = time.time()
        min_received_time = None
        while True:
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            if self._process_error:
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            current_time = time.time()
            if current_time - start_time > self.aggr_timeout:
                break

            # have we received all results?
            received, total = self._pending_clients_status()
            if received < self.min_responses:
                # we have not received min responses
                continue

            if not min_received_time:
                min_received_time = current_time

            if current_time - min_received_time >= self.wait_time_after_min_resps_received:
                # we have waited long enough after min resps received
                break

            time.sleep(0.5)

        # return aggregation result
        received, total = self._pending_clients_status()
        self.log_info(fl_ctx, f"process done after {time.time() - start_time} secs: {received=} {total=}")

        if self._process_error:
            self.log_error(fl_ctx, "there is process error")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        if received == 0:
            # nothing received!
            self.log_info(fl_ctx, "nothing received - timeout")
            return make_reply(ReturnCode.TIMEOUT)

        try:
            self.log_info(fl_ctx, "return aggregated result!")
            return self.aggregator.aggregate(fl_ctx)
        except Exception as ex:
            self.log_error(fl_ctx, f"exception 'aggregate' from {type(self.aggregator)}: {secure_format_exception(ex)}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
