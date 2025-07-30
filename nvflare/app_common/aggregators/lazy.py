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
import queue
import threading

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.fuel.utils.validation_utils import check_positive_number, check_str
from nvflare.fuel.utils.waiter_utils import WaiterRC, conditional_wait
from nvflare.security.logging import secure_format_exception

_SHORT_WAIT = 0.1


class _AcceptWaitRC(WaiterRC):
    END_RUN = 10


class _Contribution:

    def __init__(self, data: Shareable, fl_ctx: FLContext):
        self.data = data

        # We need to make a copy of the fl_ctx since it could be used for multiple contributions and its
        # content could be overwritten.
        self.fl_ctx = fl_ctx.clone()


class LazyAggregator(Aggregator):

    def __init__(self, aggregator_id: str, accept_timeout: float = 600.0):
        """Constructor of LazyAggregator.
        LazyAggregator is a wrapper for other aggregators to do accept processing in a separate thread.

        During a typical SAG-based training, updates from clients are processed by the aggregator's "accept" method.
        To ensure the integrity of training task, The SAG workflow processes client updates sequentially.
        If the "accept" method is time-consuming and there are many clients, then the update processing will
        become bottleneck.

        Using the LazyAggregator, the updates from clients are simply added to a queue quickly.
        The actual "accept" processing is done in a separate thread that processes the queued updates sequentially.

        Args:
            aggregator_id: component ID of the real aggregator
            accept_timeout: max amount of time to wait for accept to finish
        """
        Aggregator.__init__(self)
        check_str("aggregator_id", aggregator_id)
        check_positive_number("accept_timeout", accept_timeout)

        self.aggregator_id = aggregator_id
        self.accept_timeout = accept_timeout
        self.aggregator = None
        self.contributions = queue.Queue()
        self._q_lock = threading.Lock()
        self.aggregating = False
        self.run_ended = False
        self.accept_done = threading.Event()
        self.register_event_handler(EventType.START_RUN, self._lazy_aggr_start_run)
        self.register_event_handler(EventType.END_RUN, self._lazy_aggr_end_run)

    def _clear_contributions(self):
        with self._q_lock:
            q = self.contributions
            while True:
                try:
                    q.get(block=False)  # Attempt to get an item without blocking
                    q.task_done()  # Mark the task as done (important for JoinableQueue)
                except queue.Empty:
                    break  # Break the loop when the queue is empty

    def _add_contribution(self, contrib: Shareable, fl_ctx: FLContext):
        self.log_debug(fl_ctx, "adding contribution to queue")
        with self._q_lock:
            self.contributions.put(_Contribution(contrib, fl_ctx))
        self.log_debug(fl_ctx, "done adding contribution to queue")

    def _lazy_aggr_start_run(self, event_type: str, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        aggr = engine.get_component(self.aggregator_id)
        if not isinstance(aggr, Aggregator):
            self.system_panic(f"component {self.aggregator_id} must be Aggregator but got {type(aggr)}", fl_ctx)
            return

        if isinstance(aggr, LazyAggregator):
            self.system_panic(f"component {self.aggregator_id} must not be LazyAggregator", fl_ctx)
            return

        self.aggregator = aggr
        accept_thread = threading.Thread(target=self._do_accept, daemon=True)
        accept_thread.start()

    def _lazy_aggr_end_run(self, event_type: str, fl_ctx: FLContext):
        self.run_ended = True

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        if not self.aggregating:
            self._add_contribution(shareable, fl_ctx)
            return True
        else:
            # when the aggregation is started, we no longer accept new contributions
            self.log_warning(fl_ctx, "dropped contribution while aggregating")
            return False

    def _do_accept(self):
        # This thread monitors the contribution queue.
        # It takes contributions from the queue and processes them one by one.
        self.logger.debug("Started accept thread")
        while True:
            if self.run_ended:
                # the job is already done or aborted
                self.logger.debug("run ended - exit")
                break

            try:
                # we wait very shortly when trying to get a contribution, so we can check other conditions
                contrib = self.contributions.get(timeout=_SHORT_WAIT)
            except queue.Empty:
                contrib = None

            if contrib:
                assert isinstance(contrib, _Contribution)
                self.log_debug(contrib.fl_ctx, "Accepting contribution")
                try:
                    accepted = self.aggregator.accept(contrib.data, contrib.fl_ctx)
                    self.log_debug(contrib.fl_ctx, f"{type(self.aggregator)} processed contribution: {accepted=}")
                except Exception as ex:
                    self.log_exception(
                        contrib.fl_ctx,
                        f"exception from {type(self.aggregator)} when accept: {secure_format_exception(ex)}",
                    )

            if self.aggregating and self.contributions.empty() and not self.accept_done.is_set():
                # When self.aggregating is set, the "aggregate" process is started and waiting for all contributions
                # to be accepted.
                # We set "accept_done" when all pending contributions are done.
                self.logger.debug("Finished accept for one round")
                self.accept_done.set()

        self.logger.debug("Finished accept thread")

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        self.log_debug(fl_ctx, "starting to aggregate")
        # set self.aggregating to notify the "accept" thread that we are starting aggregation.
        self.aggregating = True

        # we wait until the "accept" thread is done with pending contributions
        rc = conditional_wait(
            waiter=self.accept_done,
            timeout=self.accept_timeout,
            abort_signal=fl_ctx.get_run_abort_signal(),
            condition_cb=self._check_end_run,
        )

        if rc in [_AcceptWaitRC.ABORTED, _AcceptWaitRC.END_RUN]:
            self.log_info(fl_ctx, "skipped aggregation since job is aborted")
            return Shareable()

        if rc != _AcceptWaitRC.IS_SET:
            self.log_warning(fl_ctx, f"abnormal result {rc} waiting for accept thread")

        # we then call the aggregator to perform actual aggregation
        result = self.aggregator.aggregate(fl_ctx)

        # reset state - some controllers may not call the aggregator's reset method for historical reason
        # we call it here to make sure the aggregator state is reset.
        self._reset(fl_ctx)
        self.log_debug(fl_ctx, "Finished aggregate for one round")
        return result

    def _reset(self, fl_ctx: FLContext):
        self.aggregating = False
        self.accept_done.clear()
        self._clear_contributions()
        self.aggregator.reset(fl_ctx)

    def _check_end_run(self):
        if self.run_ended:
            return _AcceptWaitRC.END_RUN
        else:
            return _AcceptWaitRC.OK

    def reset(self, fl_ctx: FLContext):
        self._reset(fl_ctx)
