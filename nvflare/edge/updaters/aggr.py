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
from typing import Optional

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.edge.updater import Updater
from nvflare.fuel.utils.validation_utils import check_positive_int, check_str


class AggrUpdater(Updater):

    def __init__(self, aggregator_id: str, min_accepted=2):
        """Constructor of AggrUpdater.
        AggrUpdater implements required logic by using an Aggregator.

        Args:
            aggregator_id: component ID of the aggregator
            min_accepted: minimum updates required before aggregating.
        """
        Updater.__init__(self)
        self.aggregator_id = aggregator_id
        self.aggregator = None
        self.num_accepted = 0
        self.min_accepted = min_accepted
        self.aggr_lock = threading.Lock()

        check_str("aggregator_id", aggregator_id)
        check_positive_int("min_accepted", min_accepted)

        self.register_event_handler(EventType.START_RUN, self._handle_start_run)

    def _handle_start_run(self, event_type: str, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        aggr = engine.get_component(self.aggregator_id)
        if not isinstance(aggr, Aggregator):
            self.system_panic(
                f"aggregator {self.aggregator_id} must be an Aggregator type object but got {type(aggr)}",
                fl_ctx,
            )
            return
        self.aggregator = aggr

    def process_parent_update_reply(self, reply: Shareable, fl_ctx: FLContext):
        # do not update my state.
        return

    def prepare_update_for_parent(self, fl_ctx: FLContext) -> Shareable:
        # return aggregation result of the aggregator
        with self.aggr_lock:
            if self.num_accepted >= self.min_accepted:
                # only when we have accepted enough updates from children
                update = self.aggregator.aggregate(fl_ctx)
                self.aggregator.reset(fl_ctx)
                self.num_accepted = 0
            else:
                # otherwise we don't update the parent
                update = None
        return update

    def process_child_update(self, update: Shareable, fl_ctx: FLContext) -> (bool, Optional[Shareable]):
        # use the aggregator to accept the update
        self.log_info(fl_ctx, f"accepting child update by {type(self.aggregator)}")
        with self.aggr_lock:
            accepted = self.aggregator.accept(update, fl_ctx)
            self.log_info(fl_ctx, f"done child update: {accepted=}")
            if accepted:
                self.num_accepted += 1
        return accepted, None
