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
from collections import defaultdict

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.app_opt.p2p.executors.base_p2p_executor import BaseP2PAlgorithmExecutor


class SyncAlgorithmExecutor(BaseP2PAlgorithmExecutor):
    """An executor to implement synchronous algorithms."""
    def __init__(self):
        super().__init__()

        self.neighbors_values = defaultdict(dict)

        self.sync_waiter = threading.Event()
        self.lock = threading.Lock()

    def _exchange_values(self, fl_ctx: FLContext, value: any, iteration: int):
        engine = fl_ctx.get_engine()

        # Clear the event before starting the exchange
        self.sync_waiter.clear()

        _ = engine.send_aux_request(
            targets=[neighbor.id for neighbor in self.neighbors],
            topic="send_value",
            request=DXO(
                data_kind=DataKind.WEIGHTS,
                data={
                    "value": self._to_message(value),
                    "iteration": iteration,
                },
            ).to_shareable(),
            timeout=10,
            fl_ctx=fl_ctx,
        )

        # check if neighbors already sent their values
        if len(self.neighbors_values[iteration]) < len(self.neighbors):
            # wait for all neighbors to send their values for the current iteration
            # if not received after timeout, abort the job
            if not self.sync_waiter.wait(timeout=10):
                self.system_panic("failed to receive values from all neighbors", fl_ctx)
                return

    def _handle_neighbor_value(
        self, topic: str, request: Shareable, fl_ctx: FLContext
    ) -> Shareable:
        sender = request.get_peer_props()["__identity_name__"]
        data = from_shareable(request).data
        iteration = data["iteration"]

        with self.lock:
            self.neighbors_values[iteration][sender] = self._from_message(data["value"])
            # Check if all neighbor values have been received
            if len(self.neighbors_values[iteration]) >= len(self.neighbors):
                self.sync_waiter.set()  # Signal that we have all neighbor values
        return make_reply(ReturnCode.OK)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        super().handle_event(event_type, fl_ctx)
        if event_type == EventType.START_RUN:
            engine = fl_ctx.get_engine()

            engine.register_aux_message_handler(
                topic="send_value", message_handle_func=self._handle_neighbor_value
            )
