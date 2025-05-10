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
from abc import abstractmethod
from typing import Optional

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.edge.constants import EdgeApiStatus, EdgeEventType, EdgeProtoKey
from nvflare.edge.simulation.simulator import Simulator
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.widgets.widget import Widget


class SimulationRunner(Widget):

    def __init__(self):
        Widget.__init__(self)
        self.simulator = None
        self.register_event_handler(EventType.ABOUT_TO_START_RUN, self._sr_about_to_start_run)
        self.register_event_handler(EventType.START_RUN, self._sr_start_run)
        self.register_event_handler(EventType.END_RUN, self._sr_end_run)

    @abstractmethod
    def create_simulator(self, fl_ctx: FLContext) -> Optional[Simulator]:
        pass

    def _sr_about_to_start_run(self, event_type: str, fl_ctx: FLContext):
        if fl_ctx.get_prop(ReservedKey.IS_LEAF):
            # devices are only for leaf nodes
            self.simulator = self.create_simulator(fl_ctx)

    def _sr_start_run(self, event_type: str, fl_ctx: FLContext):
        if not fl_ctx.get_prop(ReservedKey.IS_LEAF):
            # devices are only for leaf nodes
            return

        self.simulator.set_send_func(self._post_request, engine=fl_ctx.get_engine())
        runner = threading.Thread(target=self._run, daemon=True)
        runner.start()

    def _sr_end_run(self, event_type: str, fl_ctx: FLContext):
        if self.simulator:
            self.simulator.stop()

    def _post_request(self, request, device, engine):
        cell_msg = CellMessage(payload=request)
        with engine.new_context() as fl_ctx:
            assert isinstance(fl_ctx, FLContext)
            fl_ctx.set_prop(FLContextKey.CELL_MESSAGE, cell_msg, private=True, sticky=False)
            self.fire_event(EdgeEventType.EDGE_REQUEST_RECEIVED, fl_ctx)
            reply_dict = fl_ctx.get_prop(FLContextKey.TASK_RESULT)

            if reply_dict is None:
                # client not ready yet
                return EdgeApiStatus.OK, None

            if not isinstance(reply_dict, dict):
                raise RuntimeError(f"prop {FLContextKey.TASK_RESULT} should be dict but got {type(reply_dict)}")

            status = reply_dict.get(EdgeProtoKey.STATUS, EdgeApiStatus.OK)
            response = reply_dict.get(EdgeProtoKey.RESPONSE)
            return status, response

    def _run(self):
        self.simulator.start()
