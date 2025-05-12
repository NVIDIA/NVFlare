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
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.edge.constants import EdgeApiStatus, EdgeEventType
from nvflare.fuel.f3.cellnet.defs import CellChannel
from nvflare.fuel.f3.cellnet.utils import make_reply
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.security.logging import secure_format_exception
from nvflare.widgets.widget import Widget


class EdgeRequestReceiver(Widget):

    def __init__(self):
        Widget.__init__(self)
        self.engine = None
        self.register_event_handler(EventType.START_RUN, self._handle_start_run)

    def _handle_start_run(self, event_type: str, fl_ctx: FLContext):
        self.engine = fl_ctx.get_engine()
        cell = self.engine.get_cell()
        cell.register_request_cb(
            channel=CellChannel.EDGE_REQUEST,
            topic="*",
            cb=self._receive_edge_request,
        )

    def _receive_edge_request(self, request: CellMessage):
        with self.engine.new_context() as fl_ctx:
            assert isinstance(fl_ctx, FLContext)
            try:
                # place the cell message into fl_ctx in case it's needed by process_edge_request.
                fl_ctx.set_prop(FLContextKey.CELL_MESSAGE, request, private=True, sticky=False)
                self.engine.fire_event(EdgeEventType.EDGE_REQUEST_RECEIVED, fl_ctx)
                exception = fl_ctx.get_prop(FLContextKey.EXCEPTIONS)
                if exception:
                    return make_reply(EdgeApiStatus.ERROR)

                reply = fl_ctx.get_prop(FLContextKey.TASK_RESULT)
                if not reply:
                    self.logger.debug("no result for edge request")
                    return make_reply(EdgeApiStatus.NO_TASK)
                else:
                    self.logger.debug("sending back edge result")
                    return make_reply(EdgeApiStatus.OK, body=reply)
            except Exception as ex:
                self.log_error(fl_ctx, f"exception from receive_edge_request: {secure_format_exception(ex)}")
                return make_reply(EdgeApiStatus.ERROR)
