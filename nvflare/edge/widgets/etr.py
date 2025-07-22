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
from nvflare.edge.constants import EdgeContextKey, EdgeEventType, EdgeMsgTopic
from nvflare.fuel.f3.cellnet.cell import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.defs import CellChannel
from nvflare.fuel.f3.cellnet.utils import make_reply
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.security.logging import secure_format_exception
from nvflare.widgets.widget import Widget


class EdgeTaskReceiver(Widget):
    """Edge Task Receiver (ETR) is to be used to receive edge requests dispatched from CP.
    ETR must be installed on CJ as a component in config_fed_client.json.

    Note: ETR does not process edge requests directly. It's another component's responsibility (e.g. ETE)
    to process edge requests.

    ETR indirectly interacts with request-processing component (also installed on the CJ) via Flare Events:
        EdgeEventType.EDGE_REQUEST_RECEIVED for receiving task requests;

    """

    def __init__(self):
        Widget.__init__(self)
        self.engine = None
        self.register_event_handler(EventType.START_RUN, self._handle_start_run)

    def _handle_start_run(self, event_type: str, fl_ctx: FLContext):
        self.engine = fl_ctx.get_engine()
        cell = self.engine.get_cell()

        cell.register_request_cb(
            channel=CellChannel.EDGE_REQUEST,
            topic=EdgeMsgTopic.TASK_REQUEST,
            cb=self._receive_edge_request,
            event_type=EdgeEventType.EDGE_TASK_REQUEST_RECEIVED,
        )

        cell.register_request_cb(
            channel=CellChannel.EDGE_REQUEST,
            topic=EdgeMsgTopic.SELECTION_REQUEST,
            cb=self._receive_edge_request,
            event_type=EdgeEventType.EDGE_SELECTION_REQUEST_RECEIVED,
        )

        cell.register_request_cb(
            channel=CellChannel.EDGE_REQUEST,
            topic=EdgeMsgTopic.RESULT_REPORT,
            cb=self._receive_edge_request,
            event_type=EdgeEventType.EDGE_RESULT_REPORT_RECEIVED,
        )

    def _receive_edge_request(self, request: CellMessage, event_type: str):
        with self.engine.new_context() as fl_ctx:
            assert isinstance(fl_ctx, FLContext)
            try:
                # place the cell message into fl_ctx in case it's needed by process_edge_request.
                fl_ctx.set_prop(EdgeContextKey.REQUEST_FROM_EDGE, request.payload, private=True, sticky=False)
                self.engine.fire_event(event_type, fl_ctx)
                exception = fl_ctx.get_prop(FLContextKey.EXCEPTIONS)
                if exception:
                    return make_reply(CellReturnCode.PROCESS_EXCEPTION)

                reply = fl_ctx.get_prop(EdgeContextKey.REPLY_TO_EDGE)
                if not reply:
                    self.logger.debug("no reply for edge request")
                    return make_reply(CellReturnCode.PROCESS_EXCEPTION)
                else:
                    self.logger.debug("sending back edge result")
                    return make_reply(CellReturnCode.OK, body=reply)
            except Exception as ex:
                self.log_error(fl_ctx, f"exception from receive_edge_request: {secure_format_exception(ex)}")
                return make_reply(CellReturnCode.PROCESS_EXCEPTION)
