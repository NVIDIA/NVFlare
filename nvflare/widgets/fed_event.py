# Copyright (c) 2021, NVIDIA CORPORATION.
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

from nvflare.apis.client_engine_spec import ClientEngineSpec
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import EventScope, FedEventHeader, FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.widgets.widget import Widget

FED_EVENT_TOPIC = "fed.event"


class FedEventRunner(Widget):
    def __init__(self, topic=FED_EVENT_TOPIC):
        """
        Args:
            topic:
        """
        Widget.__init__(self)
        self.topic = topic
        self.abort_signal = None
        self.asked_to_stop = False
        self.engine = None
        self.last_timestamps = {}  # client name => last_timestamp
        self.in_events = []
        self.in_lock = threading.Lock()
        self.poster = threading.Thread(target=self._post, args=())

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.engine = fl_ctx.get_engine()
            self.engine.register_aux_message_handler(topic=self.topic, message_handle_func=self._receive)
            self.abort_signal = fl_ctx.get_run_abort_signal()
            self.asked_to_stop = False
            self.poster.start()
        elif event_type == EventType.END_RUN:
            self.asked_to_stop = True
            if self.poster.is_alive():
                self.poster.join()
        else:
            # handle outgoing fed events
            event_scope = fl_ctx.get_prop(key=FLContextKey.EVENT_SCOPE, default=EventScope.LOCAL)
            if event_scope != EventScope.FEDERATION:
                return

            event_data = fl_ctx.get_prop(FLContextKey.EVENT_DATA, None)
            if not isinstance(event_data, Shareable):
                self.log_error(fl_ctx, "bad fed event: expect data to be Shareable but got {}".format(type(event_data)))
                return

            direction = event_data.get_header(FedEventHeader.DIRECTION, "out")
            if direction != "out":
                # ignore incoming events
                return

            event_data.set_header(FedEventHeader.EVENT_TYPE, event_type)
            event_data.set_header(FedEventHeader.ORIGIN, fl_ctx.get_identity_name())
            event_data.set_header(FedEventHeader.TIMESTAMP, time.time())

            targets = event_data.get_header(FedEventHeader.TARGETS, None)
            self.fire_and_forget_request(request=event_data, fl_ctx=fl_ctx, targets=targets)

    def fire_and_forget_request(self, request: Shareable, fl_ctx: FLContext, targets=None):
        pass

    def _receive(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        peer_name = request.get_peer_prop(ReservedKey.IDENTITY_NAME, None)
        if not peer_name:
            self.log_error(fl_ctx, "missing identity name of the data sender")
            return make_reply(ReturnCode.MISSING_PEER_CONTEXT)

        timestamp = request.get_header(FedEventHeader.TIMESTAMP, None)
        if timestamp is None:
            self.log_error(fl_ctx, "missing timestamp in incoming fed event")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        event_type = request.get_header(FedEventHeader.EVENT_TYPE, None)
        if event_type is None:
            self.log_error(fl_ctx, "missing event_type in incoming fed event")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        with self.in_lock:
            last_timestamp = self.last_timestamps.get(peer_name, None)
            if last_timestamp is None or timestamp > last_timestamp:
                # we only keep new items, in case the peer somehow sent old items
                request.set_header(FedEventHeader.DIRECTION, "in")
                self.in_events.append(request)
                self.last_timestamps[peer_name] = timestamp

        # NOTE: we do not fire event here since event process could take time.
        # Instead we simply add the package to the queue and return quickly.
        # The posting of events will be handled in the poster thread
        return make_reply(ReturnCode.OK)

    def _post(self):
        sleep_time = 0.1
        while True:
            time.sleep(sleep_time)
            if self.asked_to_stop or self.abort_signal.triggered:
                break

            with self.in_lock:
                if len(self.in_events) <= 0:
                    continue
                event_to_post = self.in_events.pop(0)
                assert isinstance(event_to_post, Shareable)

            if self.asked_to_stop or self.abort_signal.triggered:
                break

            with self.engine.new_context() as fl_ctx:
                assert isinstance(fl_ctx, FLContext)

                fl_ctx.set_prop(key=FLContextKey.EVENT_DATA, value=event_to_post, private=True, sticky=False)
                fl_ctx.set_prop(key=FLContextKey.EVENT_SCOPE, value=EventScope.FEDERATION, private=True, sticky=False)

                event_type = event_to_post.get_header(FedEventHeader.EVENT_TYPE)
                self.engine.fire_event(event_type=event_type, fl_ctx=fl_ctx)


class ServerFedEventRunner(FedEventRunner):
    def __init__(self, topic=FED_EVENT_TOPIC):
        FedEventRunner.__init__(self, topic)

    def fire_and_forget_request(self, request: Shareable, fl_ctx: FLContext, targets=None):
        assert isinstance(self.engine, ServerEngineSpec)
        self.engine.fire_and_forget_aux_request(
            topic=self.topic,
            targets=targets,
            request=request,
            fl_ctx=fl_ctx,
        )


class ClientFedEventRunner(FedEventRunner):
    def __init__(self, topic=FED_EVENT_TOPIC):
        FedEventRunner.__init__(self, topic)

    def fire_and_forget_request(self, request: Shareable, fl_ctx: FLContext, targets=None):
        assert isinstance(self.engine, ClientEngineSpec)
        self.engine.fire_and_forget_aux_request(topic=self.topic, request=request, fl_ctx=fl_ctx)
