# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
    def __init__(self, topic=FED_EVENT_TOPIC, regular_interval=0.01, grace_period=2.0):
        """Init FedEventRunner.

        The FedEventRunner handles posting and receiving of fed events.
        The system will do its best to fire off all events in the queue before shutdown
        using the ABOUT_TO_END_RUN event and a grace period during END_RUN.

        Args:
            topic: the fed event topic to be handled. Defaults to 'fed.event'
        """
        Widget.__init__(self)
        self.topic = topic
        self.abort_signal = None
        self.asked_to_stop = False
        self.regular_interval = regular_interval
        self.grace_period = grace_period
        self.engine = None
        self.last_timestamps = {}  # client name => last_timestamp
        self.in_events = []
        self.in_lock = threading.Lock()
        self.poster = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.engine = fl_ctx.get_engine()
            self.engine.register_aux_message_handler(topic=self.topic, message_handle_func=self._receive)
            self.abort_signal = fl_ctx.get_run_abort_signal()
            self.asked_to_stop = False
        elif event_type == EventType.END_RUN:
            self.asked_to_stop = True
            if self.poster is not None and self.poster.is_alive():
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
            if self.poster is None:
                # create the poster thread now
                self.poster = threading.Thread(target=self._post, name="fed_event_poster")
                self.poster.start()

            last_timestamp = self.last_timestamps.get(peer_name, None)
            if last_timestamp is None or timestamp > last_timestamp:
                # we only keep new items, in case the peer somehow sent old items
                request.set_header(FedEventHeader.DIRECTION, "in")
                self.in_events.append(request)
                self.last_timestamps[peer_name] = timestamp

        # NOTE: we do not fire event here since event process could take time.
        # Instead, we simply add the package to the queue and return quickly.
        # The posting of events will be handled in the poster thread
        return make_reply(ReturnCode.OK)

    def _post(self):
        """Post an event.

        During ABOUT_TO_END_RUN, sleep_time is 0 and system will flush
         in_events by firing events without delay.

        During END_RUN, system will wait for self.grace_period, even the queue is empty,
        so any new item can be processed.

        However, since the system does not guarantee the receiving side of _post is still
        alive, we catch the exception and show warning messages to users if events can not
        be handled by receiving side.
        """
        sleep_time = self.regular_interval
        while True:
            time.sleep(sleep_time)
            if self.abort_signal.triggered:
                break
            n = len(self.in_events)
            if n > 0:
                sleep_time = 0.0
                with self.in_lock:
                    event_to_post = self.in_events.pop(0)
            elif self.asked_to_stop:
                time.sleep(self.grace_period)
                if len(self.in_events) > 0:
                    continue
                else:
                    break
            else:
                sleep_time = self.regular_interval
                continue

            with self.engine.new_context() as fl_ctx:
                if self.asked_to_stop:
                    self.log_warning(fl_ctx, f"{n} items remained in in_events.  Will stop when it reaches 0.")
                fl_ctx.set_prop(key=FLContextKey.EVENT_DATA, value=event_to_post, private=True, sticky=False)
                fl_ctx.set_prop(key=FLContextKey.EVENT_SCOPE, value=EventScope.FEDERATION, private=True, sticky=False)

                event_type = event_to_post.get_header(FedEventHeader.EVENT_TYPE)
                try:
                    self.engine.fire_event(event_type=event_type, fl_ctx=fl_ctx)
                except Exception as e:
                    if self.asked_to_stop:
                        self.log_warning(fl_ctx, f"event {event_to_post} fired unsuccessfully during END_RUN")
                    else:
                        raise e


class ServerFedEventRunner(FedEventRunner):
    def __init__(self, topic=FED_EVENT_TOPIC, regular_interval=0.01, grace_period=2.0):
        """Init ServerFedEventRunner."""
        FedEventRunner.__init__(self, topic, regular_interval, grace_period)

    def fire_and_forget_request(self, request: Shareable, fl_ctx: FLContext, targets=None):
        if not isinstance(self.engine, ServerEngineSpec):
            raise TypeError("self.engine must be ServerEngineSpec but got {}".format(type(self.engine)))
        self.engine.fire_and_forget_aux_request(
            topic=self.topic,
            targets=targets,
            request=request,
            fl_ctx=fl_ctx,
        )


class ClientFedEventRunner(FedEventRunner):
    def __init__(self, topic=FED_EVENT_TOPIC):
        """Init ClientFedEventRunner."""
        FedEventRunner.__init__(self, topic)
        self.ready = False

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        super().handle_event(event_type, fl_ctx)

        if event_type == EventType.START_RUN:
            self.ready = True

    def fire_and_forget_request(self, request: Shareable, fl_ctx: FLContext, targets=None):
        if not self.ready:
            self.log_warning(fl_ctx, "Engine in not ready, skip the fed event firing.")
            return

        if not isinstance(self.engine, ClientEngineSpec):
            raise TypeError("self.engine must be ClientEngineSpec but got {}".format(type(self.engine)))
        self.engine.fire_and_forget_aux_request(topic=self.topic, request=request, fl_ctx=fl_ctx)
