# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.client import Client
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.fuel.f3.cellnet.defs import ReturnCode
from nvflare.private.aux_runner import AuxRunner

from .client_engine_executor_spec import ClientEngineExecutorSpec


class ClientAuxRunner(AuxRunner):
    """ClientAuxRunner to send the aux messages to the server.

    Note: The ClientEngine must create a new ClientAuxRunner object for each RUN, and make sure
    it is added as an event handler!

    """

    def __init__(self):
        """To init the ClientAuxRunner."""
        AuxRunner.__init__(self)
        self.abort_signal = None
        self.sender = None
        self.asked_to_stop = False
        self.engine = None
        self.fnf_requests = []
        self.fnf_lock = threading.Lock()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        AuxRunner.handle_event(self, event_type, fl_ctx)
        if event_type == EventType.START_RUN:
            self.engine = fl_ctx.get_engine()
            self.abort_signal = fl_ctx.get_run_abort_signal()
            self.sender = threading.Thread(target=self._send_fnf_requests, args=())
            self.sender.start()
        elif event_type == EventType.END_RUN:
            self.asked_to_stop = True
            if self.sender and self.sender.is_alive():
                self.sender.join()

    def send_aux_request(self, targets: [], topic: str, request: Shareable, timeout: float, fl_ctx: FLContext):
        if not isinstance(topic, str):
            raise TypeError("invalid topic: expects str but got {}".format(type(topic)))

        if not topic:
            raise ValueError("invalid topic: must not be empty")

        if topic == self.TOPIC_BULK:
            raise ValueError('topic value "{}" is reserved'.format(topic))

        if not isinstance(timeout, float):
            raise TypeError("invalid timeout: expects float but got {}".format(type(timeout)))

        if timeout < 0:
            raise ValueError("invalid timeout value {}: must >= 0.0".format(timeout))

        if not isinstance(fl_ctx, FLContext):
            raise TypeError("fl_ctx must be FLContext but got {}".format(type(fl_ctx)))

        req_to_send = request
        req_to_send.set_header(ReservedHeaderKey.TOPIC, topic)
        req_to_send.set_peer_props(fl_ctx.get_all_public_props())

        if timeout <= 0.0:
            # this is fire-and-forget request
            with self.fnf_lock:
                self.fnf_requests.append(req_to_send)
            return make_reply(ReturnCode.OK)

        # send regular request
        engine = fl_ctx.get_engine()
        if not isinstance(engine, ClientEngineExecutorSpec):
            raise TypeError("engine must be ClientEngineExecutorSpec, but got {}".format(type(engine)))

        target_names = []
        for t in targets:
            if isinstance(t, str):
                name = t
            elif isinstance(t, Client):
                name = t.name
            else:
                raise ValueError("invalid target in list: got {}".format(type(t)))

            if name not in target_names:
                target_names.append(t)

        valid_targets, invalid_names = engine.validate_clients(target_names, fl_ctx)
        if invalid_names:
            raise ValueError("invalid target(s): {}".format(invalid_names))

        reply = engine.aux_send(targets=valid_targets, topic=topic, request=req_to_send, timeout=timeout, fl_ctx=fl_ctx)

        return reply

    def _send_fnf_requests(self):
        topic = self.TOPIC_BULK
        sleep_time = 0.5
        while True:
            time.sleep(sleep_time)
            if self.abort_signal.triggered:
                break

            if len(self.fnf_requests) <= 0:
                if self.asked_to_stop:
                    break
                else:
                    sleep_time = 1.0
                    continue

            with self.engine.new_context() as fl_ctx:
                bulk = Shareable()
                bulk.set_header(ReservedHeaderKey.TOPIC, topic)
                bulk.set_peer_props(fl_ctx.get_all_public_props())
                with self.fnf_lock:
                    bulk[self.DATA_KEY_BULK] = self.fnf_requests
                    reply = self.engine.aux_send(targets=None, topic=topic, request=bulk, timeout=15.0, fl_ctx=fl_ctx)
                    rc = reply.get_return_code()
                    if rc != ReturnCode.COMMUNICATION_ERROR:
                        # if communication error we'll retry
                        self.fnf_requests = []
            sleep_time = 0.5
